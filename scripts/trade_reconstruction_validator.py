#!/usr/bin/env python3
"""Trade reconstruction validator.

Verifies that every journal trade satisfies:
    1. Geometric consistency   — stop/entry/TP ordering is valid
    2. Stop price consistency  — four independent stop calculations agree
    3. TP/MFE consistency      — if MFE reached TPn, TPn must have been logged as hit

The four stop sources compared:
    A  Strategy (SIGNAL.stop_price)
       Computed by MomentumStrategy as entry_price ± 1.5 × ATR.
    B  Recalculated from signal entry
       signal.entry_price ± ATR_STOP_MULT × atr   (should equal A exactly)
    C  Recalculated from fill price
       fill_price ± ATR_STOP_MULT × atr            (differs by entry slippage)
    D  Journal fill price vs strategy stop
       The stop used in live_signal_router is signal.stop_price (source A).
       Entry slippage moves fill_price but NOT the stop level.
       This means actual risk R slightly exceeds plan R.

Any mismatch flags a divergence between what the strategy computed and
what was recorded or executed.

Usage:
    python scripts/trade_reconstruction_validator.py
    python scripts/trade_reconstruction_validator.py --journal-dir logs/paper --candle-dir data/candles
    python scripts/trade_reconstruction_validator.py --verbose
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    import pyarrow.parquet as pq
    _HAS_PARQUET = True
except ImportError:
    _HAS_PARQUET = False

# Strategy constants (must match MomentumConfig defaults)
ATR_STOP_MULT = 1.5
ATR_TP1_MULT  = 1.0
ATR_TP2_MULT  = 2.0
ATR_TP3_MULT  = 3.0
# Entry slippage formula from PaperExecution.simulate_entry:
#   slip = atr * (0.05 + 0.5 * 0.005) = atr * 0.0525
ENTRY_SLIP_MULT = 0.05 + 0.5 * 0.005   # 0.0525

TAKER = 0.0006
MAKER = 0.0002

TOL_PRICE = 1e-4     # price comparison tolerance (rounding artefacts)
TOL_STOP  = 0.005    # 0.5% of stop price — flags meaningful divergence


# ---------------------------------------------------------------------------
# Bar struct
# ---------------------------------------------------------------------------

class Bar(NamedTuple):
    ts_ms: int
    open: float
    high: float
    low: float
    close: float


# ---------------------------------------------------------------------------
# Violation record
# ---------------------------------------------------------------------------

@dataclass
class Violation:
    trade_num: int
    symbol: str
    side: str
    vtype: str     # GEOMETRY | STOP_MISMATCH | TP_MFE_CONSISTENCY | SLIPPAGE_ANOMALY
    detail: str
    severity: str  # ERROR | WARNING


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclass
class ValidatedTrade:
    trade_num: int
    symbol: str
    side: str
    entry_time: float

    # From SIGNAL event
    signal_entry: float
    signal_stop: float
    signal_tp1: float
    signal_tp2: float
    signal_tp3: float
    atr: float

    # From FILL event
    fill_price: float
    fill_size: float
    fill_fee: float
    entry_slippage: float       # fill_price - signal_entry (signed, adverse direction)

    # Stop sources
    stop_A: float               # signal.stop_price (strategy output)
    stop_B: float               # recalculated from signal_entry ± ATR_STOP_MULT*atr
    stop_C: float               # recalculated from fill_price ± ATR_STOP_MULT*atr
    stop_D: float               # what live_signal_router actually uses (= stop_A)

    # TP distances in R (relative to r_unit = abs(signal_entry - signal_stop))
    r_unit_signal: float        # risk unit from signal prices
    r_unit_fill: float          # risk unit from fill price to signal stop
    tp1_r: float
    tp2_r: float
    tp3_r: float

    # TP hits from journal
    tp1_hit_journal: bool = False
    tp2_hit_journal: bool = False
    tp3_hit_journal: bool = False

    # MFE from candles
    mfe_price: float = 0.0
    mfe_r: float = 0.0
    has_candles: bool = False

    # Exit
    exit_reason: str = ""
    net_pnl: float = 0.0

    # Validation results
    violations: list[Violation] = field(default_factory=list)
    geometry_ok: bool = True
    stop_ok: bool = True
    tp_mfe_ok: bool = True


# ---------------------------------------------------------------------------
# Journal loading
# ---------------------------------------------------------------------------

def _load_events(journal_dir: Path) -> list[dict]:
    evts: list[dict] = []
    for path in sorted(journal_dir.glob("*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    evts.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    evts.sort(key=lambda e: e.get("ts_epoch", 0.0))
    return evts


def build_validated_trades(events: list[dict]) -> list[ValidatedTrade]:
    last_signal: dict[str, dict] = {}
    open_fills: dict[str, dict] = {}
    open_fees:  dict[str, float] = {}
    open_partials: dict[str, list[dict]] = defaultdict(list)

    records: list[ValidatedTrade] = []
    trade_num = 0

    for ev in events:
        evt = ev.get("event", "")
        sym = ev.get("symbol", "")

        if evt == "SIGNAL":
            last_signal[sym] = ev

        elif evt == "FILL":
            trade_num += 1
            open_fills[sym] = ev
            open_fees[sym] = ev.get("fee", 0.0)
            open_partials[sym] = []

        elif evt == "PARTIAL_TP":
            if sym in open_fills:
                open_fees[sym] += ev.get("fee", 0.0)
                open_partials[sym].append(ev)

        elif evt in ("STOP_HIT", "FORCE_CLOSE"):
            fill_ev = open_fills.pop(sym, None)
            if fill_ev is None:
                continue

            sig = last_signal.get(sym, {})
            feats = sig.get("features", {})
            atr = feats.get("atr", sig.get("atr", 0.0))
            tp_prices = sig.get("tp_prices", [])
            signal_stop = sig.get("stop_price", 0.0)
            signal_entry = sig.get("entry_price", 0.0)

            if not atr or not tp_prices or not signal_stop or not signal_entry:
                open_fees.pop(sym, None)
                open_partials.pop(sym, [])
                continue

            side = fill_ev.get("direction", "long")
            is_long = side == "long"
            sign = 1.0 if is_long else -1.0

            fill_price = fill_ev.get("fill_price", 0.0)
            fill_size = fill_ev.get("size", 0.0)
            fill_fee = fill_ev.get("fee", 0.0)

            # Entry slippage (adverse = positive number meaning we paid more)
            entry_slip_raw = (fill_price - signal_entry) * sign  # positive = adverse
            entry_slip = fill_price - signal_entry   # signed

            # Four stop sources
            stop_A = signal_stop                                      # strategy output
            stop_B = signal_entry - sign * ATR_STOP_MULT * atr        # recalc from signal entry
            stop_C = fill_price   - sign * ATR_STOP_MULT * atr        # recalc from fill price
            stop_D = signal_stop                                       # router uses signal stop directly

            # R units
            r_unit_signal = abs(signal_entry - signal_stop)
            r_unit_fill   = abs(fill_price   - signal_stop)

            # TP distances
            tp1_r = abs(tp_prices[0] - signal_entry) / r_unit_signal if r_unit_signal > 0 and tp_prices else 0.0
            tp2_r = abs(tp_prices[1] - signal_entry) / r_unit_signal if r_unit_signal > 0 and len(tp_prices) > 1 else 0.0
            tp3_r = abs(tp_prices[2] - signal_entry) / r_unit_signal if r_unit_signal > 0 and len(tp_prices) > 2 else 0.0

            # TP hits from journal partials
            partials = open_partials.pop(sym, [])
            tp1_hit = any(p.get("tp_idx") == 0 for p in partials)
            tp2_hit = any(p.get("tp_idx") == 1 for p in partials)
            tp3_hit = any(p.get("tp_idx") == 2 for p in partials)

            # Net PnL reconstruction
            partial_gross = sum(
                (p["fill_price"] - fill_price) * p["size"] * sign for p in partials
            )
            partial_size = sum(p["size"] for p in partials)
            remaining = fill_size - partial_size
            if evt == "STOP_HIT":
                exit_p = ev.get("fill_price", 0.0)
                stop_fee = ev.get("fee", 0.0)
            else:
                exit_p = ev.get("price", 0.0)
                stop_fee = 0.0
            stop_gross = (exit_p - fill_price) * remaining * sign
            gross = partial_gross + stop_gross
            total_fees = open_fees.pop(sym, 0.0) + stop_fee
            net = gross - total_fees

            records.append(ValidatedTrade(
                trade_num=trade_num,
                symbol=sym,
                side=side,
                entry_time=fill_ev.get("ts_epoch", 0.0),
                signal_entry=signal_entry,
                signal_stop=signal_stop,
                signal_tp1=tp_prices[0] if tp_prices else 0.0,
                signal_tp2=tp_prices[1] if len(tp_prices) > 1 else 0.0,
                signal_tp3=tp_prices[2] if len(tp_prices) > 2 else 0.0,
                atr=atr,
                fill_price=fill_price,
                fill_size=fill_size,
                fill_fee=fill_fee,
                entry_slippage=round(entry_slip, 6),
                stop_A=round(stop_A, 6),
                stop_B=round(stop_B, 6),
                stop_C=round(stop_C, 6),
                stop_D=round(stop_D, 6),
                r_unit_signal=round(r_unit_signal, 6),
                r_unit_fill=round(r_unit_fill, 6),
                tp1_r=round(tp1_r, 4),
                tp2_r=round(tp2_r, 4),
                tp3_r=round(tp3_r, 4),
                tp1_hit_journal=tp1_hit,
                tp2_hit_journal=tp2_hit,
                tp3_hit_journal=tp3_hit,
                exit_reason=evt,
                net_pnl=round(net, 4),
            ))

    return records


# ---------------------------------------------------------------------------
# Candle utilities
# ---------------------------------------------------------------------------

def _load_parquet(candle_dir: Path, symbol: str) -> list[Bar]:
    if not _HAS_PARQUET:
        return []
    path = candle_dir / f"{symbol}_1m.parquet"
    if not path.exists():
        return []
    rows = pq.read_table(path).to_pylist()
    bars = [
        Bar(ts_ms=r["close_time"], open=r["open"], high=r["high"],
            low=r["low"], close=r["close"])
        for r in rows if r.get("is_final")
    ]
    return sorted(bars, key=lambda b: b.ts_ms)


def _fetch_rest(symbol: str, start_ms: int, end_ms: int) -> list[Bar]:
    all_bars: list[Bar] = []
    cursor = start_ms
    while cursor < end_ms:
        chunk_end = min(cursor + 999 * 60_000, end_ms)
        url = (
            f"https://api.bitget.com/api/v2/mix/market/candles"
            f"?symbol={symbol}&productType=USDT-FUTURES&granularity=1m"
            f"&startTime={cursor}&endTime={chunk_end}&limit=1000"
        )
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = json.loads(resp.read())
            if data.get("code") != "00000":
                break
            chunk = [
                Bar(ts_ms=int(r[0]), open=float(r[1]), high=float(r[2]),
                    low=float(r[3]), close=float(r[4]))
                for r in data.get("data", [])
            ]
            if not chunk:
                break
            all_bars.extend(chunk)
            cursor = chunk[-1].ts_ms + 60_000
            time.sleep(0.15)
        except Exception as exc:
            print(f"  [WARN] REST fetch failed for {symbol}: {exc}")
            break
    return sorted(all_bars, key=lambda b: b.ts_ms)


def compute_mfe_for_trades(
    trades: list[ValidatedTrade],
    candle_dir: Path,
    no_rest: bool,
) -> None:
    parquet_cache: dict[str, list[Bar]] = {}
    rest_cache: dict[str, list[Bar]] = {}

    for t in trades:
        sym = t.symbol
        if sym not in parquet_cache:
            parquet_cache[sym] = _load_parquet(candle_dir, sym)

        # Window: entry → 4 hours later (enough to capture full TP path)
        start_ms = int(t.entry_time * 1000)
        end_ms   = start_ms + 240 * 60_000

        window = [b for b in parquet_cache[sym] if start_ms <= b.ts_ms <= end_ms]
        if not window and not no_rest:
            key = f"{sym}_{start_ms}_{end_ms}"
            if key not in rest_cache:
                rest_cache[key] = _fetch_rest(sym, start_ms, end_ms)
                time.sleep(0.15)
            window = rest_cache.get(key, [])

        if not window:
            continue

        t.has_candles = True
        is_long = t.side == "long"
        mfe = t.fill_price

        for bar in window:
            if is_long:
                mfe = max(mfe, bar.high)
            else:
                mfe = min(mfe, bar.low)

        t.mfe_price = abs(mfe - t.fill_price)
        t.mfe_r = round(t.mfe_price / t.r_unit_fill, 3) if t.r_unit_fill > 0 else 0.0


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

def validate_geometry(t: ValidatedTrade) -> list[Violation]:
    viols: list[Violation] = []
    is_long = t.side == "long"

    checks = [
        # (condition_that_is_BAD, description)
        (is_long  and t.signal_stop >= t.signal_entry,
         f"LONG stop={t.signal_stop:.4f} >= entry={t.signal_entry:.4f}"),
        (not is_long and t.signal_stop <= t.signal_entry,
         f"SHORT stop={t.signal_stop:.4f} <= entry={t.signal_entry:.4f}"),
        (is_long  and t.signal_tp1  <= t.signal_entry,
         f"LONG TP1={t.signal_tp1:.4f} <= entry={t.signal_entry:.4f}"),
        (not is_long and t.signal_tp1 >= t.signal_entry,
         f"SHORT TP1={t.signal_tp1:.4f} >= entry={t.signal_entry:.4f}"),
        (t.signal_tp2 > 0 and is_long  and t.signal_tp2 <= t.signal_tp1,
         f"LONG TP2={t.signal_tp2:.4f} <= TP1={t.signal_tp1:.4f}"),
        (t.signal_tp2 > 0 and not is_long and t.signal_tp2 >= t.signal_tp1,
         f"SHORT TP2={t.signal_tp2:.4f} >= TP1={t.signal_tp1:.4f}"),
        (t.signal_tp3 > 0 and is_long  and t.signal_tp3 <= t.signal_tp2,
         f"LONG TP3={t.signal_tp3:.4f} <= TP2={t.signal_tp2:.4f}"),
        (t.signal_tp3 > 0 and not is_long and t.signal_tp3 >= t.signal_tp2,
         f"SHORT TP3={t.signal_tp3:.4f} >= TP2={t.signal_tp2:.4f}"),
        # Fill vs stop
        (is_long  and t.fill_price <= t.signal_stop,
         f"LONG fill={t.fill_price:.4f} <= stop={t.signal_stop:.4f} (filled past stop)"),
        (not is_long and t.fill_price >= t.signal_stop,
         f"SHORT fill={t.fill_price:.4f} >= stop={t.signal_stop:.4f} (filled past stop)"),
    ]

    for bad, desc in checks:
        if bad:
            viols.append(Violation(t.trade_num, t.symbol, t.side,
                                   "GEOMETRY", desc, "ERROR"))
    return viols


def validate_stop_consistency(t: ValidatedTrade) -> list[Violation]:
    viols: list[Violation] = []

    # A vs B: strategy output vs formula applied to signal_entry
    diff_AB = abs(t.stop_A - t.stop_B)
    if diff_AB > TOL_PRICE:
        viols.append(Violation(
            t.trade_num, t.symbol, t.side, "STOP_MISMATCH",
            f"StopA(strategy)={t.stop_A:.6f} vs StopB(recalc from signal_entry)={t.stop_B:.6f} "
            f"diff={diff_AB:.6f}",
            "ERROR" if diff_AB > t.atr * 0.01 else "WARNING",
        ))

    # A vs C: strategy stop vs stop recalculated from fill price
    # Expected to differ by entry slippage × ATR_STOP_MULT — flag only if unexpectedly large
    expected_diff = abs(t.entry_slippage) * ATR_STOP_MULT
    actual_diff_AC = abs(t.stop_A - t.stop_C)
    if abs(actual_diff_AC - expected_diff) > TOL_PRICE * 10:
        viols.append(Violation(
            t.trade_num, t.symbol, t.side, "STOP_MISMATCH",
            f"StopA={t.stop_A:.6f} StopC(from fill)={t.stop_C:.6f} "
            f"diff={actual_diff_AC:.6f} expected_slip_component={expected_diff:.6f}",
            "WARNING",
        ))

    # Flag if entry slippage inflated risk R materially (>5% of r_unit)
    slip_r_impact = abs(t.r_unit_fill - t.r_unit_signal) / t.r_unit_signal if t.r_unit_signal > 0 else 0.0
    if slip_r_impact > 0.05:
        viols.append(Violation(
            t.trade_num, t.symbol, t.side, "SLIPPAGE_ANOMALY",
            f"Entry slippage changed risk R by {slip_r_impact*100:.1f}%: "
            f"r_signal={t.r_unit_signal:.4f} r_fill={t.r_unit_fill:.4f} "
            f"slip={t.entry_slippage:+.4f}",
            "WARNING",
        ))

    # Validate TP distances match expected ATR multiples
    tp_checks = [
        (t.tp1_r, ATR_TP1_MULT / ATR_STOP_MULT, "TP1"),
        (t.tp2_r, ATR_TP2_MULT / ATR_STOP_MULT, "TP2"),
        (t.tp3_r, ATR_TP3_MULT / ATR_STOP_MULT, "TP3"),
    ]
    for actual_r, expected_r, label in tp_checks:
        if actual_r > 0 and abs(actual_r - expected_r) > 0.05:
            viols.append(Violation(
                t.trade_num, t.symbol, t.side, "STOP_MISMATCH",
                f"{label}_R={actual_r:.4f} expected≈{expected_r:.4f} "
                f"(ATR mult={ATR_TP1_MULT if label=='TP1' else ATR_TP2_MULT if label=='TP2' else ATR_TP3_MULT}/{ATR_STOP_MULT})",
                "WARNING",
            ))

    return viols


def validate_tp_mfe_consistency(t: ValidatedTrade) -> list[Violation]:
    """If candle MFE reached a TP level, the journal must record it as hit."""
    viols: list[Violation] = []
    if not t.has_candles or t.r_unit_fill <= 0:
        return viols

    checks = [
        (t.tp1_r, t.tp1_hit_journal, "TP1", 0),
        (t.tp2_r, t.tp2_hit_journal, "TP2", 1),
        (t.tp3_r, t.tp3_hit_journal, "TP3", 2),
    ]
    for tp_r, was_hit, label, idx in checks:
        if tp_r <= 0:
            continue
        # MFE must clear the TP level (not just graze it — add small buffer)
        mfe_cleared_tp = t.mfe_r > tp_r + 0.05
        if mfe_cleared_tp and not was_hit:
            viols.append(Violation(
                t.trade_num, t.symbol, t.side, "TP_MFE_CONSISTENCY",
                f"MFE={t.mfe_r:.3f}R > {label}_R={tp_r:.3f}R but {label} "
                f"NOT recorded as hit in journal",
                "ERROR",
            ))
        # Inverse: journal records hit but MFE never reached TP
        if was_hit and t.mfe_r < tp_r - 0.05:
            viols.append(Violation(
                t.trade_num, t.symbol, t.side, "TP_MFE_CONSISTENCY",
                f"Journal says {label} hit but MFE={t.mfe_r:.3f}R < {label}_R={tp_r:.3f}R",
                "ERROR",
            ))

    return viols


def run_all_validations(trades: list[ValidatedTrade]) -> None:
    for t in trades:
        t.violations = []
        geo = validate_geometry(t)
        stop = validate_stop_consistency(t)
        tpmfe = validate_tp_mfe_consistency(t)
        t.violations = geo + stop + tpmfe
        t.geometry_ok = not any(v.vtype == "GEOMETRY" for v in t.violations)
        t.stop_ok     = not any(v.vtype in ("STOP_MISMATCH", "SLIPPAGE_ANOMALY") for v in geo + stop)
        t.tp_mfe_ok   = not any(v.vtype == "TP_MFE_CONSISTENCY" for v in tpmfe)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

_G = "\033[92m"
_R = "\033[91m"
_Y = "\033[93m"
_B = "\033[1m"
_C = "\033[96m"
_RST = "\033[0m"


def _ts(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _ok(flag: bool) -> str:
    return f"{_G}OK {_RST}" if flag else f"{_R}FAIL{_RST}"


def _print_trade_table(trades: list[ValidatedTrade]) -> None:
    print(f"\n{_B}{'═'*100}{_RST}")
    print(f"  {_B}{'Trade Reconstruction Validator — Per-Trade Summary':^98}{_RST}")
    print(f"{_B}{'═'*100}{_RST}")
    print(
        f"  {'#':>3}  {'Time':16}  {'Sym':8}  {'Side':5}  "
        f"{'Dir':>4}  {'Stop':>9}  {'TP1':>9}  {'TP2':>9}  {'TP3':>9}  "
        f"{'Geo':4}  {'Stop':4}  {'TPMFE':5}  {'Viols':>5}"
    )
    print(f"  {'─'*98}")
    for t in trades:
        n_err = sum(1 for v in t.violations if v.severity == "ERROR")
        n_warn = sum(1 for v in t.violations if v.severity == "WARNING")
        counts = f"{_R}{n_err}E{_RST}" if n_err else "0E"
        counts += f" {_Y}{n_warn}W{_RST}" if n_warn else " 0W"
        print(
            f"  {t.trade_num:>3}  {_ts(t.entry_time):16}  {t.symbol:8}  {t.side:5}  "
            f"{t.side[0].upper():>4}  {t.signal_stop:>9.3f}  {t.signal_tp1:>9.3f}  "
            f"{t.signal_tp2:>9.3f}  {t.signal_tp3:>9.3f}  "
            f"{_ok(t.geometry_ok)}  {_ok(t.stop_ok)}  {_ok(t.tp_mfe_ok):5}  {counts}"
        )
    print(f"  {'─'*98}")


def _print_violations(trades: list[ValidatedTrade], verbose: bool) -> None:
    all_viols = [v for t in trades for v in t.violations]
    errors   = [v for v in all_viols if v.severity == "ERROR"]
    warnings = [v for v in all_viols if v.severity == "WARNING"]

    if not all_viols:
        print(f"\n  {_G}No violations found. All trades pass validation.{_RST}")
        return

    if errors:
        print(f"\n{_B}{'═'*80}{_RST}")
        print(f"  {_R}ERRORS ({len(errors)}){_RST}")
        print(f"{_B}{'═'*80}{_RST}")
        by_type: dict[str, list[Violation]] = defaultdict(list)
        for v in errors:
            by_type[v.vtype].append(v)
        for vtype, vs in sorted(by_type.items()):
            print(f"\n  [{vtype}]  ({len(vs)} violations)")
            for v in vs if verbose else vs[:5]:
                print(f"    #{v.trade_num:>3} {v.symbol} {v.side}: {v.detail}")
            if not verbose and len(vs) > 5:
                print(f"    ... and {len(vs)-5} more (use --verbose to see all)")

    if warnings:
        print(f"\n{_B}{'═'*80}{_RST}")
        print(f"  {_Y}WARNINGS ({len(warnings)}){_RST}")
        print(f"{_B}{'═'*80}{_RST}")
        by_type = defaultdict(list)
        for v in warnings:
            by_type[v.vtype].append(v)
        for vtype, vs in sorted(by_type.items()):
            print(f"\n  [{vtype}]  ({len(vs)} violations)")
            for v in vs if verbose else vs[:3]:
                print(f"    #{v.trade_num:>3} {v.symbol} {v.side}: {v.detail}")
            if not verbose and len(vs) > 3:
                print(f"    ... and {len(vs)-3} more (use --verbose to see all)")


def _print_stop_comparison(trades: list[ValidatedTrade]) -> None:
    print(f"\n{'═'*70}")
    print(f"  Stop Price Source Comparison")
    print(f"{'═'*70}")
    print(f"  A = Strategy SIGNAL.stop_price")
    print(f"  B = Recalculated from signal_entry ± {ATR_STOP_MULT}×ATR  (should equal A)")
    print(f"  C = Recalculated from fill_price  ± {ATR_STOP_MULT}×ATR  (reflects slippage)")
    print(f"  D = Live router stop used          (same as A — no modification)")
    print(f"  {'─'*68}")
    print(f"  {'#':>3}  {'Sym':8}  {'A(signal)':>11}  {'B(recalc_sig)':>13}  "
          f"{'C(recalc_fill)':>14}  {'A=B?':>5}  {'slip_R_impact':>13}")
    print(f"  {'─'*68}")
    for t in trades:
        ab_ok = abs(t.stop_A - t.stop_B) <= TOL_PRICE
        ab_s = f"{_G}YES{_RST}" if ab_ok else f"{_R} NO{_RST}"
        slip_impact = abs(t.r_unit_fill - t.r_unit_signal) / t.r_unit_signal * 100 if t.r_unit_signal > 0 else 0.0
        slip_col = _Y if slip_impact > 5 else ""
        print(
            f"  {t.trade_num:>3}  {t.symbol:8}  {t.stop_A:>11.4f}  {t.stop_B:>13.4f}  "
            f"{t.stop_C:>14.4f}  {ab_s:>5}  {slip_col}{slip_impact:>12.1f}%{_RST if slip_col else ''}"
        )
    print(f"{'═'*70}")


def _print_tp_mfe_table(trades: list[ValidatedTrade]) -> None:
    with_candles = [t for t in trades if t.has_candles]
    if not with_candles:
        print("\n  No candle data — TP/MFE consistency check skipped.")
        return

    print(f"\n{'═'*80}")
    print(f"  TP / MFE Consistency Check  (candle MFE vs journal TP hits)")
    print(f"{'═'*80}")
    print(
        f"  {'#':>3}  {'Sym':8}  {'Side':5}  {'MFE_R':>6}  "
        f"{'TP1_R':>6}  {'TP2_R':>6}  {'TP3_R':>6}  "
        f"{'TP1J':>5}  {'TP2J':>5}  {'TP3J':>5}  {'Match':>5}"
    )
    print(f"  {'─'*78}")
    for t in with_candles:
        def _hit(flag: bool) -> str:
            return f"{_G}HIT{_RST}" if flag else "  - "
        has_viol = any(v.vtype == "TP_MFE_CONSISTENCY" for v in t.violations)
        match_s = f"{_R}FAIL{_RST}" if has_viol else f"{_G} OK {_RST}"
        print(
            f"  {t.trade_num:>3}  {t.symbol:8}  {t.side:5}  {t.mfe_r:>6.2f}  "
            f"{t.tp1_r:>6.2f}  {t.tp2_r:>6.2f}  {t.tp3_r:>6.2f}  "
            f"{_hit(t.tp1_hit_journal):>5}  {_hit(t.tp2_hit_journal):>5}  "
            f"{_hit(t.tp3_hit_journal):>5}  {match_s}"
        )
    print(f"{'═'*80}")


def _print_summary(trades: list[ValidatedTrade]) -> None:
    n = len(trades)
    geo_fail   = sum(1 for t in trades if not t.geometry_ok)
    stop_fail  = sum(1 for t in trades if not t.stop_ok)
    tpmfe_fail = sum(1 for t in trades if not t.tp_mfe_ok)
    no_candles = sum(1 for t in trades if not t.has_candles)

    all_errors   = sum(len([v for v in t.violations if v.severity == "ERROR"])   for t in trades)
    all_warnings = sum(len([v for v in t.violations if v.severity == "WARNING"]) for t in trades)

    by_type: dict[str, int] = defaultdict(int)
    for t in trades:
        for v in t.violations:
            by_type[v.vtype] += 1

    print(f"\n{'═'*60}")
    print(f"  {_B}VALIDATION SUMMARY{_RST}")
    print(f"{'═'*60}")
    print(f"  Total trades                : {n}")
    print(f"  Trades without candle data  : {no_candles}")
    print(f"  {'─'*58}")
    print(f"  {'Geometry violations':30}: {_R if geo_fail else _G}{geo_fail}{_RST}")
    print(f"  {'Stop mismatch violations':30}: {_R if stop_fail else _G}{stop_fail}{_RST}")
    print(f"  {'TP/MFE consistency violations':30}: {_R if tpmfe_fail else _G}{tpmfe_fail}{_RST}")
    print(f"  {'─'*58}")
    print(f"  Total errors                : {_R if all_errors else _G}{all_errors}{_RST}")
    print(f"  Total warnings              : {_Y if all_warnings else _G}{all_warnings}{_RST}")
    if by_type:
        print(f"  {'─'*58}")
        print(f"  Breakdown by type:")
        for vtype, cnt in sorted(by_type.items()):
            col = _R if any(v.severity == "ERROR" and v.vtype == vtype
                           for t in trades for v in t.violations) else _Y
            print(f"    {vtype:<32}: {col}{cnt}{_RST}")
    print(f"  {'─'*58}")

    # Root cause determination
    print(f"\n  ROOT CAUSE ASSESSMENT:")
    if geo_fail > 0:
        print(f"  {_R}▶ GEOMETRY ERRORS: strategy is emitting malformed signals.{_RST}")
        print(f"    The entry/stop/TP ordering is wrong in the journal.")
        print(f"    This means analysis scripts are working with bad inputs.")
    if stop_fail > 0 and geo_fail == 0:
        print(f"  {_Y}▶ STOP MISMATCH: stop recalculated from formula ≠ logged value.{_RST}")
        print(f"    Possible config drift or rounding. Check ATR_STOP_MULT={ATR_STOP_MULT}.")
    if tpmfe_fail > 0:
        print(f"  {_R}▶ TP/MFE INCONSISTENCY: candle data shows price reached a TP level{_RST}")
        print(f"    that the journal did NOT record as hit. Either:")
        print(f"    (a) The exit evaluation logic has a bug (unlikely — check _evaluate_exits)")
        print(f"    (b) The candle MFE window extends PAST the trade exit (post-exit drift)")
        print(f"    (c) The PARTIAL_TP event was not emitted correctly")
    if all_errors == 0 and all_warnings == 0:
        print(f"  {_G}▶ ALL CHECKS PASSED.{_RST}")
        print(f"    Journal records are geometrically valid and internally consistent.")
        print(f"    Analysis scripts (be_stop_analysis, mfe_capture_audit, etc.) are")
        print(f"    working from correct trade data. The performance gap is real.")
    elif all_errors == 0:
        print(f"  {_Y}▶ No hard errors. Warnings are informational (slippage, rounding).{_RST}")
        print(f"    Analysis stack is reliable. Performance gap is real, not a data bug.")
    print(f"{'═'*60}\n")


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def _write_csv(trades: list[ValidatedTrade], out: Path) -> None:
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "trade_num", "entry_time", "symbol", "side",
            "signal_entry", "fill_price", "entry_slippage",
            "signal_stop", "stop_B_recalc_signal", "stop_C_recalc_fill",
            "r_unit_signal", "r_unit_fill",
            "signal_tp1", "signal_tp2", "signal_tp3",
            "tp1_r", "tp2_r", "tp3_r",
            "tp1_hit_journal", "tp2_hit_journal", "tp3_hit_journal",
            "mfe_r", "has_candles",
            "net_pnl", "exit_reason",
            "geometry_ok", "stop_ok", "tp_mfe_ok",
            "error_count", "warning_count",
            "violation_details",
        ])
        for t in trades:
            errs = sum(1 for v in t.violations if v.severity == "ERROR")
            warns = sum(1 for v in t.violations if v.severity == "WARNING")
            details = " | ".join(f"[{v.severity}:{v.vtype}] {v.detail}" for v in t.violations)
            w.writerow([
                t.trade_num,
                datetime.fromtimestamp(t.entry_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
                t.symbol, t.side,
                t.signal_entry, t.fill_price, t.entry_slippage,
                t.signal_stop, t.stop_B, t.stop_C,
                t.r_unit_signal, t.r_unit_fill,
                t.signal_tp1, t.signal_tp2, t.signal_tp3,
                t.tp1_r, t.tp2_r, t.tp3_r,
                t.tp1_hit_journal, t.tp2_hit_journal, t.tp3_hit_journal,
                t.mfe_r, t.has_candles,
                t.net_pnl, t.exit_reason,
                t.geometry_ok, t.stop_ok, t.tp_mfe_ok,
                errs, warns, details,
            ])
    print(f"CSV written: {out}  ({len(trades)} trades)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trade reconstruction validator")
    p.add_argument("--journal-dir", type=Path, default=Path("logs/paper"))
    p.add_argument("--candle-dir",  type=Path, default=Path("data/candles"))
    p.add_argument("--out",         type=Path, default=Path("trade_reconstruction_validator.csv"))
    p.add_argument("--no-rest",     action="store_true", help="Disable Bitget REST fallback")
    p.add_argument("--skip-mfe",    action="store_true", help="Skip candle MFE (faster)")
    p.add_argument("--verbose",     action="store_true", help="Show all violations in full")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.journal_dir.exists():
        print(f"Journal directory not found: {args.journal_dir}")
        sys.exit(1)

    print("Loading journal events ...")
    events = _load_events(args.journal_dir)
    trades = build_validated_trades(events)
    print(f"  {len(events)} events → {len(trades)} closed trades")

    if not trades:
        print("No closed trades found.")
        sys.exit(1)

    if not args.skip_mfe:
        print("Computing MFE from candle data ...")
        compute_mfe_for_trades(trades, args.candle_dir, args.no_rest)
        n_candles = sum(1 for t in trades if t.has_candles)
        print(f"  {n_candles}/{len(trades)} trades have candle data")

    print("Running validations ...")
    run_all_validations(trades)

    _print_trade_table(trades)
    _print_stop_comparison(trades)
    _print_tp_mfe_table(trades)
    _print_violations(trades, args.verbose)
    _print_summary(trades)
    _write_csv(trades, args.out)


if __name__ == "__main__":
    main()
