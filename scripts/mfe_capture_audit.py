#!/usr/bin/env python3
"""MFE capture audit.

Finds every trade where MFE >= 3R, reconstructs the full bar-by-bar
path from entry to exit, and measures how much of the available move
was actually captured.

Capture Efficiency = Realized_R / MFE_R

Example: MFE = 8.0R, Realized = 0.5R → Efficiency = 6.25%

R is defined as 1 × initial stop distance (abs(entry - original_stop)).

Usage:
    python scripts/mfe_capture_audit.py
    python scripts/mfe_capture_audit.py --journal-dir logs/paper --candle-dir data/candles
    python scripts/mfe_capture_audit.py --min-mfe 2.0 --verbose
"""

from __future__ import annotations

import argparse
import csv
import json
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

TAKER = 0.0006
MAKER = 0.0002


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
# Path event — one per notable moment in the bar-by-bar replay
# ---------------------------------------------------------------------------

@dataclass
class PathEvent:
    bar_idx: int
    bar_ts: float            # epoch seconds
    event_type: str          # "TP1_TOUCH" | "TP2_TOUCH" | "TP3_TOUCH" | "MFE_PEAK"
                             # | "STOP_MOVED_BE" | "STOP_HIT" | "TRAIL_MOVED"
    price: float
    note: str = ""


# ---------------------------------------------------------------------------
# Full trade record
# ---------------------------------------------------------------------------

@dataclass
class AuditTrade:
    # Identity
    trade_num: int
    symbol: str
    side: str
    entry_time: float
    exit_time: float

    # Prices from journal
    entry_price: float
    original_stop: float
    tp1_price: float
    tp2_price: float
    tp3_price: float
    atr: float
    size: float

    # R unit: abs(entry - original_stop)
    r_unit: float

    # Signal features
    signal_score: float
    rel_vol: float
    range_expansion: float
    spread_regime: float
    buy_sell_imbalance: float
    momentum_5m: float

    # PnL (from journal reconstruction)
    gross_pnl: float
    total_fees: float
    net_pnl: float
    exit_reason: str          # STOP_HIT | FORCE_CLOSE

    # MFE / MAE (price units and R)
    mfe_price: float = 0.0
    mfe_r: float = 0.0
    mfe_bar_idx: int = 0
    mfe_bar_ts: float = 0.0

    mae_price: float = 0.0
    mae_r: float = 0.0

    # Realized R
    realized_r: float = 0.0

    # Capture efficiency (0–1)
    capture_efficiency: float = 0.0

    # TP touches (bar index when first touched; -1 = never)
    tp1_bar: int = -1
    tp2_bar: int = -1
    tp3_bar: int = -1

    # Stop events
    be_move_bar: int = -1        # bar where stop moved to BE
    final_stop_bar: int = -1     # bar where stop was triggered
    final_stop_price: float = 0.0

    # Path events (full chronology)
    path: list[PathEvent] = field(default_factory=list)

    # Data quality
    has_candles: bool = False


# ---------------------------------------------------------------------------
# Journal parsing
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


def reconstruct_trades(events: list[dict]) -> list[AuditTrade]:
    last_signal: dict[str, dict] = {}
    open_fills: dict[str, dict] = {}
    open_fees: dict[str, float] = {}
    open_partials: dict[str, list[dict]] = defaultdict(list)

    records: list[AuditTrade] = []
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
            stop_price = sig.get("stop_price", 0.0)

            if atr <= 0 or not tp_prices or not stop_price:
                open_fees.pop(sym, None)
                open_partials.pop(sym, [])
                continue

            side = fill_ev.get("direction", "long")
            is_long = side == "long"
            sign = 1.0 if is_long else -1.0
            entry_price = fill_ev.get("fill_price", 0.0)
            entry_size = fill_ev.get("size", 0.0)
            entry_fee = fill_ev.get("fee", 0.0)

            partials = open_partials.pop(sym, [])
            partial_gross = sum(
                (p["fill_price"] - entry_price) * p["size"] * sign
                for p in partials
            )
            partial_size = sum(p["size"] for p in partials)
            remaining = entry_size - partial_size

            if evt == "STOP_HIT":
                exit_price = ev.get("fill_price", 0.0)
                stop_fee = ev.get("fee", 0.0)
            else:
                exit_price = ev.get("price", 0.0)
                stop_fee = 0.0

            stop_gross = (exit_price - entry_price) * remaining * sign
            gross = partial_gross + stop_gross
            total_fees = open_fees.pop(sym, 0.0) + stop_fee
            net = gross - total_fees

            r_unit = abs(entry_price - stop_price)
            realized_r = net / (r_unit * entry_size) if r_unit > 0 and entry_size > 0 else 0.0

            # Feature extraction for score context
            rel_vol = feats.get("rel_vol", 0.0)
            range_exp = feats.get("range_expansion", 0.0)
            spread_reg = feats.get("spread_regime", 0.0)
            imbalance = feats.get("buy_sell_imbalance", 0.5)
            momentum = feats.get("momentum_5m", 0.0)

            records.append(AuditTrade(
                trade_num=trade_num,
                symbol=sym,
                side=side,
                entry_time=fill_ev.get("ts_epoch", 0.0),
                exit_time=ev.get("ts_epoch", 0.0),
                entry_price=entry_price,
                original_stop=stop_price,
                tp1_price=tp_prices[0] if len(tp_prices) > 0 else 0.0,
                tp2_price=tp_prices[1] if len(tp_prices) > 1 else 0.0,
                tp3_price=tp_prices[2] if len(tp_prices) > 2 else 0.0,
                atr=atr,
                size=entry_size,
                r_unit=round(r_unit, 6),
                signal_score=0.0,  # filled later if signal_quality_study is run first
                rel_vol=rel_vol,
                range_expansion=range_exp,
                spread_regime=spread_reg,
                buy_sell_imbalance=imbalance,
                momentum_5m=momentum,
                gross_pnl=round(gross, 4),
                total_fees=round(total_fees, 4),
                net_pnl=round(net, 4),
                exit_reason=evt,
                realized_r=round(realized_r, 4),
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
    # Bitget allows max 1000 bars per request; if window > 1000 min, page
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


def _get_bars(
    t: AuditTrade,
    parquet_cache: dict[str, list[Bar]],
    rest_cache: dict[str, list[Bar]],
    candle_dir: Path,
    no_rest: bool,
    extra_bars: int = 60,      # bars past exit to see if price kept going
) -> list[Bar]:
    sym = t.symbol
    if sym not in parquet_cache:
        parquet_cache[sym] = _load_parquet(candle_dir, sym)

    start_ms = int(t.entry_time * 1000)
    end_ms   = int(t.exit_time  * 1000) + extra_bars * 60_000

    window = [b for b in parquet_cache[sym] if start_ms <= b.ts_ms <= end_ms]
    if window:
        return window

    if no_rest:
        return []

    key = f"{sym}_{start_ms}_{end_ms}"
    if key not in rest_cache:
        rest_cache[key] = _fetch_rest(sym, start_ms, end_ms)
    return rest_cache[key]


# ---------------------------------------------------------------------------
# Bar-by-bar path reconstruction
# ---------------------------------------------------------------------------

def reconstruct_path(t: AuditTrade, bars: list[Bar]) -> None:
    """Fill in MFE, MAE, TP touches, stop events, and path events."""
    if not bars:
        return

    t.has_candles = True
    is_long = t.side == "long"
    sign = 1.0 if is_long else -1.0

    mfe_price_level = t.entry_price   # absolute price level of MFE
    mae_price_level = t.entry_price

    current_stop = t.original_stop
    be_moved = False
    tp1_hit = False
    tp2_hit = False
    tp3_hit = False

    path: list[PathEvent] = []

    exit_ts_ms = int(t.exit_time * 1000)

    for i, bar in enumerate(bars):
        is_pre_exit = bar.ts_ms <= exit_ts_ms

        # MFE / MAE tracking (full window including post-exit bars for context)
        if is_long:
            if bar.high > mfe_price_level:
                mfe_price_level = bar.high
                t.mfe_bar_idx = i
                t.mfe_bar_ts = bar.ts_ms / 1000
            if bar.low < mae_price_level:
                mae_price_level = bar.low
        else:
            if bar.low < mfe_price_level:
                mfe_price_level = bar.low
                t.mfe_bar_idx = i
                t.mfe_bar_ts = bar.ts_ms / 1000
            if bar.high > mae_price_level:
                mae_price_level = bar.high

        if not is_pre_exit:
            # Only track path events within the actual holding period
            continue

        # TP1
        if not tp1_hit and t.tp1_price > 0:
            if (is_long and bar.high >= t.tp1_price) or (not is_long and bar.low <= t.tp1_price):
                tp1_hit = True
                t.tp1_bar = i
                path.append(PathEvent(i, bar.ts_ms / 1000, "TP1_TOUCH", t.tp1_price,
                                      f"TP1 touched @ bar {i}"))
                # BE stop move
                if not be_moved:
                    be_moved = True
                    current_stop = t.entry_price
                    t.be_move_bar = i
                    path.append(PathEvent(i, bar.ts_ms / 1000, "STOP_MOVED_BE", t.entry_price,
                                          f"Stop moved to BE={t.entry_price:.4f}"))

        # TP2
        if tp1_hit and not tp2_hit and t.tp2_price > 0:
            if (is_long and bar.high >= t.tp2_price) or (not is_long and bar.low <= t.tp2_price):
                tp2_hit = True
                t.tp2_bar = i
                path.append(PathEvent(i, bar.ts_ms / 1000, "TP2_TOUCH", t.tp2_price,
                                      f"TP2 touched @ bar {i}"))

        # TP3
        if tp2_hit and not tp3_hit and t.tp3_price > 0:
            if (is_long and bar.high >= t.tp3_price) or (not is_long and bar.low <= t.tp3_price):
                tp3_hit = True
                t.tp3_bar = i
                path.append(PathEvent(i, bar.ts_ms / 1000, "TP3_TOUCH", t.tp3_price,
                                      f"TP3 touched @ bar {i}"))

        # Stop trigger
        stop_hit = (is_long and bar.low <= current_stop) or (not is_long and bar.high >= current_stop)
        if stop_hit:
            t.final_stop_bar = i
            t.final_stop_price = current_stop
            path.append(PathEvent(i, bar.ts_ms / 1000, "STOP_HIT", current_stop,
                                  f"Stop hit @ {current_stop:.4f} (bar {i})"))
            break   # stop terminates the holding period

    # MFE peak event
    mfe_bar = bars[t.mfe_bar_idx] if t.mfe_bar_idx < len(bars) else bars[-1]
    path.append(PathEvent(t.mfe_bar_idx, mfe_bar.ts_ms / 1000, "MFE_PEAK", mfe_price_level,
                          f"MFE peak = {mfe_price_level:.4f}"))
    path.sort(key=lambda e: (e.bar_idx, e.event_type))
    t.path = path

    # Compute MFE / MAE in price and R
    t.mfe_price = abs(mfe_price_level - t.entry_price)
    t.mae_price = abs(mae_price_level - t.entry_price)
    t.mfe_r = round(t.mfe_price / t.r_unit, 3) if t.r_unit > 0 else 0.0
    t.mae_r = round(t.mae_price / t.r_unit, 3) if t.r_unit > 0 else 0.0

    # Capture efficiency
    t.capture_efficiency = (
        round(t.realized_r / t.mfe_r, 4) if t.mfe_r > 0 else 0.0
    )


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


def _eff_colour(eff: float) -> str:
    if eff >= 0.5:
        return _G
    if eff >= 0.2:
        return _Y
    return _R


def _print_trade_table(trades: list[AuditTrade]) -> None:
    bar = "─" * 110
    print(f"\n{_B}{'═'*110}{_RST}")
    print(f"  {_B}{'MFE Capture Audit — Trades with MFE ≥ threshold':^108}{_RST}")
    print(f"{_B}{'═'*110}{_RST}")
    print(
        f"  {'#':>3}  {'Time':16}  {'Sym':8}  {'Side':5}  "
        f"{'ATR':>8}  {'Entry':>9}  {'Stop':>9}  {'TP1':>9}  "
        f"{'MFE_R':>6}  {'Real_R':>7}  {'Eff%':>6}  "
        f"{'TP1':>3}  {'TP2':>3}  {'TP3':>3}  {'Exit'}"
    )
    print(f"  {bar}")

    for t in trades:
        eff_col = _eff_colour(t.capture_efficiency)
        tp1_s = f"{_G}HIT{_RST}" if t.tp1_bar >= 0 else f"{_R} - {_RST}"
        tp2_s = f"{_G}HIT{_RST}" if t.tp2_bar >= 0 else f"{_R} - {_RST}"
        tp3_s = f"{_G}HIT{_RST}" if t.tp3_bar >= 0 else f"{_R} - {_RST}"
        net_col = _G if t.net_pnl > 0 else _R
        print(
            f"  {t.trade_num:>3}  {_ts(t.entry_time):16}  {t.symbol:8}  {t.side:5}  "
            f"{t.atr:>8.4f}  {t.entry_price:>9.2f}  {t.original_stop:>9.2f}  {t.tp1_price:>9.2f}  "
            f"{t.mfe_r:>6.2f}  {net_col}{t.realized_r:>+7.3f}{_RST}  "
            f"{eff_col}{t.capture_efficiency*100:>5.1f}%{_RST}  "
            f"{tp1_s}  {tp2_s}  {tp3_s}  {t.exit_reason}"
        )
    print(f"  {bar}")


def _print_path(t: AuditTrade) -> None:
    print(f"\n  {_C}Trade #{t.trade_num} — {t.symbol} {t.side.upper()}  "
          f"entry={t.entry_price:.4f}  MFE={t.mfe_r:.2f}R  realized={t.realized_r:.3f}R  "
          f"eff={t.capture_efficiency*100:.1f}%{_RST}")
    print(f"  {'Bar':>4}  {'Time':16}  {'Event':16}  {'Price':>10}  Note")
    print(f"  {'─'*70}")
    for ev in t.path:
        print(f"  {ev.bar_idx:>4}  {_ts(ev.bar_ts):16}  {ev.event_type:16}  {ev.price:>10.4f}  {ev.note}")
    print()


def _print_efficiency_distribution(trades: list[AuditTrade]) -> None:
    effs = [t.capture_efficiency for t in trades if t.has_candles]
    if not effs:
        print("  No candle data available for efficiency distribution.")
        return

    buckets = {
        "0–5%":   0, "5–10%":  0, "10–20%": 0,
        "20–30%": 0, "30–50%": 0, "50–75%": 0, "75–100%": 0, ">100%": 0,
    }
    for e in effs:
        pct = e * 100
        if pct < 5:      buckets["0–5%"]    += 1
        elif pct < 10:   buckets["5–10%"]   += 1
        elif pct < 20:   buckets["10–20%"]  += 1
        elif pct < 30:   buckets["20–30%"]  += 1
        elif pct < 50:   buckets["30–50%"]  += 1
        elif pct < 75:   buckets["50–75%"]  += 1
        elif pct <= 100: buckets["75–100%"] += 1
        else:            buckets[">100%"]   += 1

    n = len(effs)
    avg_eff = sum(effs) / n
    median  = sorted(effs)[n // 2]

    print(f"\n{'═'*55}")
    print(f"  Capture Efficiency Distribution  (n={n})")
    print(f"{'═'*55}")
    print(f"  Average : {avg_eff*100:.1f}%   Median : {median*100:.1f}%")
    print(f"  {'─'*53}")
    for label, count in buckets.items():
        bar_w = int(count / n * 30) if n > 0 else 0
        bar_s = "█" * bar_w
        pct_of_n = count / n * 100 if n > 0 else 0.0
        print(f"  {label:>8}  {bar_s:<30}  {count:>3}  ({pct_of_n:.0f}%)")
    print(f"{'═'*55}")


def _print_worst(trades: list[AuditTrade], n: int = 10) -> None:
    ranked = sorted(
        [t for t in trades if t.has_candles and t.mfe_r > 0],
        key=lambda t: t.capture_efficiency
    )
    worst = ranked[:n]
    if not worst:
        return

    print(f"\n{'═'*80}")
    print(f"  Worst {n} Efficiency Examples")
    print(f"{'═'*80}")
    for t in worst:
        missed_r = t.mfe_r - max(t.realized_r, 0)
        tp_hits = (
            f"TP1={'Y' if t.tp1_bar>=0 else 'N'}  "
            f"TP2={'Y' if t.tp2_bar>=0 else 'N'}  "
            f"TP3={'Y' if t.tp3_bar>=0 else 'N'}"
        )
        print(
            f"  #{t.trade_num:>3} {t.symbol} {t.side:5} {_ts(t.entry_time)}  "
            f"MFE={t.mfe_r:.2f}R  realized={t.realized_r:.3f}R  "
            f"eff={_R}{t.capture_efficiency*100:.1f}%{_RST}  "
            f"missed={missed_r:.2f}R  {tp_hits}"
        )
        # Path summary (just key events)
        key_types = {"TP1_TOUCH", "TP2_TOUCH", "TP3_TOUCH", "STOP_MOVED_BE", "STOP_HIT", "MFE_PEAK"}
        for ev in [e for e in t.path if e.event_type in key_types]:
            print(f"    bar {ev.bar_idx:>3}: {ev.event_type:16} @ {ev.price:.4f}")
        print()
    print(f"{'═'*80}")


def _print_summary(trades: list[AuditTrade], all_trades: list[AuditTrade]) -> None:
    n_all = len(all_trades)
    n_large = len(trades)
    effs = [t.capture_efficiency for t in trades if t.has_candles]
    avg_eff = sum(effs) / len(effs) if effs else 0.0
    total_mfe_r = sum(t.mfe_r for t in trades if t.has_candles)
    total_realized_r = sum(max(t.realized_r, 0) for t in trades if t.has_candles)
    total_missed_r = total_mfe_r - total_realized_r

    tp1_rate = sum(1 for t in trades if t.tp1_bar >= 0) / n_large if n_large else 0.0
    tp2_rate = sum(1 for t in trades if t.tp2_bar >= 0) / n_large if n_large else 0.0
    tp3_rate = sum(1 for t in trades if t.tp3_bar >= 0) / n_large if n_large else 0.0

    print(f"\n{'═'*60}")
    print(f"  SUMMARY")
    print(f"{'═'*60}")
    print(f"  Total closed trades      : {n_all}")
    print(f"  Trades with MFE >= thresh: {n_large}  ({n_large/n_all*100:.0f}% of total)")
    print(f"  {'─'*58}")
    print(f"  Avg MFE (R)              : {total_mfe_r/n_large:.2f}R" if n_large else "  No data.")
    print(f"  Avg realized (R)         : {total_realized_r/n_large:.3f}R" if n_large else "")
    print(f"  Avg missed (R)           : {total_missed_r/n_large:.2f}R" if n_large else "")
    print(f"  Avg capture efficiency   : {avg_eff*100:.1f}%")
    print(f"  {'─'*58}")
    print(f"  TP1 hit rate             : {tp1_rate*100:.1f}%")
    print(f"  TP2 hit rate             : {tp2_rate*100:.1f}%")
    print(f"  TP3 hit rate             : {tp3_rate*100:.1f}%")
    print(f"  {'─'*58}")

    # Verdict
    if avg_eff < 0.10:
        print(f"  {_R}VERDICT: Large moves ARE being generated but almost nothing")
        print(f"  is being captured. Avg efficiency {avg_eff*100:.1f}% on MFE≥threshold")
        print(f"  trades. Exit structure is destroying edge.{_RST}")
    elif avg_eff < 0.25:
        print(f"  {_Y}VERDICT: Significant capture gap. Avg {avg_eff*100:.1f}% efficiency.")
        print(f"  Exit management leaves most of the move on the table.{_RST}")
    elif avg_eff < 0.50:
        print(f"  {_Y}VERDICT: Moderate capture. {avg_eff*100:.1f}% efficiency average.")
        print(f"  Room for improvement but exit structure is functional.{_RST}")
    else:
        print(f"  {_G}VERDICT: Good capture efficiency ({avg_eff*100:.1f}%). Exit structure")
        print(f"  is converting large MFE moves into realized gains.{_RST}")

    print(f"{'═'*60}\n")


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def _write_csv(trades: list[AuditTrade], out: Path) -> None:
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "trade_num", "entry_time", "symbol", "side",
            "entry_price", "original_stop", "tp1_price", "tp2_price", "tp3_price",
            "atr", "r_unit", "size",
            "rel_vol", "range_expansion", "spread_regime", "buy_sell_imbalance", "momentum_5m",
            "gross_pnl", "total_fees", "net_pnl", "exit_reason",
            "mfe_price", "mfe_r", "mfe_time",
            "mae_price", "mae_r",
            "realized_r", "capture_efficiency",
            "tp1_bar", "tp2_bar", "tp3_bar",
            "be_move_bar", "final_stop_bar", "final_stop_price",
            "has_candles",
        ])
        for t in trades:
            w.writerow([
                t.trade_num, _ts(t.entry_time), t.symbol, t.side,
                t.entry_price, t.original_stop, t.tp1_price, t.tp2_price, t.tp3_price,
                t.atr, t.r_unit, t.size,
                t.rel_vol, t.range_expansion, t.spread_regime, t.buy_sell_imbalance, t.momentum_5m,
                t.gross_pnl, t.total_fees, t.net_pnl, t.exit_reason,
                t.mfe_price, t.mfe_r, _ts(t.mfe_bar_ts) if t.mfe_bar_ts else "",
                t.mae_price, t.mae_r,
                t.realized_r, t.capture_efficiency,
                t.tp1_bar, t.tp2_bar, t.tp3_bar,
                t.be_move_bar, t.final_stop_bar, t.final_stop_price,
                t.has_candles,
            ])
    print(f"CSV written: {out}  ({len(trades)} trades)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MFE capture audit")
    p.add_argument("--journal-dir", type=Path, default=Path("logs/paper"))
    p.add_argument("--candle-dir",  type=Path, default=Path("data/candles"))
    p.add_argument("--out",         type=Path, default=Path("mfe_capture_audit.csv"))
    p.add_argument("--min-mfe",     type=float, default=3.0,
                   help="Minimum MFE in R to include (default: 3.0)")
    p.add_argument("--no-rest",     action="store_true")
    p.add_argument("--verbose",     action="store_true",
                   help="Print full path chronology for every large-MFE trade")
    p.add_argument("--worst",       type=int, default=10,
                   help="Number of worst-efficiency examples to highlight (default: 10)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.journal_dir.exists():
        print(f"Journal directory not found: {args.journal_dir}")
        sys.exit(1)

    print("Loading journal events ...")
    events = _load_events(args.journal_dir)
    all_trades = reconstruct_trades(events)
    print(f"  {len(events)} events → {len(all_trades)} closed trades")

    if not all_trades:
        print("No closed trades found.")
        sys.exit(1)

    # First pass: compute MFE without candles to find candidates
    # We need candles to get proper MFE — load for all, filter after
    print("Loading candle data and reconstructing paths ...")
    parquet_cache: dict[str, list[Bar]] = {}
    rest_cache: dict[str, list[Bar]] = {}
    no_data = 0

    for t in all_trades:
        bars = _get_bars(t, parquet_cache, rest_cache, args.candle_dir, args.no_rest)
        if not bars:
            no_data += 1
        reconstruct_path(t, bars)

    if no_data > 0:
        print(f"  {no_data} trades had no candle data")

    # Filter to large MFE trades
    large_mfe = [t for t in all_trades if t.mfe_r >= args.min_mfe]
    # Also include trades with no candle data but where we can flag them
    no_candle_trades = [t for t in all_trades if not t.has_candles]

    print(f"\n  Trades with MFE >= {args.min_mfe}R: {len(large_mfe)} / {len(all_trades)}")
    if no_candle_trades:
        print(f"  Trades without candle data (excluded from MFE filter): {len(no_candle_trades)}")

    if not large_mfe:
        print(f"\n  No trades found with MFE >= {args.min_mfe}R.")
        print(f"  Try --min-mfe 1.0 to lower the threshold.")
        # Still write CSV of all trades
        _write_csv(all_trades, args.out)
        sys.exit(0)

    # Print main table
    _print_trade_table(large_mfe)

    # Full path for verbose mode
    if args.verbose:
        print(f"\n  Full path chronology ({len(large_mfe)} trades):")
        for t in sorted(large_mfe, key=lambda x: x.mfe_r, reverse=True):
            _print_path(t)

    _print_efficiency_distribution(large_mfe)
    _print_worst(large_mfe, n=args.worst)
    _print_summary(large_mfe, all_trades)

    _write_csv(all_trades, args.out)


if __name__ == "__main__":
    main()
