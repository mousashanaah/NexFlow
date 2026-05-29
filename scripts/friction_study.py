#!/usr/bin/env python3
"""Friction reduction study: 5 execution variants of Model B (2-ATR trailing stop).

Entries are FROZEN.  Candle data is the same.  Only execution parameters vary.

Variants
--------
  B1  Current Model B     Full size, taker exit, all symbols, 2-ATR trail
  B2  Half position size  Size × 0.5                     isolates C) position sizing
  B3  Fixed stop exit     Original stop_price, no trail   isolates D) execution structure
  B4  Maker-style exits   Exit fee at 0.02% maker rate    isolates A) fee drag
  B5  ETHUSDT only        Skip all BTCUSDT entries        isolates B) BTC signal quality

Diagnosis map
-------------
  A) Fee drag           → compare B4 vs B1
  B) BTC signal quality → compare B5 vs B1
  C) Position sizing    → compare B2 ratios vs B1  (PF / fee-drag % unchanged → not root cause)
  D) Execution struct   → compare B3 vs B1  (fixed stop vs trail)

Reported per variant
--------------------
  Net PnL / Gross PnL / Total fees / Fee drag % / Profit factor / Trades executed

Usage
-----
  python scripts/friction_study.py --journal-dir logs/paper --candle-dir data/candles
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import pyarrow.parquet as pq

from nexflow.services.candles.candle_engine import Candle

# ── Execution constants (mirrors PaperExecution) ──────────────────────────────

TAKER_FEE   = 0.0006   # 0.06 %
MAKER_FEE   = 0.0002   # 0.02 %
STOP_SLIP_K = 0.025    # stop_distance × 0.025 adverse fill
ATR_WINDOW  = 14       # EMA window for rolling ATR
MAX_BARS    = 180      # max look-forward bars per trade

VARIANT_LABELS = {
    "B1": "Current Model B    ",
    "B2": "Half Position Size ",
    "B3": "Fixed Stop (no trail)",
    "B4": "Maker-Style Exits  ",
    "B5": "ETHUSDT Only       ",
}

DIAGNOSES = {
    "B1": "baseline",
    "B2": "isolates C) position sizing",
    "B3": "isolates D) execution structure",
    "B4": "isolates A) fee drag",
    "B5": "isolates B) BTC signal quality",
}

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class EntryRecord:
    seq: int
    symbol: str
    direction: str
    signal_entry: float     # signal entry_price = candle.close (for matching)
    fill_price: float
    stop_price: float
    tp_prices: list[float]
    atr: float
    total_size: float
    entry_fee: float
    entry_ts: float
    candle_idx: int = -1


@dataclass
class TradeResult:
    seq: int
    symbol: str
    direction: str
    variant: str
    entry_price: float
    exit_price: float
    total_size: float
    entry_fee: float
    exit_fees: float
    gross_pnl: float
    net_pnl: float
    hold_bars: int
    exit_reason: str


@dataclass
class VariantMetrics:
    variant: str
    label: str
    diagnosis: str
    n_trades: int
    net_pnl: float
    gross_pnl: float
    total_fees: float
    fee_drag_pct: float
    profit_factor: float
    win_rate: float
    avg_hold_min: float
    results: list[TradeResult] = field(default_factory=list)


# ── Journal parsing ───────────────────────────────────────────────────────────

def _load_events(journal_dir: Path) -> list[dict]:
    events: list[dict] = []
    for p in sorted(journal_dir.glob("journal_*.jsonl")):
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return sorted(events, key=lambda e: e.get("ts_epoch", 0.0))


def _extract_entries(events: list[dict]) -> list[EntryRecord]:
    pending: dict[str, dict] = {}
    entries: list[EntryRecord] = []
    for ev in events:
        etype = ev.get("event", "")
        sym = ev.get("symbol", "")
        if etype == "SIGNAL":
            pending[sym] = ev
        elif etype == "FILL":
            sig = pending.pop(sym, None)
            if sig is None:
                continue
            tp_prices = sig.get("tp_prices", [])
            if len(tp_prices) < 3:
                continue
            atr = sig.get("atr") or sig.get("features", {}).get("atr", 0.0) or 0.0
            entries.append(EntryRecord(
                seq=len(entries),
                symbol=sym,
                direction=ev.get("direction", "long").lower(),
                signal_entry=sig["entry_price"],
                fill_price=ev["fill_price"],
                stop_price=sig["stop_price"],
                tp_prices=tp_prices,
                atr=float(atr),
                total_size=ev["size"],
                entry_fee=ev["fee"],
                entry_ts=ev.get("ts_epoch", 0.0),
            ))
    return entries


# ── Candle loading and matching ───────────────────────────────────────────────

def _load_candles_1m(candle_dir: Path, symbols: list[str]) -> dict[str, list[Candle]]:
    out: dict[str, list[Candle]] = {}
    for sym in symbols:
        path = candle_dir / f"{sym}_1m.parquet"
        if not path.exists():
            out[sym] = []
            continue
        rows = pq.read_table(path).to_pylist()
        candles = [
            Candle(
                symbol=r["symbol"], timeframe=r["timeframe"],
                open_time=r["open_time"], close_time=r["close_time"],
                open=r["open"], high=r["high"], low=r["low"], close=r["close"],
                volume=r["volume"], buy_volume=r["buy_volume"],
                sell_volume=r["sell_volume"], trade_count=r["trade_count"],
                vwap=r["vwap"], spread_avg=r["spread_avg"],
                spread_max=r["spread_max"],
                volatility_estimate=r["volatility_estimate"],
                is_final=r["is_final"],
            )
            for r in rows if r.get("is_final")
        ]
        out[sym] = sorted(candles, key=lambda c: c.close_time)
    return out


def _match_to_candles(
    entries: list[EntryRecord],
    candles_1m: dict[str, list[Candle]],
) -> int:
    ptrs: dict[str, int] = defaultdict(int)
    matched = 0
    for entry in entries:
        sym = entry.symbol
        candles = candles_1m.get(sym, [])
        if not candles:
            continue
        ptr = ptrs[sym]
        best_i, best_diff = -1, float("inf")
        for i in range(ptr, min(ptr + 600, len(candles))):
            diff = abs(candles[i].close - entry.signal_entry) / max(entry.signal_entry, 1e-9)
            if diff < 0.001 and diff < best_diff:
                best_diff = diff
                best_i = i
            elif best_i >= 0 and i > best_i + 10:
                break
        if best_i >= 0:
            entry.candle_idx = best_i
            ptrs[sym] = best_i + 1
            matched += 1
    return matched


def _bars_after(candles: list[Candle], idx: int) -> list[Candle]:
    return candles[idx + 1 : idx + 1 + MAX_BARS]


# ── Primitives ────────────────────────────────────────────────────────────────

def _fee(price: float, size: float, *, maker: bool) -> float:
    return price * size * (MAKER_FEE if maker else TAKER_FEE)


def _raw_pnl(direction: str, entry: float, exit_p: float, size: float) -> float:
    return (exit_p - entry) * size if direction == "long" else (entry - exit_p) * size


def _stop_fill(stop: float, direction: str, entry: float) -> float:
    slip = abs(entry - stop) * STOP_SLIP_K
    return (stop - slip) if direction == "long" else (stop + slip)


def _rolling_atr(bars: list[Candle]) -> list[float]:
    out: list[float] = []
    ema = 0.0
    k = 2.0 / (ATR_WINDOW + 1)
    for i, b in enumerate(bars):
        r = b.high - b.low
        ema = r * k + ema * (1 - k) if i > 0 else r
        out.append(ema)
    return out


# ── Core Model B (2-ATR trailing stop) ───────────────────────────────────────
#
# Parameterised so each variant only tweaks what it needs.
#
#   size_mult   : 1.0 (full) or 0.5 (half)
#   use_trail   : True → 2-ATR trail; False → fixed stop at stop_price
#   exit_maker  : True → MAKER_FEE for exit; False → TAKER_FEE

def _simulate_model_B(
    entry: EntryRecord,
    bars: list[Candle],
    variant: str,
    *,
    size_mult: float = 1.0,
    use_trail: bool = True,
    exit_maker: bool = False,
) -> TradeResult:
    is_long   = entry.direction == "long"
    size      = entry.total_size * size_mult
    seed_atr  = entry.atr if entry.atr > 0 else abs(entry.fill_price - entry.stop_price) / 1.5
    atrs      = _rolling_atr(bars)
    trail     = entry.stop_price   # both fixed and trailing start here

    for i, bar in enumerate(bars):
        atr_i = atrs[i] if atrs else seed_atr

        stop_hit = (is_long and bar.low <= trail) or (not is_long and bar.high >= trail)
        if stop_hit:
            fp       = _stop_fill(trail, entry.direction, entry.fill_price)
            fees_out = _fee(fp, size, maker=exit_maker)
            gross    = _raw_pnl(entry.direction, entry.fill_price, fp, size)
            # Entry fee also scales with size_mult
            net = gross - entry.entry_fee * size_mult - fees_out
            return TradeResult(
                seq=entry.seq, symbol=entry.symbol, direction=entry.direction,
                variant=variant, entry_price=entry.fill_price, exit_price=fp,
                total_size=size, entry_fee=entry.entry_fee * size_mult,
                exit_fees=fees_out, gross_pnl=gross, net_pnl=net,
                hold_bars=i + 1, exit_reason="TRAIL" if use_trail else "STOP",
            )

        if use_trail:
            new_trail = (bar.close - 2 * atr_i) if is_long else (bar.close + 2 * atr_i)
            trail = max(trail, new_trail) if is_long else min(trail, new_trail)
        # if not use_trail: stop stays fixed at entry.stop_price

    # EOD flat
    last     = bars[-1].close if bars else entry.fill_price
    fees_out = _fee(last, size, maker=exit_maker)
    gross    = _raw_pnl(entry.direction, entry.fill_price, last, size)
    net      = gross - entry.entry_fee * size_mult - fees_out
    return TradeResult(
        seq=entry.seq, symbol=entry.symbol, direction=entry.direction,
        variant=variant, entry_price=entry.fill_price, exit_price=last,
        total_size=size, entry_fee=entry.entry_fee * size_mult,
        exit_fees=fees_out, gross_pnl=gross, net_pnl=net,
        hold_bars=len(bars), exit_reason="EOD",
    )


# ── Per-variant dispatchers ───────────────────────────────────────────────────

def run_B1(entry: EntryRecord, bars: list[Candle]) -> TradeResult:
    return _simulate_model_B(entry, bars, "B1",
                              size_mult=1.0, use_trail=True, exit_maker=False)

def run_B2(entry: EntryRecord, bars: list[Candle]) -> TradeResult:
    return _simulate_model_B(entry, bars, "B2",
                              size_mult=0.5, use_trail=True, exit_maker=False)

def run_B3(entry: EntryRecord, bars: list[Candle]) -> TradeResult:
    return _simulate_model_B(entry, bars, "B3",
                              size_mult=1.0, use_trail=False, exit_maker=False)

def run_B4(entry: EntryRecord, bars: list[Candle]) -> TradeResult:
    return _simulate_model_B(entry, bars, "B4",
                              size_mult=1.0, use_trail=True, exit_maker=True)

def run_B5(entry: EntryRecord, bars: list[Candle]) -> TradeResult | None:
    if entry.symbol != "ETHUSDT":
        return None
    return _simulate_model_B(entry, bars, "B5",
                              size_mult=1.0, use_trail=True, exit_maker=False)


# ── Metrics ───────────────────────────────────────────────────────────────────

INITIAL_EQUITY = 100_000.0


def _compute_metrics(variant: str, results: list[TradeResult]) -> VariantMetrics:
    if not results:
        return VariantMetrics(
            variant=variant, label=VARIANT_LABELS[variant],
            diagnosis=DIAGNOSES[variant], n_trades=0,
            net_pnl=0, gross_pnl=0, total_fees=0,
            fee_drag_pct=0, profit_factor=0, win_rate=0, avg_hold_min=0,
        )

    gross_wins = sum(r.gross_pnl for r in results if r.gross_pnl > 0)
    gross_loss = sum(abs(r.gross_pnl) for r in results if r.gross_pnl < 0)
    total_fees = sum(r.entry_fee + r.exit_fees for r in results)
    gross_pnl  = sum(r.gross_pnl for r in results)
    net_pnl    = sum(r.net_pnl for r in results)
    pf         = gross_wins / gross_loss if gross_loss > 0 else float("inf")
    wr         = sum(1 for r in results if r.net_pnl > 0) / len(results)
    avg_hold   = sum(r.hold_bars for r in results) / len(results)
    fee_drag   = (total_fees / abs(gross_pnl) * 100) if gross_pnl != 0 else 0.0

    return VariantMetrics(
        variant=variant, label=VARIANT_LABELS[variant],
        diagnosis=DIAGNOSES[variant], n_trades=len(results),
        net_pnl=net_pnl, gross_pnl=gross_pnl, total_fees=total_fees,
        fee_drag_pct=fee_drag, profit_factor=pf, win_rate=wr,
        avg_hold_min=avg_hold, results=results,
    )


# ── Printing ──────────────────────────────────────────────────────────────────

_G   = "\033[92m"
_R   = "\033[91m"
_Y   = "\033[93m"
_C   = "\033[96m"
_B   = "\033[1m"
_DIM = "\033[2m"
_RST = "\033[0m"


def _col(v: float, w: int = 11) -> str:
    c = _G if v > 0 else (_R if v < 0 else "")
    return f"{c}{v:>+{w}.2f}{_RST}"


def _print_results(metrics: list[VariantMetrics]) -> None:
    bar = "─" * 88

    print()
    print(f"{_B}╔══ FRICTION REDUCTION STUDY ═══════════════════════════════════════════════════════╗{_RST}")
    print(f"  {_C}Model B (2-ATR trailing stop) is the baseline.  Entries are FROZEN.{_RST}")
    print(f"  Each variant changes exactly ONE execution parameter to isolate friction source.")
    print(bar)

    # ── Summary table ─────────────────────────────────────────────────────────
    col_w = 15
    print(f"  {'Metric':<22}", end="")
    for m in metrics:
        print(f"  {m.variant:>{col_w}}", end="")
    print()
    print(f"  {'':22}", end="")
    for m in metrics:
        lbl = m.label.strip()[:col_w]
        print(f"  {_C}{lbl:>{col_w}}{_RST}", end="")
    print()
    print(bar)

    def _row(label: str, getter, fmt):
        print(f"  {label:<22}", end="")
        for m in metrics:
            v = getter(m)
            print(f"  {fmt(v, col_w)}", end="")
        print()

    def _net(v, w):   return _col(v, w)
    def _gross(v, w): return _col(v, w)
    def _fees(v, w):
        c = _R if v > 0 else ""
        return f"{c}{-v:>{w}.2f}{_RST}"
    def _drag(v, w):
        c = _G if v < 50 else (_Y if v < 100 else _R)
        return f"{c}{v:>{w-1}.1f}%{_RST}"
    def _pf(v, w):
        c = _G if v >= 1.5 else (_Y if v >= 1.0 else _R)
        s = f"{v:.3f}" if v != float("inf") else "∞"
        return f"{c}{s:>{w}}{_RST}"
    def _wr(v, w):
        c = _G if v >= 50 else (_Y if v >= 40 else _R)
        return f"{c}{v:>{w-1}.1f}%{_RST}"
    def _n(v, w):  return f"{v:>{w}}"
    def _hold(v, w): return f"{v:>{w-1}.1f}m"

    _row("Net PnL (USDT)",   lambda m: m.net_pnl,       _net)
    _row("Gross PnL (USDT)", lambda m: m.gross_pnl,     _gross)
    _row("Total Fees (USDT)",lambda m: -m.total_fees,   _fees)
    _row("Fee Drag %",       lambda m: m.fee_drag_pct,  _drag)
    print(bar)
    _row("Profit Factor",    lambda m: m.profit_factor, _pf)
    _row("Win Rate",         lambda m: m.win_rate * 100,_wr)
    _row("Avg Hold Time",    lambda m: m.avg_hold_min,  _hold)
    _row("Trades Executed",  lambda m: m.n_trades,      _n)
    print(f"{_B}╚{'═'*86}╝{_RST}")
    print()

    # ── Delta table vs B1 ──────────────────────────────────────────────────────
    base = metrics[0]  # B1
    others = metrics[1:]

    print(f"{_B}DELTA vs B1 (baseline){_RST}")
    print(bar)
    print(f"  {'Metric':<22}", end="")
    for m in others:
        print(f"  {m.variant:>{col_w}}", end="")
    print()
    print(bar)

    def _dcol(v, w, positive_good=True):
        c = (_G if positive_good else _R) if v > 0 else ((_R if positive_good else _G) if v < 0 else "")
        return f"{c}{v:>+{w}.2f}{_RST}"

    def _dpct(v, w, positive_good=True):
        c = (_G if positive_good else _R) if v > 0 else ((_R if positive_good else _G) if v < 0 else "")
        return f"{c}{v:>+{w-1}.1f}%{_RST}"

    print(f"  {'Net PnL Δ':<22}", end="")
    for m in others:
        print(f"  {_dcol(m.net_pnl - base.net_pnl, col_w)}", end="")
    print()

    print(f"  {'Fee Drag Δ (%pts)':<22}", end="")
    for m in others:
        print(f"  {_dpct(m.fee_drag_pct - base.fee_drag_pct, col_w, positive_good=False)}", end="")
    print()

    print(f"  {'Profit Factor Δ':<22}", end="")
    for m in others:
        delta = m.profit_factor - base.profit_factor if base.profit_factor != float("inf") else 0
        print(f"  {_dcol(delta, col_w)}", end="")
    print()

    print(f"  {'Win Rate Δ (%pts)':<22}", end="")
    for m in others:
        print(f"  {_dpct((m.win_rate - base.win_rate) * 100, col_w)}", end="")
    print()

    print(f"  {'Trades Δ':<22}", end="")
    for m in others:
        delta = m.n_trades - base.n_trades
        c = "" if delta == 0 else (_DIM if delta < 0 else "")
        print(f"  {c}{delta:>+{col_w}}{_RST}", end="")
    print()
    print(bar)
    print()

    # ── Diagnosis ──────────────────────────────────────────────────────────────
    _print_diagnosis(base, others)


def _print_diagnosis(base: VariantMetrics, others: list[VariantMetrics]) -> None:
    print(f"{_B}DIAGNOSIS{_RST}")
    bar = "─" * 88
    print(bar)

    by_variant = {m.variant: m for m in others}
    b2 = by_variant.get("B2")
    b3 = by_variant.get("B3")
    b4 = by_variant.get("B4")
    b5 = by_variant.get("B5")

    findings: list[tuple[str, str, str]] = []

    # A) Fee drag (B4 vs B1)
    if b4:
        fee_improvement = b4.net_pnl - base.net_pnl
        drag_reduction  = base.fee_drag_pct - b4.fee_drag_pct
        severity = "HIGH" if drag_reduction > 20 else ("MODERATE" if drag_reduction > 5 else "LOW")
        colour = _R if severity == "HIGH" else (_Y if severity == "MODERATE" else _G)
        findings.append((
            "A) Fee drag",
            f"{colour}{severity}{_RST}",
            f"Maker exits save {fee_improvement:+.2f} USDT net, "
            f"reduce fee drag by {drag_reduction:+.1f}%pts "
            f"(B1={base.fee_drag_pct:.1f}% → B4={b4.fee_drag_pct:.1f}%)",
        ))

    # B) BTC signal quality (B5 vs B1)
    if b5:
        btc_trades  = base.n_trades - b5.n_trades
        eth_net     = b5.net_pnl
        per_trade_b1 = base.net_pnl / base.n_trades if base.n_trades else 0
        per_trade_b5 = b5.net_pnl / b5.n_trades if b5.n_trades else 0
        pf_diff = b5.profit_factor - base.profit_factor
        severity = "HIGH" if pf_diff > 0.3 else ("MODERATE" if pf_diff > 0.1 else "LOW")
        colour = _R if severity == "HIGH" else (_Y if severity == "MODERATE" else _G)
        findings.append((
            "B) BTC signal quality",
            f"{colour}{severity}{_RST}",
            f"ETH-only ({b5.n_trades} trades, {btc_trades} BTC dropped): "
            f"net {eth_net:+.2f} vs {base.net_pnl:+.2f}, "
            f"PF {b5.profit_factor:.3f} vs {base.profit_factor:.3f}, "
            f"per-trade avg {per_trade_b5:+.2f} vs {per_trade_b1:+.2f}",
        ))

    # C) Position sizing (B2 vs B1 — ratios)
    if b2:
        pf_diff = abs(b2.profit_factor - base.profit_factor)
        drag_diff = abs(b2.fee_drag_pct - base.fee_drag_pct)
        # Ratios should be identical; if not, there's a rounding/slippage interaction
        if pf_diff < 0.001 and drag_diff < 0.1:
            verdict = f"{_G}NOT ROOT CAUSE{_RST}"
            note = (f"Half-size halves absolute PnL ({b2.net_pnl:+.2f} vs {base.net_pnl:+.2f}) "
                    f"but PF and fee drag % are identical — sizing scales the problem, doesn't fix it")
        else:
            verdict = f"{_Y}MINOR EFFECT{_RST}"
            note = (f"Small ratio differences (PF Δ={pf_diff:.3f}, drag Δ={drag_diff:.1f}%pts) "
                    f"suggest slippage or rounding interaction with size")
        findings.append(("C) Position sizing", verdict, note))

    # D) Execution structure (B3 vs B1 — fixed stop vs trail)
    if b3:
        net_diff = b3.net_pnl - base.net_pnl
        pf_diff  = b3.profit_factor - base.profit_factor
        hold_diff = base.avg_hold_min - b3.avg_hold_min
        if net_diff > 0:
            verdict = f"{_R}TRAIL IS HURTING{_RST}"
            note = (f"Fixed stop outperforms trail by {net_diff:+.2f} USDT net; "
                    f"trail adds {hold_diff:.1f} extra bars on average with no PnL benefit")
        elif net_diff < -10:
            verdict = f"{_G}TRAIL IS HELPING{_RST}"
            note = (f"Trail beats fixed stop by {abs(net_diff):.2f} USDT; "
                    f"holding longer ({hold_diff:+.1f} bars) captures more of the move")
        else:
            verdict = f"{_DIM}NEUTRAL{_RST}"
            note = f"Fixed vs trail delta is negligible ({net_diff:+.2f} USDT)"
        findings.append(("D) Execution structure", verdict, note))

    for label, verdict, note in findings:
        print(f"  {_B}{label:<25}{_RST}  {verdict}")
        print(f"  {_DIM}{note}{_RST}")
        print()

    # Root cause summary
    print(bar)
    print(f"  {_B}ROOT CAUSE SUMMARY{_RST}")
    print(bar)
    active: list[str] = []
    if b4 and (b4.net_pnl - base.net_pnl) > 5:
        active.append("A) fee drag (maker exits meaningfully improve net PnL)")
    if b5 and b5.profit_factor > base.profit_factor + 0.1:
        active.append("B) BTC signal quality (ETH-only improves PF)")
    if b2 and abs(b2.profit_factor - base.profit_factor) < 0.001:
        active.append("  → C) position sizing is NOT the root cause (ratios unchanged at half size)")
    if b3 and b3.net_pnl > base.net_pnl:
        active.append("D) execution structure (trailing may be increasing exposure without reward)")
    for finding in active:
        colour = _G if finding.startswith("  →") else _Y
        print(f"  {colour}▸ {finding}{_RST}")
    if not active:
        print(f"  {_DIM}No variant shows clear improvement — losses may be driven by raw signal quality.{_RST}")
    print(bar)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Friction reduction study — Model B variants")
    p.add_argument("--journal-dir", type=Path, default=Path("logs/paper"))
    p.add_argument("--candle-dir",  type=Path, default=Path("data/candles"))
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.journal_dir.exists():
        print(f"Journal directory not found: {args.journal_dir}")
        sys.exit(1)

    events  = _load_events(args.journal_dir)
    entries = _extract_entries(events)

    if not entries:
        print("No FILL entries found. Run the paper trader first.")
        sys.exit(1)

    print(f"Loaded {len(entries)} entries from {args.journal_dir}")

    symbols = list({e.symbol for e in entries})

    if not args.candle_dir.exists():
        print(f"Candle directory not found ({args.candle_dir}). "
              "1m parquet files required for exit simulation.")
        sys.exit(1)

    candles_1m = _load_candles_1m(args.candle_dir, symbols)
    n_matched  = _match_to_candles(entries, candles_1m)
    print(f"Candle match: {n_matched}/{len(entries)} entries → 1m bars")

    if n_matched == 0:
        print("No candle matches. Ensure parquet files cover the same session as the journal.")
        sys.exit(1)

    dispatchers = {
        "B1": run_B1,
        "B2": run_B2,
        "B3": run_B3,
        "B4": run_B4,
        "B5": run_B5,
    }

    all_results: dict[str, list[TradeResult]] = {v: [] for v in dispatchers}

    for entry in entries:
        if entry.candle_idx < 0:
            continue
        bars = _bars_after(candles_1m[entry.symbol], entry.candle_idx)
        if not bars:
            continue
        for variant, fn in dispatchers.items():
            result = fn(entry, bars)
            if result is not None:
                all_results[variant].append(result)

    n_sim = sum(1 for e in entries if e.candle_idx >= 0)
    print(f"Simulated {n_sim} trades × 5 variants\n")

    metrics = [_compute_metrics(v, all_results[v]) for v in ["B1", "B2", "B3", "B4", "B5"]]
    _print_results(metrics)


if __name__ == "__main__":
    main()
