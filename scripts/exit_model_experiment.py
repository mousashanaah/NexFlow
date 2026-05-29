#!/usr/bin/env python3
"""Trade Management Experiment: 4 competing exit models on identical entries.

Entries are FROZEN. Only exit logic varies. Signal generation is untouched.

Models
------
  A  Current  TP1 50% / TP2 25% / TP3 25%  — BE after TP1, maker fees at TPs
  B  2-ATR trailing stop, full position, taker fee at exit
  C  50% at TP1, then break-even + 5-bar swing-low trail for remainder
  D  Time-based: exit full position at close of bar 10 (≈10 minutes)

Fees
----
  Entry  : taker 0.06%  (taken from journal, same for all models)
  TP exit: maker 0.02%  (limit order assumption)
  Stop / trail / time exit: taker 0.06%

Stop slippage (adverse fill)
-----------------------------
  stop_fill = stop ∓ (stop_distance × 0.025)   [mirrors PaperExecution.simulate_stop]

Usage
-----
  python scripts/exit_model_experiment.py --journal-dir logs/paper --candle-dir data/candles
  python scripts/exit_model_experiment.py --journal-dir logs/paper --candle-dir data/candles --no-per-trade
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import pyarrow.parquet as pq

from nexflow.services.candles.candle_engine import Candle

# ── Constants ─────────────────────────────────────────────────────────────────

TAKER_FEE   = 0.0006   # 0.06 %
MAKER_FEE   = 0.0002   # 0.02 %
STOP_SLIP_K = 0.025    # stop_distance × 0.025 adverse fill
ATR_WINDOW  = 14       # rolling ATR window for Model B
SWING_BARS  = 5        # swing-low/high lookback for Model C
TIME_BARS   = 10       # hold bars for Model D
MAX_BARS    = 180      # max bars to scan per trade (3 hours)

MODEL_LABELS = {
    "A": "Current  TP1/TP2/TP3",
    "B": "2-ATR    trailing stop",
    "C": "BE+Swing TP1+trail",
    "D": "10-min   time exit",
}

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class EntryRecord:
    seq: int
    symbol: str
    direction: str          # "long" | "short"
    signal_entry: float     # signal entry_price (= candle.close, for matching)
    fill_price: float       # actual fill with slippage
    stop_price: float
    tp_prices: list[float]  # [tp1, tp2, tp3]
    atr: float
    total_size: float
    entry_fee: float
    entry_ts: float         # ts_epoch for ordering
    candle_idx: int = -1    # index into sorted 1m candle list after matching


@dataclass
class TradeResult:
    seq: int
    symbol: str
    direction: str
    entry_price: float
    exit_price: float       # weighted average
    total_size: float
    entry_fee: float
    exit_fees: float
    gross_pnl: float
    net_pnl: float
    hold_bars: int
    exit_reason: str        # STOP / TP1 / TP2 / TP3 / TRAIL / TIME / EOD


@dataclass
class ModelMetrics:
    name: str
    label: str
    net_pnl: float
    gross_pnl: float
    total_fees: float
    profit_factor: float
    win_rate: float
    max_drawdown: float     # fraction, e.g. 0.03 = 3 %
    avg_hold_min: float
    fee_drag_pct: float     # fees / abs(gross_pnl) * 100
    n_trades: int
    n_matched: int          # entries with candle data
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
    """Pair each FILL with its preceding SIGNAL for the same symbol."""
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


# ── Candle loading ────────────────────────────────────────────────────────────

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


def _match_entries_to_candles(
    entries: list[EntryRecord],
    candles_1m: dict[str, list[Candle]],
) -> int:
    """Match each entry to a 1m candle by signal_entry ≈ candle.close (0.1% tol).

    Advances a per-symbol pointer to prevent re-use of the same candle.
    Returns count of successfully matched entries.
    """
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
                break  # found a match; don't wander too far

        if best_i >= 0:
            entry.candle_idx = best_i
            ptrs[sym] = best_i + 1
            matched += 1

    return matched


def _bars_after(candles: list[Candle], idx: int) -> list[Candle]:
    start = idx + 1
    return candles[start : start + MAX_BARS]


# ── Fee / PnL primitives ──────────────────────────────────────────────────────

def _fee(price: float, size: float, maker: bool = False) -> float:
    return price * size * (MAKER_FEE if maker else TAKER_FEE)


def _raw_pnl(direction: str, entry: float, exit_p: float, size: float) -> float:
    return (exit_p - entry) * size if direction == "long" else (entry - exit_p) * size


def _stop_fill(stop: float, direction: str, entry: float) -> float:
    """Adverse fill at stop level (mirrors PaperExecution.simulate_stop)."""
    stop_dist = abs(entry - stop)
    slip = stop_dist * STOP_SLIP_K
    return (stop - slip) if direction == "long" else (stop + slip)


def _wavg(fills: list[tuple[float, float]]) -> float:
    total_size = sum(s for _, s in fills)
    if total_size == 0:
        return 0.0
    return sum(p * s for p, s in fills) / total_size


# ── Rolling ATR helper ────────────────────────────────────────────────────────

def _rolling_atr(bars: Sequence[Candle], window: int = ATR_WINDOW) -> list[float]:
    """Return smoothed ATR for each bar using EMA of (high-low) range."""
    out: list[float] = []
    ema = 0.0
    k = 2.0 / (window + 1)
    for i, b in enumerate(bars):
        r = b.high - b.low
        ema = r * k + ema * (1 - k) if i > 0 else r
        out.append(ema)
    return out


# ── Model A: Current TP1/TP2/TP3 with BE after TP1 ───────────────────────────

def simulate_A(entry: EntryRecord, bars: list[Candle]) -> TradeResult:
    is_long = entry.direction == "long"
    stop = entry.stop_price
    tp1, tp2, tp3 = entry.tp_prices[0], entry.tp_prices[1], entry.tp_prices[2]
    s1 = entry.total_size * 0.50
    s2 = entry.total_size * 0.25
    s3 = entry.total_size * 0.25

    fills: list[tuple[float, float]] = []
    exit_fees = 0.0
    tp1_hit = tp2_hit = tp3_hit = False
    hold_bars = len(bars)
    reason = "EOD"

    for i, bar in enumerate(bars):
        # Stop check (before TPs — stop is set to BE after TP1)
        stop_hit = (is_long and bar.low <= stop) or (not is_long and bar.high >= stop)
        if stop_hit:
            fp = _stop_fill(stop, entry.direction, entry.fill_price)
            rem = entry.total_size - sum(s for _, s in fills)
            fills.append((fp, rem))
            exit_fees += _fee(fp, rem, maker=False)
            hold_bars = i + 1
            reason = "STOP"
            break

        if not tp1_hit:
            hit = (is_long and bar.high >= tp1) or (not is_long and bar.low <= tp1)
            if hit:
                tp1_hit = True
                fills.append((tp1, s1))
                exit_fees += _fee(tp1, s1, maker=True)
                stop = entry.fill_price   # move stop to break-even

        if tp1_hit and not tp2_hit:
            hit = (is_long and bar.high >= tp2) or (not is_long and bar.low <= tp2)
            if hit:
                tp2_hit = True
                fills.append((tp2, s2))
                exit_fees += _fee(tp2, s2, maker=True)

        if tp2_hit and not tp3_hit:
            hit = (is_long and bar.high >= tp3) or (not is_long and bar.low <= tp3)
            if hit:
                tp3_hit = True
                fills.append((tp3, s3))
                exit_fees += _fee(tp3, s3, maker=True)
                hold_bars = i + 1
                reason = "TP3"
                break

    if not fills or reason == "EOD":
        rem = entry.total_size - sum(s for _, s in fills)
        if rem > 0:
            last = bars[-1].close if bars else entry.fill_price
            fills.append((last, rem))
            exit_fees += _fee(last, rem, maker=False)
            reason = ("TP1" if tp1_hit and not tp2_hit else
                      "TP2" if tp2_hit and not tp3_hit else
                      "EOD")

    gross = sum(_raw_pnl(entry.direction, entry.fill_price, p, s) for p, s in fills)
    net = gross - entry.entry_fee - exit_fees
    return TradeResult(
        seq=entry.seq, symbol=entry.symbol, direction=entry.direction,
        entry_price=entry.fill_price, exit_price=_wavg(fills),
        total_size=entry.total_size, entry_fee=entry.entry_fee, exit_fees=exit_fees,
        gross_pnl=gross, net_pnl=net, hold_bars=hold_bars, exit_reason=reason,
    )


# ── Model B: 2-ATR trailing stop, full position ───────────────────────────────

def simulate_B(entry: EntryRecord, bars: list[Candle]) -> TradeResult:
    is_long = entry.direction == "long"
    # Seed ATR from signal; warm up rolling ATR with available bars
    seed_atr = entry.atr if entry.atr > 0 else (abs(entry.fill_price - entry.stop_price) / 1.5)
    atrs = _rolling_atr(bars)

    # Initial trail stop = signal stop (1.5 ATR away)
    trail = entry.stop_price
    hold_bars = len(bars)
    reason = "EOD"

    for i, bar in enumerate(bars):
        atr_i = atrs[i] if atrs[i] > 0 else seed_atr

        # Check current trail stop
        stop_hit = (is_long and bar.low <= trail) or (not is_long and bar.high >= trail)
        if stop_hit:
            fp = _stop_fill(trail, entry.direction, entry.fill_price)
            fill = (fp, entry.total_size)
            exit_fees = _fee(fp, entry.total_size, maker=False)
            hold_bars = i + 1
            reason = "TRAIL"
            gross = _raw_pnl(entry.direction, entry.fill_price, fp, entry.total_size)
            net = gross - entry.entry_fee - exit_fees
            return TradeResult(
                seq=entry.seq, symbol=entry.symbol, direction=entry.direction,
                entry_price=entry.fill_price, exit_price=fp,
                total_size=entry.total_size, entry_fee=entry.entry_fee, exit_fees=exit_fees,
                gross_pnl=gross, net_pnl=net, hold_bars=hold_bars, exit_reason=reason,
            )

        # Update trail: only move in the profitable direction
        new_trail = (bar.close - 2 * atr_i) if is_long else (bar.close + 2 * atr_i)
        if is_long:
            trail = max(trail, new_trail)
        else:
            trail = min(trail, new_trail)

    # No exit hit — flat at last close
    last = bars[-1].close if bars else entry.fill_price
    exit_fees = _fee(last, entry.total_size, maker=False)
    gross = _raw_pnl(entry.direction, entry.fill_price, last, entry.total_size)
    net = gross - entry.entry_fee - exit_fees
    return TradeResult(
        seq=entry.seq, symbol=entry.symbol, direction=entry.direction,
        entry_price=entry.fill_price, exit_price=last,
        total_size=entry.total_size, entry_fee=entry.entry_fee, exit_fees=exit_fees,
        gross_pnl=gross, net_pnl=net, hold_bars=len(bars), exit_reason="EOD",
    )


# ── Model C: 50% at TP1 then BE + 5-bar swing trail ──────────────────────────

def simulate_C(entry: EntryRecord, bars: list[Candle]) -> TradeResult:
    is_long = entry.direction == "long"
    stop = entry.stop_price
    tp1 = entry.tp_prices[0]
    s1 = entry.total_size * 0.50
    s2 = entry.total_size * 0.50   # remainder after TP1

    fills: list[tuple[float, float]] = []
    exit_fees = 0.0
    tp1_hit = False
    hold_bars = len(bars)
    reason = "EOD"
    swing_buf: list[float] = []  # lows (long) or highs (short)

    for i, bar in enumerate(bars):
        # Update swing buffer
        swing_buf.append(bar.low if is_long else bar.high)
        if len(swing_buf) > SWING_BARS:
            swing_buf.pop(0)

        # Stop check
        stop_hit = (is_long and bar.low <= stop) or (not is_long and bar.high >= stop)
        if stop_hit:
            fp = _stop_fill(stop, entry.direction, entry.fill_price)
            rem = entry.total_size - sum(s for _, s in fills)
            fills.append((fp, rem))
            exit_fees += _fee(fp, rem, maker=False)
            hold_bars = i + 1
            reason = "STOP"
            break

        if not tp1_hit:
            hit = (is_long and bar.high >= tp1) or (not is_long and bar.low <= tp1)
            if hit:
                tp1_hit = True
                fills.append((tp1, s1))
                exit_fees += _fee(tp1, s1, maker=True)
                stop = entry.fill_price   # break-even
                reason = "TP1+TRAIL"
        else:
            # Phase 2: trail swing low (long) or swing high (short)
            if len(swing_buf) >= SWING_BARS:
                swing_level = min(swing_buf) if is_long else max(swing_buf)
                if is_long:
                    stop = max(stop, swing_level)
                else:
                    stop = min(stop, swing_level)

    if not fills or reason in ("TP1+TRAIL", "EOD"):
        rem = entry.total_size - sum(s for _, s in fills)
        if rem > 0:
            last = bars[-1].close if bars else entry.fill_price
            fills.append((last, rem))
            exit_fees += _fee(last, rem, maker=False)
            reason = "TP1+EOD" if tp1_hit else "EOD"

    gross = sum(_raw_pnl(entry.direction, entry.fill_price, p, s) for p, s in fills)
    net = gross - entry.entry_fee - exit_fees
    return TradeResult(
        seq=entry.seq, symbol=entry.symbol, direction=entry.direction,
        entry_price=entry.fill_price, exit_price=_wavg(fills),
        total_size=entry.total_size, entry_fee=entry.entry_fee, exit_fees=exit_fees,
        gross_pnl=gross, net_pnl=net, hold_bars=hold_bars, exit_reason=reason,
    )


# ── Model D: 10-bar time exit ─────────────────────────────────────────────────

def simulate_D(entry: EntryRecord, bars: list[Candle]) -> TradeResult:
    is_long = entry.direction == "long"
    stop = entry.stop_price
    target_bar = min(TIME_BARS - 1, len(bars) - 1)   # 0-indexed

    for i, bar in enumerate(bars):
        stop_hit = (is_long and bar.low <= stop) or (not is_long and bar.high >= stop)
        if stop_hit:
            fp = _stop_fill(stop, entry.direction, entry.fill_price)
            exit_fees = _fee(fp, entry.total_size, maker=False)
            gross = _raw_pnl(entry.direction, entry.fill_price, fp, entry.total_size)
            net = gross - entry.entry_fee - exit_fees
            return TradeResult(
                seq=entry.seq, symbol=entry.symbol, direction=entry.direction,
                entry_price=entry.fill_price, exit_price=fp,
                total_size=entry.total_size, entry_fee=entry.entry_fee, exit_fees=exit_fees,
                gross_pnl=gross, net_pnl=net, hold_bars=i + 1, exit_reason="STOP",
            )
        if i == target_bar:
            break

    exit_price = bars[target_bar].close if bars else entry.fill_price
    exit_fees = _fee(exit_price, entry.total_size, maker=False)
    gross = _raw_pnl(entry.direction, entry.fill_price, exit_price, entry.total_size)
    net = gross - entry.entry_fee - exit_fees
    return TradeResult(
        seq=entry.seq, symbol=entry.symbol, direction=entry.direction,
        entry_price=entry.fill_price, exit_price=exit_price,
        total_size=entry.total_size, entry_fee=entry.entry_fee, exit_fees=exit_fees,
        gross_pnl=gross, net_pnl=net, hold_bars=target_bar + 1, exit_reason="TIME",
    )


# ── Metrics ───────────────────────────────────────────────────────────────────

INITIAL_EQUITY = 100_000.0


def _compute_metrics(name: str, results: list[TradeResult]) -> ModelMetrics:
    if not results:
        return ModelMetrics(name=name, label=MODEL_LABELS[name], net_pnl=0, gross_pnl=0,
                            total_fees=0, profit_factor=0, win_rate=0, max_drawdown=0,
                            avg_hold_min=0, fee_drag_pct=0, n_trades=0, n_matched=0,
                            results=results)

    gross_wins = sum(r.gross_pnl for r in results if r.gross_pnl > 0)
    gross_loss = sum(abs(r.gross_pnl) for r in results if r.gross_pnl < 0)
    net_wins = sum(r.net_pnl for r in results if r.net_pnl > 0)
    net_loss = sum(abs(r.net_pnl) for r in results if r.net_pnl <= 0)

    profit_factor = (gross_wins / gross_loss) if gross_loss > 0 else float("inf")
    win_rate = sum(1 for r in results if r.net_pnl > 0) / len(results)
    total_fees = sum(r.entry_fee + r.exit_fees for r in results)
    gross_pnl = sum(r.gross_pnl for r in results)
    net_pnl = sum(r.net_pnl for r in results)
    avg_hold = sum(r.hold_bars for r in results) / len(results)
    fee_drag = (total_fees / abs(gross_pnl) * 100) if gross_pnl != 0 else 0.0

    # Max drawdown from simulated equity curve
    equity = INITIAL_EQUITY
    peak = equity
    max_dd = 0.0
    for r in results:
        equity += r.net_pnl
        peak = max(peak, equity)
        dd = (peak - equity) / peak
        max_dd = max(max_dd, dd)

    return ModelMetrics(
        name=name, label=MODEL_LABELS[name],
        net_pnl=net_pnl, gross_pnl=gross_pnl, total_fees=total_fees,
        profit_factor=profit_factor, win_rate=win_rate,
        max_drawdown=max_dd, avg_hold_min=avg_hold,
        fee_drag_pct=fee_drag,
        n_trades=len(results), n_matched=len(results),
        results=results,
    )


# ── Printing ──────────────────────────────────────────────────────────────────

_G = "\033[92m"
_R = "\033[91m"
_Y = "\033[93m"
_C = "\033[96m"
_B = "\033[1m"
_RST = "\033[0m"


def _col(v: float, w: int = 10, prefix: str = "") -> str:
    colour = _G if v > 0 else (_R if v < 0 else "")
    return f"{colour}{prefix}{v:>{w}.2f}{_RST}"


def _pct(v: float, w: int = 7) -> str:
    colour = _G if v > 0 else (_R if v < 0 else "")
    return f"{colour}{v:>{w}.1f}%{_RST}"


def _print_summary(metrics: list[ModelMetrics]) -> None:
    bar = "─" * 76
    print()
    print(f"{_B}╔══ EXIT MODEL EXPERIMENT ═══════════════════════════════════════════════╗{_RST}")
    print(f"  {_C}Entries are IDENTICAL across all models.  Only exit logic varies.{_RST}")
    print(f"  Initial equity: {INITIAL_EQUITY:,.0f} USDT   "
          f"Trades: {metrics[0].n_trades}   "
          f"Matched to candles: {metrics[0].n_matched}")
    print(bar)

    # Header row
    print(f"  {'Metric':<22}", end="")
    for m in metrics:
        print(f"  {'Model '+m.name:>12}", end="")
    print()
    print(f"  {'':22}", end="")
    for m in metrics:
        lbl = MODEL_LABELS[m.name][:12]
        print(f"  {_C}{lbl:>12}{_RST}", end="")
    print()
    print(bar)

    # Net PnL
    print(f"  {'Net PnL (USDT)':<22}", end="")
    for m in metrics:
        print(f"  {_col(m.net_pnl, 12)}", end="")
    print()

    # Gross PnL
    print(f"  {'Gross PnL (USDT)':<22}", end="")
    for m in metrics:
        print(f"  {_col(m.gross_pnl, 12)}", end="")
    print()

    # Total Fees
    print(f"  {'Total Fees (USDT)':<22}", end="")
    for m in metrics:
        print(f"  {_col(-m.total_fees, 12)}", end="")
    print()

    # Fee Drag
    print(f"  {'Fee Drag %':<22}", end="")
    for m in metrics:
        colour = _G if m.fee_drag_pct < 50 else (_Y if m.fee_drag_pct < 100 else _R)
        print(f"  {colour}{m.fee_drag_pct:>11.1f}%{_RST}", end="")
    print()

    print(bar)

    # Profit Factor
    print(f"  {'Profit Factor':<22}", end="")
    for m in metrics:
        colour = _G if m.profit_factor >= 1.5 else (_Y if m.profit_factor >= 1.0 else _R)
        pf = f"{m.profit_factor:.3f}" if m.profit_factor != float("inf") else "  ∞"
        print(f"  {colour}{pf:>12}{_RST}", end="")
    print()

    # Win Rate
    print(f"  {'Win Rate':<22}", end="")
    for m in metrics:
        wr = m.win_rate * 100
        colour = _G if wr >= 50 else (_Y if wr >= 40 else _R)
        print(f"  {colour}{wr:>10.1f}%{_RST}", end="")
    print()

    print(bar)

    # Max Drawdown
    print(f"  {'Max Drawdown':<22}", end="")
    for m in metrics:
        dd = m.max_drawdown * 100
        colour = _G if dd < 1 else (_Y if dd < 3 else _R)
        print(f"  {colour}{dd:>10.3f}%{_RST}", end="")
    print()

    # Avg Hold Time
    print(f"  {'Avg Hold Time':<22}", end="")
    for m in metrics:
        print(f"  {m.avg_hold_min:>11.1f}m", end="")
    print()

    print(bar)

    # Exit reason breakdown (one line per model to handle variable width)
    print(f"  {'Exit Reasons':<22}")
    for m in metrics:
        reasons: dict[str, int] = defaultdict(int)
        for r in m.results:
            reasons[r.exit_reason] += 1
        summary = "  ".join(f"{k}:{v}" for k, v in sorted(reasons.items()))
        print(f"    Model {m.name}: {summary}")

    print(f"{_B}╚{'═'*74}╝{_RST}")
    print()


def _print_per_trade(entries: list[EntryRecord], all_results: dict[str, list[TradeResult]]) -> None:
    bar = "─" * 110
    print(f"{_B}PER-TRADE DETAIL{_RST}")
    print(bar)
    header = (f"  {'#':>3}  {'Sym':>8}  {'Dir':>5}  {'Entry':>10}  "
              f"{'A Net':>9}  {'B Net':>9}  {'C Net':>9}  {'D Net':>9}  "
              f"{'A Bars':>6}  {'B Bars':>6}  {'C Bars':>6}  {'D Bars':>6}  "
              f"{'A Exit':>8}  {'B Exit':>8}  {'C Exit':>8}  {'D Exit':>8}")
    print(header)
    print(bar)

    results_by_seq: dict[str, dict[int, TradeResult]] = {}
    for model, results in all_results.items():
        results_by_seq[model] = {r.seq: r for r in results}

    for entry in entries:
        if entry.candle_idx < 0:
            print(f"  {entry.seq:>3}  {entry.symbol:>8}  {entry.direction:>5}  "
                  f"{entry.fill_price:>10.2f}  [no candle data]")
            continue

        row = (f"  {entry.seq:>3}  {entry.symbol:>8}  {entry.direction:>5}  "
               f"{entry.fill_price:>10.4f}")

        for m in ["A", "B", "C", "D"]:
            r = results_by_seq.get(m, {}).get(entry.seq)
            if r:
                c = _G if r.net_pnl > 0 else _R
                row += f"  {c}{r.net_pnl:>+9.2f}{_RST}"
            else:
                row += f"  {'N/A':>9}"

        for m in ["A", "B", "C", "D"]:
            r = results_by_seq.get(m, {}).get(entry.seq)
            row += f"  {r.hold_bars:>6}" if r else f"  {'N/A':>6}"

        for m in ["A", "B", "C", "D"]:
            r = results_by_seq.get(m, {}).get(entry.seq)
            row += f"  {r.exit_reason:>8}" if r else f"  {'N/A':>8}"

        print(row)

    print(bar)
    print()


def _print_delta_analysis(metrics: list[ModelMetrics]) -> None:
    """Show improvement/degradation of B/C/D versus baseline Model A."""
    base = next((m for m in metrics if m.name == "A"), None)
    if base is None:
        return

    bar = "─" * 76
    print(f"{_B}DELTA vs MODEL A (Current){_RST}")
    print(bar)
    print(f"  {'Metric':<22}  {'B vs A':>12}  {'C vs A':>12}  {'D vs A':>12}")
    print(bar)

    comparisons = [m for m in metrics if m.name != "A"]

    def _delta_col(v: float, w: int = 12) -> str:
        c = _G if v > 0 else (_R if v < 0 else "")
        return f"{c}{v:>+{w}.2f}{_RST}"

    print(f"  {'Net PnL Δ (USDT)':<22}", end="")
    for m in comparisons:
        print(f"  {_delta_col(m.net_pnl - base.net_pnl)}", end="")
    print()

    print(f"  {'Gross PnL Δ (USDT)':<22}", end="")
    for m in comparisons:
        print(f"  {_delta_col(m.gross_pnl - base.gross_pnl)}", end="")
    print()

    print(f"  {'Fee Drag Δ (%pts)':<22}", end="")
    for m in comparisons:
        delta = m.fee_drag_pct - base.fee_drag_pct
        c = _G if delta < 0 else _R
        print(f"  {c}{delta:>+12.1f}{_RST}", end="")
    print()

    print(f"  {'Win Rate Δ (%pts)':<22}", end="")
    for m in comparisons:
        delta = (m.win_rate - base.win_rate) * 100
        c = _G if delta > 0 else _R
        print(f"  {c}{delta:>+12.1f}{_RST}", end="")
    print()

    print(f"  {'Max DD Δ (%pts)':<22}", end="")
    for m in comparisons:
        delta = (m.max_drawdown - base.max_drawdown) * 100
        c = _G if delta < 0 else _R
        print(f"  {c}{delta:>+12.3f}{_RST}", end="")
    print()

    print(f"  {'Hold Time Δ (bars)':<22}", end="")
    for m in comparisons:
        delta = m.avg_hold_min - base.avg_hold_min
        print(f"  {delta:>+12.1f}", end="")
    print()

    print(bar)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exit model experiment — identical entries, 4 exit strategies")
    p.add_argument("--journal-dir", type=Path, default=Path("logs/paper"))
    p.add_argument("--candle-dir",  type=Path, default=Path("data/candles"))
    p.add_argument("--no-per-trade", action="store_true", help="Skip per-trade detail table")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Load journal ──────────────────────────────────────────────────────────
    if not args.journal_dir.exists():
        print(f"Journal directory not found: {args.journal_dir}")
        sys.exit(1)

    events = _load_events(args.journal_dir)
    entries = _extract_entries(events)

    if not entries:
        print("No FILL entries found in journal. Run the paper trader first.")
        sys.exit(1)

    print(f"Loaded {len(entries)} entries from {args.journal_dir}")

    # ── Load candles ──────────────────────────────────────────────────────────
    symbols = list({e.symbol for e in entries})
    candles_available = False

    if args.candle_dir.exists():
        candles_1m = _load_candles_1m(args.candle_dir, symbols)
        n_matched = _match_entries_to_candles(entries, candles_1m)
        candles_available = n_matched > 0
        print(f"Candle data: {n_matched}/{len(entries)} entries matched to 1m bars")
    else:
        candles_1m = {s: [] for s in symbols}
        n_matched = 0
        print(f"Candle directory not found ({args.candle_dir}). "
              "Models B/C/D require 1m candles.")

    if not candles_available:
        print("\nNo candle data available. Cannot run exit simulations.")
        print("Ensure data/candles/ contains <SYMBOL>_1m.parquet files from the same session.")
        print("Run 'python scripts/run_candle_engine.py' to collect candle data.")
        sys.exit(1)

    # ── Simulate all models ───────────────────────────────────────────────────
    sim_funcs = {"A": simulate_A, "B": simulate_B, "C": simulate_C, "D": simulate_D}
    all_results: dict[str, list[TradeResult]] = {m: [] for m in sim_funcs}

    for entry in entries:
        if entry.candle_idx < 0:
            continue
        bars = _bars_after(candles_1m[entry.symbol], entry.candle_idx)
        if not bars:
            continue
        for model, fn in sim_funcs.items():
            all_results[model].append(fn(entry, bars))

    matched_entries = [e for e in entries if e.candle_idx >= 0]
    n_sim = len(matched_entries)
    print(f"Simulated {n_sim} trades × 4 models = {n_sim * 4} trade outcomes\n")

    # ── Compute metrics ───────────────────────────────────────────────────────
    metrics = [_compute_metrics(m, all_results[m]) for m in ["A", "B", "C", "D"]]

    # ── Print results ─────────────────────────────────────────────────────────
    _print_summary(metrics)
    _print_delta_analysis(metrics)

    if not args.no_per_trade:
        _print_per_trade(matched_entries, all_results)

    # ── Best model by net PnL ─────────────────────────────────────────────────
    best = max(metrics, key=lambda m: m.net_pnl)
    print(f"{_B}Best net PnL: Model {best.name} ({MODEL_LABELS[best.name]})  "
          f"{_col(best.net_pnl, 10)} USDT{_RST}")
    print()


if __name__ == "__main__":
    main()
