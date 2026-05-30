#!/usr/bin/env python3
"""Inversion backtest: run the strategy with all directions flipped.

Answers one question:
    Is the current MomentumStrategy an exhaustion signal in momentum clothing?

Method:
    Run BacktestRunner twice on the same candle data, same config:
      1. Original strategy  (LONG where signal says LONG)
      2. Inverted strategy  (SHORT where signal says LONG, LONG where SHORT)

    Stop and TP levels are recalculated for the inverted direction —
    stops remain 1.5×ATR away, TPs remain at 1/2/3×ATR. Only direction flips.

    Filter attribution table shows which filters help or hurt when inverted.

Usage:
    python scripts/inversion_backtest.py --candle-dir data/candles
    python scripts/inversion_backtest.py --candle-dir data/candles --start 2023-01 --end 2025-05
"""

from __future__ import annotations

import argparse
import math
import sys
from calendar import monthrange
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import pyarrow.parquet as pq

from nexflow.services.candles.candle_engine import Candle
from nexflow.services.strategy.backtest_runner import BacktestConfig, BacktestRunner
from nexflow.services.strategy.base_strategy import BaseStrategy
from nexflow.services.strategy.momentum_strategy import (
    MomentumConfig, MomentumStrategy,
    compute_atr, compute_relative_volume, compute_breakout_level,
    compute_range_expansion, compute_buy_sell_imbalance, compute_momentum_slope,
)
from nexflow.services.strategy.paper_execution import ExecutionConfig
from nexflow.services.strategy.risk_engine import RiskConfig
from nexflow.services.strategy.signal_models import (
    BacktestMetrics, ClosedTrade, Direction, Signal,
)

_W = 76
TAKER = 0.0006


# ---------------------------------------------------------------------------
# Inverted strategy wrapper
# ---------------------------------------------------------------------------

class InvertedMomentumStrategy(BaseStrategy):
    """Wraps MomentumStrategy and flips every signal direction.

    Stop and TP prices are recomputed for the inverted direction so that
    risk/reward geometry is preserved:
      - stop remains 1.5×ATR away from entry (opposite side)
      - TPs remain at 1/2/3×ATR away from entry (opposite side)
    """

    TRIGGER_TF = "1m"
    CONTEXT_TFS = {"5m", "15m"}

    def __init__(self, cfg: MomentumConfig | None = None) -> None:
        self._inner = MomentumStrategy(cfg)
        self._cfg = cfg or MomentumConfig()

    def on_candle(self, candle: Candle) -> Signal | None:
        sig = self._inner.on_candle(candle)
        if sig is None:
            return None
        return _invert_signal(sig, self._cfg)

    def reset(self) -> None:
        self._inner.reset()


def _invert_signal(sig: Signal, cfg: MomentumConfig) -> Signal:
    """Return a new Signal with direction flipped and levels recomputed."""
    if sig.direction is Direction.LONG:
        new_dir   = Direction.SHORT
        new_stop  = sig.entry_price + cfg.atr_stop_mult * sig.atr
        new_tps   = [
            sig.entry_price - cfg.atr_tp1_mult * sig.atr,
            sig.entry_price - cfg.atr_tp2_mult * sig.atr,
            sig.entry_price - cfg.atr_tp3_mult * sig.atr,
        ]
    else:
        new_dir   = Direction.LONG
        new_stop  = sig.entry_price - cfg.atr_stop_mult * sig.atr
        new_tps   = [
            sig.entry_price + cfg.atr_tp1_mult * sig.atr,
            sig.entry_price + cfg.atr_tp2_mult * sig.atr,
            sig.entry_price + cfg.atr_tp3_mult * sig.atr,
        ]

    return Signal(
        symbol=sig.symbol,
        direction=new_dir,
        timeframe=sig.timeframe,
        bar_close_time=sig.bar_close_time,
        entry_price=sig.entry_price,
        stop_price=new_stop,
        tp_prices=new_tps,
        atr=sig.atr,
        features=sig.features,
    )


# ---------------------------------------------------------------------------
# Candle loading
# ---------------------------------------------------------------------------

def _rows_to_candles(rows: list[dict]) -> list[Candle]:
    return [
        Candle(
            symbol=r["symbol"], timeframe=r["timeframe"],
            open_time=r["open_time"], close_time=r["close_time"],
            open=r["open"], high=r["high"], low=r["low"], close=r["close"],
            volume=r["volume"], buy_volume=r["buy_volume"],
            sell_volume=r["sell_volume"], trade_count=r["trade_count"],
            vwap=r["vwap"], spread_avg=r["spread_avg"],
            spread_max=r["spread_max"],
            volatility_estimate=r["volatility_estimate"],
            is_final=True,
        )
        for r in rows
        if r.get("is_final")
    ]


def _load_all_candles(candle_dir: Path, symbol: str,
                      start_s: int, end_s: int
                      ) -> dict[str, dict[str, list[Candle]]]:
    """Return {symbol: {timeframe: [Candle]}} for use with BacktestRunner."""
    result: dict[str, dict[str, list[Candle]]] = {symbol: {}}

    for tf in ("1m", "5m", "15m"):
        path = candle_dir / f"{symbol}_{tf}.parquet"
        if not path.exists():
            continue
        rows = pq.read_table(path).to_pylist()
        candles = sorted(
            [c for c in _rows_to_candles(rows)
             if start_s <= c.close_time <= end_s],
            key=lambda c: c.close_time,
        )
        if candles:
            result[symbol][tf] = candles

    if "1m" not in result[symbol]:
        print(f"[ERROR] {candle_dir}/{symbol}_1m.parquet not found.")
        sys.exit(1)

    return result


# ---------------------------------------------------------------------------
# Month-by-month regime comparison
# ---------------------------------------------------------------------------

@dataclass
class MonthResult:
    ym: str
    orig_trades: int
    orig_wr: float
    orig_pf: float
    orig_pnl: float
    inv_trades: int
    inv_wr: float
    inv_pf: float
    inv_pnl: float


def _run_month(all_candles: dict[str, dict[str, list[Candle]]],
               symbol: str,
               ym: str,
               bt_cfg: BacktestConfig) -> tuple[BacktestMetrics, BacktestMetrics]:
    """Run original + inverted on one month's candles. Returns (orig, inv)."""
    y, m = int(ym[:4]), int(ym[5:7])
    _, last_day = monthrange(y, m)
    mo_start = int(datetime(y, m, 1, tzinfo=timezone.utc).timestamp())
    mo_end   = int(datetime(y, m, last_day, 23, 59, 59, tzinfo=timezone.utc).timestamp())

    month_candles: dict[str, dict[str, list[Candle]]] = {symbol: {}}
    for tf, bars in all_candles[symbol].items():
        filtered = [c for c in bars if mo_start <= c.close_time <= mo_end]
        if filtered:
            month_candles[symbol][tf] = filtered

    if not month_candles[symbol].get("1m"):
        empty = BacktestMetrics(
            total_trades=0, win_rate=0.0, expectancy=0.0, sharpe=0.0,
            max_drawdown=0.0, profit_factor=0.0, avg_hold_bars=0.0,
            net_pnl=0.0, total_fees=0.0,
            long_trades=0, long_win_rate=0.0,
            short_trades=0, short_win_rate=0.0,
            pnl_distribution=[], equity_curve=[],
        )
        return empty, empty

    orig = BacktestRunner(MomentumStrategy(), bt_cfg).run(month_candles)
    inv  = BacktestRunner(InvertedMomentumStrategy(), bt_cfg).run(month_candles)
    return orig, inv


# ---------------------------------------------------------------------------
# Filter attribution for inverted signal
# ---------------------------------------------------------------------------

@dataclass
class FilterEntry:
    bar_idx: int
    direction: int          # inverted direction (+1 long, -1 short)
    close_at_entry: float
    atr: float
    passed_relVol: bool
    passed_range: bool
    passed_imbalance: bool
    passed_momentum: bool
    returns: dict[int, float] = field(default_factory=dict)


HORIZONS = [1, 3, 5, 10, 20]


def _aggregate_5m(candles_1m: list[Candle]) -> list[Candle]:
    buckets: dict[int, list[Candle]] = defaultdict(list)
    for c in candles_1m:
        key = (c.open_time // 300) * 300
        buckets[key].append(c)
    result: list[Candle] = []
    for key in sorted(buckets):
        bars = sorted(buckets[key], key=lambda c: c.open_time)
        sym = bars[0].symbol
        result.append(Candle(
            symbol=sym, timeframe="5m",
            open_time=bars[0].open_time,
            close_time=bars[-1].close_time,
            open=bars[0].open,
            high=max(b.high for b in bars),
            low=min(b.low  for b in bars),
            close=bars[-1].close,
            volume=sum(b.volume for b in bars),
            buy_volume=sum(b.buy_volume for b in bars),
            sell_volume=sum(b.sell_volume for b in bars),
            trade_count=sum(b.trade_count for b in bars),
            vwap=bars[-1].close,
            spread_avg=0.0, spread_max=0.0,
            volatility_estimate=0.0, is_final=True,
        ))
    return result


def collect_filter_attribution(candles_1m: list[Candle],
                                candles_5m: list[Candle],
                                inverted: bool = False) -> list[FilterEntry]:
    """Collect all breakout bars with per-filter pass/fail, optionally inverted."""
    cfg = MomentumConfig()
    entries: list[FilterEntry] = []

    p5 = 0
    n5 = len(candles_5m)
    win1m: list[Candle] = []
    win5m_buf: list[Candle] = []

    for i, c in enumerate(candles_1m):
        win1m.append(c)
        while p5 < n5 and candles_5m[p5].close_time <= c.close_time:
            win5m_buf.append(candles_5m[p5])
            p5 += 1
        win5m = win5m_buf[-(cfg.min_bars_5m + 10):]

        if len(win1m) < cfg.min_bars_1m:
            continue

        trigger  = win1m[-1]
        atr      = compute_atr(win1m, cfg.atr_period)
        if atr <= 0:
            continue

        rel_vol   = compute_relative_volume(win1m, cfg.vol_period)
        range_exp = compute_range_expansion(trigger, atr)
        imbalance = compute_buy_sell_imbalance(trigger)
        rh, rl    = compute_breakout_level(win1m, cfg.vol_period)
        momentum  = (compute_momentum_slope(win5m, cfg.momentum_period_5m)
                     if len(win5m) >= cfg.min_bars_5m else 0.0)

        long_breakout  = trigger.close > rh
        short_breakout = trigger.close < rl
        if not (long_breakout or short_breakout):
            continue

        signal_dir = +1 if long_breakout else -1
        direction  = -signal_dir if inverted else signal_dir

        mom_ok = ((signal_dir == +1 and momentum > 0) or
                  (signal_dir == -1 and momentum < 0))
        imb_ok = (imbalance >= cfg.imbalance_min if signal_dir == +1
                  else imbalance <= (1.0 - cfg.imbalance_min))

        entries.append(FilterEntry(
            bar_idx=i,
            direction=direction,
            close_at_entry=trigger.close,
            atr=atr,
            passed_relVol=(rel_vol >= cfg.rel_vol_threshold),
            passed_range=(range_exp >= cfg.range_expansion_min),
            passed_imbalance=imb_ok,
            passed_momentum=mom_ok,
        ))

    return entries


def fill_filter_returns(entries: list[FilterEntry],
                        candles_1m: list[Candle]) -> None:
    closes = [c.close for c in candles_1m]
    n = len(closes)
    for e in entries:
        for h in HORIZONS:
            idx = e.bar_idx + h
            if idx < n:
                e.returns[h] = e.direction * (closes[idx] - e.close_at_entry) / e.close_at_entry
            else:
                e.returns[h] = float("nan")


def _mean(vals: list[float]) -> float:
    v = [x for x in vals if not math.isnan(x)]
    return sum(v) / len(v) if v else float("nan")


def _dir_acc(vals: list[float]) -> float:
    v = [x for x in vals if not math.isnan(x)]
    return sum(1 for x in v if x > 0) / len(v) if v else float("nan")


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _pf(gross_profit: float, gross_loss: float) -> float:
    return gross_profit / gross_loss if gross_loss > 0 else float("inf")


def _fmt_pct(v: float, dp: int = 1) -> str:
    if math.isnan(v) or math.isinf(v):
        return "  —  "
    return f"{v*100:+.{dp}f}%"


def _fmt_f(v: float, dp: int = 2) -> str:
    if math.isnan(v) or math.isinf(v):
        return "  —  "
    return f"{v:.{dp}f}"


def _bar(v: float, lo: float, hi: float, width: int = 20) -> str:
    """ASCII bar: position v within [lo, hi]."""
    if math.isnan(v) or hi == lo:
        return "─" * width
    frac = max(0.0, min(1.0, (v - lo) / (hi - lo)))
    filled = int(frac * width)
    return "█" * filled + "░" * (width - filled)


def _print_side_by_side(orig: BacktestMetrics, inv: BacktestMetrics,
                         label_orig: str = "ORIGINAL",
                         label_inv: str  = "INVERTED") -> None:
    col = 28
    row_fmt = f"  {{:<24}}  {{:>{col}}}  {{:>{col}}}"
    sep = "  " + "─" * (24 + 2 + col + 2 + col)

    print(row_fmt.format("", label_orig, label_inv))
    print(sep)
    print(row_fmt.format("Total trades",
                          f"{orig.total_trades:,}", f"{inv.total_trades:,}"))
    print(row_fmt.format("Win rate",
                          _fmt_pct(orig.win_rate), _fmt_pct(inv.win_rate)))
    print(row_fmt.format("Profit factor",
                          _fmt_f(orig.profit_factor), _fmt_f(inv.profit_factor)))
    print(row_fmt.format("Expectancy (R)",
                          _fmt_f(orig.expectancy, 4), _fmt_f(inv.expectancy, 4)))
    print(row_fmt.format("Sharpe (ann.)",
                          _fmt_f(orig.sharpe), _fmt_f(inv.sharpe)))
    print(row_fmt.format("Max drawdown",
                          _fmt_pct(orig.max_drawdown), _fmt_pct(inv.max_drawdown)))
    print(row_fmt.format("Net PnL (USD)",
                          f"{orig.net_pnl:+,.0f}", f"{inv.net_pnl:+,.0f}"))
    print(row_fmt.format("Total fees (USD)",
                          f"{orig.total_fees:,.0f}", f"{inv.total_fees:,.0f}"))
    print(row_fmt.format("Long WR / Short WR",
                          f"{orig.long_win_rate*100:.1f}% / {orig.short_win_rate*100:.1f}%",
                          f"{inv.long_win_rate*100:.1f}% / {inv.short_win_rate*100:.1f}%"))
    print(sep)


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

def print_report(orig: BacktestMetrics,
                 inv:  BacktestMetrics,
                 month_results: list[MonthResult],
                 orig_filter: list[FilterEntry],
                 inv_filter:  list[FilterEntry],
                 candles_1m:  list[Candle],
                 symbol: str,
                 initial_equity: float) -> None:

    period_start = datetime.fromtimestamp(candles_1m[0].close_time,  tz=timezone.utc).strftime("%Y-%m-%d")
    period_end   = datetime.fromtimestamp(candles_1m[-1].close_time, tz=timezone.utc).strftime("%Y-%m-%d")

    print()
    print("═" * _W)
    print(f"  INVERSION BACKTEST — {symbol}")
    print(f"  Period: {period_start} → {period_end}")
    print(f"  Bars: {len(candles_1m):,} × 1m   Initial equity: ${initial_equity:,.0f}")
    print("═" * _W)

    # -----------------------------------------------------------------------
    # Section 1: Side-by-side headline metrics
    # -----------------------------------------------------------------------
    print()
    print("  1. HEADLINE METRICS")
    print()
    _print_side_by_side(orig, inv)

    # -----------------------------------------------------------------------
    # Section 2: Monthly profitability
    # -----------------------------------------------------------------------
    print()
    print("  2. MONTHLY REGIME COMPARISON")
    print()
    mr = [m for m in month_results if m.orig_trades > 0 or m.inv_trades > 0]
    if mr:
        hdr = f"  {'Month':<10}  {'Orig tr':>7}  {'Orig WR':>8}  {'Orig PF':>8}  {'Orig PnL':>10}  {'Inv tr':>7}  {'Inv WR':>8}  {'Inv PF':>8}  {'Inv PnL':>10}  {'Winner':<8}"
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))
        orig_wins = inv_wins = 0
        for m in mr:
            winner = ""
            if m.inv_pnl > 0 and m.orig_pnl <= 0:
                winner = "INV ▲"
                inv_wins += 1
            elif m.orig_pnl > 0 and m.inv_pnl <= 0:
                winner = "ORIG ▲"
                orig_wins += 1
            elif m.orig_pnl > 0 and m.inv_pnl > 0:
                winner = "BOTH ▲"
                orig_wins += 1; inv_wins += 1
            else:
                winner = "both ▼"
            opf = f"{m.orig_pf:.2f}" if not math.isinf(m.orig_pf) else "∞"
            ipf = f"{m.inv_pf:.2f}"  if not math.isinf(m.inv_pf)  else "∞"
            print(f"  {m.ym:<10}  {m.orig_trades:>7}  {m.orig_wr*100:>7.1f}%  {opf:>8}  {m.orig_pnl:>+10.0f}  "
                  f"{m.inv_trades:>7}  {m.inv_wr*100:>7.1f}%  {ipf:>8}  {m.inv_pnl:>+10.0f}  {winner:<8}")
        print()
        orig_profitable = sum(1 for m in mr if m.orig_pnl > 0)
        inv_profitable  = sum(1 for m in mr if m.inv_pnl  > 0)
        print(f"  Profitable months — Original: {orig_profitable}/{len(mr)}  "
              f"({orig_profitable/len(mr)*100:.0f}%)   "
              f"Inverted: {inv_profitable}/{len(mr)}  "
              f"({inv_profitable/len(mr)*100:.0f}%)")
    else:
        print("  No monthly data available.")

    # -----------------------------------------------------------------------
    # Section 3: Filter attribution — Original vs Inverted at +5 bars
    # -----------------------------------------------------------------------
    print()
    print("  3. FILTER ATTRIBUTION  (mean signed forward return at +5 bars)")
    print("     Positive = direction was correct.  Layers are cumulative.")
    print()

    def _layer_stats(ents: list[FilterEntry], mask_fn) -> tuple[float, float, int]:
        vals = [e.returns.get(5, float("nan")) for e in ents if mask_fn(e)]
        vals = [v for v in vals if not math.isnan(v)]
        if not vals:
            return float("nan"), float("nan"), 0
        m = sum(vals) / len(vals)
        acc = sum(1 for v in vals if v > 0) / len(vals)
        return m, acc, len(vals)

    layers = [
        ("Breakout only",                  lambda e: True),
        ("+ rel_vol ≥ 1.5",               lambda e: e.passed_relVol),
        ("+ range_exp ≥ 0.8",             lambda e: e.passed_relVol and e.passed_range),
        ("+ momentum_5m aligned",         lambda e: e.passed_relVol and e.passed_range and e.passed_momentum),
        ("+ imbalance (= full signal)",   lambda e: e.passed_relVol and e.passed_range and e.passed_momentum and e.passed_imbalance),
    ]

    col2 = 22
    print(f"  {'Layer':<40}  {'ORIG ret':>{col2}}  {'ORIG acc':>{col2}}  {'INV ret':>{col2}}  {'INV acc':>{col2}}  {'N (orig)':>8}")
    print("  " + "─" * (40 + 4 + col2 * 4 + 3 * 2 + 10))
    for label, mask in layers:
        om, oa, on = _layer_stats(orig_filter, mask)
        im, ia, _  = _layer_stats(inv_filter,  mask)
        mark = " ◀ SIGNAL" if "full signal" in label else ""
        print(f"  {label:<40}  {_fmt_pct(om, 4):>{col2}}  {_fmt_pct(oa, 1):>{col2}}  "
              f"{_fmt_pct(im, 4):>{col2}}  {_fmt_pct(ia, 1):>{col2}}  {on:>8,}{mark}")
    print()

    # -----------------------------------------------------------------------
    # Section 4: PnL distribution comparison
    # -----------------------------------------------------------------------
    print("  4. PnL DISTRIBUTION COMPARISON")
    print()
    for label, m in [("Original", orig), ("Inverted", inv)]:
        if m.pnl_distribution:
            s = sorted(m.pnl_distribution)
            n = len(s)
            p5  = s[max(0, int(n * 0.05))]
            p25 = s[max(0, int(n * 0.25))]
            p50 = s[n // 2]
            p75 = s[min(n-1, int(n * 0.75))]
            p95 = s[min(n-1, int(n * 0.95))]
            avg_win  = sum(x for x in s if x > 0) / max(1, sum(1 for x in s if x > 0))
            avg_loss = sum(x for x in s if x < 0) / max(1, sum(1 for x in s if x < 0))
            print(f"  {label}:")
            print(f"    p5={p5:+.1f}  p25={p25:+.1f}  median={p50:+.1f}  p75={p75:+.1f}  p95={p95:+.1f}")
            print(f"    avg_win={avg_win:+.2f}  avg_loss={avg_loss:+.2f}  "
                  f"win/loss ratio={(avg_win/abs(avg_loss)):+.2f}" if avg_loss != 0 else "")
            print()

    # -----------------------------------------------------------------------
    # Section 5: Verdict
    # -----------------------------------------------------------------------
    print("═" * _W)
    print("  VERDICT")
    print("═" * _W)
    print()

    orig_pnl_positive  = orig.net_pnl > 0
    inv_pnl_positive   = inv.net_pnl  > 0
    orig_months_pct    = sum(1 for m in mr if m.orig_pnl > 0) / len(mr) if mr else 0
    inv_months_pct     = sum(1 for m in mr if m.inv_pnl  > 0) / len(mr) if mr else 0
    inv_beats_orig_pnl = inv.net_pnl > orig.net_pnl
    inv_beats_orig_pf  = inv.profit_factor > orig.profit_factor

    print(f"  Original  — PnL: {orig.net_pnl:+,.0f}  PF: {orig.profit_factor:.2f}  "
          f"WR: {orig.win_rate*100:.1f}%  profitable months: {orig_months_pct*100:.0f}%")
    print(f"  Inverted  — PnL: {inv.net_pnl:+,.0f}  PF: {inv.profit_factor:.2f}  "
          f"WR: {inv.win_rate*100:.1f}%  profitable months: {inv_months_pct*100:.0f}%")
    print()

    if inv_pnl_positive and not orig_pnl_positive:
        verdict = "A"
        print("  FINDING: Case A — GENUINELY ANTI-PREDICTIVE")
        print()
        print("  The inverted strategy is profitable where the original is not.")
        print("  This confirms the entry logic is an exhaustion signal, not a")
        print("  momentum signal. When price closes above the 20-bar high with")
        print("  high volume + range expansion + aligned 5m momentum + buy imbalance,")
        print("  the move is statistically complete — not beginning.")
        print()
        print("  The strategy is entering at local exhaustion peaks/troughs.")
        print("  The correct trade is the OPPOSITE: fade the breakout, not follow it.")
        print()
        if inv_months_pct > 0.55:
            print(f"  The inverted strategy is profitable in {inv_months_pct*100:.0f}% of months,")
            print("  suggesting a real and consistent edge in the opposite direction.")
        else:
            print(f"  The inverted strategy profits in {inv_months_pct*100:.0f}% of months —")
            print("  the edge is genuine but not consistent across all regimes.")

    elif not inv_pnl_positive and not orig_pnl_positive:
        verdict = "B"
        print("  FINDING: Case B — NO EDGE IN EITHER DIRECTION")
        print()
        print("  Both the original and inverted strategies lose money.")
        print("  The signal selects bars where price is about to chop sideways.")
        print("  There is no directional edge encoded in the filter combination.")
        print()
        print("  The filters (rel_vol, range_exp, momentum, imbalance) are selecting")
        print("  for high-activity bars, but high-activity does not predict direction.")
        print("  The 57% historical win rate was most likely regime-specific.")

    elif inv_pnl_positive and orig_pnl_positive:
        verdict = "AB"
        print("  FINDING: Case A/B — BOTH DIRECTIONS PROFITABLE (check for data artefact)")
        print()
        print("  Both strategies show positive PnL. This is unexpected and should")
        print("  be treated with suspicion — verify that there is no look-ahead bias")
        print("  or overfitting in the candle data before drawing conclusions.")

    else:
        verdict = "C"
        print("  FINDING: Case C — INVERTED STRATEGY BEATS ORIGINAL BUT NOT PROFITABLE")
        print()
        print("  The inverted strategy loses less than the original, suggesting")
        print("  the signal has anti-predictive directional content but the raw")
        print("  edge is still consumed by fees and stop distances.")
        if inv_beats_orig_pf:
            print(f"  Inverted PF ({inv.profit_factor:.2f}) > Original PF ({orig.profit_factor:.2f})")
            print("  — directional flip helps, but execution costs remain the problem.")

    print()
    print("  EXHAUSTION SIGNAL DIAGNOSIS:")
    print()

    # The definitive test: is the inverted direction accuracy above 50% at +5 bars?
    inv_5_vals = [e.returns.get(5, float("nan")) for e in inv_filter
                  if (e.passed_relVol and e.passed_range and
                      e.passed_momentum and e.passed_imbalance)]
    inv_5_vals = [v for v in inv_5_vals if not math.isnan(v)]
    if inv_5_vals:
        inv_dir_acc = sum(1 for v in inv_5_vals if v > 0) / len(inv_5_vals)
        inv_mean_ret = sum(inv_5_vals) / len(inv_5_vals)
        print(f"  Inverted full-signal direction accuracy at +5 bars: {inv_dir_acc*100:.1f}%")
        print(f"  Inverted full-signal mean return at +5 bars:        {inv_mean_ret*100:+.4f}%")
        print(f"  Fee breakeven (2×taker):                            {TAKER*2*100:+.4f}%")
        print()
        if inv_dir_acc > 0.55 and inv_mean_ret > TAKER * 2:
            print("  ✓ Inverted signal: direction accuracy > 55% AND mean return > fee breakeven.")
            print("  The current filters are EXHAUSTION INDICATORS.")
            print("  All five conditions (breakout + high vol + range expansion +")
            print("  aligned momentum + imbalance) together mark LOCAL TOPS/BOTTOMS,")
            print("  not the start of sustained trends.")
        elif inv_dir_acc > 0.52:
            print("  ~ Inverted signal has modest directional edge (>52%) but mean")
            print("  return may not clear fees after stops.")
        else:
            print("  ✗ Inverted signal direction accuracy ≤ 52%: no clear directional edge")
            print("  in either direction at the +5 bar horizon.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inversion backtest")
    p.add_argument("--symbol",      default="ETHUSDT")
    p.add_argument("--candle-dir",  default="data/candles", type=Path)
    p.add_argument("--start",       default="2023-01",
                   help="Start month YYYY-MM")
    p.add_argument("--end",         default=None,
                   help="End month YYYY-MM (default: all)")
    p.add_argument("--equity",      default=100_000.0, type=float)
    p.add_argument("--risk",        default=0.005,     type=float)
    p.add_argument("--no-monthly",  action="store_true",
                   help="Skip month-by-month breakdown (faster)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    candle_dir = _REPO_ROOT / args.candle_dir
    if not candle_dir.exists():
        candle_dir = Path(args.candle_dir)

    sy, sm = [int(x) for x in args.start.split("-")]
    if args.end:
        ey, em = [int(x) for x in args.end.split("-")]
    else:
        ey, em = 2099, 12

    _, last_day = monthrange(min(ey, 2099), min(em, 12))
    start_s = int(datetime(sy, sm, 1, tzinfo=timezone.utc).timestamp())
    end_s   = int(datetime(min(ey, 2099), min(em, 12), last_day,
                            23, 59, 59, tzinfo=timezone.utc).timestamp())

    print(f"\nInversion backtest — {args.symbol}")
    print(f"Loading candles from {candle_dir} …")
    all_candles = _load_all_candles(candle_dir, args.symbol, start_s, end_s)
    n_1m = len(all_candles[args.symbol].get("1m", []))
    n_5m = len(all_candles[args.symbol].get("5m", []))
    print(f"  {n_1m:,} × 1m   {n_5m:,} × 5m   {len(all_candles[args.symbol].get('15m', [])):,} × 15m")

    if n_1m < 1000:
        print("[ERROR] Too few bars.")
        sys.exit(1)

    bt_cfg = BacktestConfig(
        initial_equity=args.equity,
        risk=RiskConfig(max_risk_per_trade=args.risk),
        execution=ExecutionConfig(),
    )

    print("Running original strategy …")
    orig = BacktestRunner(MomentumStrategy(), bt_cfg).run(all_candles)
    print(f"  {orig.total_trades} trades  WR={orig.win_rate*100:.1f}%  PF={orig.profit_factor:.2f}  PnL={orig.net_pnl:+,.0f}")

    print("Running inverted strategy …")
    inv = BacktestRunner(InvertedMomentumStrategy(), bt_cfg).run(all_candles)
    print(f"  {inv.total_trades} trades  WR={inv.win_rate*100:.1f}%  PF={inv.profit_factor:.2f}  PnL={inv.net_pnl:+,.0f}")

    # Month-by-month
    month_results: list[MonthResult] = []
    if not args.no_monthly:
        print("Running month-by-month regime comparison …")
        # Enumerate months in range
        y, mo = sy, sm
        while (y < ey) or (y == ey and mo <= em):
            ym = f"{y:04d}-{mo:02d}"
            o_m, i_m = _run_month(all_candles, args.symbol, ym, bt_cfg)
            if o_m.total_trades > 0 or i_m.total_trades > 0:
                month_results.append(MonthResult(
                    ym=ym,
                    orig_trades=o_m.total_trades, orig_wr=o_m.win_rate,
                    orig_pf=o_m.profit_factor,    orig_pnl=o_m.net_pnl,
                    inv_trades=i_m.total_trades,  inv_wr=i_m.win_rate,
                    inv_pf=i_m.profit_factor,     inv_pnl=i_m.net_pnl,
                ))
                print(f"  {ym}: orig={o_m.total_trades}tr/{o_m.net_pnl:+.0f}  "
                      f"inv={i_m.total_trades}tr/{i_m.net_pnl:+.0f}")
            mo += 1
            if mo > 12:
                mo = 1; y += 1

    # Filter attribution
    print("Computing filter attribution …")
    candles_1m = all_candles[args.symbol]["1m"]
    candles_5m = all_candles[args.symbol].get("5m") or _aggregate_5m(candles_1m)

    orig_filter = collect_filter_attribution(candles_1m, candles_5m, inverted=False)
    inv_filter  = collect_filter_attribution(candles_1m, candles_5m, inverted=True)
    fill_filter_returns(orig_filter, candles_1m)
    fill_filter_returns(inv_filter,  candles_1m)
    print(f"  {len(orig_filter):,} breakout bars analysed")

    print_report(orig, inv, month_results, orig_filter, inv_filter,
                 candles_1m, args.symbol, args.equity)


if __name__ == "__main__":
    main()
