#!/usr/bin/env python3
"""Straddle audit — test whether signal bars have a volatility-forecasting edge.

Measures post-signal volatility expansion in four independent ways:

  1. ATR expansion ratio
       ATR(bars[N+1 .. N+14]) / ATR(bars[N-14 .. N])
       > 1.0 = volatility regime shifted upward after signal

  2. Bar range expansion ratio
       mean(high-low, bars[N+1..N+K]) / mean(high-low, bars[N-K..N])
       Measures whether individual bar ranges widen post-signal

  3. Maximum excursion probability (uses bar high/low, not close)
       P(max_excursion_either_direction ≥ k×ATR within H bars)
       for k = 0.5, 1.0, 1.5, 2.0, 3.0 and H = 5, 10, 20
       This is the correct measure for straddle viability

  4. Straddle expectancy simulation
       Open BOTH a long and short at the signal close price:
         long leg : entry=close, stop=close-1.5×ATR, TP=close+TP_mult×ATR
         short leg: entry=close, stop=close+1.5×ATR, TP=close-TP_mult×ATR
       Walk bars using high/low, find first event for each leg.
       Net PnL = long_PnL + short_PnL − fees (4× taker, all-taker worst case)
       Tested at TP multiples [1.0, 1.5, 2.0, 2.5, 3.0]
       Size = equal unit (1 contract each leg, ATR-normalised results)

All metrics compared: signal entries vs random entries vs breakout-only.
Welch t-tests flag statistically significant differences.

Usage:
  python scripts/straddle_audit.py --candle-dir data/candles --start 2023-01
  python scripts/straddle_audit.py --candle-dir data/candles --stop-mult 1.0
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    import pyarrow.parquet as pq
except ImportError:
    print("[ERROR] pyarrow required: pip install pyarrow")
    sys.exit(1)

from nexflow.services.candles.candle_engine import Candle
from nexflow.services.strategy.momentum_strategy import (
    MomentumConfig, MomentumStrategy, compute_atr,
)

_W = 80
TAKER    = 0.0006
MAKER    = 0.0002
HORIZONS = [5, 10, 20]
TP_MULTS = [1.0, 1.5, 2.0, 2.5, 3.0]
STOP_DEFAULT = 1.5   # ATR multiples — matches live strategy


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StraddleEntry:
    bar_idx:        int
    close_at_entry: float
    atr:            float            # ATR at entry bar
    pre_atr:        float            # ATR of pre-signal window
    entry_type:     str              # "signal" | "random" | "breakout"
    direction:      int              # original signal direction (+1/-1), 0 for random

    # Filled by compute_metrics()
    post_atr:       dict[int, float] = field(default_factory=dict)  # window → ATR
    atr_ratio:      dict[int, float] = field(default_factory=dict)  # post/pre ATR
    range_ratio:    dict[int, float] = field(default_factory=dict)  # post/pre mean range
    max_excursion:  dict[int, float] = field(default_factory=dict)  # h → max |price−entry| / ATR
    # straddle results: {(tp_mult, horizon): net_pnl_in_atr_units}
    straddle_pnl:   dict[tuple[float, int], float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Candle loading (shared pattern)
# ---------------------------------------------------------------------------

def _rows_to_candles(rows: list[dict], start_s: int, end_s: int) -> list[Candle]:
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
        if r.get("is_final") and start_s <= r.get("close_time", 0) <= end_s
    ]


def _aggregate_5m(c1m: list[Candle]) -> list[Candle]:
    buckets: dict[int, list[Candle]] = defaultdict(list)
    for c in c1m:
        buckets[(c.open_time // 300) * 300].append(c)
    result: list[Candle] = []
    for key in sorted(buckets):
        bars = sorted(buckets[key], key=lambda c: c.open_time)
        sym = bars[0].symbol
        result.append(Candle(
            symbol=sym, timeframe="5m",
            open_time=bars[0].open_time, close_time=bars[-1].close_time,
            open=bars[0].open, high=max(b.high for b in bars),
            low=min(b.low for b in bars), close=bars[-1].close,
            volume=sum(b.volume for b in bars),
            buy_volume=sum(b.buy_volume for b in bars),
            sell_volume=sum(b.sell_volume for b in bars),
            trade_count=sum(b.trade_count for b in bars),
            vwap=bars[-1].close, spread_avg=0.0, spread_max=0.0,
            volatility_estimate=0.0, is_final=True,
        ))
    return result


def _load(candle_dir: Path, symbol: str,
          start_s: int, end_s: int) -> tuple[list[Candle], list[Candle]]:
    p1 = candle_dir / f"{symbol}_1m.parquet"
    p5 = candle_dir / f"{symbol}_5m.parquet"
    if not p1.exists():
        print(f"[ERROR] {p1} not found.")
        sys.exit(1)
    c1m = sorted(_rows_to_candles(pq.read_table(p1).to_pylist(), start_s, end_s),
                 key=lambda c: c.close_time)
    if p5.exists():
        c5m = sorted(_rows_to_candles(pq.read_table(p5).to_pylist(), start_s, end_s),
                     key=lambda c: c.close_time)
    else:
        print("  [WARN] 5m parquet missing — aggregating from 1m")
        c5m = _aggregate_5m(c1m)
    return c1m, sorted(c1m + c5m, key=lambda c: c.close_time)


# ---------------------------------------------------------------------------
# Entry collection
# ---------------------------------------------------------------------------

_MIN_PRE = 16   # bars needed before entry for pre-ATR window


def collect_signal_entries(c1m: list[Candle],
                            merged: list[Candle]) -> list[StraddleEntry]:
    idx_by_close: dict[int, int] = {c.close_time: i for i, c in enumerate(c1m)}
    strategy = MomentumStrategy()
    entries: list[StraddleEntry] = []
    for c in merged:
        sig = strategy.on_candle(c)
        if sig is None or c.timeframe != "1m":
            continue
        i = idx_by_close.get(c.close_time, -1)
        if i < _MIN_PRE or i + max(HORIZONS) + 1 >= len(c1m):
            continue
        pre_atr = compute_atr(c1m[i - _MIN_PRE : i], 14)
        d = 1 if sig.direction.value == "long" else -1
        entries.append(StraddleEntry(
            bar_idx=i, close_at_entry=c.close,
            atr=sig.atr, pre_atr=pre_atr,
            entry_type="signal", direction=d,
        ))
    return entries


def collect_random_entries(c1m: list[Candle],
                            n: int, seed: int = 42) -> list[StraddleEntry]:
    rng = random.Random(seed)
    eligible = list(range(_MIN_PRE + 1, len(c1m) - max(HORIZONS) - 2))
    sample = rng.sample(eligible, min(n, len(eligible)))
    entries: list[StraddleEntry] = []
    for i in sample:
        c = c1m[i]
        atr     = compute_atr(c1m[max(0, i-14):i+1], 14)
        pre_atr = compute_atr(c1m[i - _MIN_PRE : i], 14)
        atr     = max(atr, c.close * 0.0001)
        pre_atr = max(pre_atr, c.close * 0.0001)
        entries.append(StraddleEntry(
            bar_idx=i, close_at_entry=c.close,
            atr=atr, pre_atr=pre_atr,
            entry_type="random", direction=rng.choice([+1, -1]),
        ))
    return entries


def collect_breakout_entries(c1m: list[Candle],
                              period: int = 20) -> list[StraddleEntry]:
    entries: list[StraddleEntry] = []
    for i in range(period + _MIN_PRE, len(c1m) - max(HORIZONS) - 2):
        trigger  = c1m[i]
        lookback = c1m[i - period : i]
        rh = max(b.high for b in lookback)
        rl = min(b.low  for b in lookback)
        if trigger.close <= rh and trigger.close >= rl:
            continue
        atr     = compute_atr(c1m[max(0, i-14):i+1], 14)
        pre_atr = compute_atr(c1m[i - _MIN_PRE : i], 14)
        atr     = max(atr, trigger.close * 0.0001)
        pre_atr = max(pre_atr, trigger.close * 0.0001)
        d = +1 if trigger.close > rh else -1
        entries.append(StraddleEntry(
            bar_idx=i, close_at_entry=trigger.close,
            atr=atr, pre_atr=pre_atr,
            entry_type="breakout", direction=d,
        ))
    return entries


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _mean_range(bars: list[Candle]) -> float:
    if not bars:
        return float("nan")
    return sum(b.high - b.low for b in bars) / len(bars)


def _simulate_straddle(entry: float, atr: float,
                        bars: list[Candle],
                        stop_mult: float, tp_mult: float,
                        ) -> float:
    """Simulate a long+short straddle.  Returns net PnL in ATR units.

    Rules (applied bar-by-bar using high/low):
      long leg  : stop = entry - stop_mult×ATR  TP = entry + tp_mult×ATR
      short leg : stop = entry + stop_mult×ATR  TP = entry - tp_mult×ATR

    Within each bar, if both stop AND TP are touched, the stop is assumed
    first (conservative — same rule as live backtester).

    For a straddle, once one leg is closed the other continues.
    If both legs closed in the same bar (TP on one, stop on the other),
    that bar's resolution is the same-bar stop-priority rule applied
    independently to each leg.

    Returns PnL in ATR units (so 1.0 = gained 1×ATR across both legs).
    Fees: 4 taker exits (2 entries + 2 exits, worst case all-taker).
    """
    if atr <= 0:
        return float("nan")

    long_stop  = entry - stop_mult * atr
    long_tp    = entry + tp_mult   * atr
    short_stop = entry + stop_mult * atr
    short_tp   = entry - tp_mult   * atr

    long_pnl  = None
    short_pnl = None
    last_close = entry

    for bar in bars:
        last_close = bar.close

        if long_pnl is None:
            stop_hit = bar.low  <= long_stop
            tp_hit   = bar.high >= long_tp
            if stop_hit:   # stop priority
                long_pnl = -stop_mult
            elif tp_hit:
                long_pnl = +tp_mult

        if short_pnl is None:
            stop_hit = bar.high >= short_stop
            tp_hit   = bar.low  <= short_tp
            if stop_hit:
                short_pnl = -stop_mult
            elif tp_hit:
                short_pnl = +tp_mult

        if long_pnl is not None and short_pnl is not None:
            break

    # Force-close any open legs at last bar close
    if long_pnl is None:
        long_pnl  = (last_close - entry) / atr
    if short_pnl is None:
        short_pnl = (entry - last_close) / atr

    # Fees in ATR units: 2 entries + 2 exits, all taker
    fee_atr = 4 * TAKER * entry / atr
    return long_pnl + short_pnl - fee_atr


def compute_metrics(entries: list[StraddleEntry],
                    c1m: list[Candle],
                    stop_mult: float) -> None:
    n = len(c1m)
    post_window = 14   # bars for post-signal ATR

    for e in entries:
        i   = e.bar_idx
        cl  = e.close_at_entry
        atr = e.atr
        if atr <= 0:
            atr = e.pre_atr
        if atr <= 0:
            continue

        # ── 1. Post-signal ATR ratio ────────────────────────────────────────
        for h in HORIZONS:
            end_idx = min(i + h + 1, n)
            post_bars = c1m[i + 1 : end_idx]
            if len(post_bars) >= 2:
                pa = compute_atr(post_bars, min(len(post_bars) - 1, post_window))
                e.post_atr[h]   = pa
                e.atr_ratio[h]  = pa / e.pre_atr if e.pre_atr > 0 else float("nan")
            else:
                e.post_atr[h]  = float("nan")
                e.atr_ratio[h] = float("nan")

        # ── 2. Bar range expansion ratio ────────────────────────────────────
        pre_bars = c1m[max(0, i - post_window) : i]
        pre_rng  = _mean_range(pre_bars)
        for h in HORIZONS:
            post_bars_h = c1m[i + 1 : min(i + h + 1, n)]
            post_rng    = _mean_range(post_bars_h)
            e.range_ratio[h] = (post_rng / pre_rng
                                 if pre_rng > 0 and not math.isnan(post_rng)
                                 else float("nan"))

        # ── 3. Maximum excursion (uses bar high/low) ─────────────────────────
        for h in HORIZONS:
            max_exc = 0.0
            for bar in c1m[i + 1 : min(i + h + 1, n)]:
                max_exc = max(max_exc,
                              abs(bar.high - cl),
                              abs(bar.low  - cl))
            e.max_excursion[h] = max_exc / atr   # in ATR units

        # ── 4. Straddle expectancy ───────────────────────────────────────────
        for tp_mult in TP_MULTS:
            for h in HORIZONS:
                bars = c1m[i + 1 : min(i + h + 1, n)]
                pnl  = _simulate_straddle(cl, atr, bars, stop_mult, tp_mult)
                e.straddle_pnl[(tp_mult, h)] = pnl


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _mean_nnan(vals: list[float]) -> float:
    v = [x for x in vals if not math.isnan(x) and not math.isinf(x)]
    return sum(v) / len(v) if v else float("nan")


def _frac_above(vals: list[float], threshold: float) -> float:
    v = [x for x in vals if not math.isnan(x)]
    return sum(1 for x in v if x >= threshold) / len(v) if v else float("nan")


def _welch_t(a: list[float], b: list[float]) -> tuple[float, float]:
    a = [x for x in a if not math.isnan(x) and not math.isinf(x)]
    b = [x for x in b if not math.isnan(x) and not math.isinf(x)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    na, nb = len(a), len(b)
    ma, mb = sum(a)/na, sum(b)/nb
    va = sum((x-ma)**2 for x in a) / (na-1)
    vb = sum((x-mb)**2 for x in b) / (nb-1)
    se = math.sqrt(va/na + vb/nb)
    if se == 0:
        return float("nan"), float("nan")
    t = (ma - mb) / se
    p = 2 * (1 - _norm_cdf(abs(t)))
    return t, p


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))


def _sig(p: float) -> str:
    if math.isnan(p): return "   "
    if p < 0.001:     return "***"
    if p < 0.01:      return "** "
    if p < 0.05:      return "*  "
    return "   "


def _pct(v: float, dp: int = 3) -> str:
    return f"{v*100:.{dp}f}%" if not math.isnan(v) else "  —  "


def _f(v: float, dp: int = 3) -> str:
    return f"{v:.{dp}f}" if (not math.isnan(v) and not math.isinf(v)) else "  —  "


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(sig: list[StraddleEntry],
                 ran: list[StraddleEntry],
                 bo:  list[StraddleEntry],
                 c1m: list[Candle],
                 symbol: str,
                 stop_mult: float) -> None:

    period_start = datetime.fromtimestamp(c1m[0].close_time,  tz=timezone.utc).strftime("%Y-%m-%d")
    period_end   = datetime.fromtimestamp(c1m[-1].close_time, tz=timezone.utc).strftime("%Y-%m-%d")

    print()
    print("═" * _W)
    print(f"  STRADDLE / VOLATILITY FORECASTING AUDIT — {symbol}")
    print(f"  Period: {period_start} → {period_end}   ({len(c1m):,} bars)")
    print(f"  Signal: {len(sig):,}   Random: {len(ran):,}   Breakout: {len(bo):,}")
    print(f"  Straddle stop: {stop_mult}×ATR each leg   Fees: 4×TAKER={4*TAKER*100:.3f}%")
    print("═" * _W)

    groups = [("Signal", sig), ("Random", ran), ("Breakout", bo)]

    # -----------------------------------------------------------------------
    # Section 1: ATR expansion ratio
    # -----------------------------------------------------------------------
    print()
    print("  1. POST-SIGNAL ATR EXPANSION RATIO  (post_ATR / pre_ATR)")
    print("     > 1.0 = volatility regime shifted upward after signal fired.")
    print(f"     Pre window: {_MIN_PRE} bars.  Post window: up to H bars.")
    print()
    print(f"  {'Horizon':<10}" + "".join(f"  {g[0]:>16}" for g in groups)
          + "  {'Sig−Ran':>10}  {'p':>8}")
    print("  " + "─" * 70)
    for h in HORIZONS:
        vals = {name: [e.atr_ratio[h] for e in ents] for name, ents in groups}
        means = {name: _mean_nnan(v) for name, v in vals.items()}
        _, p = _welch_t(vals["Signal"], vals["Random"])
        diff = means["Signal"] - means["Random"]
        print(f"  +{h:<9d}"
              + "".join(f"  {_f(means[g[0]], 3):>16}" for g in groups)
              + f"  {diff:>+9.3f}  {p:>6.4f}{_sig(p)}")

    # -----------------------------------------------------------------------
    # Section 2: Bar range expansion ratio
    # -----------------------------------------------------------------------
    print()
    print("  2. BAR RANGE EXPANSION RATIO  (mean post-signal bar range / pre-signal)")
    print("     > 1.0 = individual bars are wider after signal (intrabar vol up).")
    print()
    print(f"  {'Horizon':<10}" + "".join(f"  {g[0]:>16}" for g in groups)
          + "  {'Sig−Ran':>10}  {'p':>8}")
    print("  " + "─" * 70)
    for h in HORIZONS:
        vals  = {name: [e.range_ratio[h] for e in ents] for name, ents in groups}
        means = {name: _mean_nnan(v) for name, v in vals.items()}
        _, p  = _welch_t(vals["Signal"], vals["Random"])
        diff  = means["Signal"] - means["Random"]
        print(f"  +{h:<9d}"
              + "".join(f"  {_f(means[g[0]], 3):>16}" for g in groups)
              + f"  {diff:>+9.3f}  {p:>6.4f}{_sig(p)}")

    # -----------------------------------------------------------------------
    # Section 3: Maximum excursion probability (high/low based)
    # -----------------------------------------------------------------------
    print()
    print("  3. MAXIMUM EXCURSION PROBABILITY  (uses bar high/low, either direction)")
    print("     P(max |price − entry| ≥ k×ATR within H bars)")
    print("     This is the correct input for straddle sizing decisions.")
    print()
    thresholds = [0.5, 1.0, 1.5, 2.0, 3.0]

    for h in HORIZONS:
        print(f"  Horizon +{h} bars:")
        print(f"    {'≥ k×ATR':<12}" + "".join(f"  {g[0]:>12}" for g in groups)
              + "  {'Sig−Ran':>10}  {'p':>8}")
        print("    " + "─" * 64)
        for k in thresholds:
            vals  = {name: [e.max_excursion[h] for e in ents] for name, ents in groups}
            fracs = {name: _frac_above(v, k) for name, v in vals.items()}
            _, p  = _welch_t(
                [1.0 if x >= k else 0.0 for x in vals["Signal"] if not math.isnan(x)],
                [1.0 if x >= k else 0.0 for x in vals["Random"] if not math.isnan(x)],
            )
            diff = fracs["Signal"] - fracs["Random"]
            print(f"    ≥{k:<11.1f}"
                  + "".join(f"  {fracs[g[0]]*100:>11.1f}%" for g in groups)
                  + f"  {diff*100:>+9.1f}%  {p:>6.4f}{_sig(p)}")
        print()

    # -----------------------------------------------------------------------
    # Section 4: Straddle expectancy vs TP multiple
    # -----------------------------------------------------------------------
    print()
    print("  4. STRADDLE EXPECTANCY  (net PnL in ATR units, both legs combined)")
    print(f"     Stop = {stop_mult}×ATR each leg.  Fees = 4×TAKER = {4*TAKER*100:.3f}%/ATR.")
    print("     > 0.0 = profitable straddle.  Break-even is where line crosses zero.")
    print()

    for h in HORIZONS:
        print(f"  Horizon +{h} bars:")
        print(f"    {'TP mult':<12}" + "".join(f"  {g[0]:>14}" for g in groups)
              + "  {'Sig−Ran':>10}  {'p':>8}  {'Sig>0':>6}")
        print("    " + "─" * 72)
        for tp in TP_MULTS:
            vals  = {name: [e.straddle_pnl.get((tp, h), float("nan")) for e in ents]
                     for name, ents in groups}
            means = {name: _mean_nnan(v) for name, v in vals.items()}
            _, p  = _welch_t(vals["Signal"], vals["Random"])
            diff  = means["Signal"] - means["Random"]
            pos_frac = _frac_above(vals["Signal"], 0.0)
            mark = " ◀" if (not math.isnan(means["Signal"]) and means["Signal"] > 0) else ""
            print(f"    TP={tp:<7.1f}×ATR"
                  + "".join(f"  {_f(means[g[0]], 3):>14}" for g in groups)
                  + f"  {diff:>+9.3f}  {p:>6.4f}{_sig(p)}  {pos_frac*100:>5.1f}%{mark}")
        print()

    # -----------------------------------------------------------------------
    # Section 5: Straddle expectancy vs TP multiple — SIGNAL ONLY DETAIL
    # -----------------------------------------------------------------------
    print()
    print("  5. SIGNAL STRADDLE DETAIL: mean, median, p5, p95  (at best TP, +10 bars)")
    print()
    # Find best TP for signal at horizon=10
    best_tp_mean = max(TP_MULTS,
                       key=lambda tp: _mean_nnan([e.straddle_pnl.get((tp, 10), float("nan"))
                                                  for e in sig]))
    vals_best = [e.straddle_pnl.get((best_tp_mean, 10), float("nan")) for e in sig]
    vals_best = sorted(v for v in vals_best if not math.isnan(v))
    if vals_best:
        n = len(vals_best)
        print(f"  Best TP multiple: {best_tp_mean}×ATR at +10 bars")
        print(f"  n={n:,}  mean={_f(_mean_nnan(vals_best))}  "
              f"median={_f(vals_best[n//2])}  "
              f"p5={_f(vals_best[max(0,int(n*0.05))])}  "
              f"p95={_f(vals_best[min(n-1,int(n*0.95))])}")
        pct_positive = sum(1 for v in vals_best if v > 0) / n
        print(f"  % positive trades: {pct_positive*100:.1f}%")
        avg_win  = _mean_nnan([v for v in vals_best if v > 0])
        avg_loss = _mean_nnan([v for v in vals_best if v < 0])
        if not math.isnan(avg_loss) and avg_loss != 0:
            print(f"  avg win: {_f(avg_win)} ATR   avg loss: {_f(avg_loss)} ATR   "
                  f"win/loss ratio: {_f(avg_win/abs(avg_loss))}")

    # -----------------------------------------------------------------------
    # Section 6: Year-by-year straddle expectancy (signal only, best TP, +10)
    # -----------------------------------------------------------------------
    print()
    print("  6. YEAR-BY-YEAR STRADDLE EXPECTANCY  (signal, best TP multiple, +10 bars)")
    print()
    by_year: dict[int, list[float]] = defaultdict(list)
    ran_by_year: dict[int, list[float]] = defaultdict(list)
    for e in sig:
        v = e.straddle_pnl.get((best_tp_mean, 10), float("nan"))
        if not math.isnan(v):
            dt = datetime.fromtimestamp(c1m[e.bar_idx].close_time, tz=timezone.utc)
            by_year[dt.year].append(v)
    for e in ran:
        v = e.straddle_pnl.get((best_tp_mean, 10), float("nan"))
        if not math.isnan(v):
            dt = datetime.fromtimestamp(c1m[e.bar_idx].close_time, tz=timezone.utc)
            ran_by_year[dt.year].append(v)

    print(f"  {'Year':<8}  {'N':>6}  {'Mean':>10}  {'Win%':>8}  {'vs Rand':>10}  {'Rand mean':>10}")
    print("  " + "─" * 60)
    for yr in sorted(by_year):
        vals_yr = by_year[yr]
        m       = _mean_nnan(vals_yr)
        win_pct = sum(1 for v in vals_yr if v > 0) / len(vals_yr)
        ran_m   = _mean_nnan(ran_by_year.get(yr, []))
        diff    = m - ran_m if not math.isnan(ran_m) else float("nan")
        flag    = " ▲" if m > 0 else (" ▼" if m < -0.1 else "")
        print(f"  {yr:<8}  {len(vals_yr):>6,}  {_f(m):>10}  {win_pct*100:>7.1f}%"
              f"  {_f(diff):>10}  {_f(ran_m):>10}{flag}")

    # -----------------------------------------------------------------------
    # Section 7: VERDICT
    # -----------------------------------------------------------------------
    print()
    print("═" * _W)
    print("  VERDICT: VOLATILITY FORECASTING EDGE ASSESSMENT")
    print("═" * _W)
    print()

    # Key metrics
    atr_ratio_sig = _mean_nnan([e.atr_ratio[10] for e in sig])
    atr_ratio_ran = _mean_nnan([e.atr_ratio[10] for e in ran])
    _, p_atr      = _welch_t([e.atr_ratio[10] for e in sig],
                              [e.atr_ratio[10] for e in ran])
    _, p_rng      = _welch_t([e.range_ratio[10] for e in sig],
                              [e.range_ratio[10] for e in ran])

    exc2_sig = _frac_above([e.max_excursion[10] for e in sig], 2.0)
    exc2_ran = _frac_above([e.max_excursion[10] for e in ran], 2.0)
    exc3_sig = _frac_above([e.max_excursion[10] for e in sig], 3.0)
    exc3_ran = _frac_above([e.max_excursion[10] for e in ran], 3.0)

    best_straddle_mean = _mean_nnan([e.straddle_pnl.get((best_tp_mean, 10), float("nan"))
                                      for e in sig])
    best_straddle_ran  = _mean_nnan([e.straddle_pnl.get((best_tp_mean, 10), float("nan"))
                                      for e in ran])
    _, p_straddle      = _welch_t(
        [e.straddle_pnl.get((best_tp_mean, 10), float("nan")) for e in sig],
        [e.straddle_pnl.get((best_tp_mean, 10), float("nan")) for e in ran],
    )

    print(f"  ATR ratio @+10:          {_f(atr_ratio_sig)} sig  vs  {_f(atr_ratio_ran)} ran  "
          f"(p={p_atr:.4f}{_sig(p_atr)})")
    print(f"  Range ratio @+10:        (p={p_rng:.4f}{_sig(p_rng)})")
    print(f"  P(≥2×ATR excursion @+10): {exc2_sig*100:.1f}% sig  vs  {exc2_ran*100:.1f}% ran  "
          f"({exc2_sig-exc2_ran:+.1%} diff)")
    print(f"  P(≥3×ATR excursion @+10): {exc3_sig*100:.1f}% sig  vs  {exc3_ran*100:.1f}% ran  "
          f"({exc3_sig-exc3_ran:+.1%} diff)")
    print(f"  Best straddle ({best_tp_mean}×ATR TP, +10): {_f(best_straddle_mean)} ATR sig  "
          f"vs  {_f(best_straddle_ran)} ATR ran  (p={p_straddle:.4f}{_sig(p_straddle)})")
    print()

    # Evaluate each hypothesis
    vol_regime_shift = (not math.isnan(p_atr) and p_atr < 0.05
                        and not math.isnan(atr_ratio_sig) and atr_ratio_sig > atr_ratio_ran)
    excursion_edge   = (exc2_sig - exc2_ran > 0.03 or exc3_sig - exc3_ran > 0.02)
    straddle_viable  = (not math.isnan(best_straddle_mean) and best_straddle_mean > 0
                        and not math.isnan(p_straddle) and p_straddle < 0.05)

    print("  ┌─────────────────────────────────────────────────────────────┐")
    print(f"  │ Vol regime shift (ATR expands post-signal):  "
          f"{'YES ***' if vol_regime_shift else 'NO     ':10}            │")
    print(f"  │ Elevated excursion probability:              "
          f"{'YES' if excursion_edge else 'NO ':10}                        │")
    print(f"  │ Straddle edge (mean PnL > 0, p<0.05):        "
          f"{'YES ***' if straddle_viable else 'NO     ':10}            │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()

    if vol_regime_shift and straddle_viable:
        print("  FINDING: GENUINE VOLATILITY-FORECASTING EDGE")
        print()
        print("  The signal predicts both ATR expansion AND straddle profitability.")
        print("  The current directional stop/TP structure is the wrong instrument.")
        print("  A symmetric, volatility-capturing exit structure could be profitable:")
        print(f"    • Enter both long and short at signal close")
        print(f"    • Stop both legs at {stop_mult}×ATR from entry")
        print(f"    • TP at {best_tp_mean}×ATR — the leg that runs first covers the leg that stops")
        print()
        print("  NEXT STEP: Test a dual-entry implementation in the backtester.")

    elif vol_regime_shift and not straddle_viable:
        print("  FINDING: VOL REGIME SHIFTS BUT STRADDLE IS NOT PROFITABLE")
        print()
        print("  ATR does expand after the signal fires, but the straddle simulation")
        print("  shows fees consume the edge. This is because:")
        print(f"  • The stop ({stop_mult}×ATR) is hit on the losing leg before the TP")
        print(f"    ({best_tp_mean}×ATR) is reached on the winning leg in many trades.")
        print("  • The fee drag (4× taker) exceeds the net expected move.")
        print()
        print("  IMPLICATION: The vol edge exists in principle but requires either:")
        print("    a) Wider TP / tighter stop ratio to improve R multiple")
        print("    b) Maker entry fills to reduce fee drag")
        print("    c) A different instrument (options, funding rate trades)")

    elif excursion_edge and not straddle_viable:
        print("  FINDING: EXCURSION EDGE WITHOUT STRADDLE PROFITABILITY")
        print()
        print("  Signal bars produce larger maximum excursions than random, but")
        print("  the straddle cannot capture this because price frequently reverses")
        print("  after touching the winning TP — the losing leg then gets stopped.")
        print("  The two-sided expansion pattern from the path analysis confirms this.")

    else:
        print("  FINDING: NO EXPLOITABLE VOLATILITY EDGE")
        print()
        print("  The signal does not predict ATR expansion, does not produce")
        print("  significantly larger excursions, and does not generate positive")
        print("  straddle expectancy at any tested TP multiple.")
        print()
        print("  The elevated absolute returns seen in the volatility audit reflect")
        print("  pre-existing high volatility (the signal is drawn from high-ATR")
        print("  environments) rather than a forward-looking vol forecast.")
        print()
        print("  CONCLUSION: The signal has no recoverable edge — not directional,")
        print("  not volatility-based, not path-dependent. The entry logic selects")
        print("  a specific bar type but that bar type has no predictive information")
        print("  about what follows. The strategy needs a new entry hypothesis.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Straddle / volatility forecasting audit")
    p.add_argument("--symbol",     default="ETHUSDT")
    p.add_argument("--candle-dir", default="data/candles", type=Path)
    p.add_argument("--start",      default="2023-01", help="YYYY-MM")
    p.add_argument("--end",        default=None,      help="YYYY-MM")
    p.add_argument("--stop-mult",  default=STOP_DEFAULT, type=float,
                   help=f"Stop distance as ATR multiple (default {STOP_DEFAULT})")
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

    from calendar import monthrange
    _, last_day = monthrange(min(ey, 2099), min(em, 12))
    start_s = int(datetime(sy, sm, 1, tzinfo=timezone.utc).timestamp())
    end_s   = int(datetime(min(ey, 2099), min(em, 12), last_day,
                            23, 59, 59, tzinfo=timezone.utc).timestamp())

    print(f"\nStraddle audit — {args.symbol}  stop={args.stop_mult}×ATR")
    print(f"Loading candles from {candle_dir} …")
    c1m, merged = _load(candle_dir, args.symbol, start_s, end_s)
    print(f"  {len(c1m):,} × 1m bars")

    if len(c1m) < 1000:
        print("[ERROR] Too few bars.")
        sys.exit(1)

    print("Collecting signal entries …")
    sig = collect_signal_entries(c1m, merged)
    print(f"  {len(sig):,} signal entries")
    if len(sig) == 0:
        print("[ERROR] Zero signal entries.")
        sys.exit(1)

    n_compare = max(len(sig), 500)
    print(f"Collecting {n_compare:,} random entries …")
    ran = collect_random_entries(c1m, n=n_compare)

    print("Collecting breakout entries …")
    bo = collect_breakout_entries(c1m)
    print(f"  {len(bo):,} breakout entries")

    print("Computing volatility metrics and straddle simulations …")
    compute_metrics(sig, c1m, args.stop_mult)
    compute_metrics(ran, c1m, args.stop_mult)
    compute_metrics(bo,  c1m, args.stop_mult)

    print_report(sig, ran, bo, c1m, args.symbol, args.stop_mult)


if __name__ == "__main__":
    main()
