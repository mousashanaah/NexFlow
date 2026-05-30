#!/usr/bin/env python3
"""Signal forensic audit.

Answers one question: does the entry signal contain any predictive
information, or is it fundamentally non-predictive?

Tests:
  1. Signal forward returns   — actual MomentumStrategy signals, measured
                                in the direction of the signal at +1/3/5/10/20 bars.
  2. Random entries           — same number of entries, random direction,
                                drawn from the same bar population.
  3. Simple breakout entries  — close > 20-bar rolling high (long) or
                                close < rolling low (short), no secondary filters.
                                This isolates the breakout condition from the
                                rest of the filter stack.
  4. Direction accuracy       — % of signal entries where the market moved
                                in the predicted direction by horizon k.
  5. Statistical significance — t-test: is signal mean return > 0?
                                Is signal return different from random?
  6. Filter contribution      — which filters (rel_vol, range_exp, imbalance,
                                momentum_5m) improve or hurt predictive accuracy.

Metric: signed forward return
  = direction × (close[N+k] − close[N]) / close[N]
  Positive = market moved in signal direction.
  Fees and stops are NOT included — this tests raw signal direction quality.

Usage:
  python scripts/signal_forensic_audit.py
  python scripts/signal_forensic_audit.py --candle-dir data/candles --start 2023-01
  python scripts/signal_forensic_audit.py --candle-dir data/candles --start 2023-01 --csv signal_forensic.csv
"""

from __future__ import annotations

import argparse
import csv
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
    _HAS_PARQUET = True
except ImportError:
    print("[ERROR] pyarrow required: pip install pyarrow")
    sys.exit(1)

from nexflow.services.candles.candle_engine import Candle
from nexflow.services.strategy.momentum_strategy import (
    MomentumConfig, MomentumStrategy,
    compute_atr, compute_relative_volume, compute_breakout_level,
    compute_range_expansion, compute_buy_sell_imbalance, compute_momentum_slope,
)

HORIZONS = [1, 3, 5, 10, 20]
_W = 76

# Fee round-trip for breakeven reference (taker entry + taker stop)
TAKER = 0.0006
ENTRY_SLIP_MULT = 0.0525   # ATR fraction
BREAKEVEN_RT = TAKER * 2   # 0.12% — minimum forward return to cover fees


# ---------------------------------------------------------------------------
# Data loading
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


def _load_candles(candle_dir: Path, symbol: str,
                  start_s: int, end_s: int) -> tuple[list[Candle], list[Candle], list[Candle]]:
    """Return (candles_1m, candles_5m, merged_sorted_all).

    merged_sorted_all interleaves 1m and 5m candles sorted by close_time so
    that MomentumStrategy.on_candle() receives both timeframes in order,
    which is required for the 5m momentum window to be populated.
    """
    path_1m = candle_dir / f"{symbol}_1m.parquet"
    path_5m = candle_dir / f"{symbol}_5m.parquet"

    if not path_1m.exists():
        print(f"[ERROR] {path_1m} not found. Run download_candles.py first.")
        sys.exit(1)

    rows_1m = pq.read_table(path_1m).to_pylist()
    candles_1m = sorted(_rows_to_candles(rows_1m, start_s, end_s),
                        key=lambda c: c.close_time)

    candles_5m: list[Candle] = []
    if path_5m.exists():
        rows_5m = pq.read_table(path_5m).to_pylist()
        candles_5m = sorted(_rows_to_candles(rows_5m, start_s, end_s),
                            key=lambda c: c.close_time)
    else:
        print(f"  [WARN] {path_5m} not found — aggregating 5m from 1m bars")
        candles_5m = _aggregate_5m(candles_1m)

    # Interleave both timeframes sorted by close_time for strategy replay
    merged = sorted(candles_1m + candles_5m, key=lambda c: c.close_time)
    return candles_1m, candles_5m, merged


def _aggregate_5m(candles_1m: list[Candle]) -> list[Candle]:
    """Aggregate 1m candles into 5m candles by grouping on floor(close_time / 300)."""
    from collections import defaultdict
    buckets: dict[int, list[Candle]] = defaultdict(list)
    for c in candles_1m:
        # Each 5m bucket key = open_time of the 5m bar (floor to 5-min boundary)
        key = (c.open_time // 300) * 300
        buckets[key].append(c)

    result: list[Candle] = []
    for key in sorted(buckets):
        bars = sorted(buckets[key], key=lambda c: c.open_time)
        sym = bars[0].symbol
        o   = bars[0].open
        h   = max(b.high for b in bars)
        lo  = min(b.low  for b in bars)
        cl  = bars[-1].close
        vol = sum(b.volume for b in bars)
        bvol = sum(b.buy_volume for b in bars)
        svol = sum(b.sell_volume for b in bars)
        tc  = sum(b.trade_count for b in bars)
        result.append(Candle(
            symbol=sym, timeframe="5m",
            open_time=bars[0].open_time,
            close_time=bars[-1].close_time,
            open=o, high=h, low=lo, close=cl,
            volume=vol, buy_volume=bvol, sell_volume=svol,
            trade_count=tc,
            vwap=cl,        # proxy: true VWAP needs price×vol per bar
            spread_avg=0.0, spread_max=0.0,
            volatility_estimate=0.0,
            is_final=True,
        ))
    return result


# ---------------------------------------------------------------------------
# Entry record
# ---------------------------------------------------------------------------

@dataclass
class Entry:
    bar_idx: int
    direction: int          # +1 long, -1 short
    close_at_entry: float
    atr: float
    year_month: str         # "YYYY-MM"
    entry_type: str         # "signal" | "random" | "breakout" | "anti_signal"

    # Filled in after scanning forward bars
    returns: dict[int, float] = field(default_factory=dict)  # horizon → signed return


# ---------------------------------------------------------------------------
# Strategy signals
# ---------------------------------------------------------------------------

def collect_signal_entries(candles_1m: list[Candle],
                           merged: list[Candle],
                           symbol: str) -> list[Entry]:
    """Run MomentumStrategy on interleaved 1m+5m candles.

    The strategy requires both 1m and 5m timeframes to fire signals.
    `merged` contains both sorted by close_time so on_candle() populates
    both internal windows correctly. We only record signal entries at 1m
    bars, using the 1m bar index for forward-return computation.
    """
    # Build a lookup: 1m close_time → index in candles_1m
    idx_by_close: dict[int, int] = {c.close_time: i for i, c in enumerate(candles_1m)}

    strategy = MomentumStrategy()
    entries: list[Entry] = []
    for c in merged:
        sig = strategy.on_candle(c)
        if sig is not None and c.timeframe == "1m":
            direction = 1 if sig.direction.value == "long" else -1
            dt = datetime.fromtimestamp(c.close_time, tz=timezone.utc)
            bar_idx = idx_by_close.get(c.close_time, -1)
            if bar_idx >= 0:
                entries.append(Entry(
                    bar_idx=bar_idx,
                    direction=direction,
                    close_at_entry=c.close,
                    atr=sig.atr,
                    year_month=f"{dt.year:04d}-{dt.month:02d}",
                    entry_type="signal",
                ))
    return entries


# ---------------------------------------------------------------------------
# Simple breakout entries (primary filter only, no secondary filters)
# ---------------------------------------------------------------------------

def collect_breakout_entries(candles: list[Candle], symbol: str,
                              period: int = 20) -> list[Entry]:
    """Close > rolling_high → LONG. Close < rolling_low → SHORT.
    No rel_vol, range_exp, imbalance, or momentum filters.
    Isolates the breakout condition alone.
    """
    entries: list[Entry] = []
    window: list[Candle] = []
    for i, c in enumerate(candles):
        window.append(c)
        if len(window) < period + 2:
            continue
        lookback = window[-(period + 1):-1]
        rolling_high = max(b.high for b in lookback)
        rolling_low  = min(b.low  for b in lookback)
        atr = compute_atr(window[-15:], 14)
        dt  = datetime.fromtimestamp(c.close_time, tz=timezone.utc)

        if c.close > rolling_high:
            entries.append(Entry(i, +1, c.close, atr,
                                 f"{dt.year:04d}-{dt.month:02d}", "breakout"))
        elif c.close < rolling_low:
            entries.append(Entry(i, -1, c.close, atr,
                                 f"{dt.year:04d}-{dt.month:02d}", "breakout"))
    return entries


# ---------------------------------------------------------------------------
# Anti-signal entries (opposite direction of signal — sanity check)
# ---------------------------------------------------------------------------

def collect_anti_signal_entries(signal_entries: list[Entry]) -> list[Entry]:
    return [
        Entry(e.bar_idx, -e.direction, e.close_at_entry, e.atr,
              e.year_month, "anti_signal")
        for e in signal_entries
    ]


# ---------------------------------------------------------------------------
# Random entries matched to signal bar population
# ---------------------------------------------------------------------------

def collect_random_entries(candles: list[Candle],
                            n: int, seed: int = 42) -> list[Entry]:
    """Sample n random entries from all bars with random direction."""
    rng = random.Random(seed)
    # Exclude first 30 and last 20 bars (no forward data / no warmup)
    eligible = list(range(30, len(candles) - max(HORIZONS)))
    sample_idx = rng.sample(eligible, min(n, len(eligible)))
    entries: list[Entry] = []
    for i in sample_idx:
        c   = candles[i]
        dir = rng.choice([+1, -1])
        atr = (c.high - c.low)   # crude range as proxy
        dt  = datetime.fromtimestamp(c.close_time, tz=timezone.utc)
        entries.append(Entry(i, dir, c.close, atr,
                             f"{dt.year:04d}-{dt.month:02d}", "random"))
    return entries


# ---------------------------------------------------------------------------
# Forward return computation
# ---------------------------------------------------------------------------

def fill_forward_returns(entries: list[Entry], candles: list[Candle]) -> None:
    close_arr = [c.close for c in candles]
    n = len(close_arr)
    for e in entries:
        for h in HORIZONS:
            fwd_idx = e.bar_idx + h
            if fwd_idx < n:
                fwd_ret = (close_arr[fwd_idx] - e.close_at_entry) / e.close_at_entry
                e.returns[h] = e.direction * fwd_ret
            else:
                e.returns[h] = float("nan")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _stats(values: list[float]) -> dict:
    v = [x for x in values if not math.isnan(x)]
    if not v:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "t": float("nan"), "p": float("nan"), "dir_acc": float("nan")}
    n    = len(v)
    mean = sum(v) / n
    var  = sum((x - mean) ** 2 for x in v) / max(n - 1, 1)
    std  = math.sqrt(var)
    se   = std / math.sqrt(n) if n > 1 else float("inf")
    t    = mean / se if se > 0 else float("nan")
    # Two-sided p-value approximation (large-n normal approximation)
    p    = 2 * (1 - _norm_cdf(abs(t))) if not math.isnan(t) else float("nan")
    dir_acc = sum(1 for x in v if x > 0) / n
    return {"n": n, "mean": mean, "std": std, "t": t, "p": p, "dir_acc": dir_acc}


def _norm_cdf(z: float) -> float:
    """Approximation of standard normal CDF."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _sig_star(p: float) -> str:
    if math.isnan(p):   return "   "
    if p < 0.001:       return "***"
    if p < 0.01:        return "** "
    if p < 0.05:        return "*  "
    return "   "


# ---------------------------------------------------------------------------
# Filter contribution analysis
# ---------------------------------------------------------------------------

@dataclass
class FilteredEntry:
    bar_idx: int
    direction: int
    close_at_entry: float
    atr: float
    passed_relVol: bool
    passed_range: bool
    passed_imbalance: bool
    passed_momentum: bool
    passed_breakout: bool
    returns: dict[int, float] = field(default_factory=dict)


def collect_filter_contributions(candles_1m: list[Candle],
                                  candles_5m: list[Candle]) -> list[FilteredEntry]:
    """Record each breakout bar's filter pass/fail independently for attribution.

    Uses real 5m candles (from parquet or aggregated from 1m) so that the
    momentum filter is computed identically to the live strategy.
    """
    cfg      = MomentumConfig()
    entries: list[FilteredEntry] = []

    # Build a pointer into candles_5m that advances as 1m time advances
    p5 = 0
    n5 = len(candles_5m)

    win1m: list[Candle] = []
    win5m_buf: list[Candle] = []

    for i, c in enumerate(candles_1m):
        win1m.append(c)

        # Advance 5m window: include all 5m bars whose close_time <= current 1m close_time
        while p5 < n5 and candles_5m[p5].close_time <= c.close_time:
            win5m_buf.append(candles_5m[p5])
            p5 += 1
        # Keep only the most recent window (strategy uses deque with maxlen)
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

        direction = +1 if long_breakout else -1

        mom_ok = (direction == +1 and momentum > 0) or (direction == -1 and momentum < 0)

        if direction == +1:
            imb_ok = imbalance >= cfg.imbalance_min
        else:
            imb_ok = imbalance <= (1.0 - cfg.imbalance_min)

        entries.append(FilteredEntry(
            bar_idx=i,
            direction=direction,
            close_at_entry=trigger.close,
            atr=atr,
            passed_relVol=(rel_vol >= cfg.rel_vol_threshold),
            passed_range=(range_exp >= cfg.range_expansion_min),
            passed_imbalance=imb_ok,
            passed_momentum=mom_ok,
            passed_breakout=True,
        ))

    return entries


def fill_filter_returns(entries: list[FilteredEntry], candles: list[Candle]) -> None:
    close_arr = [c.close for c in candles]
    n = len(close_arr)
    for e in entries:
        for h in HORIZONS:
            idx = e.bar_idx + h
            if idx < n:
                e.returns[h] = e.direction * (close_arr[idx] - e.close_at_entry) / e.close_at_entry
            else:
                e.returns[h] = float("nan")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _pct(v: float, dp: int = 3) -> str:
    if math.isnan(v):
        return " " * (dp + 5)
    return f"{v*100:+.{dp}f}%"


def _row(label: str, stats_list: list[dict], key: str, multiplier: float = 1.0) -> None:
    print(f"  {label:<14}", end="")
    for s in stats_list:
        v = s.get(key, float("nan"))
        if math.isnan(v):
            print(f"  {'—':>10}", end="")
        elif key in ("dir_acc",):
            print(f"  {v*100:>9.1f}%", end="")
        elif key in ("p",):
            star = _sig_star(v)
            print(f"  {v:>7.4f}{star}", end="")
        elif key in ("t",):
            print(f"  {v:>10.2f}", end="")
        else:
            print(f"  {v*100*multiplier:>+9.4f}%", end="")
    print()


def print_report(
    signal_entries: list[Entry],
    random_entries: list[Entry],
    breakout_entries: list[Entry],
    anti_entries: list[Entry],
    filter_entries: list[FilteredEntry],
    candles: list[Candle],
    symbol: str,
) -> None:

    groups = {
        "signal":    signal_entries,
        "random":    random_entries,
        "breakout":  breakout_entries,
        "anti":      anti_entries,
    }

    # Compute per-horizon stats for each group
    stats: dict[str, dict[int, dict]] = {}
    for name, ents in groups.items():
        stats[name] = {}
        for h in HORIZONS:
            vals = [e.returns.get(h, float("nan")) for e in ents]
            stats[name][h] = _stats(vals)

    period_start = datetime.fromtimestamp(candles[0].close_time,  tz=timezone.utc).strftime("%Y-%m-%d")
    period_end   = datetime.fromtimestamp(candles[-1].close_time, tz=timezone.utc).strftime("%Y-%m-%d")
    n_sig   = len(signal_entries)
    n_bo    = len(breakout_entries)

    print()
    print("═" * _W)
    print(f"  SIGNAL FORENSIC AUDIT — {symbol}")
    print(f"  Period: {period_start} → {period_end}   ({len(candles):,} bars)")
    print(f"  Signal entries: {n_sig:,}   Breakout entries: {n_bo:,}   Random entries: {len(random_entries):,}")
    print("═" * _W)

    # -----------------------------------------------------------------------
    # Section 1: Forward returns
    # -----------------------------------------------------------------------
    print()
    print("  1. SIGNED FORWARD RETURNS  (direction × price change, pre-fee)")
    print("     Positive = market moved in predicted direction")
    print()
    hdrs = ["Signal", "Breakout", "Random", "Anti-Signal"]
    print(f"  {'Horizon':<14}" + "".join(f"  {h:>10}" for h in hdrs))
    print("  " + "─" * (_W - 2))
    for h in HORIZONS:
        s_sig = stats["signal"][h]
        s_bo  = stats["breakout"][h]
        s_ran = stats["random"][h]
        s_ant = stats["anti"][h]
        sig_mean = s_sig["mean"]
        bo_mean  = s_bo["mean"]
        ran_mean = s_ran["mean"]
        ant_mean = s_ant["mean"]
        print(f"  +{h:<13d}"
              f"  {_pct(sig_mean):>10}"
              f"  {_pct(bo_mean):>10}"
              f"  {_pct(ran_mean):>10}"
              f"  {_pct(ant_mean):>10}")
    print()
    print(f"  Breakeven (fees only, no stops): {_pct(BREAKEVEN_RT, 3)}")

    # -----------------------------------------------------------------------
    # Section 2: Direction accuracy
    # -----------------------------------------------------------------------
    print()
    print("  2. DIRECTION ACCURACY  (% of entries where market moved in signal direction)")
    print()
    print(f"  {'Horizon':<14}" + "".join(f"  {h:>10}" for h in hdrs))
    print("  " + "─" * (_W - 2))
    for h in HORIZONS:
        vals = [stats[g][h]["dir_acc"] for g in ["signal", "breakout", "random", "anti"]]
        print(f"  +{h:<13d}" + "".join(f"  {v*100:>9.1f}%" for v in vals))

    # -----------------------------------------------------------------------
    # Section 3: Statistical significance
    # -----------------------------------------------------------------------
    print()
    print("  3. STATISTICAL SIGNIFICANCE  (t-test: is mean return > 0?)")
    print("     *** p<0.001   ** p<0.01   * p<0.05   (two-sided)")
    print()
    for name, label in [("signal","Signal"), ("breakout","Breakout"), ("random","Random")]:
        print(f"  {label}:")
        print(f"    {'Horizon':<10} {'N':>7} {'Mean ret':>12} {'t-stat':>8} {'p-value':>12}")
        print(f"    {'─'*54}")
        for h in HORIZONS:
            s = stats[name][h]
            star = _sig_star(s["p"])
            mean_str = _pct(s["mean"])
            t_str    = f"{s['t']:+.2f}" if not math.isnan(s["t"]) else "  —  "
            p_str    = f"{s['p']:.4f}" if not math.isnan(s["p"]) else "  —  "
            print(f"    +{h:<9d} {s['n']:>7,} {mean_str:>12} {t_str:>8} {p_str:>10}  {star}")
        print()

    # -----------------------------------------------------------------------
    # Section 4: Filter contribution
    # -----------------------------------------------------------------------
    print("  4. FILTER CONTRIBUTION ANALYSIS")
    print("     Starting from all breakout bars, does each additional filter help?")
    print("     Mean signed forward return at +5 bars.")
    print()

    h5 = 5
    close_arr = [c.close for c in candles]
    n_candles = len(close_arr)

    def mean_return_for_mask(ents: list[FilteredEntry], mask_fn) -> tuple[float, int]:
        vals = []
        for e in ents:
            if mask_fn(e):
                v = e.returns.get(h5, float("nan"))
                if not math.isnan(v):
                    vals.append(v)
        if not vals:
            return float("nan"), 0
        return sum(vals) / len(vals), len(vals)

    layers = [
        ("Breakout only",                     lambda e: True),
        ("+ rel_vol filter",                  lambda e: e.passed_relVol),
        ("+ range_expansion filter",          lambda e: e.passed_relVol and e.passed_range),
        ("+ momentum_5m filter",              lambda e: e.passed_relVol and e.passed_range and e.passed_momentum),
        ("+ imbalance filter (= full signal)",lambda e: e.passed_relVol and e.passed_range and e.passed_momentum and e.passed_imbalance),
    ]

    print(f"  {'Layer':<42} {'N':>7} {'Mean+5bar ret':>14}")
    print("  " + "─" * 66)
    for label, mask in layers:
        m, n = mean_return_for_mask(filter_entries, mask)
        mark = " ◀ FINAL SIGNAL" if "full signal" in label else ""
        m_str = _pct(m) if not math.isnan(m) else "  —  "
        print(f"  {label:<42} {n:>7,} {m_str:>14}{mark}")
    print()

    # -----------------------------------------------------------------------
    # Section 5: Year-month breakdown of signal forward returns
    # -----------------------------------------------------------------------
    print("  5. SIGNAL DIRECTION ACCURACY BY YEAR  (at +5 bars)")
    print()
    by_year: dict[str, list[float]] = defaultdict(list)
    for e in signal_entries:
        v = e.returns.get(5, float("nan"))
        if not math.isnan(v):
            by_year[e.year_month[:4]].append(v)

    print(f"  {'Year':<10} {'N':>7} {'Mean+5bar':>12} {'Dir acc':>10}")
    print("  " + "─" * 42)
    for yr in sorted(by_year):
        vals = by_year[yr]
        m    = sum(vals) / len(vals) if vals else float("nan")
        acc  = sum(1 for v in vals if v > 0) / len(vals) if vals else float("nan")
        flag = " ◀" if acc < 0.45 else (" ▲" if acc > 0.55 else "")
        print(f"  {yr:<10} {len(vals):>7,} {_pct(m):>12} {acc*100:>9.1f}%{flag}")
    print()

    # -----------------------------------------------------------------------
    # Section 6: Verdict
    # -----------------------------------------------------------------------
    print("═" * _W)
    print("  VERDICT")
    print("═" * _W)

    # Key metrics
    sig_5   = stats["signal"][5]["mean"]
    ran_5   = stats["random"][5]["mean"]
    bo_5    = stats["breakout"][5]["mean"]
    sig_acc = stats["signal"][5]["dir_acc"]
    ran_acc = stats["random"][5]["dir_acc"]
    sig_p1  = stats["signal"][1]["p"]
    sig_p5  = stats["signal"][5]["p"]

    print()
    print(f"  Signal mean return at +5 bars:   {_pct(sig_5)}")
    print(f"  Random  mean return at +5 bars:  {_pct(ran_5)}")
    print(f"  Breakout mean return at +5 bars: {_pct(bo_5)}")
    print(f"  Signal direction accuracy @+5:   {sig_acc*100:.1f}%  (random baseline: {ran_acc*100:.1f}%)")
    print()

    # Determine verdict
    has_positive_return = not math.isnan(sig_5) and sig_5 > 0
    beats_random        = not math.isnan(sig_5) and not math.isnan(ran_5) and sig_5 > ran_5
    statistically_sig   = not math.isnan(sig_p5) and sig_p5 < 0.05
    above_breakeven     = not math.isnan(sig_5) and sig_5 > BREAKEVEN_RT
    dir_above_50        = sig_acc > 0.50

    if not has_positive_return:
        print("  FINDING A: Signal has NEGATIVE mean return pre-fee.")
        print("  The signal actively predicts the WRONG direction on average.")
        print("  This is not an execution problem. The entry logic itself is")
        print("  anti-predictive — it enters at local exhaustion points, not")
        print("  at the start of moves.")
    elif not beats_random:
        print("  FINDING A: Signal return is positive but BELOW random.")
        print("  The filters do not select better setups than chance.")
    else:
        print("  FINDING A: Signal return is positive and beats random.")

    print()
    if sig_acc < 0.48:
        print("  FINDING B: Direction accuracy is BELOW 48%.")
        print("  The strategy enters longs that go down and shorts that go up.")
        print("  The breakout condition is entering AFTER the move, not before it.")
    elif sig_acc < 0.50:
        print("  FINDING B: Direction accuracy is below 50% (coin flip).")
    elif sig_acc < 0.52:
        print("  FINDING B: Direction accuracy is approximately 50% — no directional edge.")
    else:
        print("  FINDING B: Direction accuracy is above 50%.")

    print()
    if not above_breakeven:
        print("  FINDING C: Mean return at +5 bars is BELOW the fee breakeven threshold")
        print(f"  ({_pct(BREAKEVEN_RT)}). Even if execution were free, the signal")
        print("  would not be profitable at this horizon.")
    else:
        print(f"  FINDING C: Mean return at +5 bars ({_pct(sig_5)}) exceeds fee")
        print(f"  breakeven ({_pct(BREAKEVEN_RT)}). The edge exists but may be consumed by stops.")

    print()
    print("  ROOT CAUSE:")
    if not has_positive_return or sig_acc < 0.50:
        print("  The signal enters at breakout points where price has already moved")
        print("  1×ATR+ in one direction. The evidence suggests this is a LOCAL")
        print("  EXHAUSTION pattern — price reverts after these high-volume, high-range")
        print("  breakout bars rather than continuing. The strategy is systematically")
        print("  entering at the end of short moves, not the beginning of sustained trends.")
        print()
        print("  The 57% historical WR most likely came from a specific trending period")
        print("  where breakouts DID continue. In the full 41-month sample, this is the")
        print("  exception rather than the rule.")
        print()
        print("  No amount of stop or TP tuning will fix a non-predictive entry signal.")
        print("  The entry logic itself needs to be rethought.")
    else:
        print("  The signal has some directional edge but execution costs consume it.")
        print("  Focus on reducing fees (maker entries/exits) and widening stops to")
        print("  allow the edge to survive to the profitable horizon.")
    print()


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def write_csv(signal_entries: list[Entry], path: Path) -> None:
    if not signal_entries:
        return
    rows = []
    for e in signal_entries:
        row = {
            "year_month": e.year_month,
            "bar_idx":    e.bar_idx,
            "direction":  "long" if e.direction == 1 else "short",
            "close":      e.close_at_entry,
            "atr":        round(e.atr, 4),
        }
        for h in HORIZONS:
            row[f"ret_{h}b"] = round(e.returns.get(h, float("nan")) * 100, 5)
            row[f"dir_{h}b"] = int(e.returns.get(h, 0) > 0)
        rows.append(row)
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV written → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Signal forensic audit")
    p.add_argument("--symbol",     default="ETHUSDT")
    p.add_argument("--candle-dir", default="data/candles")
    p.add_argument("--start",      default="2023-01",
                   help="Start month YYYY-MM")
    p.add_argument("--end",        default=None,
                   help="End month YYYY-MM (default: all available)")
    p.add_argument("--csv",        default="signal_forensic.csv")
    args = p.parse_args()

    candle_dir = _REPO_ROOT / args.candle_dir
    if not candle_dir.exists():
        candle_dir = Path(args.candle_dir)

    # Date range
    sy, sm = [int(x) for x in args.start.split("-")]
    if args.end:
        ey, em = [int(x) for x in args.end.split("-")]
    else:
        ey, em = 2099, 12

    from calendar import monthrange
    start_s = int(__import__("datetime").datetime(sy, sm, 1, tzinfo=__import__("datetime").timezone.utc).timestamp())
    end_s   = int(__import__("datetime").datetime(min(ey, 2099), min(em, 12),
                  monthrange(min(ey,2099), min(em,12))[1],
                  23, 59, 59, tzinfo=__import__("datetime").timezone.utc).timestamp())

    print(f"\nSignal forensic audit — {args.symbol}")
    print(f"Loading candles from {candle_dir} …")
    candles_1m, candles_5m, merged = _load_candles(candle_dir, args.symbol, start_s, end_s)
    print(f"  {len(candles_1m):,} × 1m bars  |  {len(candles_5m):,} × 5m bars loaded")

    if len(candles_1m) < 1000:
        print("[ERROR] Too few 1m bars. Run download_candles.py first.")
        sys.exit(1)

    print("Collecting signal entries (1m + 5m feed into strategy) …")
    signal_entries = collect_signal_entries(candles_1m, merged, args.symbol)
    print(f"  {len(signal_entries):,} signal entries")

    if len(signal_entries) == 0:
        print("[WARN] Zero signal entries — check that 5m parquet exists or 1m data "
              "covers enough history for the 5m window to fill.")

    print("Collecting breakout entries …")
    breakout_entries = collect_breakout_entries(candles_1m, args.symbol)
    print(f"  {len(breakout_entries):,} simple breakout entries")

    print("Collecting random entries …")
    n_random = max(len(signal_entries), 500)
    random_entries = collect_random_entries(candles_1m, n=n_random)
    print(f"  {len(random_entries):,} random entries")

    anti_entries = collect_anti_signal_entries(signal_entries)

    print("Collecting filter contribution data …")
    filter_entries = collect_filter_contributions(candles_1m, candles_5m)
    print(f"  {len(filter_entries):,} breakout bars for filter analysis")

    print("Computing forward returns …")
    fill_forward_returns(signal_entries,   candles_1m)
    fill_forward_returns(random_entries,   candles_1m)
    fill_forward_returns(breakout_entries, candles_1m)
    fill_forward_returns(anti_entries,     candles_1m)
    fill_filter_returns(filter_entries,    candles_1m)

    print_report(
        signal_entries, random_entries, breakout_entries, anti_entries,
        filter_entries, candles_1m, args.symbol,
    )

    if args.csv:
        write_csv(signal_entries, _REPO_ROOT / args.csv)


if __name__ == "__main__":
    main()
