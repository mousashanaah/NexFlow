#!/usr/bin/env python3
"""Path analysis — classify signal bars by post-entry price path.

For each signal entry, walk the subsequent bars using high/low (not just
the closing price) to determine which 2×ATR target was touched first:
  - continuation  : signal-direction target hit, opposite NOT hit first
  - reversal      : opposite-direction target hit first
  - two_sided     : both targets hit (whipsaw)
  - neither       : neither target hit within the horizon

The first-touch rule is applied bar-by-bar:
  For each bar from i+1 onward, check bar.low against the stop target
  and bar.high against the continuation target (for a long signal):
    → If continuation target (high ≥ entry + 2×ATR) is touched first: CONTINUATION
    → If reversal target (low ≤ entry − 2×ATR) is touched first: REVERSAL
    → If both are touched in the same bar: TWO_SIDED (whipsaw)
    → If neither is touched by bar i+horizon: NEITHER

Compares signal against random entries (same target distances, same horizon)
to determine whether the classification distribution differs from baseline.

Usage:
  python scripts/path_analysis.py --candle-dir data/candles --start 2023-01
  python scripts/path_analysis.py --candle-dir data/candles --mult 1.5
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
HORIZONS = [5, 10, 20]


# ---------------------------------------------------------------------------
# Path classification
# ---------------------------------------------------------------------------

class PathClass:
    CONTINUATION = "continuation"
    REVERSAL     = "reversal"
    TWO_SIDED    = "two_sided"
    NEITHER      = "neither"


@dataclass
class PathEntry:
    bar_idx:       int
    direction:     int           # +1 long signal, -1 short signal
    close_at_entry: float
    atr:           float
    entry_type:    str           # "signal" | "random"
    year:          int
    month:         int

    # Filter flags (signals only)
    passed_relVol:    bool = False
    passed_range:     bool = False
    passed_momentum:  bool = False
    passed_imbalance: bool = False

    # Filled by classify_paths()
    # For each horizon: {h: PathClass string}
    path_class:   dict[int, str]   = field(default_factory=dict)
    # bars to first touch (NaN if target never touched)
    bars_to_cont: dict[int, float] = field(default_factory=dict)
    bars_to_rev:  dict[int, float] = field(default_factory=dict)
    # signed return at target touch (or at horizon close)
    ret_at_touch: dict[int, float] = field(default_factory=dict)


def _classify_path(entry: PathEntry,
                   candles: list[Candle],
                   mult: float,
                   horizon: int) -> None:
    """Classify the path for one entry at a given horizon using bar high/low."""
    i     = entry.bar_idx
    cl    = entry.close_at_entry
    atr   = entry.atr
    d     = entry.direction       # +1 = long, -1 = short

    # Target levels (always measured from entry close)
    cont_target = cl + d * mult * atr   # 2×ATR in signal direction
    rev_target  = cl - d * mult * atr   # 2×ATR against signal direction

    n = len(candles)
    first_cont = None   # bar offset at which continuation target first touched
    first_rev  = None   # bar offset at which reversal target first touched

    for offset in range(1, horizon + 1):
        idx = i + offset
        if idx >= n:
            break
        bar = candles[idx]

        cont_hit = (d == +1 and bar.high >= cont_target) or \
                   (d == -1 and bar.low  <= cont_target)
        rev_hit  = (d == +1 and bar.low  <= rev_target)  or \
                   (d == -1 and bar.high >= rev_target)

        if cont_hit and first_cont is None:
            first_cont = offset
        if rev_hit  and first_rev  is None:
            first_rev  = offset

        if first_cont is not None and first_rev is not None:
            break   # both found; no need to look further

    # Classify
    if first_cont is not None and first_rev is not None:
        if first_cont < first_rev:
            pc = PathClass.CONTINUATION
        elif first_rev < first_cont:
            pc = PathClass.REVERSAL
        else:
            pc = PathClass.TWO_SIDED     # same bar touched both
    elif first_cont is not None:
        pc = PathClass.CONTINUATION
    elif first_rev is not None:
        pc = PathClass.REVERSAL
    else:
        pc = PathClass.NEITHER

    entry.path_class[horizon]   = pc
    entry.bars_to_cont[horizon] = float(first_cont) if first_cont is not None else float("nan")
    entry.bars_to_rev[horizon]  = float(first_rev)  if first_rev  is not None else float("nan")

    # Signed return at first meaningful touch (or horizon close)
    if pc == PathClass.CONTINUATION:
        entry.ret_at_touch[horizon] = d * (cont_target - cl) / cl
    elif pc == PathClass.REVERSAL:
        entry.ret_at_touch[horizon] = d * (rev_target  - cl) / cl   # negative
    else:
        fwd_idx = min(i + horizon, n - 1)
        entry.ret_at_touch[horizon] = d * (candles[fwd_idx].close - cl) / cl


def classify_paths(entries: list[PathEntry],
                   candles: list[Candle],
                   mult: float) -> None:
    for e in entries:
        for h in HORIZONS:
            _classify_path(e, candles, mult, h)


# ---------------------------------------------------------------------------
# Candle loading (same pattern as other audit scripts)
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
        print("  [WARN] 5m parquet not found — aggregating from 1m")
        c5m = _aggregate_5m(c1m)
    merged = sorted(c1m + c5m, key=lambda c: c.close_time)
    return c1m, merged


# ---------------------------------------------------------------------------
# Entry collection
# ---------------------------------------------------------------------------

def collect_signal_entries(c1m: list[Candle],
                            merged: list[Candle]) -> list[PathEntry]:
    from nexflow.services.strategy.momentum_strategy import (
        compute_relative_volume, compute_breakout_level,
        compute_range_expansion, compute_buy_sell_imbalance,
        compute_momentum_slope,
    )
    idx_by_close: dict[int, int] = {c.close_time: i for i, c in enumerate(c1m)}
    cfg      = MomentumConfig()
    strategy = MomentumStrategy(cfg)
    entries: list[PathEntry] = []

    # We need the 1m window at each signal to compute filter flags.
    # Rebuild a parallel 1m window alongside the merged feed.
    win1m: list[Candle] = []
    win5m: list[Candle] = []

    for c in merged:
        if c.timeframe == "1m":
            win1m.append(c)
            if len(win1m) > cfg.min_bars_1m + 30:
                win1m.pop(0)
        elif c.timeframe == "5m":
            win5m.append(c)
            if len(win5m) > cfg.min_bars_5m + 10:
                win5m.pop(0)

        sig = strategy.on_candle(c)
        if sig is None or c.timeframe != "1m":
            continue

        i = idx_by_close.get(c.close_time, -1)
        if i < 0 or i + max(HORIZONS) >= len(c1m):
            continue

        d = 1 if sig.direction.value == "long" else -1
        dt = datetime.fromtimestamp(c.close_time, tz=timezone.utc)

        # Recompute filter flags from the window at signal time
        atr      = sig.atr
        rel_vol  = compute_relative_volume(win1m, cfg.vol_period) if len(win1m) >= 2 else 1.0
        rng_exp  = compute_range_expansion(c, atr)
        imb      = compute_buy_sell_imbalance(c)
        momentum = compute_momentum_slope(win5m, cfg.momentum_period_5m) if len(win5m) >= cfg.min_bars_5m else 0.0

        mom_ok = (d == +1 and momentum > 0) or (d == -1 and momentum < 0)
        imb_ok = (imb >= cfg.imbalance_min if d == +1 else imb <= (1.0 - cfg.imbalance_min))

        entries.append(PathEntry(
            bar_idx=i, direction=d,
            close_at_entry=c.close, atr=atr,
            entry_type="signal",
            year=dt.year, month=dt.month,
            passed_relVol=(rel_vol >= cfg.rel_vol_threshold),
            passed_range=(rng_exp >= cfg.range_expansion_min),
            passed_momentum=mom_ok,
            passed_imbalance=imb_ok,
        ))
    return entries


def collect_random_entries(c1m: list[Candle],
                            n: int, seed: int = 42) -> list[PathEntry]:
    rng = random.Random(seed)
    eligible = list(range(30, len(c1m) - max(HORIZONS) - 1))
    sample = rng.sample(eligible, min(n, len(eligible)))
    entries: list[PathEntry] = []
    for i in sample:
        c   = c1m[i]
        atr = compute_atr(c1m[max(0, i-14):i+1], 14)
        d   = rng.choice([+1, -1])
        dt  = datetime.fromtimestamp(c.close_time, tz=timezone.utc)
        entries.append(PathEntry(
            bar_idx=i, direction=d,
            close_at_entry=c.close, atr=max(atr, c.close * 0.0001),
            entry_type="random",
            year=dt.year, month=dt.month,
        ))
    return entries


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _rates(entries: list[PathEntry], horizon: int) -> dict[str, float]:
    """Return fraction for each path class at this horizon."""
    valid = [e for e in entries if horizon in e.path_class]
    if not valid:
        return {pc: 0.0 for pc in (PathClass.CONTINUATION, PathClass.REVERSAL,
                                    PathClass.TWO_SIDED,    PathClass.NEITHER)}
    n = len(valid)
    return {pc: sum(1 for e in valid if e.path_class[horizon] == pc) / n
            for pc in (PathClass.CONTINUATION, PathClass.REVERSAL,
                       PathClass.TWO_SIDED,    PathClass.NEITHER)}


def _mean(vals: list[float]) -> float:
    v = [x for x in vals if not math.isnan(x)]
    return sum(v) / len(v) if v else float("nan")


def _fmt(v: float, dp: int = 1) -> str:
    return f"{v*100:.{dp}f}%" if not math.isnan(v) else "  —  "


def _chi2_2x2(a_cont: int, a_rev: int, b_cont: int, b_rev: int) -> tuple[float, float]:
    """Chi-squared test on a 2×2 table [cont vs rev] for two groups.
    Returns (chi2_stat, p_value).  Uses normal approximation for large n."""
    n1 = a_cont + a_rev
    n2 = b_cont + b_rev
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    n  = n1 + n2
    r1 = a_cont + b_cont
    r2 = a_rev  + b_rev
    # Expected
    def E(r, c): return r * c / n
    cells = [(a_cont, E(r1, n1)), (a_rev, E(r2, n1)),
             (b_cont, E(r1, n2)), (b_rev, E(r2, n2))]
    chi2 = sum((o - e)**2 / e for o, e in cells if e > 0)
    # p-value from chi2 with 1 df: approximate via normal
    # P(X² ≥ x) ≈ 2 × P(Z ≥ sqrt(x)) for 1 df
    p = 2 * (1 - _norm_cdf(math.sqrt(chi2))) if chi2 >= 0 else float("nan")
    return chi2, p


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))


def _sig(p: float) -> str:
    if math.isnan(p): return "   "
    if p < 0.001:     return "***"
    if p < 0.01:      return "** "
    if p < 0.05:      return "*  "
    return "   "


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(sig: list[PathEntry],
                 ran: list[PathEntry],
                 c1m: list[Candle],
                 symbol: str,
                 mult: float) -> None:

    period_start = datetime.fromtimestamp(c1m[0].close_time,  tz=timezone.utc).strftime("%Y-%m-%d")
    period_end   = datetime.fromtimestamp(c1m[-1].close_time, tz=timezone.utc).strftime("%Y-%m-%d")

    print()
    print("═" * _W)
    print(f"  PATH ANALYSIS — {symbol}")
    print(f"  Target: ±{mult}×ATR from entry close  |  First-touch classification")
    print(f"  Period: {period_start} → {period_end}   ({len(c1m):,} bars)")
    print(f"  Signal entries: {len(sig):,}   Random entries: {len(ran):,}")
    print("═" * _W)

    # -----------------------------------------------------------------------
    # Section 1: Path class rates — signal vs random
    # -----------------------------------------------------------------------
    print()
    print("  1. PATH CLASSIFICATION RATES  (% of entries in each class)")
    print(f"     Target distance: {mult}×ATR from entry close, using bar high/low")
    print()

    classes = [PathClass.CONTINUATION, PathClass.REVERSAL,
               PathClass.TWO_SIDED,    PathClass.NEITHER]
    labels  = {"continuation": "Continuation",
                "reversal":    "Reversal",
                "two_sided":   "Two-sided",
                "neither":     "Neither"}

    for h in HORIZONS:
        sig_r = _rates(sig, h)
        ran_r = _rates(ran, h)

        sig_valid = [e for e in sig if h in e.path_class]
        ran_valid = [e for e in ran if h in e.path_class]
        a_cont = sum(1 for e in sig_valid if e.path_class[h] == PathClass.CONTINUATION)
        a_rev  = sum(1 for e in sig_valid if e.path_class[h] == PathClass.REVERSAL)
        b_cont = sum(1 for e in ran_valid if e.path_class[h] == PathClass.CONTINUATION)
        b_rev  = sum(1 for e in ran_valid if e.path_class[h] == PathClass.REVERSAL)
        chi2, pval = _chi2_2x2(a_cont, a_rev, b_cont, b_rev)
        star = _sig(pval)

        print(f"  Horizon +{h} bars   (signal n={len(sig_valid):,}  random n={len(ran_valid):,})")
        print(f"    Chi² (cont vs rev, signal vs random) = {chi2:.2f}  p={pval:.4f}{star}")
        print(f"    {'Class':<16}  {'Signal':>10}  {'Random':>10}  {'Diff':>10}")
        print("    " + "─" * 50)
        for pc in classes:
            sv = sig_r[pc]
            rv = ran_r[pc]
            diff = sv - rv
            flag = ""
            if pc in (PathClass.CONTINUATION, PathClass.REVERSAL):
                if abs(diff) > 0.02:
                    flag = " ▲" if diff > 0 else " ▼"
            print(f"    {labels[pc]:<16}  {sv*100:>9.1f}%  {rv*100:>9.1f}%  {diff*100:>+9.1f}%{flag}")
        print()

    # -----------------------------------------------------------------------
    # Section 2: Speed of touch — how many bars to first hit
    # -----------------------------------------------------------------------
    print()
    print("  2. SPEED TO FIRST TARGET TOUCH  (mean bars, only when touched)")
    print()
    print(f"  {'Horizon':<10}  {'Sig cont':>10}  {'Ran cont':>10}  {'Sig rev':>10}  {'Ran rev':>10}")
    print("  " + "─" * 56)
    for h in HORIZONS:
        sc = _mean([e.bars_to_cont[h] for e in sig if h in e.bars_to_cont and not math.isnan(e.bars_to_cont[h])])
        rc = _mean([e.bars_to_cont[h] for e in ran if h in e.bars_to_cont and not math.isnan(e.bars_to_cont[h])])
        sr = _mean([e.bars_to_rev[h]  for e in sig if h in e.bars_to_rev  and not math.isnan(e.bars_to_rev[h])])
        rr = _mean([e.bars_to_rev[h]  for e in ran if h in e.bars_to_rev  and not math.isnan(e.bars_to_rev[h])])
        def fmt_bars(v): return f"{v:.1f}" if not math.isnan(v) else "  —  "
        print(f"  +{h:<9d}  {fmt_bars(sc):>10}  {fmt_bars(rc):>10}  {fmt_bars(sr):>10}  {fmt_bars(rr):>10}")

    # -----------------------------------------------------------------------
    # Section 3: Long vs Short split
    # -----------------------------------------------------------------------
    print()
    print("  3. LONG vs SHORT DIRECTION SPLIT  (at +5 and +10 bars)")
    print()
    for label, subset in [("LONG  signals", [e for e in sig if e.direction == +1]),
                           ("SHORT signals", [e for e in sig if e.direction == -1])]:
        if not subset:
            continue
        print(f"  {label}  (n={len(subset):,})")
        print(f"    {'Class':<16}" + "".join(f"  {'+'+str(h)+'b':>10}" for h in [5, 10]))
        print("    " + "─" * 38)
        for pc in classes:
            row = f"    {labels[pc]:<16}"
            for h in [5, 10]:
                valid = [e for e in subset if h in e.path_class]
                frac = sum(1 for e in valid if e.path_class[h] == pc) / len(valid) if valid else 0.0
                row += f"  {frac*100:>9.1f}%"
            print(row)
        print()

    # -----------------------------------------------------------------------
    # Section 4: Filter attribution — does any filter predict continuation?
    # -----------------------------------------------------------------------
    print()
    print("  4. FILTER ATTRIBUTION  (continuation rate at +5 bars by filter stack)")
    print("     Does adding a filter increase the continuation fraction?")
    print()

    h5 = 5
    layers = [
        ("All signal entries",               lambda e: True),
        ("+ rel_vol ≥ 1.5",                 lambda e: e.passed_relVol),
        ("+ range_exp ≥ 0.8",               lambda e: e.passed_relVol and e.passed_range),
        ("+ momentum_5m aligned",           lambda e: e.passed_relVol and e.passed_range and e.passed_momentum),
        ("+ imbalance (= full signal)",     lambda e: e.passed_relVol and e.passed_range and e.passed_momentum and e.passed_imbalance),
    ]

    print(f"  {'Layer':<42}  {'N':>6}  {'Cont':>8}  {'Rev':>8}  {'TwoSid':>8}  {'Neither':>8}")
    print("  " + "─" * 84)
    for label, mask_fn in layers:
        subset = [e for e in sig if mask_fn(e) and h5 in e.path_class]
        if not subset:
            print(f"  {label:<42}  {'—':>6}")
            continue
        n   = len(subset)
        r   = _rates(subset, h5)
        mark = " ◀" if "full signal" in label else ""
        print(f"  {label:<42}  {n:>6,}  {r['continuation']*100:>7.1f}%  "
              f"{r['reversal']*100:>7.1f}%  {r['two_sided']*100:>7.1f}%  "
              f"{r['neither']*100:>7.1f}%{mark}")

    # Also show random baseline for comparison
    ran_valid_h5 = [e for e in ran if h5 in e.path_class]
    if ran_valid_h5:
        rr = _rates(ran_valid_h5, h5)
        print(f"  {'Random baseline (same n)':<42}  {len(ran_valid_h5):>6,}  "
              f"{rr['continuation']*100:>7.1f}%  {rr['reversal']*100:>7.1f}%  "
              f"{rr['two_sided']*100:>7.1f}%  {rr['neither']*100:>7.1f}%")

    # -----------------------------------------------------------------------
    # Section 5: Year-by-year continuation rate
    # -----------------------------------------------------------------------
    print()
    print("  5. YEAR-BY-YEAR CONTINUATION RATE  (at +5 bars)")
    print()
    by_year: dict[int, list[PathEntry]] = defaultdict(list)
    for e in sig:
        if h5 in e.path_class:
            by_year[e.year].append(e)

    # Random baseline cont rate at +5
    ran_cont_base = (_rates([e for e in ran if h5 in e.path_class], h5)
                     .get(PathClass.CONTINUATION, float("nan")))

    print(f"  {'Year':<8}  {'N':>6}  {'Cont':>8}  {'Rev':>8}  {'TwoSid':>8}  {'Neither':>8}  {'vs Rand':>9}")
    print("  " + "─" * 66)
    for yr in sorted(by_year):
        ents = by_year[yr]
        r    = _rates(ents, h5)
        diff = r[PathClass.CONTINUATION] - ran_cont_base
        flag = " ▲" if diff > 0.03 else (" ▼" if diff < -0.03 else "")
        print(f"  {yr:<8}  {len(ents):>6,}  {r['continuation']*100:>7.1f}%  "
              f"{r['reversal']*100:>7.1f}%  {r['two_sided']*100:>7.1f}%  "
              f"{r['neither']*100:>7.1f}%  {diff*100:>+8.1f}%{flag}")

    # -----------------------------------------------------------------------
    # Section 6: Conditional expected value
    # -----------------------------------------------------------------------
    print()
    print("  6. CONDITIONAL EXPECTED VALUE  (mean signed return at path-class touch)")
    print("     Continuation: positive = good. Reversal: negative = bad.")
    print("     Neither: return measured at horizon close.")
    print()
    print(f"  {'Class':<16}" + "".join(f"  {'+'+str(h)+'b':>12}" for h in HORIZONS))
    print("  " + "─" * 52)
    for pc in classes:
        row = f"  {labels[pc]:<16}"
        for h in HORIZONS:
            vals = [e.ret_at_touch[h] for e in sig
                    if h in e.path_class and e.path_class[h] == pc
                    and not math.isnan(e.ret_at_touch.get(h, float("nan")))]
            m = _mean(vals)
            row += f"  {m*100:>+11.4f}%" if not math.isnan(m) else f"  {'—':>12}"
        print(row)

    # -----------------------------------------------------------------------
    # Section 7: Verdict
    # -----------------------------------------------------------------------
    print()
    print("═" * _W)
    print("  VERDICT")
    print("═" * _W)
    print()

    h_key = 5
    sig_r5  = _rates([e for e in sig if h_key in e.path_class], h_key)
    ran_r5  = _rates([e for e in ran if h_key in e.path_class], h_key)

    cont_sig  = sig_r5[PathClass.CONTINUATION]
    rev_sig   = sig_r5[PathClass.REVERSAL]
    two_sig   = sig_r5[PathClass.TWO_SIDED]
    cont_ran  = ran_r5[PathClass.CONTINUATION]
    rev_ran   = ran_r5[PathClass.REVERSAL]

    # Chi2 test on cont vs rev
    sig_v5   = [e for e in sig if h_key in e.path_class]
    ran_v5   = [e for e in ran if h_key in e.path_class]
    ac = sum(1 for e in sig_v5 if e.path_class[h_key] == PathClass.CONTINUATION)
    ar = sum(1 for e in sig_v5 if e.path_class[h_key] == PathClass.REVERSAL)
    bc = sum(1 for e in ran_v5 if e.path_class[h_key] == PathClass.CONTINUATION)
    br = sum(1 for e in ran_v5 if e.path_class[h_key] == PathClass.REVERSAL)
    chi2, pval = _chi2_2x2(ac, ar, bc, br)

    print(f"  At +{h_key} bars, {mult}×ATR targets:")
    print(f"    Signal:  Continuation={cont_sig*100:.1f}%  Reversal={rev_sig*100:.1f}%  "
          f"Two-sided={two_sig*100:.1f}%")
    print(f"    Random:  Continuation={cont_ran*100:.1f}%  Reversal={rev_ran*100:.1f}%")
    print(f"    Chi² (distribution differs from random): {chi2:.2f}  p={pval:.4f}{_sig(pval)}")
    print()

    # Determine the character of the signal
    cont_bias = cont_sig - cont_ran    # positive = signal has more continuations than random
    rev_bias  = rev_sig  - rev_ran     # positive = signal has more reversals than random

    if pval < 0.05:
        if rev_sig > cont_sig and rev_bias > 0.02:
            print("  FINDING: REVERSAL BIAS  (statistically significant)")
            print()
            print(f"  Signal entries reverse {rev_sig*100:.1f}% of the time vs "
                  f"{rev_ran*100:.1f}% for random.")
            print(f"  Reversal rate exceeds continuation rate by "
                  f"{(rev_sig-cont_sig)*100:.1f}pp.")
            print()
            print("  The filters are identifying exhaustion bars. After the full-signal")
            print("  condition fires, price more often reverses than continues through")
            f"  a second {mult}×ATR move in the breakout direction."
            print(f"  a second {mult}×ATR move in the breakout direction.")
            print()
            print("  This confirms the anti-predictive character seen in direction accuracy")
            print("  (42.9%). The signal is NOT random noise — it actively selects bars")
            print("  where the market is at a local exhaustion point.")
            print()
            if two_sig > 0.15:
                print(f"  Two-sided expansion ({two_sig*100:.1f}%) is elevated — many bars hit")
                print(f"  both targets within {h_key} bars. This is consistent with high-vol")
                print("  bars that spike in both directions before settling.")

        elif cont_sig > rev_sig and cont_bias > 0.02:
            print("  FINDING: CONTINUATION BIAS  (statistically significant)")
            print()
            print(f"  Signal entries continue in breakout direction {cont_sig*100:.1f}% of")
            print(f"  the time vs {cont_ran*100:.1f}% for random.")
            print()
            print("  The signal has a genuine momentum edge at the path level,")
            print("  contradicting the direction-accuracy result. The discrepancy")
            print("  may be explained by the first-touch vs close-price difference:")
            print("  price often touches the continuation target intrabar before reversing.")

        elif two_sig > ran_r5[PathClass.TWO_SIDED] + 0.03:
            print("  FINDING: ELEVATED TWO-SIDED EXPANSION")
            print()
            print(f"  Two-sided expansion ({two_sig*100:.1f}% vs {ran_r5[PathClass.TWO_SIDED]*100:.1f}% random)")
            print("  is the dominant signal characteristic. Price hits BOTH the continuation")
            print("  and reversal targets more often than random, but in unpredictable order.")
            print()
            print("  A non-directional volatility strategy (e.g. dual-entry or options-like")
            print("  payoff) would capture this pattern. A directional stop/TP strategy")
            print("  cannot because the stop is hit roughly as often as the TP.")

        else:
            print("  FINDING: STATISTICALLY DIFFERENT FROM RANDOM (mixed character)")
            print()
            print("  The path distribution differs from random but has no clear directional")
            print("  or reversal bias. The signal may be interacting with other market")
            print("  structure not captured in a simple ±ATR framework.")

    else:
        print("  FINDING: PATH DISTRIBUTION IS NOT STATISTICALLY DIFFERENT FROM RANDOM")
        print()
        print(f"  Chi² p={pval:.4f} — the signal's continuation/reversal breakdown")
        print("  cannot be distinguished from a random-entry baseline.")
        print("  The larger absolute moves seen in the volatility audit are not")
        print("  directionally structured — they are equally likely to go either way.")

    print()
    print("  SUMMARY:")
    print()
    if cont_bias < -0.01 or rev_bias > 0.01:
        print("  Directional:        ANTI-PREDICTIVE  (reversal bias)")
    elif cont_bias > 0.01:
        print("  Directional:        CONTINUATION BIAS  (but verify with live data)")
    else:
        print("  Directional:        NO EDGE  (signal ~ random)")

    if two_sig > ran_r5[PathClass.TWO_SIDED] + 0.03:
        print("  Volatility:         TWO-SIDED EXPANSION bias — undirected vol edge")
    else:
        print("  Volatility:         No elevated two-sided expansion vs random")

    longs  = [e for e in sig if e.direction == +1 and h_key in e.path_class]
    shorts = [e for e in sig if e.direction == -1 and h_key in e.path_class]
    if longs and shorts:
        l_cont = sum(1 for e in longs  if e.path_class[h_key] == PathClass.CONTINUATION) / len(longs)
        s_cont = sum(1 for e in shorts if e.path_class[h_key] == PathClass.CONTINUATION) / len(shorts)
        if abs(l_cont - s_cont) > 0.05:
            print(f"  Long vs Short:      ASYMMETRIC  "
                  f"(long cont={l_cont*100:.1f}%  short cont={s_cont*100:.1f}%)")
            print("  One direction is more exploitable than the other.")
        else:
            print(f"  Long vs Short:      SYMMETRIC  (long={l_cont*100:.1f}%  short={s_cont*100:.1f}%)")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Path analysis: classify signal bars by post-entry path")
    p.add_argument("--symbol",     default="ETHUSDT")
    p.add_argument("--candle-dir", default="data/candles", type=Path)
    p.add_argument("--start",      default="2023-01", help="YYYY-MM")
    p.add_argument("--end",        default=None,      help="YYYY-MM")
    p.add_argument("--mult",       default=2.0, type=float,
                   help="ATR multiple for target (default 2.0)")
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

    print(f"\nPath analysis — {args.symbol}  target={args.mult}×ATR")
    print(f"Loading candles from {candle_dir} …")
    c1m, merged = _load(candle_dir, args.symbol, start_s, end_s)
    print(f"  {len(c1m):,} × 1m bars")

    if len(c1m) < 1000:
        print("[ERROR] Too few bars.")
        sys.exit(1)

    print("Collecting signal entries …")
    sig_entries = collect_signal_entries(c1m, merged)
    print(f"  {len(sig_entries):,} signal entries")

    if len(sig_entries) == 0:
        print("[ERROR] Zero signal entries — check 1m + 5m parquet files.")
        sys.exit(1)

    n_ran = max(len(sig_entries), 500)
    print(f"Collecting {n_ran:,} random entries …")
    ran_entries = collect_random_entries(c1m, n=n_ran)

    print("Classifying paths …")
    classify_paths(sig_entries, c1m, args.mult)
    classify_paths(ran_entries, c1m, args.mult)

    print_report(sig_entries, ran_entries, c1m, args.symbol, args.mult)


if __name__ == "__main__":
    main()
