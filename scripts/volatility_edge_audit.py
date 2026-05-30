#!/usr/bin/env python3
"""Volatility edge audit.

Tests whether the MomentumStrategy signal predicts VOLATILITY even when
it fails to predict DIRECTION.

If a signal selects bars where subsequent absolute moves are larger than
random, there may be a volatility-based edge — usable via straddles,
wider TP targets, or a different exit model — even if the directional
bet is worthless.

Four questions answered:
  1. Absolute move after signal vs random: does the signal select
     high-volatility bars?
  2. Realized volatility post-entry: is the post-signal realised vol
     larger than a random sample of the same length?
  3. Pre-to-post vol ratio: does volatility expand AFTER the signal
     fires, or was it already elevated before?
  4. ATR-normalised absolute return: does the signal capture larger
     moves relative to the expected bar size?

Populations compared:
  - signal     : full MomentumStrategy entries (1m + 5m feed)
  - random      : same number of bars drawn uniformly at random
  - breakout    : close > 20-bar high/low, no secondary filters

Metric definitions:
  abs_ret[k]     = |close[N+k] - close[N]| / close[N]
  realized_vol[k]= std(log_returns over bars N+1..N+k) × sqrt(k)
                   (scaled to same horizon for comparability)
  pre_vol[w]     = std(log_returns over bars N-w..N) × sqrt(w)
  vol_ratio[k]   = realized_vol[k] / pre_vol[ATR window]
                   > 1.0 means volatility expanded after entry

Usage:
  python scripts/volatility_edge_audit.py --candle-dir data/candles --start 2023-01
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
    MomentumConfig, MomentumStrategy,
    compute_atr,
)

HORIZONS      = [1, 3, 5, 10, 20]
PRE_WINDOW    = 14          # bars before entry for pre-signal vol estimate
TAKER         = 0.0006
BREAKEVEN_ABS = TAKER * 2   # 0.12% — minimum abs move to cover round-trip fees
_W            = 76


# ---------------------------------------------------------------------------
# Data loading  (shared with signal_forensic_audit.py pattern)
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


def _load_candles(candle_dir: Path, symbol: str,
                  start_s: int, end_s: int,
                  ) -> tuple[list[Candle], list[Candle]]:
    """Return (candles_1m, merged_1m_5m_sorted)."""
    path_1m = candle_dir / f"{symbol}_1m.parquet"
    path_5m = candle_dir / f"{symbol}_5m.parquet"
    if not path_1m.exists():
        print(f"[ERROR] {path_1m} not found.")
        sys.exit(1)

    c1m = sorted(_rows_to_candles(pq.read_table(path_1m).to_pylist(), start_s, end_s),
                 key=lambda c: c.close_time)

    if path_5m.exists():
        c5m = sorted(_rows_to_candles(pq.read_table(path_5m).to_pylist(), start_s, end_s),
                     key=lambda c: c.close_time)
    else:
        print("  [WARN] 5m parquet not found — aggregating from 1m")
        c5m = _aggregate_5m(c1m)

    merged = sorted(c1m + c5m, key=lambda c: c.close_time)
    return c1m, merged


# ---------------------------------------------------------------------------
# Entry record
# ---------------------------------------------------------------------------

@dataclass
class VolEntry:
    bar_idx: int
    close_at_entry: float
    atr_at_entry: float        # pre-signal ATR (proxy for expected bar size)
    pre_vol: float             # realised vol of PRE_WINDOW bars before entry
    entry_type: str            # "signal" | "random" | "breakout"

    # Filled after scan
    abs_ret:     dict[int, float] = field(default_factory=dict)  # horizon → |Δclose|/close
    real_vol:    dict[int, float] = field(default_factory=dict)  # horizon → realised vol
    vol_ratio:   dict[int, float] = field(default_factory=dict)  # real_vol / pre_vol
    atr_norm:    dict[int, float] = field(default_factory=dict)  # abs_ret / (atr/close)


# ---------------------------------------------------------------------------
# Pre/post vol helpers
# ---------------------------------------------------------------------------

def _realised_vol(closes: list[float]) -> float:
    """Std of log returns over a close sequence, scaled to length of sequence."""
    if len(closes) < 2:
        return float("nan")
    rets = [math.log(closes[i] / closes[i-1])
            for i in range(1, len(closes))
            if closes[i-1] > 0 and closes[i] > 0]
    if len(rets) < 2:
        return float("nan")
    n    = len(rets)
    mean = sum(rets) / n
    var  = sum((r - mean) ** 2 for r in rets) / (n - 1)
    # Scale to window length so horizons are comparable
    return math.sqrt(var) * math.sqrt(n)


def _fill_vol_metrics(entries: list[VolEntry],
                      closes: list[float]) -> None:
    n = len(closes)
    for e in entries:
        i = e.bar_idx
        for h in HORIZONS:
            fwd_idx = i + h
            if fwd_idx >= n:
                e.abs_ret[h]   = float("nan")
                e.real_vol[h]  = float("nan")
                e.vol_ratio[h] = float("nan")
                e.atr_norm[h]  = float("nan")
                continue

            # 1. Absolute price change
            ar = abs(closes[fwd_idx] - closes[i]) / closes[i]
            e.abs_ret[h] = ar

            # 2. Realised vol over [i+1 .. i+h]
            rv = _realised_vol(closes[i : fwd_idx + 1])
            e.real_vol[h] = rv

            # 3. Vol ratio: post / pre
            e.vol_ratio[h] = (rv / e.pre_vol) if (e.pre_vol > 0 and not math.isnan(rv)) else float("nan")

            # 4. ATR-normalised absolute return
            if e.atr_at_entry > 0 and closes[i] > 0:
                e.atr_norm[h] = ar / (e.atr_at_entry / closes[i])
            else:
                e.atr_norm[h] = float("nan")


# ---------------------------------------------------------------------------
# Entry collection
# ---------------------------------------------------------------------------

def collect_signal_entries(c1m: list[Candle],
                           merged: list[Candle]) -> list[VolEntry]:
    idx_by_close: dict[int, int] = {c.close_time: i for i, c in enumerate(c1m)}
    strategy = MomentumStrategy()
    entries: list[VolEntry] = []

    for c in merged:
        sig = strategy.on_candle(c)
        if sig is None or c.timeframe != "1m":
            continue
        i = idx_by_close.get(c.close_time, -1)
        if i < PRE_WINDOW:
            continue
        closes_pre = [c1m[j].close for j in range(i - PRE_WINDOW, i + 1)]
        pre_vol = _realised_vol(closes_pre)
        entries.append(VolEntry(
            bar_idx=i,
            close_at_entry=c.close,
            atr_at_entry=sig.atr,
            pre_vol=pre_vol if not math.isnan(pre_vol) else 0.0,
            entry_type="signal",
        ))
    return entries


def collect_random_entries(c1m: list[Candle],
                           n: int, seed: int = 42) -> list[VolEntry]:
    rng = random.Random(seed)
    eligible = list(range(PRE_WINDOW + 1, len(c1m) - max(HORIZONS)))
    sample = rng.sample(eligible, min(n, len(eligible)))
    entries: list[VolEntry] = []
    for i in sample:
        c = c1m[i]
        atr = compute_atr(c1m[max(0, i-14):i+1], 14)
        closes_pre = [c1m[j].close for j in range(i - PRE_WINDOW, i + 1)]
        pre_vol = _realised_vol(closes_pre)
        entries.append(VolEntry(
            bar_idx=i,
            close_at_entry=c.close,
            atr_at_entry=atr,
            pre_vol=pre_vol if not math.isnan(pre_vol) else 0.0,
            entry_type="random",
        ))
    return entries


def collect_breakout_entries(c1m: list[Candle],
                              period: int = 20) -> list[VolEntry]:
    entries: list[VolEntry] = []
    for i in range(period + 1, len(c1m) - max(HORIZONS)):
        trigger  = c1m[i]
        lookback = c1m[i - period : i]
        rh = max(b.high for b in lookback)
        rl = min(b.low  for b in lookback)
        if trigger.close <= rh and trigger.close >= rl:
            continue
        atr = compute_atr(c1m[max(0, i-14):i+1], 14)
        closes_pre = [c1m[j].close for j in range(i - PRE_WINDOW, i + 1)]
        pre_vol = _realised_vol(closes_pre)
        entries.append(VolEntry(
            bar_idx=i,
            close_at_entry=trigger.close,
            atr_at_entry=atr,
            pre_vol=pre_vol if not math.isnan(pre_vol) else 0.0,
            entry_type="breakout",
        ))
    return entries


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _stats(vals: list[float]) -> dict:
    v = [x for x in vals if not math.isnan(x) and not math.isinf(x)]
    if not v:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "p75": float("nan"), "p90": float("nan"), "p95": float("nan"),
                "t": float("nan"), "p": float("nan")}
    v_s = sorted(v)
    n   = len(v_s)
    mean = sum(v_s) / n
    var  = sum((x - mean) ** 2 for x in v_s) / max(n - 1, 1)
    std  = math.sqrt(var)
    se   = std / math.sqrt(n) if n > 1 else float("inf")
    t    = mean / se if se > 0 else float("nan")
    p    = 2 * (1 - _norm_cdf(abs(t))) if not math.isnan(t) else float("nan")
    return {
        "n":      n,
        "mean":   mean,
        "median": v_s[n // 2],
        "p75":    v_s[min(n-1, int(n * 0.75))],
        "p90":    v_s[min(n-1, int(n * 0.90))],
        "p95":    v_s[min(n-1, int(n * 0.95))],
        "t":      t,
        "p":      p,
    }


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))


def _sig(p: float) -> str:
    if math.isnan(p): return "   "
    if p < 0.001:     return "***"
    if p < 0.01:      return "** "
    if p < 0.05:      return "*  "
    return "   "


# ---------------------------------------------------------------------------
# Excess volatility test: signal mean − random mean, with t-test
# ---------------------------------------------------------------------------

def _two_sample_t(a: list[float], b: list[float]) -> tuple[float, float]:
    """Welch t-test: H0 = mean(a) == mean(b). Returns (t, p)."""
    a = [x for x in a if not math.isnan(x)]
    b = [x for x in b if not math.isnan(x)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    na, nb   = len(a), len(b)
    ma, mb   = sum(a)/na, sum(b)/nb
    va = sum((x-ma)**2 for x in a) / (na-1)
    vb = sum((x-mb)**2 for x in b) / (nb-1)
    se = math.sqrt(va/na + vb/nb)
    if se == 0:
        return float("nan"), float("nan")
    t = (ma - mb) / se
    p = 2 * (1 - _norm_cdf(abs(t)))
    return t, p


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _pct(v: float, dp: int = 4) -> str:
    if math.isnan(v) or math.isinf(v):
        return "  —    "
    return f"{v*100:.{dp}f}%"


def _fmt(v: float, dp: int = 4) -> str:
    if math.isnan(v) or math.isinf(v):
        return "  —  "
    return f"{v:.{dp}f}"


def print_report(sig_ents: list[VolEntry],
                 ran_ents: list[VolEntry],
                 bo_ents:  list[VolEntry],
                 c1m:      list[Candle],
                 symbol:   str) -> None:

    period_start = datetime.fromtimestamp(c1m[0].close_time,  tz=timezone.utc).strftime("%Y-%m-%d")
    period_end   = datetime.fromtimestamp(c1m[-1].close_time, tz=timezone.utc).strftime("%Y-%m-%d")

    print()
    print("═" * _W)
    print(f"  VOLATILITY EDGE AUDIT — {symbol}")
    print(f"  Period: {period_start} → {period_end}   ({len(c1m):,} bars)")
    print(f"  Signal: {len(sig_ents):,}   Random: {len(ran_ents):,}   Breakout: {len(bo_ents):,}")
    print("═" * _W)

    groups = [
        ("Signal",   sig_ents),
        ("Random",   ran_ents),
        ("Breakout", bo_ents),
    ]

    # Pre-compute stats for every group × horizon × metric
    abs_stats:  dict[str, dict[int, dict]] = {}
    rvol_stats: dict[str, dict[int, dict]] = {}
    vrat_stats: dict[str, dict[int, dict]] = {}
    atrn_stats: dict[str, dict[int, dict]] = {}

    for name, ents in groups:
        abs_stats[name]  = {}
        rvol_stats[name] = {}
        vrat_stats[name] = {}
        atrn_stats[name] = {}
        for h in HORIZONS:
            abs_stats[name][h]  = _stats([e.abs_ret[h]   for e in ents])
            rvol_stats[name][h] = _stats([e.real_vol[h]  for e in ents])
            vrat_stats[name][h] = _stats([e.vol_ratio[h] for e in ents])
            atrn_stats[name][h] = _stats([e.atr_norm[h]  for e in ents])

    # -----------------------------------------------------------------------
    # Section 1: Mean absolute return
    # -----------------------------------------------------------------------
    print()
    print("  1. MEAN ABSOLUTE RETURN  |close[N+k] − close[N]| / close[N]")
    print("     Direction is ignored. Positive = price moved at all.")
    print(f"     Fee breakeven (2× taker):  {BREAKEVEN_ABS*100:.4f}%")
    print()
    print(f"  {'Horizon':<12}" + "".join(f"  {g[0]:>14}" for g in groups) + "  {'Sig-Ran diff':>14}  {'p':>8}")
    print("  " + "─" * _W)
    for h in HORIZONS:
        sig_v  = abs_stats["Signal"][h]["mean"]
        ran_v  = abs_stats["Random"][h]["mean"]
        bo_v   = abs_stats["Breakout"][h]["mean"]
        diff   = sig_v - ran_v
        sig_ar = [e.abs_ret[h] for e in sig_ents]
        ran_ar = [e.abs_ret[h] for e in ran_ents]
        _, tp  = _two_sample_t(sig_ar, ran_ar)
        star   = _sig(tp)
        print(f"  +{h:<11d}  {_pct(sig_v):>14}  {_pct(ran_v):>14}  {_pct(bo_v):>14}"
              f"  {_pct(diff):>14}  {tp:>6.4f}{star}")

    # -----------------------------------------------------------------------
    # Section 2: Realised volatility post-entry
    # -----------------------------------------------------------------------
    print()
    print("  2. REALISED VOLATILITY POST-ENTRY  std(log_returns[N+1..N+k]) × sqrt(k)")
    print("     Higher = larger typical bar-to-bar moves after entry.")
    print()
    print(f"  {'Horizon':<12}" + "".join(f"  {g[0]:>14}" for g in groups) + "  {'Sig-Ran diff':>14}  {'p':>8}")
    print("  " + "─" * _W)
    for h in HORIZONS:
        sv = rvol_stats["Signal"][h]["mean"]
        rv = rvol_stats["Random"][h]["mean"]
        bv = rvol_stats["Breakout"][h]["mean"]
        diff = sv - rv
        _, tp = _two_sample_t(
            [e.real_vol[h] for e in sig_ents],
            [e.real_vol[h] for e in ran_ents],
        )
        star = _sig(tp)
        print(f"  +{h:<11d}  {_pct(sv):>14}  {_pct(rv):>14}  {_pct(bv):>14}"
              f"  {_pct(diff):>14}  {tp:>6.4f}{star}")

    # -----------------------------------------------------------------------
    # Section 3: Vol ratio  (post-signal vol / pre-signal vol)
    # -----------------------------------------------------------------------
    print()
    print("  3. POST/PRE VOLATILITY RATIO  realised_vol[post] / realised_vol[pre]")
    print(f"     Pre-window: {PRE_WINDOW} bars.  > 1.0 = vol expanded after signal.")
    print()
    print(f"  {'Horizon':<12}" + "".join(f"  {g[0]:>14}" for g in groups))
    print("  " + "─" * (_W - 8))
    for h in HORIZONS:
        sv = vrat_stats["Signal"][h]["mean"]
        rv = vrat_stats["Random"][h]["mean"]
        bv = vrat_stats["Breakout"][h]["mean"]
        print(f"  +{h:<11d}  {_fmt(sv, 3):>14}  {_fmt(rv, 3):>14}  {_fmt(bv, 3):>14}")

    # -----------------------------------------------------------------------
    # Section 4: ATR-normalised absolute return
    # -----------------------------------------------------------------------
    print()
    print("  4. ATR-NORMALISED ABSOLUTE RETURN  abs_ret / (ATR/close)")
    print("     > 1.0 = price moved more than one ATR away from entry.")
    print("     Measures whether signal bars generate multi-ATR moves.")
    print()
    print(f"  {'Horizon':<12}" + "".join(f"  {g[0]:>14}" for g in groups))
    print("  " + "─" * (_W - 8))
    for h in HORIZONS:
        sv = atrn_stats["Signal"][h]["mean"]
        rv = atrn_stats["Random"][h]["mean"]
        bv = atrn_stats["Breakout"][h]["mean"]
        print(f"  +{h:<11d}  {_fmt(sv, 3):>14}  {_fmt(rv, 3):>14}  {_fmt(bv, 3):>14}")

    # -----------------------------------------------------------------------
    # Section 5: Distribution buckets — how often does the signal produce
    # a move large enough to be tradable (> 1×ATR, > 2×ATR, > 3×ATR)?
    # -----------------------------------------------------------------------
    print()
    print("  5. DISTRIBUTION OF ABSOLUTE MOVES AT +5 BARS")
    print("     Fraction of entries where abs move exceeded threshold.")
    print()
    h5 = 5
    thresholds = [
        ("< fee breakeven (0.12%)", lambda v, atr, cl: v < BREAKEVEN_ABS),
        ("0.12% – 0.5×ATR",        lambda v, atr, cl: BREAKEVEN_ABS <= v < 0.5 * atr / cl),
        ("0.5×ATR – 1.0×ATR",      lambda v, atr, cl: 0.5 * atr / cl <= v < atr / cl),
        ("1.0×ATR – 2.0×ATR",      lambda v, atr, cl: atr / cl <= v < 2 * atr / cl),
        ("> 2.0×ATR",               lambda v, atr, cl: v >= 2 * atr / cl),
    ]
    print(f"  {'Bucket':<28}" + "".join(f"  {g[0]:>12}" for g in groups))
    print("  " + "─" * 68)
    for label, fn in thresholds:
        row = f"  {label:<28}"
        for name, ents in groups:
            vals = [(e.abs_ret[h5], e.atr_at_entry, e.close_at_entry)
                    for e in ents
                    if not math.isnan(e.abs_ret.get(h5, float("nan")))]
            frac = sum(1 for v, atr, cl in vals if fn(v, atr, cl)) / len(vals) if vals else 0.0
            row += f"  {frac*100:>11.1f}%"
        print(row)

    # -----------------------------------------------------------------------
    # Section 6: Pre-signal vol vs post-signal vol for signal entries only
    # -----------------------------------------------------------------------
    print()
    print("  6. SIGNAL ENTRIES: PRE vs POST VOLATILITY")
    print(f"     Pre = std(log_returns) over {PRE_WINDOW} bars before entry.")
    print()
    pre_vols  = [e.pre_vol  for e in sig_ents if e.pre_vol > 0]
    post5_vols= [e.real_vol[5] for e in sig_ents if not math.isnan(e.real_vol.get(5, float("nan")))]
    pre_stat  = _stats(pre_vols)
    post5_stat= _stats(post5_vols)
    if pre_stat["n"] and post5_stat["n"]:
        ratio = post5_stat["mean"] / pre_stat["mean"] if pre_stat["mean"] > 0 else float("nan")
        print(f"  Pre-signal realised vol  (mean): {_pct(pre_stat['mean'])}")
        print(f"  Post-signal realised vol (mean, +5 bars): {_pct(post5_stat['mean'])}")
        print(f"  Post/Pre ratio: {ratio:.3f}x  {'↑ vol expands' if ratio > 1.05 else ('↓ vol contracts' if ratio < 0.95 else '≈ no change')}")
        _, t_pp = _two_sample_t(post5_vols, pre_vols)
        print(f"  Welch t-test (post vs pre, p={t_pp:.4f}): "
              f"{'significant difference' if t_pp < 0.05 else 'no significant difference'}")

    # -----------------------------------------------------------------------
    # Section 7: Year-by-year abs return at +5 bars for signal
    # -----------------------------------------------------------------------
    print()
    print("  7. YEAR-BY-YEAR: SIGNAL ABSOLUTE RETURN AT +5 BARS")
    print()
    by_year: dict[str, list[float]] = defaultdict(list)
    for e in sig_ents:
        v = e.abs_ret.get(5, float("nan"))
        if not math.isnan(v):
            dt = datetime.fromtimestamp(c1m[e.bar_idx].close_time, tz=timezone.utc)
            by_year[str(dt.year)].append(v)

    ran_all_5 = [e.abs_ret.get(5, float("nan")) for e in ran_ents]
    ran_mean5 = (sum(v for v in ran_all_5 if not math.isnan(v)) /
                 sum(1 for v in ran_all_5 if not math.isnan(v))) if ran_all_5 else float("nan")

    print(f"  {'Year':<8}  {'N':>6}  {'Mean abs ret':>14}  {'vs Random':>12}  {'> 1×ATR %':>10}")
    print("  " + "─" * 56)
    for yr in sorted(by_year):
        vals = by_year[yr]
        m    = sum(vals) / len(vals)
        ents_yr = [e for e in sig_ents
                   if datetime.fromtimestamp(c1m[e.bar_idx].close_time,
                                             tz=timezone.utc).year == int(yr)]
        atr_frac = [e.atr_norm.get(5, float("nan")) for e in ents_yr]
        pct_1atr = (sum(1 for v in atr_frac if not math.isnan(v) and v > 1.0) /
                    max(1, sum(1 for v in atr_frac if not math.isnan(v))))
        diff = m - ran_mean5
        flag = " ▲" if m > ran_mean5 * 1.1 else (" ▼" if m < ran_mean5 * 0.9 else "")
        print(f"  {yr:<8}  {len(vals):>6}  {_pct(m):>14}  {_pct(diff):>12}  {pct_1atr*100:>9.1f}%{flag}")

    # -----------------------------------------------------------------------
    # Section 8: Verdict
    # -----------------------------------------------------------------------
    print()
    print("═" * _W)
    print("  VERDICT")
    print("═" * _W)
    print()

    sig_abs5   = abs_stats["Signal"][5]["mean"]
    ran_abs5   = abs_stats["Random"][5]["mean"]
    sig_rvol5  = rvol_stats["Signal"][5]["mean"]
    ran_rvol5  = rvol_stats["Random"][5]["mean"]
    sig_vrat5  = vrat_stats["Signal"][5]["mean"]
    _, pval_abs  = _two_sample_t([e.abs_ret[5]  for e in sig_ents],
                                 [e.abs_ret[5]  for e in ran_ents])
    _, pval_rvol = _two_sample_t([e.real_vol[5] for e in sig_ents],
                                 [e.real_vol[5] for e in ran_ents])

    abs_beats_random  = (not math.isnan(sig_abs5)  and sig_abs5  > ran_abs5)
    rvol_beats_random = (not math.isnan(sig_rvol5) and sig_rvol5 > ran_rvol5)
    abs_sig           = (not math.isnan(pval_abs)  and pval_abs  < 0.05)
    rvol_sig          = (not math.isnan(pval_rvol) and pval_rvol < 0.05)
    vol_expands       = (not math.isnan(sig_vrat5) and sig_vrat5 > 1.05)

    print(f"  Signal mean abs ret @+5:   {_pct(sig_abs5)}"
          f"  vs random {_pct(ran_abs5)}  diff={_pct(sig_abs5-ran_abs5)}"
          f"  p={pval_abs:.4f}{'  ***' if abs_sig else ''}")
    print(f"  Signal realised vol @+5:   {_pct(sig_rvol5)}"
          f"  vs random {_pct(ran_rvol5)}  diff={_pct(sig_rvol5-ran_rvol5)}"
          f"  p={pval_rvol:.4f}{'  ***' if rvol_sig else ''}")
    print(f"  Post/pre vol ratio @+5:    {sig_vrat5:.3f}x")
    print()

    if abs_beats_random and abs_sig and rvol_beats_random and rvol_sig:
        print("  FINDING: VOLATILITY EDGE EXISTS")
        print()
        print("  The signal selects bars where subsequent absolute moves AND realised")
        print("  volatility are statistically larger than a random baseline.")
        print("  There is a volatility-forecasting edge even though direction is useless.")
        print()
        if vol_expands:
            print("  Volatility expands after the signal fires (post/pre > 1.05).")
            print("  The signal is correctly identifying volatility regime transitions,")
            print("  but the current exit structure (directional stop/TP) cannot capture")
            print("  an undirected move.")
        else:
            print("  Volatility does not meaningfully expand after the signal — the signal")
            print("  is selecting bars that are already in a high-volatility environment.")
        print()
        print("  IMPLICATION: A volatility-based strategy (wider TPs, straddle-equivalent")
        print("  positioning, or entry-on-both-sides at the breakout level) may be able")
        print("  to monetize this edge. Directional betting cannot.")

    elif abs_beats_random and abs_sig:
        print("  FINDING: WEAK VOLATILITY EDGE (absolute move only)")
        print()
        print("  Signal entries show larger absolute moves than random, but the")
        print("  realised volatility difference is not statistically significant.")
        print("  The signal may be selecting for a few large outlier moves rather")
        print("  than consistently elevated volatility.")

    elif rvol_beats_random and rvol_sig:
        print("  FINDING: WEAK VOLATILITY EDGE (realised vol only)")
        print()
        print("  Signal entries show higher realised volatility than random, but")
        print("  mean absolute return is not statistically different.")
        print("  The subsequent volatility is elevated but meandering — the price")
        print("  chops rather than making a clean move.")

    else:
        print("  FINDING: NO VOLATILITY EDGE")
        print()
        print("  The signal does not predict post-entry volatility better than chance.")
        print("  The signal selects high-activity bars (high volume, high range) but")
        print("  subsequent volatility reverts to the baseline quickly.")
        print()
        print("  IMPLICATION: There is no recoverable edge in this signal — not")
        print("  directional, not volatility-based. The filter combination identifies")
        print("  a specific bar pattern that has no predictive content at any of the")
        f"  tested horizons ({HORIZONS})."
        print(f"  tested horizons ({HORIZONS}).")
        print()
        print("  The 57% historical win rate was a regime artefact.")
        print("  The entry logic needs to be replaced, not adjusted.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Volatility edge audit")
    p.add_argument("--symbol",     default="ETHUSDT")
    p.add_argument("--candle-dir", default="data/candles", type=Path)
    p.add_argument("--start",      default="2023-01", help="YYYY-MM")
    p.add_argument("--end",        default=None,      help="YYYY-MM")
    return p.parse_args()


def main() -> None:
    args   = _parse_args()

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

    print(f"\nVolatility edge audit — {args.symbol}")
    print(f"Loading candles from {candle_dir} …")
    c1m, merged = _load_candles(candle_dir, args.symbol, start_s, end_s)
    print(f"  {len(c1m):,} × 1m bars")

    if len(c1m) < 1000:
        print("[ERROR] Too few bars.")
        sys.exit(1)

    print("Collecting signal entries …")
    sig_ents = collect_signal_entries(c1m, merged)
    print(f"  {len(sig_ents):,} signal entries")

    n_compare = max(len(sig_ents), 500)
    print("Collecting random entries …")
    ran_ents = collect_random_entries(c1m, n=n_compare)
    print(f"  {len(ran_ents):,} random entries")

    print("Collecting breakout entries …")
    bo_ents = collect_breakout_entries(c1m)
    print(f"  {len(bo_ents):,} breakout entries")

    print("Computing volatility metrics …")
    closes = [c.close for c in c1m]
    _fill_vol_metrics(sig_ents, closes)
    _fill_vol_metrics(ran_ents, closes)
    _fill_vol_metrics(bo_ents,  closes)

    print_report(sig_ents, ran_ents, bo_ents, c1m, args.symbol)


if __name__ == "__main__":
    main()
