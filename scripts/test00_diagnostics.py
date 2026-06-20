#!/usr/bin/env python3
"""
TEST 00 — Momentum Overlap Diagnostic Suite

Three tests:
  00A: Monthly overlap between V8.63 held coins and Top-3 momentum selection
  00B: Momentum leadership persistence at 1m / 2m / 3m lag
  00C: Selection contribution analysis — regime timing vs asset selection
"""
from __future__ import annotations
import random
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

_REPO  = Path(__file__).parent.parent
_CDIR  = _REPO / "data" / "candles"
_SYMS  = [
    "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT",
    "XRPUSDT","ADAUSDT","DOGEUSDT","AVAXUSDT",
    "LINKUSDT","LTCUSDT","DOTUSDT","TRXUSDT",
]
_FROM_MS = int(dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
_TO_MS   = int(dt.datetime(2026, 6, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
_CAPITAL = 10_000.0


# ── data loading ──────────────────────────────────────────────────────────────

def _load(sym: str) -> pd.DataFrame:
    path = _CDIR / f"{sym}_1D.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path, columns=["open_time", "close"])
    df = df.rename(columns={"open_time": "ts"})
    df["ts"] = df["ts"].astype(int)
    df["close"] = df["close"].astype(float)
    df = df.sort_values("ts").reset_index(drop=True)
    mask = (df["ts"] >= _FROM_MS) & (df["ts"] <= _TO_MS)
    return df[mask].reset_index(drop=True)


# ── indicators ────────────────────────────────────────────────────────────────

def _ema(s: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(s), np.nan)
    alpha = 2.0 / (n + 1)
    val = float(s[0])
    for i in range(len(s)):
        val = alpha * float(s[i]) + (1 - alpha) * val
        if i >= n - 1:
            out[i] = val
    return out


def _sma(s: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(s), np.nan)
    for i in range(n - 1, len(s)):
        out[i] = np.mean(s[i - n + 1:i + 1])
    return out


def _macd_long_state(s: np.ndarray) -> np.ndarray:
    e12 = _ema(s, 12); e26 = _ema(s, 26)
    macd = e12 - e26
    valid_idx = np.where(~np.isnan(macd))[0]
    sig = np.full(len(s), np.nan)
    if len(valid_idx) >= 9:
        sig_vals = _ema(macd[valid_idx], 9)
        sig[valid_idx] = sig_vals
    hist = macd - sig
    state = np.zeros(len(s), dtype=bool)
    for i in range(1, len(s)):
        h_prev, h_cur = hist[i-1], hist[i]
        if np.isnan(h_prev) or np.isnan(h_cur):
            state[i] = state[i-1]
        elif h_prev <= 0 < h_cur:
            state[i] = True
        elif h_prev >= 0 > h_cur:
            state[i] = False
        else:
            state[i] = state[i-1]
    return state


def _build_all() -> dict[str, dict]:
    """Returns per-symbol arrays aligned to their own date series."""
    result = {}
    for sym in _SYMS:
        df = _load(sym)
        if df.empty:
            continue
        c = df["close"].values
        ts = df["ts"].values
        e8    = _ema(c, 8)
        e21   = _ema(c, 21)
        s200  = _sma(c, 200)
        ml    = _macd_long_state(c)
        # EMA long state (crossover-based)
        above = e8 > e21
        ema_long = np.zeros(len(c), dtype=bool)
        for i in range(1, len(c)):
            if above[i] and not above[i-1]:   ema_long[i] = True
            elif not above[i] and above[i-1]: ema_long[i] = False
            else:                              ema_long[i] = ema_long[i-1]
        # momentum windows
        mom20 = np.full(len(c), np.nan)
        mom60 = np.full(len(c), np.nan)
        mom30 = np.full(len(c), np.nan)
        for i in range(20, len(c)): mom20[i] = (c[i] - c[i-20]) / c[i-20]
        for i in range(60, len(c)): mom60[i] = (c[i] - c[i-60]) / c[i-60]
        for i in range(30, len(c)): mom30[i] = (c[i] - c[i-30]) / c[i-30]
        # coin signal: ema_long AND macd_long AND mom20>0
        coin_sig = ema_long & ml & (mom20 > 0)
        above200 = np.where(np.isnan(s200), False, c > s200)
        result[sym] = {
            "ts":       ts,
            "close":    c,
            "ema_long": ema_long,
            "macd_long":ml,
            "coin_sig": coin_sig,
            "above200": above200,
            "mom20":    mom20,
            "mom30":    mom30,
            "mom60":    mom60,
            "by_ts":    {int(t): i for i, t in enumerate(ts)},
        }
    return result


def _build_btc_regime(syms_data: dict, common_ts: list[int]) -> np.ndarray:
    """Asymmetric regime: True = BULL."""
    btc = syms_data["BTCUSDT"]
    bear = False; streak = 0
    bull = np.ones(len(common_ts), dtype=bool)
    for j, ts in enumerate(common_ts):
        i = btc["by_ts"].get(ts)
        if i is None:
            bull[j] = not bear
            continue
        above200 = bool(btc["above200"][i])
        mom30 = float(btc["mom30"][i]) if not np.isnan(btc["mom30"][i]) else 0.0
        streak = streak + 1 if above200 else 0
        if not bear:
            if (not above200) and (mom30 < -0.20):
                bear = True
        else:
            if streak >= 10:
                bear = False
        bull[j] = not bear
    return bull


# ── helpers ───────────────────────────────────────────────────────────────────

def _cagr(eq: np.ndarray, n_days: int) -> float:
    if eq[0] <= 0 or eq[-1] <= 0 or n_days <= 0: return 0.0
    return (eq[-1] / eq[0]) ** (365.0 / n_days) - 1.0

def _sharpe(eq: np.ndarray) -> float:
    rets = np.diff(eq) / eq[:-1]
    rets = rets[np.isfinite(rets)]
    if len(rets) < 2: return 0.0
    s = rets.std()
    return (rets.mean() / s * np.sqrt(252)) if s > 0 else 0.0

def _maxdd(eq: np.ndarray) -> float:
    peak = np.maximum.accumulate(eq)
    return float(np.max((peak - eq) / np.where(peak > 0, peak, 1)))

def _year_returns(eq: np.ndarray, dates: pd.DatetimeIndex) -> dict[int, float]:
    yr_start: dict[int, float] = {}; yr_end: dict[int, float] = {}
    for i, d in enumerate(dates):
        yr = d.year
        if yr not in yr_start: yr_start[yr] = eq[i]
        yr_end[yr] = eq[i]
    return {yr: (yr_end[yr] - yr_start[yr]) / yr_start[yr] for yr in yr_start}


# ── TEST 00A — Monthly Overlap ────────────────────────────────────────────────

def run_00A(syms_data: dict, bull: np.ndarray, common_ts: list[int]) -> None:
    print("\n" + "=" * 80)
    print("  TEST 00A — MONTHLY OVERLAP: V8.63 SIGNALS vs TOP-3 60d MOMENTUM")
    print("=" * 80)

    dates = pd.to_datetime(common_ts, unit="ms", utc=True)

    # Month groups
    month_groups: dict[tuple, list[int]] = {}
    for j, ts in enumerate(common_ts):
        d = dates[j]; key = (d.year, d.month)
        month_groups.setdefault(key, []).append(j)

    print(f"\n  {'Month':8s}  {'V8.63 Held':32s}  {'Top-3 Mom':24s}  {'Overlap':8s}  {'Regime'}")
    print("  " + "-" * 100)

    all_ov, bull_ov, bear_ov, trans_ov = [], [], [], []

    for (yr, mo) in sorted(month_groups):
        idxs = month_groups[(yr, mo)]
        bull_days = sum(1 for j in idxs if bull[j])
        bear_days = len(idxs) - bull_days
        if bull_days >= len(idxs) * 0.7:   regime = "BULL"
        elif bear_days >= len(idxs) * 0.7: regime = "BEAR"
        else:                               regime = "TRANS"

        # V8.63 held: coins where coin_sig is True on >30% of bull days this month
        v863 = set()
        for sym in _SYMS:
            sd = syms_data[sym]
            active = sum(1 for j in idxs
                         if bull[j] and sd["by_ts"].get(common_ts[j]) is not None
                         and sd["coin_sig"][sd["by_ts"][common_ts[j]]])
            if active > max(1, len(idxs) * 0.3):
                v863.add(sym)

        # Top-3 momentum: ranked at start of month, SMA200 gate
        first_j = idxs[0]
        scores = {}
        for sym in _SYMS:
            i = syms_data[sym]["by_ts"].get(common_ts[first_j])
            if i is not None:
                m = syms_data[sym]["mom60"][i]
                a = syms_data[sym]["above200"][i]
                if not np.isnan(m) and a:
                    scores[sym] = m
        top3 = set(sorted(scores, key=scores.get, reverse=True)[:3])

        inter = v863 & top3; union = v863 | top3
        ov = len(inter) / len(union) if union else 1.0
        all_ov.append(ov)
        if regime == "BULL":   bull_ov.append(ov)
        elif regime == "BEAR": bear_ov.append(ov)
        else:                  trans_ov.append(ov)

        v_s = "+".join(s.replace("USDT","") for s in sorted(v863)) or "(cash)"
        t_s = "+".join(s.replace("USDT","") for s in sorted(top3)) or "(none)"
        print(f"  {yr}-{mo:02d}   {v_s:32s}  {t_s:24s}  {ov:6.0%}   {regime}")

    print("  " + "-" * 100)
    print(f"\n  SUMMARY:")
    print(f"  Average overlap — all months:        {np.mean(all_ov):.1%}  (n={len(all_ov)})")
    print(f"  Average overlap — BULL months:       {np.mean(bull_ov) if bull_ov else 0:.1%}  (n={len(bull_ov)})")
    print(f"  Average overlap — BEAR months:       {np.mean(bear_ov) if bear_ov else 0:.1%}  (n={len(bear_ov)})")
    print(f"  Average overlap — TRANS months:      {np.mean(trans_ov) if trans_ov else 0:.1%}  (n={len(trans_ov)})")
    print(f"  Random baseline (expected overlap):  25.0%  (3 from 12 by chance)")
    ov_mean = np.mean(all_ov)
    if ov_mean > 0.60:
        print(f"\n  VERDICT: HIGH overlap ({ov_mean:.0%}). V8.63 is already tracking momentum leaders.")
        print(f"           Dynamic rotation unlikely to produce dramatic improvement.")
    elif ov_mean < 0.35:
        print(f"\n  VERDICT: LOW overlap ({ov_mean:.0%}). V8.63 and momentum rotation select DIFFERENT coins.")
        print(f"           Dynamic selection introduces genuinely new information.")
    else:
        print(f"\n  VERDICT: MEDIUM overlap ({ov_mean:.0%}). Partial overlap — some new information available.")


# ── TEST 00B — Momentum Persistence ──────────────────────────────────────────

def run_00B(syms_data: dict, common_ts: list[int]) -> None:
    print("\n" + "=" * 80)
    print("  TEST 00B — MOMENTUM LEADERSHIP PERSISTENCE")
    print("=" * 80)

    dates = pd.to_datetime(common_ts, unit="ms", utc=True)
    month_starts: list[int] = []
    seen = set()
    for j, ts in enumerate(common_ts):
        d = dates[j]; key = (d.year, d.month)
        if key not in seen:
            seen.add(key); month_starts.append(j)

    def top3_at(j: int) -> set[str]:
        scores = {}
        for sym in _SYMS:
            i = syms_data[sym]["by_ts"].get(common_ts[j])
            if i is not None:
                m = syms_data[sym]["mom60"][i]
                if not np.isnan(m): scores[sym] = m
        return set(sorted(scores, key=scores.get, reverse=True)[:3])

    lag1, lag2, lag3 = [], [], []
    for k in range(len(month_starts)):
        cur = top3_at(month_starts[k])
        if len(cur) < 3: continue
        if k + 1 < len(month_starts):
            lag1.append(len(cur & top3_at(month_starts[k+1])) / 3.0)
        if k + 2 < len(month_starts):
            lag2.append(len(cur & top3_at(month_starts[k+2])) / 3.0)
        if k + 3 < len(month_starts):
            lag3.append(len(cur & top3_at(month_starts[k+3])) / 3.0)

    rb = 3 / 12  # random baseline: chance any coin is in top-3
    print(f"\n  Probability top-3 assets remain in top-3 after N months:")
    print(f"\n  {'Lag':10s}  {'Persistence':14s}  {'vs Random (25%)':16s}  n")
    print("  " + "-" * 52)
    for lag, vals, name in [(1,lag1,"1 month"),(2,lag2,"2 months"),(3,lag3,"3 months")]:
        m = np.mean(vals) if vals else 0
        print(f"  {name:10s}  {m:13.1%}  {m/rb:>12.1f}×           {len(vals)}")
    print(f"  {'Random':10s}  {rb:13.1%}  {'1.0×':>16s}")

    m1 = np.mean(lag1) if lag1 else 0
    print()
    if m1 > rb * 1.5:
        print("  CONCLUSION: STRONG persistence. Momentum rotation has a valid foundation.")
    elif m1 > rb * 1.1:
        print("  CONCLUSION: WEAK persistence. Slight edge but mostly noise.")
    else:
        print("  CONCLUSION: NO persistence. Rotation would be chasing noise.")


# ── TEST 00C — Selection Contribution ────────────────────────────────────────

def run_00C(syms_data: dict, bull: np.ndarray, common_ts: list[int]) -> None:
    print("\n" + "=" * 80)
    print("  TEST 00C — SELECTION CONTRIBUTION ANALYSIS")
    print("  Isolates regime timing alpha from asset selection alpha")
    print("=" * 80)

    dates = pd.to_datetime(common_ts, unit="ms", utc=True)
    N = len(common_ts)

    # Daily returns array per symbol
    daily_ret = np.zeros((len(_SYMS), N))
    sym_idx = {s: k for k, s in enumerate(_SYMS)}
    for sym in _SYMS:
        sd = syms_data[sym]; k = sym_idx[sym]
        closes = np.array([sd["close"][sd["by_ts"][ts]] if ts in sd["by_ts"] else np.nan
                           for ts in common_ts])
        for j in range(1, N):
            if not np.isnan(closes[j]) and not np.isnan(closes[j-1]) and closes[j-1] > 0:
                daily_ret[k, j] = (closes[j] - closes[j-1]) / closes[j-1]

    # Monthly rebalance dates
    month_starts = []
    seen = set()
    for j, ts in enumerate(common_ts):
        d = dates[j]; key = (d.year, d.month)
        if key not in seen:
            seen.add(key); month_starts.append(j)
    reb_set = set(month_starts)

    def simulate(get_holdings) -> np.ndarray:
        eq = np.full(N, _CAPITAL)
        holdings: list[int] = []
        for j in range(1, N):
            if j in reb_set or not holdings:
                holdings = get_holdings(j)
            if not holdings:
                eq[j] = eq[j-1]
                continue
            port_ret = np.mean(daily_ret[holdings, j])
            eq[j] = eq[j-1] * (1 + port_ret)
        return eq

    # Version A: V8.63 coin signals + BTC regime
    def get_v863(j: int) -> list[int]:
        if not bull[j]: return []
        h = []
        for sym in _SYMS:
            sd = syms_data[sym]; i = sd["by_ts"].get(common_ts[j])
            if i is not None and sd["coin_sig"][i]:
                h.append(sym_idx[sym])
        return h if h else []

    # Version C: Top-3 by 60d momentum with SMA200 gate
    def get_mom3(j: int) -> list[int]:
        if not bull[j]: return []
        scores = {}
        for sym in _SYMS:
            sd = syms_data[sym]; i = sd["by_ts"].get(common_ts[j])
            if i is not None:
                m = sd["mom60"][i]; a = sd["above200"][i]
                if not np.isnan(m) and a: scores[sym] = m
        top = sorted(scores, key=scores.get, reverse=True)[:3]
        return [sym_idx[s] for s in top]

    print("\n  Running Version A (V8.63 signals)...")
    eq_a = simulate(get_v863)

    print("  Running Version C (Top-3 momentum)...")
    eq_c = simulate(get_mom3)

    print("  Running Version B (Random, 500 seeds)...")
    all_finals = []; sum_eq = np.zeros(N)

    def get_random(j: int, rng_obj: random.Random) -> list[int]:
        if not bull[j]: return []
        pool = [sym_idx[s] for s in _SYMS
                if syms_data[s]["by_ts"].get(common_ts[j]) is not None
                and syms_data[s]["above200"][syms_data[s]["by_ts"][common_ts[j]]]]
        if len(pool) <= 3: return pool
        return rng_obj.sample(pool, 3)

    for seed in range(500):
        rng_obj = random.Random(seed)
        eq_b = simulate(lambda j, r=rng_obj: get_random(j, r))
        all_finals.append(eq_b[-1]); sum_eq += eq_b
    avg_eq_b = sum_eq / 500

    n_days = int((common_ts[-1] - common_ts[0]) / 86_400_000)
    yr_a = _year_returns(eq_a, dates)
    yr_b = _year_returns(avg_eq_b, dates)
    yr_c = _year_returns(eq_c, dates)

    print(f"\n  Starting capital: ${_CAPITAL:,.0f}")
    print(f"\n  {'Metric':22s}  {'A: V8.63 Signals':>18s}  {'B: Random (avg)':>18s}  {'C: Top-3 Momentum':>18s}")
    print("  " + "-" * 84)
    for label, val_a, val_b, val_c in [
        ("Final equity", f"${eq_a[-1]:,.0f}", f"${avg_eq_b[-1]:,.0f}", f"${eq_c[-1]:,.0f}"),
        ("CAGR",         f"{_cagr(eq_a,n_days):.1%}", f"{_cagr(avg_eq_b,n_days):.1%}", f"{_cagr(eq_c,n_days):.1%}"),
        ("Sharpe",       f"{_sharpe(eq_a):.2f}", f"{_sharpe(avg_eq_b):.2f}", f"{_sharpe(eq_c):.2f}"),
        ("Max DD",       f"{_maxdd(eq_a):.1%}", f"{_maxdd(avg_eq_b):.1%}", f"{_maxdd(eq_c):.1%}"),
    ]:
        print(f"  {label:22s}  {val_a:>18s}  {val_b:>18s}  {val_c:>18s}")

    print(f"\n  Random distribution (500 seeds):")
    print(f"    Final: min=${min(all_finals):,.0f}  P25=${np.percentile(all_finals,25):,.0f}"
          f"  median=${np.median(all_finals):,.0f}  P75=${np.percentile(all_finals,75):,.0f}"
          f"  max=${max(all_finals):,.0f}")

    print(f"\n  Year-by-year returns:")
    print(f"  {'Year':6s}  {'A: V8.63':>12s}  {'B: Random':>12s}  {'C: Top-3 Mom':>12s}")
    print("  " + "-" * 50)
    for yr in sorted(yr_a):
        print(f"  {yr}    {yr_a.get(yr,0):>11.1%}  {yr_b.get(yr,0):>11.1%}  {yr_c.get(yr,0):>11.1%}")

    cagr_a = _cagr(eq_a, n_days)
    cagr_b = _cagr(avg_eq_b, n_days)
    cagr_c = _cagr(eq_c, n_days)

    print(f"\n  ATTRIBUTION:")
    print(f"  Regime timing vs random:     {cagr_a - cagr_b:>+.1%} CAGR  (A minus B)")
    print(f"  Momentum selection vs random:{cagr_c - cagr_b:>+.1%} CAGR  (C minus B)")
    print(f"  Momentum vs V8.63 signals:   {cagr_c - cagr_a:>+.1%} CAGR  (C minus A)")

    gap_regime = cagr_a - cagr_b
    gap_mom    = cagr_c - cagr_b
    gap_ca     = cagr_c - cagr_a
    print()
    if abs(gap_regime) < 0.03 and abs(gap_mom) < 0.03:
        print("  → A ≈ B ≈ C: REGIME TIMING is everything. Selection barely matters.")
    elif gap_regime > 0.05 and gap_ca < 0.03:
        print("  → V8.63 signals add value, but momentum adds little on top.")
        print("     EMA/MACD IS the selection edge — dynamic rotation unlikely to help much.")
    elif gap_mom > gap_regime + 0.03:
        print("  → MOMENTUM SELECTION is the larger alpha source.")
        print("     Which coins you hold matters more than the EMA/MACD timing signals.")
        print("     Dynamic selection deserves high research priority.")
    elif gap_ca > 0.05:
        print(f"  → Top-3 momentum beats V8.63 by {gap_ca:.1%} CAGR.")
        print("     Dynamic rotation is a material improvement over current coin selection.")
    else:
        print("  → Results are mixed. No clear dominant alpha source.")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  TEST 00 — MOMENTUM OVERLAP DIAGNOSTIC SUITE")
    print(f"  Universe: {len(_SYMS)} coins | Period: 2021-01 → 2026-06")
    print("=" * 80)

    print("\n  Loading data and computing indicators...")
    syms_data = _build_all()
    missing = [s for s in _SYMS if s not in syms_data]
    if missing:
        print(f"  WARNING: missing data for {missing}")

    # Common timestamp grid from BTC
    btc_ts = sorted(syms_data["BTCUSDT"]["ts"].tolist())
    common_ts = [ts for ts in btc_ts if _FROM_MS <= ts <= _TO_MS]
    print(f"  Common grid: {len(common_ts)} daily bars")

    bull = _build_btc_regime(syms_data, common_ts)
    bull_pct = bull.mean()
    print(f"  BTC regime — BULL: {bull_pct:.1%}  BEAR: {1-bull_pct:.1%}")

    run_00A(syms_data, bull, common_ts)
    run_00B(syms_data, common_ts)
    run_00C(syms_data, bull, common_ts)

    print("\n" + "=" * 80)
    print("  TEST 00 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
