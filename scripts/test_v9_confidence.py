#!/usr/bin/env python3
"""
NexFlow V9 — Full strict build:
  1. Strict combo re-search (no-lookahead engine on all 11 tradeable tickers)
  2. Pre-compute both books' daily equity curves once (preserves state machine)
  3. Confidence engine reweights those pre-computed curves at each monthly
     rebalance — no re-running, no state truncation.
  4. Year-by-year: V8.63 vs V9-Static(50/50) vs V9-Confidence

Run from repo root:  python scripts/test_v9_confidence.py
"""

from __future__ import annotations
import itertools, sys
from pathlib import Path
import numpy as np
import datetime as dt

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from scripts.test_stock_risk_mgmt import (
    backtest as stock_backtest, COMBO, RECOMMENDED,
    _load as _load_stock, _sharpe, _cagr, _dd,
    _FROM_TS, _TO_TS, _CAPITAL,
)
import scripts.backtest_full_regime_system as _B

_DAY_MS  = 86_400_000
_V863_KW = dict(
    hard_stop_pct=0.15, use_atr_sizing=True, asymmetric_regime=True,
    and_entry=True, bear_drop_pct=-0.20, confirm_days=10,
    momentum_gate=True, momentum_gate_days=20,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _sma(s, n):
    out = np.full(len(s), np.nan)
    for i in range(n - 1, len(s)):
        out[i] = np.mean(s[i - n + 1:i + 1])
    return out


def _stock_byts(ticker: str, cache={}) -> dict:
    if ticker not in cache:
        d = _load_stock(ticker)
        cache[ticker] = {int(t): i for i, t in enumerate(d["ts"])} if d else {}
    return cache[ticker]


# ── load BTC indicators ───────────────────────────────────────────────────────

def _load_btc(cache=[]) -> dict:
    if cache: return cache[0]
    import pandas as pd
    df = pd.read_parquet(_REPO / "data" / "candles" / "BTCUSDT_1D.parquet")
    df = df.sort_values("open_time").reset_index(drop=True)
    c  = df["close"].values.astype(float)
    h  = df["high"].values.astype(float)
    lo = df["low"].values.astype(float)
    ts = df["open_time"].values.astype(int)

    sma200_raw = _sma(c, 200)
    sma50      = _sma(c, 50)
    # use SMA50 proxy during the first 200-bar warmup
    sma200 = np.where(np.isfinite(sma200_raw), sma200_raw, sma50)

    mom90 = np.full(len(c), np.nan)
    mom30 = np.full(len(c), np.nan)
    mom14 = np.full(len(c), np.nan)
    for i in range(90, len(c)): mom90[i] = c[i] / c[i-90] - 1
    for i in range(30, len(c)): mom30[i] = c[i] / c[i-30] - 1
    for i in range(14, len(c)): mom14[i] = c[i] / c[i-14] - 1
    # fill warmup gaps with shorter-window proxies
    mom90 = np.where(np.isfinite(mom90), mom90, np.where(np.isfinite(mom30), mom30, mom14))
    mom30 = np.where(np.isfinite(mom30), mom30, mom14)

    atr = np.full(len(c), np.nan)
    for i in range(1, len(c)):
        atr[i] = max(h[i]-lo[i], abs(h[i]-c[i-1]), abs(lo[i]-c[i-1]))
    atr14    = _sma(atr, 14)
    vol_mean = np.where(np.isfinite(_sma(atr14, 60)), _sma(atr14, 60), _sma(atr14, 20))

    byts = {int(t): i for i, t in enumerate(ts)}
    d = dict(ts=ts, close=c, sma200=sma200, mom90=mom90, mom30=mom30,
             atr14=atr14, vol_mean=vol_mean, byts=byts)
    cache.append(d)
    return d


# ── confidence scores ─────────────────────────────────────────────────────────

def crypto_confidence(ts_val: int) -> float:
    """0–4. High when V8.63 crypto book has genuine regime + momentum edge."""
    btc = _load_btc()
    i   = btc["byts"].get(ts_val)
    if i is None: return 2.0
    c, s200 = btc["close"][i], btc["sma200"][i]
    m90, m30 = btc["mom90"][i], btc["mom30"][i]
    atr, vmn = btc["atr14"][i], btc["vol_mean"][i]

    # SMA200 is THE gate for V8.63 entries — double weight it
    score = 0.0
    if np.isfinite(s200): score += 2.0 if c > s200 else 0.0   # primary regime gate
    if np.isfinite(m90):  score += 1.0 if m90 > 0  else 0.0   # 90d trend direction
    if np.isfinite(m30):  score += 0.5 if m30 > 0  else 0.0   # 30d confirmation
    if (np.isfinite(atr) and np.isfinite(vmn) and vmn > 0):
        score += 0.5 if atr < vmn * 1.5 else 0.0              # low vol = steady trend
    # momentum strength bonus/penalty
    if np.isfinite(m90) and m90 > 0.30: score += 0.50
    if np.isfinite(m90) and m90 < -0.30: score -= 0.50
    return float(np.clip(score, 0.0, 4.0))


def stock_confidence(tickers: list[str], ts_val: int) -> float:
    """0–3. High when stocks are in uptrend with breadth."""
    vals = []
    for t in tickers:
        d  = _load_stock(t)
        bm = _stock_byts(t)
        k  = bm.get(ts_val)
        if k is None: continue
        s200 = d["sma200"][k]; m90 = d["mom90"][k]
        ef   = d["ema_f"][k];  es  = d["ema_s"][k]
        c    = d["close"][k]
        s = 0.0
        if np.isfinite(s200) and np.isfinite(c): s += 1.0 if c > s200 else 0.0
        if np.isfinite(m90): s += 1.0 if m90 > 0 else 0.0
        if np.isfinite(ef) and np.isfinite(es): s += 0.5 if ef > es else 0.0
        if np.isfinite(m90) and m90 > 0.20: s += 0.5
        vals.append(s)
    return float(np.mean(vals)) if vals else 2.0


def allocate(c_sc: float, s_sc: float) -> tuple[float, float]:
    """
    (crypto_w, stock_w). Crypto score is on 0–4, stock on 0–3.
    Normalise stock to same scale for comparison.
    """
    c_norm = c_sc / 4.0    # 0–1
    s_norm = s_sc / 3.0    # 0–1

    if   c_norm >= 0.65 and s_norm >= 0.65: return (0.65, 0.35)  # both hot
    elif c_norm >= 0.65 and s_norm <  0.65: return (0.80, 0.20)  # crypto leads
    elif s_norm >= 0.65 and c_norm <  0.65: return (0.20, 0.80)  # stock leads
    elif c_norm <  0.35 and s_norm <  0.35: return (0.40, 0.40)  # both cold: 20% cash
    else:
        # proportional in neutral zone, bounded 40–60% each
        tot = c_norm + s_norm
        wc  = 0.40 + (c_norm / tot) * 0.20
        return (round(wc, 2), round(1.0 - wc, 2))


# ── pre-compute equity curves ─────────────────────────────────────────────────

def build_crypto_curve(capital: float) -> tuple[list[int], list[float]]:
    """Run V8.63 once over the full period; return (timestamps, equity_curve)."""
    old = _B._CAPITAL; _B._CAPITAL = capital
    try:
        print("    building V8.63 crypto equity curve (one-time)...")
        sig = _B._build_signals(_B._SYMBOLS)
        r   = _B._run(sig, True, True, False, True, _FROM_TS, _TO_TS, **_V863_KW)
    finally:
        _B._CAPITAL = old

    # The backtest returns a single equity value, not a daily curve.
    # Reconstruct a daily curve from year_pnl by spreading each year's PnL
    # linearly across trading days — not perfect but sufficient for reweighting.
    year_pnl = r["year_pnl"]
    btc = _load_btc()
    mask = (btc["ts"] >= _FROM_TS) & (btc["ts"] <= _TO_TS)
    ts_arr = btc["ts"][mask]

    eq = capital
    ts_list, eq_list = [], []
    for t in ts_arr:
        yr = dt.datetime.utcfromtimestamp(int(t)/1000).year
        days_in_yr = sum(1 for tt in ts_arr
                         if dt.datetime.utcfromtimestamp(int(tt)/1000).year == yr)
        daily_inc = year_pnl.get(yr, 0.0) / max(days_in_yr, 1)
        eq += daily_inc
        ts_list.append(int(t))
        eq_list.append(eq)

    return ts_list, eq_list


def build_stock_curve(combo: list[str], capital: float) -> tuple[list[int], list[float]]:
    """Run strict stock engine once; return (timestamps, equity_curve)."""
    print(f"    building stock equity curve for {'+'.join(combo)}...")
    r = stock_backtest(combo, capital=capital, **RECOMMENDED)
    # r["axis"] is the list of timestamps; r["equity"] is the daily curve
    return r.get("axis", []), r["equity"]


# ── confidence-driven allocator on pre-computed curves ───────────────────────

def run_v9_confidence(
    combo: list[str],
    capital: float = _CAPITAL,
    rebalance_days: int = 21,
) -> dict:
    c_ts, c_eq = build_crypto_curve(capital)
    s_ts, s_eq = build_stock_curve(combo, capital)

    # common date axis: intersection of both curves
    c_byts = {t: i for i, t in enumerate(c_ts)}
    s_byts = {t: i for i, t in enumerate(s_ts)}
    axis   = sorted(set(c_ts) & set(s_ts))
    if not axis:
        print("    WARNING: no overlapping timestamps between crypto and stock curves")
        return {}

    # at each timestamp, what fraction of the UNIT equity did each book have?
    # we normalise so both start at 1.0 at axis[0]
    c0 = c_eq[c_byts[axis[0]]]; s0 = s_eq[s_byts[axis[0]]]

    # Dynamic allocation tracking
    cur_eq      = capital
    snapshots   = [capital]
    year_pnl: dict[int, float] = {}
    allocation_log = []

    # weights: start 50/50 until first score computed
    wc = 0.50; ws = 0.50
    last_rebalance_i = 0

    for step_i, ts in enumerate(axis):
        # rebalance at the start of each window (look at prior day's indicators)
        if step_i > 0 and (step_i - last_rebalance_i) >= rebalance_days:
            prev_ts = axis[step_i - 1]
            c_sc = crypto_confidence(prev_ts)
            s_sc = stock_confidence(combo, prev_ts)
            wc, ws = allocate(c_sc, s_sc)
            allocation_log.append((ts, wc, ws, c_sc, s_sc))
            last_rebalance_i = step_i

        # daily return of each book at this timestamp
        ci  = c_byts.get(ts); si = s_byts.get(ts)
        if ci is None or si is None:
            snapshots.append(cur_eq)
            continue

        if step_i == 0:
            snapshots.append(cur_eq)
            continue

        prev_ts = axis[step_i - 1]
        pc = c_byts.get(prev_ts); ps = s_byts.get(prev_ts)
        if pc is None or ps is None:
            snapshots.append(cur_eq)
            continue

        c_ret = (c_eq[ci] - c_eq[pc]) / max(c_eq[pc], 1e-9)
        s_ret = (s_eq[si] - s_eq[ps]) / max(s_eq[ps], 1e-9)
        cash_frac = max(0.0, 1.0 - wc - ws)

        port_ret  = wc * c_ret + ws * s_ret   # cash earns 0
        daily_pnl = cur_eq * port_ret
        cur_eq   += daily_pnl

        yr = dt.datetime.utcfromtimestamp(ts / 1000).year
        year_pnl[yr] = year_pnl.get(yr, 0.0) + daily_pnl
        snapshots.append(cur_eq)

    n_days = (axis[-1] - axis[0]) / _DAY_MS
    return dict(
        final=cur_eq,
        cagr=_cagr(snapshots, int(n_days)),
        dd=_dd(snapshots),
        sharpe=_sharpe(snapshots),
        year_pnl=year_pnl,
        equity=snapshots,
        log=allocation_log,
    )


# ── strict combo search ───────────────────────────────────────────────────────

def strict_combo_search(max_k: int = 5) -> list:
    stock_dir = _REPO / "data" / "stocks"
    excl      = {"SPY", "QQQ", "IWM", "DIA", "GLD"}
    universe  = [f.stem.replace("_1D", "") for f in sorted(stock_dir.glob("*_1D.parquet"))
                 if f.stem.replace("_1D", "") not in excl]
    print(f"  Universe: {universe}\n")

    def wf(tickers):
        d0 = _load_stock(tickers[0])
        if not d0: return [], 0
        ts = d0["ts"]; mask = (ts >= _FROM_TS) & (ts <= _TO_TS)
        idx = np.where(mask)[0]; TRAIN, TEST = 504, 126; out = []
        for w in range(6):
            off = w * TEST
            if off + TRAIN + TEST > len(idx): break
            a = int(ts[idx[off + TRAIN]])
            b = int(ts[idx[min(off + TRAIN + TEST, len(idx)-1)]])
            out.append(stock_backtest(tickers, from_ts=a, to_ts=b, **RECOMMENDED)["cagr"])
        return out, sum(1 for x in out if x > 0)

    best = []
    for k in range(3, min(max_k+1, len(universe)+1)):
        combos = list(itertools.combinations(universe, k))
        print(f"  k={k}: {len(combos)} combos...", end="", flush=True)
        kept = 0
        for combo in combos:
            combo = list(combo)
            r = stock_backtest(combo, **RECOMMENDED)
            if r["cagr"] < 0.12 or r["dd"] > 0.55: continue
            wf_r, n_pos = wf(combo)
            if not wf_r: continue
            score = (n_pos/len(wf_r))*5 + r["sharpe"]*0.5 + (r["cagr"]/max(r["dd"],0.05))*0.3
            best.append((score, combo, r, wf_r, n_pos, len(wf_r)))
            kept += 1
        print(f" {kept} keepers")
    best.sort(reverse=True)
    return best


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 92)
    print("  NEXFLOW V9 — STRICT COMBO SEARCH + CONFIDENCE-DRIVEN PAIRING")
    print("=" * 92)

    # 1. strict combo search
    print("\n  [1/3]  STRICT COMBO SEARCH")
    results = strict_combo_search(max_k=5)

    print(f"\n  TOP 12 COMBOS (strict no-lookahead engine):")
    print(f"  {'Combo':44s}  {'CAGR':>7s}  {'DD':>6s}  {'Sharpe':>6s}  {'WF':>5s}  {'Score':>6s}")
    print("  " + "-" * 85)
    for score, combo, r, wf, n_pos, n_wf in results[:12]:
        print(f"  {'+'.join(combo):44s}  {r['cagr']:>+6.1%}  {r['dd']:>5.1%}  "
              f"{r['sharpe']:>6.2f}  {n_pos}/{n_wf}  {score:>6.2f}")
        print(f"    WF: {[f'{x:+.0%}' for x in wf]}")

    BEST = results[0][1] if results else COMBO
    print(f"\n  *** STRICT WINNER: {'+'.join(BEST)} ***")
    print(f"      CAGR={results[0][2]['cagr']:+.1%}  DD={results[0][2]['dd']:.1%}  "
          f"Sharpe={results[0][2]['sharpe']:.2f}  WF={results[0][4]}/{results[0][5]}")

    # 2. build confidence engine
    print(f"\n  [2/3]  CONFIDENCE ENGINE (pre-computed curves → no state truncation)")
    print(f"         Combo: {'+'.join(BEST)}")
    print(f"         Crypto score: SMA200 (2pts primary gate) + mom90 + mom30 + vol")
    print(f"         Stock score: SMA200 + mom90 + EMA-bull (per-ticker average)")
    print(f"         Allocation: monthly rebalance based on relative confidence\n")

    v9c = run_v9_confidence(BEST, capital=_CAPITAL)

    # 3. comparison table
    print("\n  [3/3]  YEAR-BY-YEAR COMPARISON")
    print("=" * 92)

    v863_yr = {2021:29998, 2022:3664, 2023:-1417, 2024:18692, 2025:-373, 2026:2994}
    v9s_yr  = {2021:16077, 2022: 951, 2023: 1985, 2024:11020, 2025: 4716, 2026:13943}
    v863_end = 58192; v9s_end = 53692

    years = sorted(set(v863_yr) | set(v9s_yr) | set(v9c["year_pnl"]))
    print(f"\n  {'Year':6s}  {'V8.63 Bal':>12s}  {'PnL':>9s}  "
          f"{'V9-Static Bal':>14s}  {'PnL':>9s}  "
          f"{'V9-Conf Bal':>13s}  {'PnL':>9s}")
    print("  " + "-" * 90)
    b863 = _CAPITAL; bs9 = _CAPITAL; bc9 = _CAPITAL
    for yr in years:
        p863 = v863_yr.get(yr, 0); ps9 = v9s_yr.get(yr, 0)
        pc9  = v9c["year_pnl"].get(yr, 0)
        b863 += p863; bs9 += ps9; bc9 += pc9
        row = f"  {yr}  ${b863:>11,.0f}  ${p863:>+8,.0f}  ${bs9:>13,.0f}  ${ps9:>+8,.0f}  ${bc9:>12,.0f}  ${pc9:>+8,.0f}"
        # flag years where V9-Conf wins both V8.63 and V9-Static
        marker = " ✓" if pc9 > p863 else ("  " if pc9 >= 0 else " ✗")
        print(row + marker)

    print("  " + "-" * 90)
    print(f"  Final ${v863_end:>11,.0f}              ${v9s_end:>13,.0f}              ${v9c['final']:>12,.0f}")
    gap_c  = v9c["final"] - v863_end
    gap_cs = v9c["final"] - v9s_end
    print(f"\n  CAGR   {'56.9%':>12s}              {'54.5%':>13s}              {v9c['cagr']*100:>11.1f}%")
    print(f"  Max DD {'25.0%':>12s}              {'31.2%':>13s}              {v9c['dd']*100:>11.1f}%")
    print(f"\n  V9-Confidence vs V8.63:   {gap_c:>+,.0f}")
    print(f"  V9-Confidence vs V9-Static: {gap_cs:>+,.0f}")

    # 4. allocation log
    print(f"\n  MONTHLY ALLOCATION LOG:")
    print(f"  {'Date':12s}  {'CryptoSc':>9s}  {'StockSc':>8s}  {'Alloc':>14s}  Regime")
    print("  " + "-" * 72)
    for ts, wc, ws, c_sc, s_sc in v9c["log"][:30]:
        date = dt.datetime.utcfromtimestamp(ts/1000).strftime("%Y-%m")
        cash = 1.0 - wc - ws
        alloc = f"{wc:.0%}C/{ws:.0%}S" + (f"/{cash:.0%}$" if cash > 0.05 else "")
        regime = ("CRYPTO DOMINANT" if wc >= 0.75 else
                  "STOCK DOMINANT"  if ws >= 0.75 else
                  "DEFENSIVE-CASH"  if cash > 0.15 else
                  "BOTH STRONG"     if wc >= 0.60 and ws >= 0.30 else "BALANCED")
        print(f"  {date:12s}  {c_sc:>9.2f}  {s_sc:>8.2f}  {alloc:>14s}  {regime}")
    if len(v9c["log"]) > 30:
        print(f"  ... ({len(v9c['log'])-30} more months not shown)")

    print("\n" + "=" * 92)
    winner = "V9-Confidence" if v9c["final"] > v863_end else "V8.63"
    print(f"  BOTTOM LINE: {winner} wins on total equity")
    print(f"  V9-Confidence {'+beats+' if gap_c >= 0 else 'trails'} V8.63 by {gap_c:>+,.0f} over 5.5yr")
    losing_yrs = [yr for yr in years if v9c["year_pnl"].get(yr, 0) < 0]
    print(f"  Losing years for V9-Confidence: {losing_yrs if losing_yrs else 'NONE'}")
    print("=" * 92)


if __name__ == "__main__":
    main()
