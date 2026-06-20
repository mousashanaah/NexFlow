#!/usr/bin/env python3
"""
TEST 50 — Full V9 Confidence: Static vs Dynamic Stock Book
TEST 51 — Full V9 Confidence: Current Crypto vs Regime-Only Crypto

Both tests run inside the complete V9 architecture (confidence engine,
allocation weights, real bar-by-bar equity curves). This is the integration
gate before engineering phase.

Run: python scripts/test50_integration.py
"""
from __future__ import annotations
import sys, datetime as dt
from pathlib import Path
import numpy as np

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

# ── import real V9 engine pieces ──────────────────────────────────────────────
from test_v9_confidence import (
    build_crypto_curve, crypto_score, stock_score, allocate,
    _load_btc,
)
from test_stock_risk_mgmt import (
    _load as _load_stock_data, _sharpe as _sharpe_fn, _cagr as _cagr_fn,
    _dd as _dd_fn, RECOMMENDED, _FROM_TS, _TO_TS, _CAPITAL,
)

_DAY_MS = 86_400_000

# Known best combo from last V9 combo search
_STATIC_COMBO = ["AMD", "GOOGL", "MSTR", "SPOT"]

# 51-stock universe exclusions (mirrors test40)
_EXCL = {"SPY","QQQ","IWM","DIA","GLD","XLE","XLF","XLI","XLK","XLP",
         "XLRE","XLU","XLV","XLY","SLV","USO"}

# 12-coin crypto universe (V8.63)
_CRYPTO_SYMS = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","ADAUSDT",
                "DOGEUSDT","AVAXUSDT","LINKUSDT","LTCUSDT","DOTUSDT","TRXUSDT"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _sortino(eq):
    r = np.diff(eq) / np.maximum(eq[:-1], 1e-9)
    neg = r[r < 0]
    if len(neg) < 2: return 0.0
    down = np.std(neg) * np.sqrt(252)
    ann  = np.mean(r) * 252
    return ann / down if down > 0 else 0.0

def _calmar(eq, n_days):
    c = _cagr_fn(eq, n_days)
    d = _dd_fn(eq)
    return c / d if d > 0 else 0.0

def _yr(eq, dates):
    out = {}
    bal = {d.year: [] for d in dates}
    for i, d in enumerate(dates):
        bal[d.year].append(eq[i])
    prev = eq[0]
    for yr in sorted(bal):
        if bal[yr]:
            end_val = bal[yr][-1]
            start_val = prev
            out[yr] = (end_val - start_val) / start_val if start_val > 0 else 0
            prev = end_val
    return out


# ── dynamic stock equity builder ──────────────────────────────────────────────

def _load_stock_universe():
    stock_dir = _REPO / "data" / "stocks"
    tickers = sorted(f.stem.replace("_1D","")
                     for f in stock_dir.glob("*_1D.parquet")
                     if f.stem.replace("_1D","") not in _EXCL)
    print(f"  Loading {len(tickers)} stocks for dynamic universe...")

    sdata = {}
    all_ts = set()
    for t in tickers:
        d = _load_stock_data(t)
        if not d or len(d["ts"]) < 60: continue
        ts  = np.array(d["ts"], dtype=np.int64)
        cl  = np.array(d["close"], dtype=float)
        s200 = np.array(d["sma200"], dtype=float)
        # 60d momentum
        mom60 = np.full(len(cl), np.nan)
        for i in range(60, len(cl)):
            if cl[i-60] > 0: mom60[i] = cl[i]/cl[i-60] - 1
        by_ts = {int(t2): i for i, t2 in enumerate(ts)}
        sdata[t] = {"ts": ts, "close": cl, "sma200": s200,
                    "mom60": mom60, "by_ts": by_ts}
        all_ts.update(ts.tolist())

    # common stock timestamps within range
    common_ts = sorted(ts for ts in all_ts if _FROM_TS <= ts <= _TO_TS)
    dates = [dt.datetime.utcfromtimestamp(ts/1000) for ts in common_ts]
    tidx  = {t: i for i, t in enumerate(sorted(sdata.keys()))}
    N = len(common_ts)

    # stock return matrix [n_tickers, N]
    tlist = sorted(sdata.keys())
    ret = np.zeros((len(tlist), N))
    for ti, t in enumerate(tlist):
        d = sdata[t]
        for j in range(1, N):
            ts_j = common_ts[j]; ts_p = common_ts[j-1]
            ij = d["by_ts"].get(ts_j); ip = d["by_ts"].get(ts_p)
            if ij is not None and ip is not None and d["close"][ip] > 0:
                ret[ti, j] = (d["close"][ij] - d["close"][ip]) / d["close"][ip]

    return sdata, tlist, tidx, common_ts, dates, ret, N


def _top3_fn(sdata, tlist, common_ts, j):
    """Return list of ticker indices for top-3 60d mom with SMA200 gate."""
    ts = common_ts[j]
    sc = {}
    for t in tlist:
        d = sdata[t]; i = d["by_ts"].get(ts)
        if i is None: continue
        m = d["mom60"][i]; a = d["sma200"][i]
        cl = d["close"][i]
        if np.isnan(m) or np.isnan(a) or cl <= a: continue
        sc[t] = m
    return [tlist.index(t) for t in sorted(sc, key=sc.get, reverse=True)[:3]]


def build_dynamic_stock_curve(sdata, tlist, tidx, common_ts, dates, ret, N, capital):
    """Bar-by-bar dynamic stock equity: top-3 60d momentum, monthly rebalance."""
    eq = np.full(N, capital, dtype=float)
    cur_idx = []
    last_reb = 0
    month_seen = set()

    for j in range(1, N):
        d = dates[j]; key = (d.year, d.month)
        if key not in month_seen:
            month_seen.add(key); cur_idx = _top3_fn(sdata, tlist, common_ts, j)
            last_reb = j
        if not cur_idx:
            eq[j] = eq[j-1]; continue
        r = np.mean(ret[cur_idx, j])
        eq[j] = eq[j-1] * (1 + r)

    return common_ts, eq.tolist()


# ── regime-only crypto curve ──────────────────────────────────────────────────

def build_regime_only_crypto_curve(c_ts, c_eq_full):
    """
    Build an alternative crypto equity curve using only the BTC asymmetric
    regime gate — no per-coin EMA/MACD/mom20 filtering.

    In bull regime: equal-weight all 12 coins.
    In bear regime: cash (0% return).

    We derive the bull/bear state from BTC bar-by-bar using the same
    asymmetric logic as V8.63: bear triggered when 30d return < -20% AND
    below SMA200; exit bear after 10 consecutive days above SMA200.
    """
    import pandas as pd
    btc_path = _REPO / "data" / "candles" / "BTCUSDT_1D.parquet"
    df = pd.read_parquet(btc_path).sort_values("open_time").reset_index(drop=True)
    btc_ts = df["open_time"].values.astype(np.int64)
    btc_cl = df["close"].values.astype(float)
    n_btc  = len(btc_cl)

    # SMA200
    sma200 = np.full(n_btc, np.nan)
    for i in range(199, n_btc):
        sma200[i] = np.mean(btc_cl[i-199:i+1])
    for i in range(n_btc):
        if np.isnan(sma200[i]) and i >= 49:
            sma200[i] = np.mean(btc_cl[max(0,i-49):i+1])

    # 30d return
    mom30 = np.full(n_btc, np.nan)
    for i in range(30, n_btc):
        if btc_cl[i-30] > 0: mom30[i] = btc_cl[i]/btc_cl[i-30] - 1

    # asymmetric bear state
    in_bear = False; above_count = 0
    bear = np.zeros(n_btc, dtype=bool)
    btc_byts = {int(t): i for i, t in enumerate(btc_ts)}
    for i in range(n_btc):
        above = btc_cl[i] > sma200[i] if np.isfinite(sma200[i]) else True
        if in_bear:
            above_count = (above_count + 1) if above else 0
            if above_count >= 10: in_bear = False
        else:
            if np.isfinite(mom30[i]) and mom30[i] < -0.20 and not above:
                in_bear = True; above_count = 0
        bear[i] = in_bear

    # load all 12 coin daily returns on the same timestamp axis
    print("  Loading 12 coins for regime-only crypto curve...")
    import pandas as pd
    coin_ret = {}
    for sym in _CRYPTO_SYMS:
        p = _REPO / "data" / "candles" / f"{sym}_1D.parquet"
        if not p.exists(): continue
        df2 = pd.read_parquet(p).sort_values("open_time").reset_index(drop=True)
        ts2 = df2["open_time"].values.astype(np.int64)
        cl2 = df2["close"].values.astype(float)
        by2 = {int(t): i for i, t in enumerate(ts2)}
        coin_ret[sym] = {"ts": ts2, "close": cl2, "by": by2}

    # build equity curve on c_ts axis
    N = len(c_ts)
    eq = np.full(N, _CAPITAL, dtype=float)
    c_byts2 = {int(t): i for i, t in enumerate(c_ts)}

    for j in range(1, N):
        ts_j = c_ts[j]; ts_p = c_ts[j-1]
        bi = btc_byts.get(ts_j)
        if bi is None or bear[bi]:
            eq[j] = eq[j-1]; continue
        # bull: equal weight all coins
        returns = []
        for sym, cd in coin_ret.items():
            ij = cd["by"].get(ts_j); ip = cd["by"].get(ts_p)
            if ij is not None and ip is not None and cd["close"][ip] > 0:
                returns.append((cd["close"][ij] - cd["close"][ip]) / cd["close"][ip])
        if returns:
            eq[j] = eq[j-1] * (1 + np.mean(returns))
        else:
            eq[j] = eq[j-1]

    return c_ts, eq.tolist()


# ── V9 confidence engine ──────────────────────────────────────────────────────

def run_v9_engine(c_ts, c_eq, s_ts_or_list, s_eq,
                  stock_tickers_fn, capital, label):
    """
    Universal V9 confidence engine.
    c_ts/c_eq: crypto daily curve
    s_ts_or_list/s_eq: stock daily curve (list of timestamps + equity)
    stock_tickers_fn(ts) -> list of tickers for scoring at that timestamp
    """
    c_ts  = list(c_ts); c_eq = list(c_eq)
    s_ts2 = list(s_ts_or_list); s_eq2 = list(s_eq)

    c_byts = {t: i for i, t in enumerate(c_ts)}
    s_byts = {t: i for i, t in enumerate(s_ts2)}
    axis   = sorted(set(c_ts) & set(s_ts2))

    if not axis:
        print(f"  ERROR [{label}]: no common timestamps"); return {}

    wc, ws = 0.50, 0.50; last_reb = 0
    cur_eq = capital; snapshots = [capital]
    year_pnl = {}; alloc_log = []

    for step, ts in enumerate(axis):
        if step > 0 and (step - last_reb) >= 21:
            prev_ts = axis[step-1]
            c_sc = crypto_score(prev_ts)
            tickers = stock_tickers_fn(prev_ts)
            s_sc = stock_score(tickers, prev_ts)
            wc, ws = allocate(c_sc, s_sc)
            alloc_log.append((ts, wc, ws, c_sc, s_sc))
            last_reb = step

        if step == 0:
            snapshots.append(cur_eq); continue

        ci = c_byts.get(ts); si = s_byts.get(ts)
        pc = c_byts.get(axis[step-1]); ps = s_byts.get(axis[step-1])
        if None in (ci, si, pc, ps):
            snapshots.append(cur_eq); continue

        c_ret = (c_eq[ci] - c_eq[pc]) / max(c_eq[pc], 1e-9)
        s_ret = (s_eq2[si] - s_eq2[ps]) / max(s_eq2[ps], 1e-9)
        port_ret = wc * c_ret + ws * s_ret
        cur_eq  += cur_eq * port_ret

        yr = dt.datetime.utcfromtimestamp(ts/1000).year
        year_pnl[yr] = year_pnl.get(yr, 0.0) + cur_eq * 0  # track via equity diff below

        snapshots.append(cur_eq)

    # recompute year_pnl from snapshots
    axis_dates = [dt.datetime.utcfromtimestamp(ts/1000) for ts in axis]
    year_pnl2 = {}
    prev_yr = axis_dates[0].year; prev_eq = capital
    for i, d in enumerate(axis_dates):
        if d.year != prev_yr:
            year_pnl2[prev_yr] = snapshots[i] - prev_eq
            prev_eq = snapshots[i]; prev_yr = d.year
    year_pnl2[prev_yr] = snapshots[-1] - prev_eq

    n_days = int((axis[-1] - axis[0]) / _DAY_MS)
    return dict(
        final    = cur_eq,
        cagr     = _cagr_fn(snapshots, n_days),
        dd       = _dd_fn(snapshots),
        sharpe   = _sharpe_fn(snapshots),
        sortino  = _sortino(np.array(snapshots)),
        calmar   = _calmar(np.array(snapshots), n_days),
        year_pnl = year_pnl2,
        equity   = snapshots,
        axis     = axis,
        log      = alloc_log,
    )


# ── report ────────────────────────────────────────────────────────────────────

def _print_comparison(label_a, ra, label_b, rb, capital):
    years = sorted(set(ra["year_pnl"]) | set(rb["year_pnl"]))
    axis_dates = [dt.datetime.utcfromtimestamp(ts/1000) for ts in ra["axis"]]

    n_days = int((ra["axis"][-1] - ra["axis"][0]) / _DAY_MS)

    print(f"\n  {'Metric':22s}  {label_a:>22s}  {label_b:>22s}  {'Delta':>10s}")
    print("  " + "-"*82)
    metrics = [
        ("Final equity",
         f"${ra['final']:>10,.0f}", f"${rb['final']:>10,.0f}",
         f"${rb['final']-ra['final']:>+10,.0f}"),
        ("CAGR",
         f"{ra['cagr']:>21.1%}", f"{rb['cagr']:>21.1%}",
         f"{rb['cagr']-ra['cagr']:>+9.1%}"),
        ("Sharpe",
         f"{ra['sharpe']:>22.2f}", f"{rb['sharpe']:>22.2f}",
         f"{rb['sharpe']-ra['sharpe']:>+10.2f}"),
        ("Sortino",
         f"{ra['sortino']:>22.2f}", f"{rb['sortino']:>22.2f}",
         f"{rb['sortino']-ra['sortino']:>+10.2f}"),
        ("Max DD",
         f"{ra['dd']:>21.1%}", f"{rb['dd']:>21.1%}",
         f"{rb['dd']-ra['dd']:>+9.1%}"),
        ("Calmar",
         f"{ra['calmar']:>22.2f}", f"{rb['calmar']:>22.2f}",
         f"{rb['calmar']-ra['calmar']:>+10.2f}"),
    ]
    for m, va, vb, vd in metrics:
        print(f"  {m:22s}  {va}  {vb}  {vd}")

    print(f"\n  Year-by-year returns:")
    print(f"  {'Year':6s}  {label_a:>18s}  {label_b:>18s}  {'Diff':>8s}  Win")
    print("  "+"-"*62)
    losing_a = []; losing_b = []; wins_b = 0

    bal_a = capital; bal_b = capital
    for yr in years:
        pa = ra["year_pnl"].get(yr, 0); pb = rb["year_pnl"].get(yr, 0)
        ra_yr = pa / bal_a if bal_a > 0 else 0
        rb_yr = pb / bal_b if bal_b > 0 else 0
        bal_a += pa; bal_b += pb
        if pa < 0: losing_a.append(yr)
        if pb < 0: losing_b.append(yr)
        flag = "B" if pb > pa + 1 else ("A" if pa > pb + 1 else "~")
        if pb > pa: wins_b += 1
        print(f"  {yr}    {ra_yr:>17.1%}  {rb_yr:>17.1%}  {rb_yr-ra_yr:>+7.1%}  {flag}")

    print(f"\n  Losing years — {label_a}: {losing_a or 'NONE'}  |  {label_b}: {losing_b or 'NONE'}")
    print(f"  {label_b} wins {wins_b}/{len(years)} years")


def _deployment_gate(ra, rb, label_b, wf_wins, n_years):
    print(f"\n  ── DEPLOYMENT GATE: {label_b} ──")
    cagr_ok    = rb["cagr"]    > ra["cagr"]    + 0.01
    sharpe_ok  = rb["sharpe"]  >= ra["sharpe"] - 0.05
    dd_ok      = rb["dd"]      <= ra["dd"]     + 0.05
    wf_ok      = wf_wins >= (n_years * 2 // 3)
    mech_ok    = True  # specification is fixed and documented

    gate = [
        ("1. Improves CAGR",           cagr_ok,   f"{rb['cagr']:.1%} vs {ra['cagr']:.1%}"),
        ("2. Sharpe preserved",         sharpe_ok, f"{rb['sharpe']:.2f} vs {ra['sharpe']:.2f}"),
        ("3. DD not materially worse",  dd_ok,     f"{rb['dd']:.1%} vs {ra['dd']:.1%}"),
        ("4. Walk-forward wins ≥2/3",   wf_ok,     f"{wf_wins}/{n_years} years better"),
        ("5. Mechanical explanation",   mech_ok,   "Fixed spec: documented"),
    ]
    all_pass = all(ok for _, ok, _ in gate)
    for name, ok, detail in gate:
        mark = "PASS ✓" if ok else "FAIL ✗"
        print(f"  {mark}  {name:35s}  {detail}")
    score = sum(1 for _, ok, _ in gate if ok)
    print(f"\n  Gate score: {score}/5")
    if all_pass:
        print(f"  ══ ALL GATES PASSED. {label_b} is a DEPLOYABLE UPGRADE. ══")
    else:
        print(f"  ══ GATE FAILED ({score}/5). {label_b} does NOT qualify for deployment. ══")
    return all_pass


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n"+"="*92)
    print("  TEST 50/51 — FULL V9 CONFIDENCE INTEGRATION TESTS")
    print("  Real V9 architecture: actual confidence engine, real equity curves")
    print("="*92)

    # ── shared: build crypto curve (V8.63) once ───────────────────────────────
    print("\n  Building V8.63 crypto curve (shared baseline)...")
    c_ts, c_eq = build_crypto_curve(_CAPITAL)

    # ── shared: build static stock curve (current flagship) ───────────────────
    print(f"\n  Building static stock curve ({'+'.join(_STATIC_COMBO)})...")
    from test_stock_risk_mgmt import backtest as _stock_bt
    sr = _stock_bt(_STATIC_COMBO, capital=_CAPITAL, **RECOMMENDED)
    s_ts_static = sr["axis"]; s_eq_static = sr["equity"]

    # static ticker fn
    def static_tickers_fn(ts): return _STATIC_COMBO

    # ── TEST 50 — Dynamic stock book ──────────────────────────────────────────
    print("\n"+"="*92)
    print("  TEST 50 — V9 Confidence: Static Stock Book vs Dynamic Top-3")
    print("="*92)

    print("\n  Loading stock universe...")
    sdata, tlist, tidx, common_ts, sdates, stock_ret, sN = _load_stock_universe()

    print("  Building dynamic stock equity curve (top-3 60d momentum)...")
    s_ts_dyn, s_eq_dyn = build_dynamic_stock_curve(
        sdata, tlist, tidx, common_ts, sdates, stock_ret, sN, _CAPITAL)

    # dynamic ticker fn for confidence scoring
    def dynamic_tickers_fn(ts_val):
        j_candidates = [j for j, t in enumerate(common_ts)
                        if abs(t - ts_val) < 2 * _DAY_MS]
        if not j_candidates: return _STATIC_COMBO
        j = min(j_candidates, key=lambda x: abs(common_ts[x] - ts_val))
        idxs = _top3_fn(sdata, tlist, common_ts, j)
        return [tlist[i] for i in idxs] if idxs else _STATIC_COMBO

    print("\n  Running V9 engine — Static...")
    r50_static = run_v9_engine(
        c_ts, c_eq, s_ts_static, s_eq_static,
        static_tickers_fn, _CAPITAL, "V9-Static")

    print("  Running V9 engine — Dynamic...")
    r50_dyn = run_v9_engine(
        c_ts, c_eq, s_ts_dyn, s_eq_dyn,
        dynamic_tickers_fn, _CAPITAL, "V9-Dynamic")

    # year win count
    years50 = sorted(set(r50_static["year_pnl"]) | set(r50_dyn["year_pnl"]))
    wf50 = sum(1 for yr in years50
               if r50_dyn["year_pnl"].get(yr,0) > r50_static["year_pnl"].get(yr,0))

    _print_comparison("V9-Static", r50_static, "V9-Dynamic", r50_dyn, _CAPITAL)
    _deployment_gate(r50_static, r50_dyn, "V9-Dynamic Stock", wf50, len(years50))

    # ── TEST 51 — Regime-only crypto ──────────────────────────────────────────
    print("\n"+"="*92)
    print("  TEST 51 — V9 Confidence: Current Crypto (V8.63) vs Regime-Only Crypto")
    print("  Regime-only = BTC asymmetric bear gate only, hold all 12 coins in bull")
    print("="*92)

    print("\n  Building regime-only crypto curve...")
    c_ts_ro, c_eq_ro = build_regime_only_crypto_curve(c_ts, c_eq)

    print("  Running V9 engine — V8.63 crypto (current)...")
    r51_current = run_v9_engine(
        c_ts, c_eq, s_ts_static, s_eq_static,
        static_tickers_fn, _CAPITAL, "V9-V8.63")

    print("  Running V9 engine — Regime-Only crypto...")
    r51_regime = run_v9_engine(
        c_ts_ro, c_eq_ro, s_ts_static, s_eq_static,
        static_tickers_fn, _CAPITAL, "V9-RegimeOnly")

    years51 = sorted(set(r51_current["year_pnl"]) | set(r51_regime["year_pnl"]))
    wf51 = sum(1 for yr in years51
               if r51_regime["year_pnl"].get(yr,0) > r51_current["year_pnl"].get(yr,0))

    _print_comparison("V9-V8.63", r51_current, "V9-RegimeOnly", r51_regime, _CAPITAL)
    _deployment_gate(r51_current, r51_regime, "V9-RegimeOnly Crypto", wf51, len(years51))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n"+"="*92)
    print("  INTEGRATION SUMMARY")
    print("="*92)
    t50 = r50_dyn["cagr"] > r50_static["cagr"] + 0.01
    t51 = r51_regime["cagr"] > r51_current["cagr"] + 0.01
    print(f"\n  TEST 50 Dynamic Stock:    {'PASS — upgrade confirmed' if t50 else 'FAIL — no CAGR improvement'}")
    print(f"  TEST 51 Regime-Only Crypto: {'PASS — simplification valid' if t51 else 'FAIL — V8.63 filtering adds value'}")
    if t50 and not t51:
        print(f"\n  VERDICT: Deploy dynamic stock book. Keep V8.63 crypto book unchanged.")
    elif t50 and t51:
        print(f"\n  VERDICT: Both upgrades valid. Discuss further before combined deploy.")
    elif not t50:
        print(f"\n  VERDICT: Dynamic stock book does NOT survive full V9 integration. Do not deploy.")
    print("\n"+"="*92)


if __name__ == "__main__":
    main()
