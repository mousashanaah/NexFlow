#!/usr/bin/env python3
"""
TEST 10/11/12 — Breadth Analysis Suite + Edge Attribution

TEST 10: Breadth mechanics — how many coins held vs portfolio return
TEST 11: Breadth sensitivity — cap holdings at 3/5/7/unlimited
TEST 12: Breadth quality — % universe signaling vs market state
ATTR:    Edge attribution — what specific components generate V8.63's advantage
"""
from __future__ import annotations
import random
import datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd

_REPO    = Path(__file__).parent.parent
_CDIR    = _REPO / "data" / "candles"
_SYMS    = [
    "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT",
    "XRPUSDT","ADAUSDT","DOGEUSDT","AVAXUSDT",
    "LINKUSDT","LTCUSDT","DOTUSDT","TRXUSDT",
]
_FROM_MS = int(dt.datetime(2021,1,1,tzinfo=dt.timezone.utc).timestamp()*1000)
_TO_MS   = int(dt.datetime(2026,6,1,tzinfo=dt.timezone.utc).timestamp()*1000)
_CAPITAL = 10_000.0


# ── data & indicators ─────────────────────────────────────────────────────────

def _ema(s, n):
    out = np.full(len(s), np.nan); alpha = 2/(n+1); v = float(s[0])
    for i in range(len(s)):
        v = alpha*float(s[i]) + (1-alpha)*v
        if i >= n-1: out[i] = v
    return out

def _sma(s, n):
    out = np.full(len(s), np.nan)
    for i in range(n-1, len(s)): out[i] = np.mean(s[i-n+1:i+1])
    return out

def _macd_long(s):
    e12=_ema(s,12); e26=_ema(s,26); macd=e12-e26
    vi = np.where(~np.isnan(macd))[0]
    sig = np.full(len(s), np.nan)
    if len(vi)>=9: sig[vi] = _ema(macd[vi], 9)
    hist = macd - sig
    state = np.zeros(len(s), dtype=bool)
    for i in range(1, len(s)):
        hp,hc = hist[i-1], hist[i]
        if np.isnan(hp) or np.isnan(hc): state[i]=state[i-1]
        elif hp<=0<hc: state[i]=True
        elif hp>=0>hc: state[i]=False
        else: state[i]=state[i-1]
    return state

def _build_all():
    data = {}
    for sym in _SYMS:
        p = _CDIR / f"{sym}_1D.parquet"
        if not p.exists(): continue
        df = pd.read_parquet(p, columns=["open_time","close"])
        df = df.rename(columns={"open_time":"ts"})
        df["ts"] = df["ts"].astype(int); df["close"] = df["close"].astype(float)
        df = df[(df.ts>=_FROM_MS)&(df.ts<=_TO_MS)].sort_values("ts").reset_index(drop=True)
        c = df["close"].values; ts = df["ts"].values
        e8=_ema(c,8); e21=_ema(c,21); s200=_sma(c,200)
        ml = _macd_long(c)
        above_e = e8>e21
        ema_long = np.zeros(len(c), dtype=bool)
        for i in range(1,len(c)):
            if above_e[i] and not above_e[i-1]: ema_long[i]=True
            elif not above_e[i] and above_e[i-1]: ema_long[i]=False
            else: ema_long[i]=ema_long[i-1]
        mom20 = np.full(len(c),np.nan); mom60=np.full(len(c),np.nan); mom30=np.full(len(c),np.nan)
        for i in range(20,len(c)): mom20[i]=(c[i]-c[i-20])/c[i-20]
        for i in range(30,len(c)): mom30[i]=(c[i]-c[i-30])/c[i-30]
        for i in range(60,len(c)): mom60[i]=(c[i]-c[i-60])/c[i-60]
        coin_sig = ema_long & ml & (mom20>0)
        above200 = np.where(np.isnan(s200), False, c>s200)
        data[sym] = {"ts":ts,"close":c,"coin_sig":coin_sig,"above200":above200,
                     "mom20":mom20,"mom30":mom30,"mom60":mom60,
                     "by_ts":{int(t):i for i,t in enumerate(ts)}}
    return data

def _btc_regime(data, common_ts):
    btc = data["BTCUSDT"]
    bear=False; streak=0; bull=np.ones(len(common_ts),dtype=bool)
    for j,ts in enumerate(common_ts):
        i = btc["by_ts"].get(ts)
        if i is None: bull[j]=not bear; continue
        above200 = bool(btc["above200"][i])
        mom30 = float(btc["mom30"][i]) if not np.isnan(btc["mom30"][i]) else 0.0
        streak = streak+1 if above200 else 0
        if not bear:
            if (not above200) and mom30<-0.20: bear=True
        else:
            if streak>=10: bear=False
        bull[j] = not bear
    return bull


# ── portfolio simulation ───────────────────────────────────────────────────────

def _daily_ret(data, common_ts):
    N = len(common_ts); K = len(_SYMS)
    ret = np.zeros((K,N))
    for k,sym in enumerate(_SYMS):
        sd = data[sym]
        closes = np.array([sd["close"][sd["by_ts"][ts]] if ts in sd["by_ts"] else np.nan
                           for ts in common_ts])
        for j in range(1,N):
            if not np.isnan(closes[j]) and not np.isnan(closes[j-1]) and closes[j-1]>0:
                ret[k,j] = (closes[j]-closes[j-1])/closes[j-1]
    return ret  # shape (K, N)

def _cagr(eq,nd): return (eq[-1]/eq[0])**(365/nd)-1 if eq[0]>0 and eq[-1]>0 and nd>0 else 0
def _sharpe(eq):
    r=np.diff(eq)/eq[:-1]; r=r[np.isfinite(r)]
    return r.mean()/r.std()*np.sqrt(252) if len(r)>1 and r.std()>0 else 0
def _maxdd(eq):
    pk=np.maximum.accumulate(eq)
    return float(np.max((pk-eq)/np.where(pk>0,pk,1)))
def _yr(eq,dates):
    ys={};ye={}
    for i,d in enumerate(dates):
        yr=d.year
        if yr not in ys: ys[yr]=eq[i]
        ye[yr]=eq[i]
    return {yr:(ye[yr]-ys[yr])/ys[yr] for yr in ys}

def simulate(holdings_fn, ret, N, dates, label=""):
    """holdings_fn(j) -> list of sym indices to hold on day j."""
    eq = np.full(N, _CAPITAL)
    cur = []
    month_starts=set(); seen=set()
    for j,d in enumerate(dates):
        key=(d.year,d.month)
        if key not in seen: seen.add(key); month_starts.add(j)
    for j in range(1,N):
        if j in month_starts or not cur: cur = holdings_fn(j)
        if not cur: eq[j]=eq[j-1]; continue
        r_day = np.mean(ret[cur,j]) if cur else 0.0
        eq[j] = eq[j-1]*(1+r_day)
    return eq


# ── TEST 10 — Breadth Analysis ────────────────────────────────────────────────

def run_10(data, bull, common_ts, ret, dates):
    print("\n"+"="*80)
    print("  TEST 10 — BREADTH MECHANICS")
    print("  Monthly breadth vs portfolio return")
    print("="*80)

    sym_idx = {s:k for k,s in enumerate(_SYMS)}
    month_groups={}
    for j,ts in enumerate(common_ts):
        d=dates[j]; key=(d.year,d.month)
        month_groups.setdefault(key,[]).append(j)

    print(f"\n  {'Month':8s} {'Reg':4s} {'N_held':6s} {'N_qual':6s} {'Breadth%':9s} {'Port Return':11s}")
    print("  "+"-"*55)

    breadth_by_regime = {"BULL":[], "BEAR":[], "TRANS":[]}
    ret_by_breadth = {"high":[], "med":[], "low":[]}  # high>66%, med 33-66%, low<33%
    monthly_data = []

    for (yr,mo) in sorted(month_groups):
        idxs = month_groups[(yr,mo)]
        bull_days = sum(1 for j in idxs if bull[j])
        if bull_days >= len(idxs)*0.7:   regime="BULL"
        elif bull_days <= len(idxs)*0.3: regime="BEAR"
        else:                            regime="TRANS"

        # Count coins held each day
        held_counts=[]
        for j in idxs:
            if not bull[j]: held_counts.append(0); continue
            n = sum(1 for s in _SYMS
                    if data[s]["by_ts"].get(common_ts[j]) is not None
                    and data[s]["coin_sig"][data[s]["by_ts"][common_ts[j]]])
            held_counts.append(n)

        # Coins eligible (above SMA200, in bull)
        qual_counts=[]
        for j in idxs:
            if not bull[j]: qual_counts.append(0); continue
            n = sum(1 for s in _SYMS
                    if data[s]["by_ts"].get(common_ts[j]) is not None
                    and data[s]["above200"][data[s]["by_ts"][common_ts[j]]])
            qual_counts.append(n)

        avg_held = np.mean(held_counts)
        avg_qual = np.mean(qual_counts) if any(q>0 for q in qual_counts) else 0
        breadth_pct = avg_held/12 if avg_held>0 else 0

        # Monthly return (sum of daily equity changes) — approximate
        # Use equal-weight of held coins
        month_rets = []
        for j in idxs[1:]:
            held = [sym_idx[s] for s in _SYMS
                    if data[s]["by_ts"].get(common_ts[j]) is not None
                    and data[s]["coin_sig"][data[s]["by_ts"][common_ts[j]]]
                    and bull[j]]
            if held:
                month_rets.append(np.mean(ret[held,j]))
            else:
                month_rets.append(0.0)
        port_ret = (np.prod([1+r for r in month_rets])-1) if month_rets else 0.0

        breadth_by_regime[regime].append(breadth_pct)
        if breadth_pct>0.50: ret_by_breadth["high"].append(port_ret)
        elif breadth_pct>0.20: ret_by_breadth["med"].append(port_ret)
        else: ret_by_breadth["low"].append(port_ret)

        monthly_data.append((yr,mo,regime,avg_held,avg_qual,breadth_pct,port_ret))
        print(f"  {yr}-{mo:02d}  {regime:4s}  {avg_held:5.1f}  {avg_qual:5.1f}  {breadth_pct:8.0%}   {port_ret:>+9.1%}")

    print("  "+"-"*55)
    print(f"\n  BREADTH BY REGIME:")
    for reg in ["BULL","BEAR","TRANS"]:
        vals = breadth_by_regime[reg]
        print(f"  {reg:5s}: avg {np.mean(vals):.0%} breadth  (n={len(vals)} months)")
    print(f"\n  RETURN BY BREADTH LEVEL:")
    for lev in ["high","med","low"]:
        vals = ret_by_breadth[lev]
        label = ">50% coins held" if lev=="high" else ("20-50%" if lev=="med" else "<20%")
        print(f"  {label}: avg monthly return {np.mean(vals):>+.1%}  (n={len(vals)} months)")

    # Attribution: 2021 specifically
    bull_2021 = [(yr,mo,bpct,pret) for yr,mo,reg,nh,nq,bpct,pret in monthly_data if yr==2021]
    print(f"\n  2021 BREAKDOWN (source of the massive bull run):")
    for yr,mo,bpct,pret in bull_2021:
        print(f"    {yr}-{mo:02d}: breadth={bpct:.0%}  return={pret:>+.1%}")


# ── TEST 11 — Breadth Sensitivity ────────────────────────────────────────────

def run_11(data, bull, common_ts, ret, dates, nd):
    print("\n"+"="*80)
    print("  TEST 11 — BREADTH SENSITIVITY")
    print("  Performance when holdings capped at 3 / 5 / 7 / unlimited")
    print("  (When >N coins qualify, randomly sample N — averaged 200 seeds)")
    print("="*80)

    sym_idx = {s:k for k,s in enumerate(_SYMS)}
    N = len(common_ts)

    def get_qualified(j):
        if not bull[j]: return []
        return [sym_idx[s] for s in _SYMS
                if data[s]["by_ts"].get(common_ts[j]) is not None
                and data[s]["coin_sig"][data[s]["by_ts"][common_ts[j]]]]

    # Unlimited (current V8.63 signal breadth)
    eq_unlim = simulate(get_qualified, ret, N, dates)

    results = {"unlimited": {"eq": eq_unlim}}

    for cap in [3, 5, 7]:
        all_eq = np.zeros(N)
        for seed in range(200):
            rng = random.Random(seed)
            def cap_fn(j, _cap=cap, _rng=rng):
                q = get_qualified(j)
                if len(q) <= _cap: return q
                return _rng.sample(q, _cap)
            eq = simulate(cap_fn, ret, N, dates)
            all_eq += eq
        results[f"cap{cap}"] = {"eq": all_eq/200}

    print(f"\n  {'Version':15s}  {'CAGR':>8s}  {'Sharpe':>7s}  {'Max DD':>7s}  {'Final $':>12s}")
    print("  "+"-"*60)

    labels = {"cap3":"Cap-3 (avg)","cap5":"Cap-5 (avg)","cap7":"Cap-7 (avg)","unlimited":"Unlimited"}
    for key in ["cap3","cap5","cap7","unlimited"]:
        eq = results[key]["eq"]
        print(f"  {labels[key]:15s}  {_cagr(eq,nd):>7.1%}  {_sharpe(eq):>7.2f}"
              f"  {_maxdd(eq):>7.1%}  ${eq[-1]:>11,.0f}")

    print(f"\n  Year-by-year CAGR:")
    print(f"  {'Year':6s}  {'Cap-3':>8s}  {'Cap-5':>8s}  {'Cap-7':>8s}  {'Unlimited':>9s}")
    print("  "+"-"*48)
    for yr in range(2021,2027):
        row = []
        for key in ["cap3","cap5","cap7","unlimited"]:
            yr_map = _yr(results[key]["eq"], dates)
            row.append(yr_map.get(yr,0))
        print(f"  {yr}    {row[0]:>7.1%}  {row[1]:>7.1%}  {row[2]:>7.1%}  {row[3]:>8.1%}")

    # Conclusion
    c_unlim = _cagr(results["unlimited"]["eq"], nd)
    c_cap3  = _cagr(results["cap3"]["eq"], nd)
    gap = c_unlim - c_cap3
    print(f"\n  BREADTH SENSITIVITY GAP (unlimited vs cap-3): {gap:>+.1%} CAGR")
    if gap > 0.10:
        print("  → STRONG evidence that breadth is a core component of the edge.")
        print("    Restricting to 3 coins materially degrades performance.")
    elif gap > 0.03:
        print("  → MODERATE evidence. Some breadth benefit, not dominant.")
    else:
        print("  → WEAK evidence. Breadth may not be the primary driver.")


# ── TEST 12 — Breadth Quality vs Quantity ────────────────────────────────────

def run_12(data, bull, common_ts, dates):
    print("\n"+"="*80)
    print("  TEST 12 — BREADTH QUALITY vs QUANTITY")
    print("  Is V8.63 acting as a market breadth indicator?")
    print("="*80)

    month_groups={}
    for j,ts in enumerate(common_ts):
        d=dates[j]; key=(d.year,d.month)
        month_groups.setdefault(key,[]).append(j)

    print(f"\n  {'Month':8s}  {'%>SMA200':9s}  {'%Signaling':11s}  {'Regime':6s}  Relationship")
    print("  "+"-"*65)

    sma200_pcts = []; sig_pcts = []; regimes_month = []

    for (yr,mo) in sorted(month_groups):
        idxs = month_groups[(yr,mo)]
        bull_days = sum(1 for j in idxs if bull[j])
        if bull_days >= len(idxs)*0.7:   reg="BULL"
        elif bull_days <= len(idxs)*0.3: reg="BEAR"
        else:                            reg="TRANS"

        daily_sma=[]; daily_sig=[]
        for j in idxs:
            ts=common_ts[j]
            n200 = sum(1 for s in _SYMS if data[s]["by_ts"].get(ts) is not None
                       and data[s]["above200"][data[s]["by_ts"][ts]])
            nsig = sum(1 for s in _SYMS if data[s]["by_ts"].get(ts) is not None
                       and data[s]["coin_sig"][data[s]["by_ts"][ts]])
            daily_sma.append(n200/12); daily_sig.append(nsig/12)

        p200=np.mean(daily_sma); psig=np.mean(daily_sig)
        sma200_pcts.append(p200); sig_pcts.append(psig); regimes_month.append(reg)

        note = "FULLY DEPLOYED" if psig>0.6 else ("DEFENSIVE" if psig<0.15 else "PARTIAL")
        print(f"  {yr}-{mo:02d}    {p200:>7.0%}  {psig:>10.0%}    {reg:6s}  {note}")

    print("  "+"-"*65)
    print(f"\n  MARKET BREADTH STATES:")
    for reg in ["BULL","BEAR","TRANS"]:
        mask=[i for i,r in enumerate(regimes_month) if r==reg]
        if not mask: continue
        avg200=np.mean([sma200_pcts[i] for i in mask])
        avgsig=np.mean([sig_pcts[i] for i in mask])
        print(f"  {reg:5s}: {avg200:.0%} of coins above SMA200  |  {avgsig:.0%} generating V8.63 signal")

    # Correlation test
    from scipy.stats import spearmanr
    corr, pval = spearmanr(sma200_pcts, sig_pcts)
    print(f"\n  Correlation (SMA200 breadth vs Signal breadth): r={corr:.3f}  p={pval:.4f}")
    if corr>0.8:
        print("  → V8.63 coin signals are a PROXY for SMA200 breadth.")
        print("    The EMA/MACD filter tracks market-wide participation health.")
    elif corr>0.5:
        print("  → Moderate correlation. Signal breadth adds some information beyond SMA200.")
    else:
        print("  → Low correlation. V8.63 signals capture something distinct from SMA200 breadth.")


# ── EDGE ATTRIBUTION ──────────────────────────────────────────────────────────

def run_attribution(data, bull, common_ts, ret, dates, nd):
    print("\n"+"="*80)
    print("  EDGE ATTRIBUTION")
    print("  Decomposing V8.63's outperformance component by component")
    print("="*80)

    sym_idx = {s:k for k,s in enumerate(_SYMS)}
    N = len(common_ts)

    # 1. Baseline: hold all 12 coins always (no regime, no signals)
    def all_always(j): return list(range(12))
    eq_all = simulate(all_always, ret, N, dates)

    # 2. BTC regime only: hold all 12 when BULL, cash when BEAR
    def regime_all(j): return list(range(12)) if bull[j] else []
    eq_reg = simulate(regime_all, ret, N, dates)

    # 3. BTC regime + SMA200 gate (hold all coins above SMA200 in bull)
    def regime_sma(j):
        if not bull[j]: return []
        return [sym_idx[s] for s in _SYMS
                if data[s]["by_ts"].get(common_ts[j]) is not None
                and data[s]["above200"][data[s]["by_ts"][common_ts[j]]]]
    eq_rsma = simulate(regime_sma, ret, N, dates)

    # 4. Full V8.63 (regime + SMA200 + EMA + MACD + mom20 — same breadth)
    def v863_full(j):
        if not bull[j]: return []
        return [sym_idx[s] for s in _SYMS
                if data[s]["by_ts"].get(common_ts[j]) is not None
                and data[s]["coin_sig"][data[s]["by_ts"][common_ts[j]]]]
    eq_v863 = simulate(v863_full, ret, N, dates)

    # 5. Random 3 with BTC regime (matched to TEST 00C Version B)
    all_rand = np.zeros(N)
    for seed in range(300):
        rng=random.Random(seed)
        def rand3(j,r=rng):
            if not bull[j]: return []
            pool=[sym_idx[s] for s in _SYMS
                  if data[s]["by_ts"].get(common_ts[j]) is not None
                  and data[s]["above200"][data[s]["by_ts"][common_ts[j]]]]
            return r.sample(pool,3) if len(pool)>=3 else pool
        all_rand += simulate(rand3, ret, N, dates)
    eq_rand3 = all_rand/300

    # 6. V8.63 signals but same breadth as random-3 (hold random 3 from qualified set)
    all_v863_cap3 = np.zeros(N)
    for seed in range(300):
        rng=random.Random(seed)
        def v863_cap3(j,r=rng):
            if not bull[j]: return []
            q=[sym_idx[s] for s in _SYMS
               if data[s]["by_ts"].get(common_ts[j]) is not None
               and data[s]["coin_sig"][data[s]["by_ts"][common_ts[j]]]]
            return r.sample(q,3) if len(q)>=3 else q
        all_v863_cap3 += simulate(v863_cap3, ret, N, dates)
    eq_v863c3 = all_v863_cap3/300

    configs = [
        ("1. Hold all always",   eq_all,  "No regime, no signals. Pure asset class return."),
        ("2. + BTC regime",      eq_reg,  "Add: go cash when BTC in bear. Timing only."),
        ("3. + SMA200 gate",     eq_rsma, "Add: only hold coins above SMA200."),
        ("4. + EMA/MACD/Mom",    eq_v863, "Add: V8.63 coin signals. Full breadth system."),
        ("5. Random-3 + regime", eq_rand3,"Regime + SMA200 gate, random 3 coins (300 seeds avg)."),
        ("6. V8.63 signals cap3",eq_v863c3,"V8.63 coins but capped at random-3 breadth (300 seeds avg)."),
    ]

    print(f"\n  {'Version':25s}  {'CAGR':>8s}  {'Sharpe':>7s}  {'Max DD':>7s}  {'Final $':>12s}")
    print("  "+"-"*70)
    for label,eq,note in configs:
        print(f"  {label:25s}  {_cagr(eq,nd):>7.1%}  {_sharpe(eq):>7.2f}"
              f"  {_maxdd(eq):>7.1%}  ${eq[-1]:>11,.0f}")

    print(f"\n  COMPONENT CONTRIBUTIONS (CAGR increments):")
    cagrs = [_cagr(eq,nd) for _,eq,_ in configs]
    c_all,c_reg,c_rsma,c_v863,c_rand3,c_v863c3 = cagrs
    print(f"  BTC regime timing alone:      {c_reg-c_all:>+.1%}  (version 2 minus 1)")
    print(f"  SMA200 coin filter:           {c_rsma-c_reg:>+.1%}  (version 3 minus 2)")
    print(f"  EMA/MACD/Mom breadth filter:  {c_v863-c_rsma:>+.1%}  (version 4 minus 3)")
    print(f"  ---")
    print(f"  Breadth vs concentration:     {c_v863-c_rand3:>+.1%}  (full V8.63 minus random-3)")
    print(f"  Signal quality vs random:     {c_v863c3-c_rand3:>+.1%}  (V8.63-cap3 minus random-3, same breadth)")
    print(f"  Breadth quantity effect:      {c_v863-c_v863c3:>+.1%}  (uncapped minus capped at 3, same signals)")

    print(f"\n  INTERPRETATION:")
    if c_reg-c_all > 0.10:
        print(f"  BTC timing is a major alpha source ({c_reg-c_all:+.1%} CAGR).")
    if c_v863-c_rsma > 0.05:
        print(f"  EMA/MACD breadth filter adds meaningful alpha ({c_v863-c_rsma:+.1%} CAGR) beyond SMA200 alone.")
    if c_v863-c_v863c3 > 0.05:
        print(f"  Breadth quantity (holding more coins) contributes {c_v863-c_v863c3:+.1%} CAGR.")
        print(f"  This is the BREADTH PREMIUM — V8.63's mechanism of maximising participation.")
    if c_v863c3-c_rand3 > 0.03:
        print(f"  EMA/MACD signals select better coins than random within same breadth ({c_v863c3-c_rand3:+.1%} CAGR).")
    else:
        print(f"  EMA/MACD signal quality vs random (same breadth): {c_v863c3-c_rand3:+.1%} — negligible.")
        print(f"  When breadth is equalized, coin-selection quality is not the driver.")


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("="*80)
    print("  TEST 10/11/12 + ATTRIBUTION — BREADTH ANALYSIS SUITE")
    print(f"  Universe: {len(_SYMS)} coins | 2021-01 → 2026-06")
    print("="*80)

    print("\n  Loading data...")
    data = _build_all()
    missing = [s for s in _SYMS if s not in data]
    if missing: print(f"  WARNING: missing {missing}")

    btc_ts = sorted(data["BTCUSDT"]["ts"].tolist())
    common_ts = [ts for ts in btc_ts if _FROM_MS<=ts<=_TO_MS]
    N = len(common_ts)
    dates = pd.to_datetime(common_ts, unit="ms", utc=True)
    nd = int((common_ts[-1]-common_ts[0])/86_400_000)

    print(f"  Grid: {N} bars | {nd} calendar days")
    bull = _btc_regime(data, common_ts)
    print(f"  BTC regime: BULL {bull.mean():.1%}  BEAR {1-bull.mean():.1%}")

    print("\n  Pre-computing daily returns...")
    ret = _daily_ret(data, common_ts)

    run_10(data, bull, common_ts, ret, dates)
    run_11(data, bull, common_ts, ret, dates, nd)
    run_12(data, bull, common_ts, dates)
    run_attribution(data, bull, common_ts, ret, dates, nd)

    print("\n"+"="*80)
    print("  BREADTH ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
