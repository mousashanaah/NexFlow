#!/usr/bin/env python3
"""
TEST 30 — Signal Edge Isolation

Decomposes the +33.4% CAGR advantage of V8.63 signals over random selection.
Tests each signal component in isolation to find which carries the edge.

Variants (all share BTC asymmetric regime gate):
  A: BTC regime only — hold ALL coins in bull, cash in bear
  B: BTC regime + EMA only (EMA8 > EMA21 crossover state)
  C: BTC regime + MACD only (MACD histogram positive state)
  D: BTC regime + 20d momentum only (20d return > 0)
  E: Full V8.63 — EMA + MACD + mom20 (all three required)
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


def _ema(s, n):
    out=np.full(len(s),np.nan); a=2/(n+1); v=float(s[0])
    for i in range(len(s)):
        v=a*float(s[i])+(1-a)*v
        if i>=n-1: out[i]=v
    return out

def _sma(s, n):
    out=np.full(len(s),np.nan)
    for i in range(n-1,len(s)): out[i]=np.mean(s[i-n+1:i+1])
    return out

def _macd_state(s):
    e12=_ema(s,12); e26=_ema(s,26); macd=e12-e26
    vi=np.where(~np.isnan(macd))[0]
    sig=np.full(len(s),np.nan)
    if len(vi)>=9: sig[vi]=_ema(macd[vi],9)
    hist=macd-sig; state=np.zeros(len(s),dtype=bool)
    for i in range(1,len(s)):
        hp,hc=hist[i-1],hist[i]
        if np.isnan(hp) or np.isnan(hc): state[i]=state[i-1]
        elif hp<=0<hc:  state[i]=True
        elif hp>=0>hc:  state[i]=False
        else:           state[i]=state[i-1]
    return state

def _ema_state(s):
    e8=_ema(s,8); e21=_ema(s,21); above=e8>e21
    state=np.zeros(len(s),dtype=bool)
    for i in range(1,len(s)):
        if above[i] and not above[i-1]:   state[i]=True
        elif not above[i] and above[i-1]: state[i]=False
        else:                              state[i]=state[i-1]
    return state

def _build():
    data={}
    for sym in _SYMS:
        p=_CDIR/f"{sym}_1D.parquet"
        if not p.exists(): continue
        df=pd.read_parquet(p,columns=["open_time","close"])
        df=df.rename(columns={"open_time":"ts"})
        df["ts"]=df["ts"].astype(int); df["close"]=df["close"].astype(float)
        df=df[(df.ts>=_FROM_MS)&(df.ts<=_TO_MS)].sort_values("ts").reset_index(drop=True)
        c=df["close"].values; ts=df["ts"].values
        s200=_sma(c,200)
        mom20=np.full(len(c),np.nan); mom30=np.full(len(c),np.nan)
        for i in range(20,len(c)): mom20[i]=(c[i]-c[i-20])/c[i-20]
        for i in range(30,len(c)): mom30[i]=(c[i]-c[i-30])/c[i-30]
        ema_long=_ema_state(c); macd_long=_macd_state(c)
        above200=np.where(np.isnan(s200),False,c>s200)
        data[sym]={"ts":ts,"close":c,"ema_long":ema_long,"macd_long":macd_long,
                   "mom20":mom20,"mom30":mom30,"above200":above200,
                   "by_ts":{int(t):i for i,t in enumerate(ts)}}
    return data

def _btc_regime(data,common_ts):
    btc=data["BTCUSDT"]; bear=False; streak=0
    bull=np.ones(len(common_ts),dtype=bool)
    for j,ts in enumerate(common_ts):
        i=btc["by_ts"].get(ts)
        if i is None: bull[j]=not bear; continue
        above200=bool(btc["above200"][i])
        mom30=float(btc["mom30"][i]) if not np.isnan(btc["mom30"][i]) else 0.0
        streak=streak+1 if above200 else 0
        if not bear:
            if (not above200) and mom30<-0.20: bear=True
        else:
            if streak>=10: bear=False
        bull[j]=not bear
    return bull

def _daily_ret(data,common_ts):
    K=len(_SYMS); N=len(common_ts); ret=np.zeros((K,N))
    for k,sym in enumerate(_SYMS):
        sd=data[sym]
        c=np.array([sd["close"][sd["by_ts"][ts]] if ts in sd["by_ts"] else np.nan
                    for ts in common_ts])
        for j in range(1,N):
            if not np.isnan(c[j]) and not np.isnan(c[j-1]) and c[j-1]>0:
                ret[k,j]=(c[j]-c[j-1])/c[j-1]
    return ret

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

def simulate(hfn, ret, N, dates):
    eq=np.full(N,_CAPITAL); cur=[]
    mstarts=set(); seen=set()
    for j,d in enumerate(dates):
        k=(d.year,d.month)
        if k not in seen: seen.add(k); mstarts.add(j)
    for j in range(1,N):
        if j in mstarts or not cur: cur=hfn(j)
        if not cur: eq[j]=eq[j-1]; continue
        eq[j]=eq[j-1]*(1+np.mean(ret[cur,j]))
    return eq

def main():
    print("="*80)
    print("  TEST 30 — SIGNAL EDGE ISOLATION")
    print("  Which component carries V8.63's +33% edge over random?")
    print("="*80)

    data=_build()
    btc_ts=sorted(data["BTCUSDT"]["ts"].tolist())
    common_ts=[ts for ts in btc_ts if _FROM_MS<=ts<=_TO_MS]
    N=len(common_ts); nd=int((common_ts[-1]-common_ts[0])/86_400_000)
    dates=pd.to_datetime(common_ts,unit="ms",utc=True)
    bull=_btc_regime(data,common_ts)
    ret=_daily_ret(data,common_ts)
    sym_idx={s:k for k,s in enumerate(_SYMS)}
    print(f"  Grid: {N} bars | BTC BULL {bull.mean():.1%}")

    def attr(sym,field,j):
        i=data[sym]["by_ts"].get(common_ts[j]); return bool(data[sym][field][i]) if i is not None else False

    # Variant A: regime only — all coins in bull
    def varA(j):
        if not bull[j]: return []
        return list(range(12))

    # Variant B: regime + EMA only
    def varB(j):
        if not bull[j]: return []
        return [sym_idx[s] for s in _SYMS if attr(s,"ema_long",j)]

    # Variant C: regime + MACD only
    def varC(j):
        if not bull[j]: return []
        return [sym_idx[s] for s in _SYMS if attr(s,"macd_long",j)]

    # Variant D: regime + mom20 > 0 only
    def varD(j):
        if not bull[j]: return []
        return [sym_idx[s] for s in _SYMS
                if data[s]["by_ts"].get(common_ts[j]) is not None
                and data[s]["mom20"][data[s]["by_ts"][common_ts[j]]] > 0]

    # Variant E: full V8.63 (EMA + MACD + mom20)
    def varE(j):
        if not bull[j]: return []
        return [sym_idx[s] for s in _SYMS
                if data[s]["by_ts"].get(common_ts[j]) is not None
                and data[s]["ema_long"][data[s]["by_ts"][common_ts[j]]]
                and data[s]["macd_long"][data[s]["by_ts"][common_ts[j]]]
                and data[s]["mom20"][data[s]["by_ts"][common_ts[j]]] > 0]

    # Baseline: random-3 with regime (from TEST 00C)
    sum_rand=np.zeros(N)
    for seed in range(300):
        rng=random.Random(seed)
        def randFn(j,r=rng):
            if not bull[j]: return []
            pool=list(range(12)); return r.sample(pool,3)
        sum_rand+=simulate(randFn,ret,N,dates)
    eq_rand=sum_rand/300

    print("\n  Running variants...")
    variants = [
        ("A: Regime only",    varA),
        ("B: Regime + EMA",   varB),
        ("C: Regime + MACD",  varC),
        ("D: Regime + Mom20", varD),
        ("E: Full V8.63",     varE),
    ]
    results={}
    for label,fn in variants:
        eq=simulate(fn,ret,N,dates)
        # avg holdings
        avg_held=np.mean([len(fn(j)) for j in range(N)])
        results[label]={"eq":eq,"avg_held":avg_held}

    print(f"\n  {'Variant':22s}  {'CAGR':>8s}  {'Sharpe':>7s}  {'Max DD':>7s}  {'Avg Held':>9s}  {'vs Random':>10s}")
    print("  "+"-"*78)
    c_rand=_cagr(eq_rand,nd)
    for label,r in results.items():
        eq=r["eq"]; c=_cagr(eq,nd)
        print(f"  {label:22s}  {c:>7.1%}  {_sharpe(eq):>7.2f}  {_maxdd(eq):>7.1%}"
              f"  {r['avg_held']:>8.1f}  {c-c_rand:>+9.1%}")
    print(f"  {'Baseline: Random-3':22s}  {c_rand:>7.1%}  {_sharpe(eq_rand):>7.2f}"
          f"  {_maxdd(eq_rand):>7.1%}  {'3.0':>9s}  {'—':>10s}")

    print(f"\n  Year-by-year CAGR:")
    print(f"  {'Year':6s}  {'A:Regime':>9s}  {'B:EMA':>8s}  {'C:MACD':>8s}  {'D:Mom20':>8s}  {'E:Full':>8s}  {'Random':>8s}")
    print("  "+"-"*68)
    for yr in range(2021,2027):
        row=[_yr(results[l]["eq"],dates).get(yr,0) for l,_ in variants]
        rb=_yr(eq_rand,dates).get(yr,0)
        print(f"  {yr}  "+("  ".join(f"{v:>8.1%}" for v in row))+f"  {rb:>7.1%}")

    # Identify the key driver
    print(f"\n  INCREMENTAL CONTRIBUTIONS:")
    cA=_cagr(results["A: Regime only"]["eq"],nd)
    cB=_cagr(results["B: Regime + EMA"]["eq"],nd)
    cC=_cagr(results["C: Regime + MACD"]["eq"],nd)
    cD=_cagr(results["D: Regime + Mom20"]["eq"],nd)
    cE=_cagr(results["E: Full V8.63"]["eq"],nd)
    print(f"  EMA alone vs regime:       {cB-cA:>+.1%}")
    print(f"  MACD alone vs regime:      {cC-cA:>+.1%}")
    print(f"  Mom20 alone vs regime:     {cD-cA:>+.1%}")
    print(f"  Full V8.63 vs regime:      {cE-cA:>+.1%}")
    print(f"  Full V8.63 vs random-3:    {cE-c_rand:>+.1%}  (the +33% being explained)")
    best=(max([("EMA",cB-cA),("MACD",cC-cA),("Mom20",cD-cA)],key=lambda x:x[1]))
    print(f"\n  Strongest single signal:   {best[0]} ({best[1]:+.1%} vs regime alone)")
    print(f"\n  CONCLUSION:")
    if cE-cA < 0.03:
        print("  The regime gate alone explains almost all of V8.63's edge.")
        print("  EMA/MACD/Mom filters add minimal value beyond BTC timing.")
    elif best[1] > (cE-cA)*0.7:
        print(f"  {best[0]} is doing most of the work ({best[1]/max(cE-cA,0.001):.0%} of total coin-filter gain).")
        print("  Other signals are largely redundant.")
    else:
        print("  All three filters contribute meaningfully — no single dominant signal.")
        print("  The combination (AND logic) is what generates the edge.")

    print("\n"+"="*80); print("  TEST 30 COMPLETE"); print("="*80)

if __name__=="__main__": main()
