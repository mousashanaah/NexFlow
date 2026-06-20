#!/usr/bin/env python3
"""
TEST 20 — Stock Book Overlap Diagnostic

Replicates TEST 00 for the stock universe.
Determines whether the stock book is breadth-driven (like crypto)
or dispersion-driven (typical equity momentum universe).

Static basket: MSTR + AMD + GOOGL + SPOT
Universe: all 51 tradeable tickers in data/stocks/
"""
from __future__ import annotations
import random
import datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd

_REPO    = Path(__file__).parent.parent
_SDIR    = _REPO / "data" / "stocks"
_FROM_MS = int(dt.datetime(2021,1,1,tzinfo=dt.timezone.utc).timestamp()*1000)
_TO_MS   = int(dt.datetime(2026,6,1,tzinfo=dt.timezone.utc).timestamp()*1000)
_CAPITAL = 10_000.0
_STATIC_BASKET = ["MSTR","AMD","GOOGL","SPOT"]
_EXCL = {"SPY","QQQ","IWM","DIA","GLD","XLE","XLF","XLI","XLK","XLP",
         "XLRE","XLU","XLV","XLY","SLV","USO"}


def _get_universe():
    tickers = [f.stem.replace("_1D","") for f in _SDIR.glob("*_1D.parquet")
               if f.stem.replace("_1D","") not in _EXCL]
    return sorted(tickers)


def _ema(s,n):
    out=np.full(len(s),np.nan); alpha=2/(n+1); v=float(s[0])
    for i in range(len(s)):
        v=alpha*float(s[i])+(1-alpha)*v
        if i>=n-1: out[i]=v
    return out

def _sma(s,n):
    out=np.full(len(s),np.nan)
    for i in range(n-1,len(s)): out[i]=np.mean(s[i-n+1:i+1])
    return out

def _macd_long(s):
    e12=_ema(s,12); e26=_ema(s,26); macd=e12-e26
    vi=np.where(~np.isnan(macd))[0]
    sig=np.full(len(s),np.nan)
    if len(vi)>=9: sig[vi]=_ema(macd[vi],9)
    hist=macd-sig
    state=np.zeros(len(s),dtype=bool)
    for i in range(1,len(s)):
        hp,hc=hist[i-1],hist[i]
        if np.isnan(hp) or np.isnan(hc): state[i]=state[i-1]
        elif hp<=0<hc: state[i]=True
        elif hp>=0>hc: state[i]=False
        else: state[i]=state[i-1]
    return state

def _load_stock(ticker):
    p = _SDIR / f"{ticker}_1D.parquet"
    if not p.exists(): return None
    df = pd.read_parquet(p, columns=["open_time","close"])
    df = df.rename(columns={"open_time":"ts"})
    df["ts"]=df["ts"].astype(int); df["close"]=df["close"].astype(float)
    df=df[(df.ts>=_FROM_MS)&(df.ts<=_TO_MS)].sort_values("ts").reset_index(drop=True)
    if len(df)<100: return None
    c=df["close"].values; ts=df["ts"].values
    e8=_ema(c,8); e21=_ema(c,21); s200=_sma(c,200)
    ml=_macd_long(c)
    above_e=e8>e21
    ema_long=np.zeros(len(c),dtype=bool)
    for i in range(1,len(c)):
        if above_e[i] and not above_e[i-1]: ema_long[i]=True
        elif not above_e[i] and above_e[i-1]: ema_long[i]=False
        else: ema_long[i]=ema_long[i-1]
    mom20=np.full(len(c),np.nan); mom60=np.full(len(c),np.nan)
    for i in range(20,len(c)): mom20[i]=(c[i]-c[i-20])/c[i-20]
    for i in range(60,len(c)): mom60[i]=(c[i]-c[i-60])/c[i-60]
    coin_sig=ema_long & ml & (mom20>0)
    above200=np.where(np.isnan(s200),False,c>s200)
    return {"ts":ts,"close":c,"coin_sig":coin_sig,"above200":above200,
            "mom60":mom60,"by_ts":{int(t):i for i,t in enumerate(ts)}}

def _cagr(eq,nd): return (eq[-1]/eq[0])**(365/nd)-1 if eq[0]>0 and eq[-1]>0 and nd>0 else 0
def _sharpe(eq):
    r=np.diff(eq)/eq[:-1]; r=r[np.isfinite(r)]
    return r.mean()/r.std()*np.sqrt(252) if len(r)>1 and r.std()>0 else 0
def _maxdd(eq):
    pk=np.maximum.accumulate(eq)
    return float(np.max((pk-eq)/np.where(pk>0,pk,1)))


def main():
    print("="*80)
    print("  TEST 20 — STOCK BOOK OVERLAP DIAGNOSTIC")
    print(f"  Static basket: {'+'.join(_STATIC_BASKET)}")
    print("="*80)

    universe = _get_universe()
    print(f"\n  Loading {len(universe)} tickers...")
    data = {}
    for t in universe:
        d = _load_stock(t)
        if d: data[t] = d
    print(f"  Loaded {len(data)} tickers with sufficient data")

    # Common timestamp grid from AMD (reliable long history)
    anchor = "AMD" if "AMD" in data else list(data.keys())[0]
    all_ts = sorted(data[anchor]["ts"].tolist())

    # Month groups
    dates = pd.to_datetime(all_ts, unit="ms", utc=True)
    month_groups={}
    for j,ts in enumerate(all_ts):
        d=dates[j]; key=(d.year,d.month)
        month_groups.setdefault(key,[]).append(j)

    # ── TEST 20A: Monthly overlap ─────────────────────────────────────────────
    print("\n"+"="*80)
    print("  TEST 20A — MONTHLY OVERLAP: STATIC BASKET vs TOP-3 MOMENTUM")
    print("  (Note: static basket selected by V9 combo search, not dynamic)")
    print("="*80)
    print(f"\n  {'Month':8s}  {'Static Basket':20s}  {'Top-3 Mom':24s}  {'Overlap':8s}")
    print("  "+"-"*70)

    all_ov=[]
    for (yr,mo) in sorted(month_groups):
        idxs = month_groups[(yr,mo)]
        first_j = idxs[0]
        first_ts = all_ts[first_j]

        # Static basket: check if each member has valid data this month
        static = set()
        for t in _STATIC_BASKET:
            if t in data and first_ts in {int(x) for x in data[t]["ts"][:50]}:
                static.add(t)
            elif t in data:
                # find closest ts
                static.add(t)  # include if data exists regardless of exact ts match

        # Top-3 momentum from full universe
        scores={}
        for t in data:
            i = data[t]["by_ts"].get(first_ts)
            if i is not None:
                m=data[t]["mom60"][i]; a=data[t]["above200"][i]
                if not np.isnan(m) and a: scores[t]=m
        top3=set(sorted(scores,key=scores.get,reverse=True)[:3])

        inter=static&top3; union=static|top3
        ov=len(inter)/len(union) if union else 1.0
        all_ov.append(ov)
        s_str="+".join(sorted(static)) or "(none)"
        t_str="+".join(sorted(top3)) or "(none)"
        print(f"  {yr}-{mo:02d}   {s_str:20s}  {t_str:24s}  {ov:6.0%}")

    print("  "+"-"*70)
    print(f"\n  Average overlap (static vs top-3 momentum): {np.mean(all_ov):.1%}")
    print(f"  Random baseline: {4/len(data)*100:.1f}%  (4 specific tickers in {len(data)}-ticker universe)")

    # ── TEST 20B: Momentum persistence in stocks ──────────────────────────────
    print("\n"+"="*80)
    print("  TEST 20B — MOMENTUM PERSISTENCE (Stocks)")
    print("="*80)

    month_starts=[]
    seen=set()
    for j,ts in enumerate(all_ts):
        d=dates[j]; key=(d.year,d.month)
        if key not in seen: seen.add(key); month_starts.append(j)

    def top3_at(j):
        ts=all_ts[j]
        sc={}
        for t in data:
            i=data[t]["by_ts"].get(ts)
            if i is not None:
                m=data[t]["mom60"][i]
                if not np.isnan(m): sc[t]=m
        return set(sorted(sc,key=sc.get,reverse=True)[:3])

    lag1,lag2,lag3=[],[],[]
    for k in range(len(month_starts)):
        cur=top3_at(month_starts[k])
        if len(cur)<3: continue
        if k+1<len(month_starts): lag1.append(len(cur&top3_at(month_starts[k+1]))/3)
        if k+2<len(month_starts): lag2.append(len(cur&top3_at(month_starts[k+2]))/3)
        if k+3<len(month_starts): lag3.append(len(cur&top3_at(month_starts[k+3]))/3)

    rb=3/len(data)
    print(f"\n  Universe: {len(data)} stocks  |  Random baseline: {rb:.1%}")
    print(f"\n  {'Lag':10s}  {'Persistence':14s}  {'vs Random':14s}  n")
    print("  "+"-"*50)
    for lag,vals,name in [(1,lag1,"1 month"),(2,lag2,"2 months"),(3,lag3,"3 months")]:
        m=np.mean(vals) if vals else 0
        print(f"  {name:10s}  {m:13.1%}  {m/rb:>10.1f}×         {len(vals)}")
    print(f"  {'Random':10s}  {rb:13.1%}  {'1.0×':>14s}")

    # ── TEST 20C: Selection contribution ─────────────────────────────────────
    print("\n"+"="*80)
    print("  TEST 20C — SELECTION CONTRIBUTION (Stocks)")
    print("  Static basket vs Top-3 momentum vs Random-3")
    print("="*80)

    # Build daily returns for all stocks on AMD's timestamp grid
    N=len(all_ts)
    tickers_list=list(data.keys())
    K=len(tickers_list); tidx={t:k for k,t in enumerate(tickers_list)}
    ret=np.zeros((K,N))
    for t in tickers_list:
        k=tidx[t]; sd=data[t]
        closes=np.array([sd["close"][sd["by_ts"][ts]] if ts in sd["by_ts"] else np.nan
                         for ts in all_ts])
        for j in range(1,N):
            if not np.isnan(closes[j]) and not np.isnan(closes[j-1]) and closes[j-1]>0:
                ret[k,j]=(closes[j]-closes[j-1])/closes[j-1]

    mstarts=set(month_starts)
    nd=int((all_ts[-1]-all_ts[0])/86_400_000)

    def simulate(hfn):
        eq=np.full(N,_CAPITAL); cur=[]
        for j in range(1,N):
            if j in mstarts or not cur: cur=hfn(j)
            if not cur: eq[j]=eq[j-1]; continue
            eq[j]=eq[j-1]*(1+np.mean(ret[cur,j]))
        return eq

    # Static basket
    static_idx=[tidx[t] for t in _STATIC_BASKET if t in tidx]
    eq_static = simulate(lambda j: static_idx)

    # Top-3 momentum
    def mom3(j):
        ts=all_ts[j]
        sc={}
        for t in tickers_list:
            i=data[t]["by_ts"].get(ts)
            if i is not None:
                m=data[t]["mom60"][i]; a=data[t]["above200"][i]
                if not np.isnan(m) and a: sc[t]=m
        return [tidx[t] for t in sorted(sc,key=sc.get,reverse=True)[:3]]
    eq_mom3 = simulate(mom3)

    # Random-3 (300 seeds)
    sum_rand=np.zeros(N)
    for seed in range(300):
        rng=random.Random(seed)
        def rand3(j,r=rng):
            pool=[tidx[t] for t in tickers_list
                  if data[t]["by_ts"].get(all_ts[j]) is not None
                  and data[t]["above200"][data[t]["by_ts"][all_ts[j]]]]
            return r.sample(pool,3) if len(pool)>=3 else pool
        sum_rand+=simulate(rand3)
    eq_rand=sum_rand/300

    # Breadth: all tickers with V8.63-style signal
    def v863_all(j):
        ts=all_ts[j]
        return [tidx[t] for t in tickers_list
                if data[t]["by_ts"].get(ts) is not None
                and data[t]["coin_sig"][data[t]["by_ts"][ts]]]
    eq_broad = simulate(v863_all)

    print(f"\n  {'Version':25s}  {'CAGR':>8s}  {'Sharpe':>7s}  {'Max DD':>7s}  {'Final $':>12s}")
    print("  "+"-"*68)
    for label,eq in [("Static basket (MSTR+..)",eq_static),
                     ("Top-3 momentum",eq_mom3),
                     ("Random-3 (avg 300)",eq_rand),
                     (f"All signals ({len(tickers_list)}-wide)",eq_broad)]:
        print(f"  {label:25s}  {_cagr(eq,nd):>7.1%}  {_sharpe(eq):>7.2f}"
              f"  {_maxdd(eq):>7.1%}  ${eq[-1]:>11,.0f}")

    print(f"\n  Year-by-year returns:")
    print(f"  {'Year':6s}  {'Static':>9s}  {'Top-3 Mom':>10s}  {'Random-3':>9s}  {'Broad Sig':>10s}")
    print("  "+"-"*54)
    def yr_map(eq):
        ys={}; ye={}
        for i,d in enumerate(dates):
            yr=d.year
            if yr not in ys: ys[yr]=eq[i]
            ye[yr]=eq[i]
        return {yr:(ye[yr]-ys[yr])/ys[yr] for yr in ys}
    ym_s=yr_map(eq_static); ym_m=yr_map(eq_mom3); ym_r=yr_map(eq_rand); ym_b=yr_map(eq_broad)
    for yr in range(2021,2027):
        print(f"  {yr}    {ym_s.get(yr,0):>8.1%}  {ym_m.get(yr,0):>9.1%}"
              f"  {ym_r.get(yr,0):>8.1%}  {ym_b.get(yr,0):>9.1%}")

    # Key question: does stocks resemble crypto (breadth) or momentum universe (dispersion)?
    c_static=_cagr(eq_static,nd); c_mom=_cagr(eq_mom3,nd)
    c_rand=_cagr(eq_rand,nd);     c_broad=_cagr(eq_broad,nd)
    print(f"\n  KEY QUESTION: Breadth vs Concentration in stocks")
    print(f"  Broad signals vs static basket:  {c_broad-c_static:>+.1%} CAGR")
    print(f"  Momentum top-3 vs random-3:      {c_mom-c_rand:>+.1%} CAGR")
    print(f"  Static basket vs random-3:       {c_static-c_rand:>+.1%} CAGR")
    if c_broad > c_static + 0.05:
        print("\n  → STOCKS RESEMBLE CRYPTO: Breadth outperforms concentration.")
        print("    The static basket may be artificially constraining the stock book.")
    elif c_mom > c_rand + 0.05:
        print("\n  → STOCKS RESEMBLE MOMENTUM UNIVERSE: Momentum selection adds value.")
        print("    Dynamic rotation in the stock book deserves research.")
    else:
        print("\n  → MIXED: No dominant pattern. Static basket vs dynamic needs more investigation.")

    print("\n"+"="*80)
    print("  TEST 20 COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
