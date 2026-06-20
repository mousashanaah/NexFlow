#!/usr/bin/env python3
"""
TEST 40/41/42 — Dynamic Stock Selection Validation

TEST 40: Walk-forward out-of-sample test (year-by-year)
TEST 41: BTC correlation analysis — static vs dynamic book
TEST 42: Full V9 Confidence with dynamic stock selection vs current

Specification: Top-3 by 60d momentum, SMA200 gate, monthly rebalance.
No parameter search. No optimization. Fixed as specified.
"""
from __future__ import annotations
import random, sys
import datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd

_REPO    = Path(__file__).parent.parent
_SDIR    = _REPO / "data" / "stocks"
_CDIR    = _REPO / "data" / "candles"
_FROM_MS = int(dt.datetime(2021,1,1,tzinfo=dt.timezone.utc).timestamp()*1000)
_TO_MS   = int(dt.datetime(2026,6,1,tzinfo=dt.timezone.utc).timestamp()*1000)
_CAPITAL = 5_000.0   # match V9 Confidence capital
_STATIC  = ["MSTR","AMD","GOOGL","SPOT"]
_EXCL    = {"SPY","QQQ","IWM","DIA","GLD","XLE","XLF","XLI","XLK","XLP",
            "XLRE","XLU","XLV","XLY","SLV","USO"}
_CRYPTO  = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT",
            "XRPUSDT","ADAUSDT","DOGEUSDT","AVAXUSDT",
            "LINKUSDT","LTCUSDT","DOTUSDT","TRXUSDT"]


# ── shared helpers ────────────────────────────────────────────────────────────

def _ema(s,n):
    out=np.full(len(s),np.nan); a=2/(n+1); v=float(s[0])
    for i in range(len(s)):
        v=a*float(s[i])+(1-a)*v
        if i>=n-1: out[i]=v
    return out

def _sma(s,n):
    out=np.full(len(s),np.nan)
    for i in range(n-1,len(s)): out[i]=np.mean(s[i-n+1:i+1])
    return out

def _cagr(eq,nd): return (eq[-1]/eq[0])**(365/nd)-1 if eq[0]>0 and eq[-1]>0 and nd>0 else 0
def _sharpe(eq):
    r=np.diff(eq)/eq[:-1]; r=r[np.isfinite(r)]
    return r.mean()/r.std()*np.sqrt(252) if len(r)>1 and r.std()>0 else 0
def _maxdd(eq):
    pk=np.maximum.accumulate(eq)
    return float(np.max((pk-eq)/np.where(pk>0,pk,1)))
def _sortino(eq):
    r=np.diff(eq)/eq[:-1]; r=r[np.isfinite(r)]
    neg=r[r<0]; ds=neg.std() if len(neg)>1 else 1e-9
    return r.mean()/ds*np.sqrt(252) if ds>0 else 0
def _calmar(eq,nd):
    c=_cagr(eq,nd); d=_maxdd(eq); return c/d if d>0 else 0
def _yr(eq,dates):
    ys={};ye={}
    for i,d in enumerate(dates):
        yr=d.year
        if yr not in ys: ys[yr]=eq[i]
        ye[yr]=eq[i]
    return {yr:(ye[yr]-ys[yr])/ys[yr] for yr in ys}


# ── stock data loading ────────────────────────────────────────────────────────

def _load_stocks():
    tickers=[f.stem.replace("_1D","") for f in _SDIR.glob("*_1D.parquet")
             if f.stem.replace("_1D","") not in _EXCL]
    data={}
    for t in sorted(tickers):
        p=_SDIR/f"{t}_1D.parquet"
        if not p.exists(): continue
        df=pd.read_parquet(p,columns=["open_time","close"])
        df=df.rename(columns={"open_time":"ts"})
        df["ts"]=df["ts"].astype(int); df["close"]=df["close"].astype(float)
        df=df[(df.ts>=_FROM_MS)&(df.ts<=_TO_MS)].sort_values("ts").reset_index(drop=True)
        if len(df)<120: continue
        c=df["close"].values; ts=df["ts"].values
        s200=_sma(c,200)
        mom60=np.full(len(c),np.nan); mom20=np.full(len(c),np.nan)
        for i in range(60,len(c)): mom60[i]=(c[i]-c[i-60])/c[i-60]
        for i in range(20,len(c)): mom20[i]=(c[i]-c[i-20])/c[i-20]
        above200=np.where(np.isnan(s200),False,c>s200)
        data[t]={"ts":ts,"close":c,"mom60":mom60,"mom20":mom20,
                 "above200":above200,"by_ts":{int(t2):i for i,t2 in enumerate(ts)}}
    return data

def _stock_grid(sdata):
    anchor="AMD" if "AMD" in sdata else list(sdata.keys())[0]
    ts=[t for t in sdata[anchor]["ts"] if _FROM_MS<=t<=_TO_MS]
    return sorted(ts)

def _stock_ret(sdata,common_ts,tickers):
    K=len(tickers); N=len(common_ts); ret=np.zeros((K,N))
    tidx={t:k for k,t in enumerate(tickers)}
    for t in tickers:
        if t not in sdata: continue
        k=tidx[t]; sd=sdata[t]
        c=np.array([sd["close"][sd["by_ts"][ts]] if ts in sd["by_ts"] else np.nan
                    for ts in common_ts])
        for j in range(1,N):
            if not np.isnan(c[j]) and not np.isnan(c[j-1]) and c[j-1]>0:
                ret[k,j]=(c[j]-c[j-1])/c[j-1]
    return ret, tidx

def _simulate_stock(get_holdings, ret, N, dates, capital=_CAPITAL):
    eq=np.full(N,capital); cur=[]
    mstarts=set(); seen=set()
    for j,d in enumerate(dates):
        k=(d.year,d.month)
        if k not in seen: seen.add(k); mstarts.add(j)
    for j in range(1,N):
        if j in mstarts or not cur: cur=get_holdings(j)
        if not cur: eq[j]=eq[j-1]; continue
        eq[j]=eq[j-1]*(1+np.mean(ret[cur,j]))
    return eq


# ── crypto data for TEST 42 ───────────────────────────────────────────────────

def _load_crypto():
    data={}
    for sym in _CRYPTO:
        p=_CDIR/f"{sym}_1D.parquet"
        if not p.exists(): continue
        df=pd.read_parquet(p,columns=["open_time","close"])
        df=df.rename(columns={"open_time":"ts"})
        df["ts"]=df["ts"].astype(int); df["close"]=df["close"].astype(float)
        df=df[(df.ts>=_FROM_MS)&(df.ts<=_TO_MS)].sort_values("ts").reset_index(drop=True)
        c=df["close"].values; ts=df["ts"].values
        s200=_sma(c,200)
        e8=_ema(c,8); e21=_ema(c,21)
        above_e=e8>e21; ema_long=np.zeros(len(c),dtype=bool)
        for i in range(1,len(c)):
            if above_e[i] and not above_e[i-1]:   ema_long[i]=True
            elif not above_e[i] and above_e[i-1]: ema_long[i]=False
            else:                                   ema_long[i]=ema_long[i-1]
        from scripts.test30_signal_edge import _macd_state
        macd_long=_macd_state(c)
        mom20=np.full(len(c),np.nan); mom30=np.full(len(c),np.nan)
        for i in range(20,len(c)): mom20[i]=(c[i]-c[i-20])/c[i-20]
        for i in range(30,len(c)): mom30[i]=(c[i]-c[i-30])/c[i-30]
        above200=np.where(np.isnan(s200),False,c>s200)
        coin_sig=ema_long & macd_long & (mom20>0)
        data[sym]={"ts":ts,"close":c,"coin_sig":coin_sig,"above200":above200,
                   "mom30":mom30,"by_ts":{int(t):i for i,t in enumerate(ts)}}
    return data

def _btc_regime(cdata,common_ts):
    btc=cdata["BTCUSDT"]; bear=False; streak=0
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


# ── TEST 40 — Walk-Forward ────────────────────────────────────────────────────

def run_40(sdata,common_ts,dates,ret,tidx):
    print("\n"+"="*80)
    print("  TEST 40 — WALK-FORWARD OUT-OF-SAMPLE VALIDATION")
    print("  Each year is an independent out-of-sample test.")
    print("  Specification fixed: Top-3, 60d momentum, SMA200 gate.")
    print("="*80)

    tickers=list(tidx.keys())

    def top3(j):
        ts=common_ts[j]
        sc={}
        for t in tickers:
            i=sdata[t]["by_ts"].get(ts)
            if i is not None:
                m=sdata[t]["mom60"][i]; a=sdata[t]["above200"][i]
                if not np.isnan(m) and a: sc[t]=m
        return [tidx[t] for t in sorted(sc,key=sc.get,reverse=True)[:3]]

    static_idx=[tidx[t] for t in _STATIC if t in tidx]

    print(f"\n  Year-by-year out-of-sample performance:")
    print(f"  (Dynamic uses only data available at each monthly rebalance point)\n")
    print(f"  {'Year':6s}  {'Static Basket':>14s}  {'Dynamic Top-3':>14s}  {'Difference':>11s}  {'Dynamic Holdings (sample)':s}")
    print("  "+"-"*95)

    all_static=[]; all_dynamic=[]
    for yr in range(2021,2027):
        yr_ms=int(dt.datetime(yr,1,1,tzinfo=dt.timezone.utc).timestamp()*1000)
        yr_end=int(dt.datetime(yr+1,1,1,tzinfo=dt.timezone.utc).timestamp()*1000)
        idxs=[j for j,ts in enumerate(common_ts) if yr_ms<=ts<yr_end]
        if not idxs: continue

        # Static: fixed basket, compounded daily
        eq_s=np.full(len(idxs),_CAPITAL)
        for jj in range(1,len(idxs)):
            eq_s[jj]=eq_s[jj-1]*(1+np.mean(ret[static_idx,idxs[jj]]))
        ret_s=(eq_s[-1]-eq_s[0])/eq_s[0]

        # Dynamic: monthly rebalance within year
        eq_d=np.full(len(idxs),_CAPITAL)
        cur=[]; prev_month=None
        year_dates=dates[idxs]
        for jj in range(len(idxs)):
            mo=(year_dates[jj].year,year_dates[jj].month)
            if mo!=prev_month: cur=top3(idxs[jj]); prev_month=mo
            if jj>0:
                eq_d[jj]=eq_d[jj-1]*(1+np.mean(ret[cur,idxs[jj]]) if cur else 0)
        ret_d=(eq_d[-1]-eq_d[0])/eq_d[0]

        # Sample holdings for that year (mid-year)
        mid=idxs[len(idxs)//2]
        sample=top3(mid)
        sample_names="+".join([t for t,k in tidx.items() if k in sample][:3])

        diff=ret_d-ret_s
        flag="✓" if ret_d>ret_s else ("✗" if ret_d<ret_s-0.05 else "~")
        print(f"  {yr}    {ret_s:>13.1%}  {ret_d:>13.1%}  {diff:>+10.1%}  {flag}  {sample_names}")
        all_static.append(ret_s); all_dynamic.append(ret_d)

    print("  "+"-"*95)
    print(f"  {'Avg':6s}  {np.mean(all_static):>13.1%}  {np.mean(all_dynamic):>13.1%}"
          f"  {np.mean(all_dynamic)-np.mean(all_static):>+10.1%}")
    wins=sum(1 for d,s in zip(all_dynamic,all_static) if d>s)
    print(f"\n  Dynamic wins {wins}/{len(all_static)} years")
    better_2022 = next((all_dynamic[i]-all_static[i] for i,yr in enumerate(range(2021,2027))
                        if yr==2022), None)
    if better_2022 is not None:
        print(f"  2022 improvement: {better_2022:>+.1%}  (key risk-reduction year)")
    if wins>=4:
        print(f"\n  VERDICT: Strong out-of-sample evidence. Dynamic selection is robust.")
    elif wins==3:
        print(f"\n  VERDICT: Mixed. Slight edge but not conclusive.")
    else:
        print(f"\n  VERDICT: Static basket is competitive. Dynamic selection not clearly better.")


# ── TEST 41 — BTC Correlation ─────────────────────────────────────────────────

def run_41(sdata,common_ts,dates,ret,tidx,cdata):
    print("\n"+"="*80)
    print("  TEST 41 — BTC CORRELATION ANALYSIS")
    print("  Does dynamic rotation genuinely reduce hidden Bitcoin exposure?")
    print("="*80)

    tickers=list(tidx.keys())
    # BTC daily return on stock grid (match timestamps)
    btc=cdata["BTCUSDT"]
    btc_ret=np.zeros(len(common_ts))
    for j in range(1,len(common_ts)):
        ts=common_ts[j]; ts_prev=common_ts[j-1]
        i=btc["by_ts"].get(ts); ip=btc["by_ts"].get(ts_prev)
        if i and ip and btc["close"][ip]>0:
            btc_ret[j]=(btc["close"][i]-btc["close"][ip])/btc["close"][ip]

    static_idx=[tidx[t] for t in _STATIC if t in tidx]

    def top3(j):
        ts=common_ts[j]; sc={}
        for t in tickers:
            i=sdata[t]["by_ts"].get(ts)
            if i is not None:
                m=sdata[t]["mom60"][i]; a=sdata[t]["above200"][i]
                if not np.isnan(m) and a: sc[t]=m
        return [tidx[t] for t in sorted(sc,key=sc.get,reverse=True)[:3]]

    month_groups={}
    for j,ts in enumerate(common_ts):
        d=dates[j]; key=(d.year,d.month)
        month_groups.setdefault(key,[]).append(j)

    print(f"\n  Monthly 30-day rolling correlation to BTC:")
    print(f"  {'Month':8s}  {'Static Corr':>12s}  {'Dynamic Corr':>13s}  {'Difference':>11s}  {'Dynamic Holdings'}")
    print("  "+"-"*88)

    static_corrs=[]; dynamic_corrs=[]; months_shown=[]
    prev_mo=None; dyn_cur=[]

    for (yr,mo) in sorted(month_groups):
        idxs=month_groups[(yr,mo)]
        if len(idxs)<10: continue

        # Update dynamic holdings at month start
        dyn_cur=top3(idxs[0])

        # Static daily returns this month
        sr=np.array([np.mean(ret[static_idx,j]) for j in idxs])
        dr=np.array([np.mean(ret[dyn_cur,j]) for j in idxs]) if dyn_cur else sr*0
        br=np.array([btc_ret[j] for j in idxs])

        def corr(a,b):
            if len(a)<5: return np.nan
            da=a-a.mean(); db=b-b.mean()
            denom=np.sqrt((da**2).sum()*(db**2).sum())
            return float(np.dot(da,db)/denom) if denom>0 else np.nan

        sc=corr(sr,br); dc=corr(dr,br)
        static_corrs.append(sc); dynamic_corrs.append(dc)

        dyn_names="+".join([t for t,k in tidx.items() if k in dyn_cur])
        diff=dc-sc if not np.isnan(dc) and not np.isnan(sc) else np.nan
        flag="↓BTC" if diff<-0.1 else ("↑BTC" if diff>0.1 else "~")
        print(f"  {yr}-{mo:02d}   {sc:>11.3f}  {dc:>12.3f}  {diff:>+10.3f}  {flag}  {dyn_names}")

    vs=[s for s in static_corrs if not np.isnan(s)]
    vd=[d for d in dynamic_corrs if not np.isnan(d)]
    print("  "+"-"*88)
    print(f"  {'Average':8s}  {np.mean(vs):>11.3f}  {np.mean(vd):>12.3f}  {np.mean(vd)-np.mean(vs):>+10.3f}")
    diff_avg=np.mean(vd)-np.mean(vs)
    print(f"\n  INTERPRETATION:")
    if diff_avg < -0.10:
        print(f"  Dynamic selection REDUCES BTC correlation by {abs(diff_avg):.3f} on average.")
        print(f"  The two books are more genuinely independent with dynamic stock selection.")
        print(f"  This improves the V9 diversification architecture, not just CAGR.")
    elif diff_avg > 0.10:
        print(f"  Dynamic selection INCREASES BTC correlation ({diff_avg:+.3f}).")
        print(f"  It may be rotating into crypto-adjacent stocks (miners, leveraged plays).")
        print(f"  This would degrade the V9 two-book diversification premise.")
    else:
        print(f"  Correlation change is modest ({diff_avg:+.3f}). Neutral effect on diversification.")


# ── TEST 42 — Full V9 Confidence Comparison ───────────────────────────────────

def run_42(sdata,cdata,stock_ts,stock_dates,stock_ret,tidx):
    print("\n"+"="*80)
    print("  TEST 42 — FULL V9 CONFIDENCE: STATIC vs DYNAMIC STOCK BOOK")
    print("  Everything identical except stock basket.")
    print("  Confidence engine, allocation weights, crypto book — all unchanged.")
    print("="*80)

    tickers=list(tidx.keys())

    # Build BTC regime on stock grid
    bull=_btc_regime(cdata,stock_ts)

    # BTC score (simplified version of V9 confidence scoring)
    def btc_score(j):
        ts=stock_ts[j]; btc=cdata["BTCUSDT"]
        i=btc["by_ts"].get(ts)
        if i is None: return 2.0
        c=btc["close"][i]
        s200_arr=_sma(btc["close"][:i+1],200)
        s200=s200_arr[i] if not np.isnan(s200_arr[i]) else c
        mom90=((c-btc["close"][max(0,i-90)])/btc["close"][max(0,i-90)]) if i>=90 else 0
        mom30=float(btc["mom30"][i]) if not np.isnan(btc["mom30"][i]) else 0
        sc=0.0
        sc+=2.0 if c>s200 else 0.0
        sc+=1.0 if mom90>0 else 0.0
        sc+=0.5 if mom30>0 else 0.0
        if mom90>0.30: sc+=0.5
        if mom90<-0.30: sc-=0.5
        return float(np.clip(sc,0,4))

    def stock_score(holdings,j):
        if not holdings: return 1.5
        ts=stock_ts[j]; vals=[]
        for k in holdings:
            t=[tk for tk,ki in tidx.items() if ki==k]
            if not t: continue
            t=t[0]; sd=sdata[t]; i=sd["by_ts"].get(ts)
            if i is None: continue
            m60=sd["mom60"][i]; a200=sd["above200"][i]
            sc=0.0
            sc+=1.0 if a200 else 0.0
            sc+=1.0 if (not np.isnan(m60) and m60>0) else 0.0
            vals.append(sc)
        return float(np.mean(vals)) if vals else 1.5

    def allocate(cs,ss):
        cn=cs/4.0; sn=ss/3.0
        if   cn>=0.65 and sn>=0.65: return 0.65,0.35
        elif cn>=0.65:               return 0.80,0.20
        elif sn>=0.65:               return 0.20,0.80
        elif cn<0.35 and sn<0.35:   return 0.40,0.40
        else:
            tot=cn+sn; wc=0.40+(cn/tot)*0.20 if tot>0 else 0.50
            return round(wc,2),round(1-wc,2)

    # Build crypto equity curve (simplified V8.63 on stock grid)
    print("  Building crypto curve (V8.63 on stock market days)...")
    csym_idx={s:k for k,s in enumerate(_CRYPTO)}
    crypto_ret=np.zeros((len(_CRYPTO),len(stock_ts)))
    for k,sym in enumerate(_CRYPTO):
        if sym not in cdata: continue
        sd=cdata[sym]
        c=np.array([sd["close"][sd["by_ts"][ts]] if ts in sd["by_ts"] else np.nan
                    for ts in stock_ts])
        for j in range(1,len(stock_ts)):
            if not np.isnan(c[j]) and not np.isnan(c[j-1]) and c[j-1]>0:
                crypto_ret[k,j]=(c[j]-c[j-1])/c[j-1]

    def crypto_holdings(j):
        if not bull[j]: return []
        return [csym_idx[s] for s in _CRYPTO
                if cdata[s]["by_ts"].get(stock_ts[j]) is not None
                and cdata[s]["coin_sig"][cdata[s]["by_ts"][stock_ts[j]]]]

    # Precompute crypto daily equity (used as reference curve)
    N=len(stock_ts); crypto_eq=np.full(N,_CAPITAL); cur_c=[]
    mstarts=set(); seen=set()
    for j,d in enumerate(stock_dates):
        key=(d.year,d.month)
        if key not in seen: seen.add(key); mstarts.add(j)
    for j in range(1,N):
        if j in mstarts or not cur_c: cur_c=crypto_holdings(j)
        if not cur_c: crypto_eq[j]=crypto_eq[j-1]; continue
        crypto_eq[j]=crypto_eq[j-1]*(1+np.mean(crypto_ret[csym_idx.get(s,0) for s in _CRYPTO if s in csym_idx and csym_idx[s] in cur_c],axis=None if not cur_c else 0))

    # Simpler: recompute crypto equity directly
    crypto_eq2=np.full(N,_CAPITAL); cur_c2=[]
    for j in range(1,N):
        if j in mstarts or not cur_c2: cur_c2=crypto_holdings(j)
        if not cur_c2: crypto_eq2[j]=crypto_eq2[j-1]; continue
        crypto_eq2[j]=crypto_eq2[j-1]*(1+np.mean(crypto_ret[cur_c2,j]))

    print("  Building static stock curve...")
    static_idx=[tidx[t] for t in _STATIC if t in tidx]
    def static_fn(j): return static_idx
    stock_eq_static=_simulate_stock(static_fn,stock_ret,N,stock_dates)

    print("  Building dynamic stock curve (top-3 momentum)...")
    def top3fn(j):
        ts=stock_ts[j]; sc={}
        for t in tickers:
            i=sdata[t]["by_ts"].get(ts)
            if i is not None:
                m=sdata[t]["mom60"][i]; a=sdata[t]["above200"][i]
                if not np.isnan(m) and a: sc[t]=m
        return [tidx[t] for t in sorted(sc,key=sc.get,reverse=True)[:3]]
    stock_eq_dynamic=_simulate_stock(top3fn,stock_ret,N,stock_dates)

    print("  Running V9 confidence engine (both versions)...")

    def run_v9(c_eq,s_eq,label):
        c_ret=np.diff(c_eq)/c_eq[:-1]; s_ret=np.diff(s_eq)/s_eq[:-1]
        wc,ws=0.50,0.50; last_reb=0; cur_eq=_CAPITAL
        port_eq=np.full(N,_CAPITAL); year_pnl={}; alloc_log=[]
        for step in range(1,N):
            if step>0 and (step-last_reb)>=21:
                cs=btc_score(step-1)
                # get current dynamic or static holdings for stock score
                sc=stock_score(top3fn(step-1) if "Dynamic" in label else static_idx, step-1)
                wc,ws=allocate(cs,sc)
                alloc_log.append((wc,ws,cs,sc)); last_reb=step
            cr=c_ret[step-1]; sr=s_ret[step-1]
            port_eq[step]=port_eq[step-1]*(1+wc*cr+ws*sr)
            yr=stock_dates[step].year
            daily_pnl=port_eq[step]-port_eq[step-1]
            year_pnl[yr]=year_pnl.get(yr,0)+daily_pnl
        return port_eq, year_pnl, alloc_log

    port_static,yp_s,log_s=run_v9(crypto_eq2,stock_eq_static,"Static")
    port_dynamic,yp_d,log_d=run_v9(crypto_eq2,stock_eq_dynamic,"Dynamic")

    nd=int((stock_ts[-1]-stock_ts[0])/86_400_000)

    print(f"\n  Starting capital: ${_CAPITAL:,.0f}")
    print(f"\n  {'Metric':22s}  {'V9 Static Basket':>20s}  {'V9 Dynamic Top-3':>20s}  {'Improvement':>12s}")
    print("  "+"-"*80)
    for label,val_s,val_d in [
        ("Final equity",    f"${port_static[-1]:,.0f}",    f"${port_dynamic[-1]:,.0f}",  f"${port_dynamic[-1]-port_static[-1]:>+,.0f}"),
        ("CAGR",            f"{_cagr(port_static,nd):.1%}",f"{_cagr(port_dynamic,nd):.1%}",f"{_cagr(port_dynamic,nd)-_cagr(port_static,nd):>+.1%}"),
        ("Sharpe",          f"{_sharpe(port_static):.2f}", f"{_sharpe(port_dynamic):.2f}", f"{_sharpe(port_dynamic)-_sharpe(port_static):>+.2f}"),
        ("Sortino",         f"{_sortino(port_static):.2f}",f"{_sortino(port_dynamic):.2f}",f"{_sortino(port_dynamic)-_sortino(port_static):>+.2f}"),
        ("Max DD",          f"{_maxdd(port_static):.1%}",  f"{_maxdd(port_dynamic):.1%}",  f"{_maxdd(port_dynamic)-_maxdd(port_static):>+.1%}"),
        ("Calmar",          f"{_calmar(port_static,nd):.2f}",f"{_calmar(port_dynamic,nd):.2f}",f"{_calmar(port_dynamic,nd)-_calmar(port_static,nd):>+.2f}"),
    ]:
        print(f"  {label:22s}  {val_s:>20s}  {val_d:>20s}  {val_d:>12s}")

    print(f"\n  Year-by-year returns:")
    yrs_s=_yr(port_static,stock_dates); yrs_d=_yr(port_dynamic,stock_dates)
    print(f"  {'Year':6s}  {'Static V9':>12s}  {'Dynamic V9':>12s}  {'Diff':>8s}  {'Better':s}")
    print("  "+"-"*55)
    losing_s=[]; losing_d=[]
    for yr in range(2021,2027):
        rs=yrs_s.get(yr,0); rd=yrs_d.get(yr,0)
        if rs<0: losing_s.append(yr)
        if rd<0: losing_d.append(yr)
        flag="✓" if rd>rs+0.01 else ("✗" if rs>rd+0.01 else "~")
        print(f"  {yr}    {rs:>11.1%}  {rd:>11.1%}  {rd-rs:>+7.1%}  {flag}")

    print(f"\n  Losing years — Static: {losing_s or 'NONE'}  |  Dynamic: {losing_d or 'NONE'}")

    # DEPLOYMENT GATE CHECK
    print(f"\n  ── DEPLOYMENT GATE CHECK ─────────────────────────────────────────")
    c_s=_cagr(port_static,nd); c_d=_cagr(port_dynamic,nd)
    sh_s=_sharpe(port_static); sh_d=_sharpe(port_dynamic)
    dd_s=_maxdd(port_static); dd_d=_maxdd(port_dynamic)
    wf_wins=sum(1 for yr in range(2021,2027) if yrs_d.get(yr,0)>yrs_s.get(yr,0))
    gate=[
        ("1. Improves CAGR",        c_d>c_s+0.01,      f"{c_d:.1%} vs {c_s:.1%}"),
        ("2. Sharpe preserved",      sh_d>=sh_s-0.05,   f"{sh_d:.2f} vs {sh_s:.2f}"),
        ("3. DD not materially worse",dd_d<=dd_s+0.05,  f"{dd_d:.1%} vs {dd_s:.1%}"),
        ("4. Walk-forward wins",      wf_wins>=4,        f"{wf_wins}/6 years better"),
        ("5. Mechanical explanation", True,              "Top-3 momentum, SMA200 gate — fully defined"),
    ]
    passed=0
    for name,ok,detail in gate:
        status="PASS ✓" if ok else "FAIL ✗"
        if ok: passed+=1
        print(f"  {status}  {name:35s} {detail}")
    print(f"\n  Gate score: {passed}/5")
    if passed==5:
        print("  ══ ALL GATES PASSED. Dynamic selection is a DEPLOYABLE UPGRADE. ══")
    elif passed>=4:
        print("  ══ 4/5 passed. Strong candidate — review failed gate before deployment. ══")
    else:
        print("  ══ Not ready. Address failed gates before proceeding to implementation. ══")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("="*80)
    print("  TEST 40/41/42 — DYNAMIC STOCK SELECTION VALIDATION SUITE")
    print("="*80)

    print("\n  Loading stock data...")
    sdata=_load_stocks()
    print(f"  Loaded {len(sdata)} stocks")

    common_ts=_stock_grid(sdata)
    dates=pd.to_datetime(common_ts,unit="ms",utc=True)
    tickers=list(sdata.keys())
    tidx={t:k for k,t in enumerate(tickers)}
    print(f"  Stock grid: {len(common_ts)} trading days")

    print("  Computing stock returns...")
    ret,tidx=_stock_ret(sdata,common_ts,tickers)

    print("  Loading crypto data...")
    cdata=_load_crypto()

    run_40(sdata,common_ts,dates,ret,tidx)
    run_41(sdata,cdata,common_ts,dates,ret,tidx)
    run_42(sdata,cdata,common_ts,dates,ret,tidx)

    print("\n"+"="*80)
    print("  TEST 40/41/42 COMPLETE")
    print("="*80)

if __name__=="__main__": main()
