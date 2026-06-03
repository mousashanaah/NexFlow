#!/usr/bin/env python3
"""Deep diagnostic: why does V8 lose money in 2025?"""

from __future__ import annotations
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from collections import defaultdict

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import pyarrow.parquet as pq

_CANDLE_DIR = _REPO_ROOT / "data" / "candles"
_TAKER_FEE  = 0.0006
_CAPITAL    = 5_000.0   # $5K as user mentioned
_SYMBOLS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT",
    "XRPUSDT","ADAUSDT","DOGEUSDT","AVAXUSDT",
    "LINKUSDT","LTCUSDT","DOTUSDT","TRXUSDT",
]
_DAY_MS = 86_400_000

def _load_daily(symbol: str) -> list[dict]:
    path = _CANDLE_DIR / f"{symbol}_1D.parquet"
    if not path.exists(): return []
    tbl = pq.read_table(path, columns=["open_time","close"])
    rows = [{"ts":int(ts),"close":float(c)}
            for ts,c in zip(tbl.column("open_time").to_pylist(),
                            tbl.column("close").to_pylist())]
    return sorted(rows, key=lambda x: x["ts"])

def _ema_series(closes, period):
    alpha = 2.0/(period+1)
    result = [None]*len(closes)
    ema = None
    for i, c in enumerate(closes):
        ema = alpha*c + (1-alpha)*ema if ema is not None else c
        if i >= period-1: result[i] = ema
    return result

def _sma_series(closes, period):
    result = [None]*len(closes)
    for i in range(period-1, len(closes)):
        result[i] = sum(closes[i-period+1:i+1])/period
    return result

def _macd_long_series(closes):
    ef = _ema_series(closes, 12)
    es = _ema_series(closes, 26)
    macds = [ef[i]-es[i] if ef[i] and es[i] else None for i in range(len(closes))]
    sig_vals = [m for m in macds if m is not None]
    sig_raw = _ema_series(sig_vals, 9)
    sig = []
    idx = 0
    for m in macds:
        if m is None: sig.append(None)
        else: sig.append(sig_raw[idx]); idx += 1
    hist = [macds[i]-sig[i] if macds[i] is not None and sig[i] is not None else None
            for i in range(len(closes))]
    result = [False]*len(closes)
    in_long = False
    for i in range(1, len(closes)):
        if hist[i-1] is not None and hist[i] is not None:
            if hist[i-1] <= 0 < hist[i]: in_long = True
            elif hist[i-1] >= 0 > hist[i]: in_long = False
        result[i] = in_long
    return result

def _build_signals(symbols):
    out = {}
    for sym in symbols:
        bars = _load_daily(sym)
        if not bars: continue
        closes = [b["close"] for b in bars]
        ts_list = [b["ts"] for b in bars]
        ef8  = _ema_series(closes, 8)
        ef21 = _ema_series(closes, 21)
        sma200 = _sma_series(closes, 200)
        sma50  = _sma_series(closes, 50)
        macd_long = _macd_long_series(closes)
        ef5  = _ema_series(closes, 5)
        ef13 = _ema_series(closes, 13)
        by_ts = {}
        ema_long_state = False
        h4_long_state  = False
        prev_ema_above = None
        prev_h4_above  = None
        for i, ts in enumerate(ts_list):
            ema_above = ef8[i] > ef21[i] if ef8[i] and ef21[i] else False
            h4_above  = ef5[i] > ef13[i] if ef5[i] and ef13[i] else False
            if prev_ema_above is not None and ema_above != prev_ema_above:
                ema_long_state = ema_above
            if prev_h4_above is not None and h4_above != prev_h4_above:
                h4_long_state = h4_above and ema_above
            prev_ema_above = ema_above
            prev_h4_above  = h4_above
            by_ts[ts] = {
                "close":       closes[i],
                "ema_long":    ema_long_state,
                "macd_long":   macd_long[i],
                "h4_long":     h4_long_state,
                "sma200_above": sma200[i] is not None and closes[i] > sma200[i],
                "sma50_above":  sma50[i]  is not None and closes[i] > sma50[i],
                "sma200":      sma200[i],
            }
        out[sym] = by_ts
    return out

def _vol_series(signals, sym, window=14):
    ts_list = sorted(signals.get(sym, {}).keys())
    closes = [signals[sym][t]["close"] for t in ts_list]
    result = {}
    for i, ts in enumerate(ts_list):
        if i < window:
            result[ts] = 0.0
            continue
        rets = [(closes[j]-closes[j-1])/closes[j-1] for j in range(i-window+1, i+1)]
        mean = sum(rets)/len(rets)
        variance = sum((r-mean)**2 for r in rets)/len(rets)
        result[ts] = variance**0.5
    return result

def main():
    print("=" * 70)
    print("V8 2025 ROOT CAUSE DIAGNOSTIC")
    print("=" * 70)

    # Timestamps
    ts_2025_start = int(datetime(2025,1,1,tzinfo=timezone.utc).timestamp()*1000)
    ts_2025_end   = int(datetime(2026,1,1,tzinfo=timezone.utc).timestamp()*1000)
    ts_full_start = int(datetime(2021,1,1,tzinfo=timezone.utc).timestamp()*1000)
    ts_full_end   = int(datetime.now(timezone.utc).timestamp()*1000)

    print("\nBuilding signals...")
    signals = _build_signals(_SYMBOLS)
    print("Done.\n")

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 1: BTC REGIME IN 2025
    # ──────────────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("SECTION 1: BTC REGIME IN 2025")
    print("=" * 70)

    btc_ts_list = sorted(signals.get("BTCUSDT", {}).keys())

    # Precompute BTC mom30 and sma50
    btc_closes = [signals["BTCUSDT"][t]["close"] for t in btc_ts_list]
    btc_sma50_vals = _sma_series(btc_closes, 50)
    btc_mom30 = {}
    for i, t in enumerate(btc_ts_list):
        if i >= 30:
            btc_mom30[t] = (btc_closes[i]-btc_closes[i-30])/btc_closes[i-30]
        else:
            btc_mom30[t] = 0.0

    # Simulate V8 asymmetric regime across full history to get correct state
    prev_btc_bear_mode = False
    btc_above_sma200_streak = 0
    all_ts = sorted(set(ts for sym in _SYMBOLS for ts in signals.get(sym, {})))

    regime_2025 = {}
    for ts in all_ts:
        btc_sig = signals.get("BTCUSDT",{}).get(ts,{})
        btc_sma200_above = btc_sig.get("sma200_above", True)
        btc_30d_ret = btc_mom30.get(ts, 0.0)
        drop_triggered = btc_30d_ret < -0.20
        # AND mode
        enter_bear = (not btc_sma200_above) and drop_triggered
        if btc_sma200_above:
            btc_above_sma200_streak += 1
        else:
            btc_above_sma200_streak = 0
        confirmed_bull = btc_above_sma200_streak >= max(1, 5)
        if prev_btc_bear_mode:
            btc_bear_mode = not confirmed_bull
        else:
            btc_bear_mode = enter_bear
        prev_btc_bear_mode = btc_bear_mode

        if ts_2025_start <= ts < ts_2025_end:
            dt = datetime.fromtimestamp(ts/1000, tz=timezone.utc)
            regime_2025[ts] = {
                "date": dt.strftime("%Y-%m-%d"),
                "month": dt.month,
                "bear": btc_bear_mode,
                "btc_close": btc_sig.get("close", 0),
                "sma200_above": btc_sma200_above,
                "mom30": btc_30d_ret,
                "drop_triggered": drop_triggered,
            }

    bear_days = sum(1 for v in regime_2025.values() if v["bear"])
    bull_days = sum(1 for v in regime_2025.values() if not v["bear"])
    total_days = len(regime_2025)
    print(f"\nTotal 2025 trading days: {total_days}")
    print(f"  Bull days: {bull_days} ({bull_days/total_days*100:.1f}%)")
    print(f"  Bear days: {bear_days} ({bear_days/total_days*100:.1f}%)")

    # Regime flips
    prev_bear = None
    flips = []
    for ts in sorted(regime_2025):
        cur_bear = regime_2025[ts]["bear"]
        if prev_bear is not None and cur_bear != prev_bear:
            flips.append({"date": regime_2025[ts]["date"], "to_bear": cur_bear,
                          "btc": regime_2025[ts]["btc_close"]})
        prev_bear = cur_bear
    print(f"\nRegime flips in 2025: {len(flips)}")
    for f in flips:
        label = "BULL→BEAR" if f["to_bear"] else "BEAR→BULL"
        print(f"  {f['date']}: {label}  BTC=${f['btc']:,.0f}")

    # Monthly bear/bull breakdown
    print("\nMonthly regime (bear days vs bull days):")
    month_names = ["","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    month_bull = defaultdict(int)
    month_bear = defaultdict(int)
    for v in regime_2025.values():
        if v["bear"]:
            month_bear[v["month"]] += 1
        else:
            month_bull[v["month"]] += 1
    for m in range(1, 13):
        bd = month_bear[m]; bud = month_bull[m]
        if bd + bud == 0: continue
        print(f"  {month_names[m]:>4}: bull={bud:>2}d  bear={bd:>2}d")

    # BTC price range in 2025
    btc_2025 = {ts: v for ts, v in regime_2025.items()}
    if btc_2025:
        prices = [v["btc_close"] for v in btc_2025.values()]
        print(f"\nBTC 2025 price range: ${min(prices):,.0f} – ${max(prices):,.0f}")
        # Monthly closes
        print("\nBTC monthly prices in 2025:")
        month_last = {}
        for ts in sorted(btc_2025):
            m = btc_2025[ts]["month"]
            month_last[m] = btc_2025[ts]["btc_close"]
        prev_p = None
        for m in range(1, 13):
            if m not in month_last: continue
            p = month_last[m]
            chg = f"({(p-prev_p)/prev_p*100:+.1f}%)" if prev_p else ""
            print(f"  {month_names[m]}: ${p:,.0f} {chg}")
            prev_p = p

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 2: RUN V8 WITH FULL TRADE-LEVEL LOGGING
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SECTION 2: V8 TRADE-LEVEL ANALYSIS FOR 2025")
    print("=" * 70)

    # Re-run full backtest from 2021 with full logging
    base_notional = _CAPITAL / len(_SYMBOLS)

    # Precompute vol series
    vol_series = {}
    for sym in _SYMBOLS:
        vol_series[sym] = _vol_series(signals, sym)

    # Precompute mom30 for momentum gate
    mom30_series = {}
    for sym in _SYMBOLS:
        ts_list = sorted(signals.get(sym, {}).keys())
        closes = [signals[sym][t]["close"] for t in ts_list]
        m30 = {}
        for i, t in enumerate(ts_list):
            if i >= 30:
                m30[t] = (closes[i]-closes[i-30])/closes[i-30]
            else:
                m30[t] = 0.0
        mom30_series[sym] = m30

    # Reset regime state
    prev_btc_bear_mode = False
    btc_above_sma200_streak = 0
    positions = {}
    trades_2025 = []
    month_pnl = defaultdict(float)
    coin_pnl = defaultdict(float)
    long_pnl_total = 0.0
    short_pnl_total = 0.0
    hard_stop_fires = []
    equity = _CAPITAL
    last_rebal_ts = 0

    # Momentum gate blocked entries count
    mom_gate_blocked = defaultdict(int)
    mom_gate_allowed = defaultdict(int)

    def _position_size(sym, ts, mult=1.0):
        n = base_notional * mult
        vol = vol_series.get(sym, {}).get(ts, 0.0)
        if vol <= 0: return n
        vol_sized = (0.01 * _CAPITAL) / vol
        return min(vol_sized * mult, base_notional * 2 * mult)

    for ts in all_ts:
        btc_sig = signals.get("BTCUSDT",{}).get(ts,{})
        btc_sma200_above = btc_sig.get("sma200_above", True)
        btc_30d_ret = btc_mom30.get(ts, 0.0)
        drop_triggered = btc_30d_ret < -0.20
        enter_bear = (not btc_sma200_above) and drop_triggered
        if btc_sma200_above:
            btc_above_sma200_streak += 1
        else:
            btc_above_sma200_streak = 0
        confirmed_bull = btc_above_sma200_streak >= 5
        if prev_btc_bear_mode:
            btc_bear_mode = not confirmed_bull
        else:
            btc_bear_mode = enter_bear
        prev_btc_bear_mode = btc_bear_mode
        btc_bull = not btc_bear_mode
        long_allowed = btc_bull

        in_2025 = ts_2025_start <= ts < ts_2025_end
        dt = datetime.fromtimestamp(ts/1000, tz=timezone.utc)

        # TSMOM short rebalance
        if not btc_bull and (ts - last_rebal_ts) >= 7*_DAY_MS:
            last_rebal_ts = ts
            scores = []
            for sym in _SYMBOLS:
                sym_ts_list = sorted(signals.get(sym,{}).keys())
                past = [t for t in sym_ts_list if t <= ts]
                if len(past) < 127: continue
                c_now  = signals[sym][past[-1]]["close"]
                c_past = signals[sym][past[-127]]["close"]
                ret = (c_now-c_past)/c_past
                scores.append((ret, sym))
            scores.sort()
            desired_shorts = {sym for ret, sym in scores if ret < -0.05}

            for sym in [s for s in list(positions) if positions[s].get("side")=="SHORT"]:
                if sym not in desired_shorts:
                    c = signals.get(sym,{}).get(ts,{}).get("close")
                    if c is None: continue
                    pos = positions.pop(sym)
                    raw = (pos["entry"]-c)/pos["entry"]*pos["notional"]
                    net = raw - _TAKER_FEE*pos["notional"]
                    equity += net
                    if in_2025:
                        trades_2025.append({"ts":ts,"date":dt.strftime("%Y-%m-%d"),"sym":sym,"net":net,"side":"SHORT","reason":"rebal_close"})
                        month_pnl[dt.month] += net
                        coin_pnl[sym] += net
                        short_pnl_total += net

            for sym in desired_shorts:
                if sym not in positions:
                    c = signals.get(sym,{}).get(ts,{}).get("close")
                    if c is None: continue
                    n = _position_size(sym, ts)
                    equity -= _TAKER_FEE*n
                    positions[sym] = {"entry":c,"notional":n,"side":"SHORT"}

        # Hard stop on shorts
        for sym in [s for s in list(positions) if positions[s].get("side")=="SHORT"]:
            c = signals.get(sym,{}).get(ts,{}).get("close")
            if c is None: continue
            pos = positions[sym]
            loss_pct = (c-pos["entry"])/pos["entry"]
            if loss_pct >= 0.15:
                positions.pop(sym)
                raw = (pos["entry"]-c)/pos["entry"]*pos["notional"]
                net = raw - _TAKER_FEE*pos["notional"]
                equity += net
                if in_2025:
                    trades_2025.append({"ts":ts,"date":dt.strftime("%Y-%m-%d"),"sym":sym,"net":net,"side":"SHORT","reason":"hard_stop"})
                    month_pnl[dt.month] += net
                    coin_pnl[sym] += net
                    short_pnl_total += net
                    hard_stop_fires.append({"date":dt.strftime("%Y-%m-%d"),"sym":sym,"net":net,"loss_pct":loss_pct})

        # Close shorts if back in bull
        if btc_bull:
            for sym in [s for s in list(positions) if positions[s].get("side")=="SHORT"]:
                c = signals.get(sym,{}).get(ts,{}).get("close")
                if c is None: continue
                pos = positions.pop(sym)
                raw = (pos["entry"]-c)/pos["entry"]*pos["notional"]
                net = raw - _TAKER_FEE*pos["notional"]
                equity += net
                if in_2025:
                    trades_2025.append({"ts":ts,"date":dt.strftime("%Y-%m-%d"),"sym":sym,"net":net,"side":"SHORT","reason":"bull_close"})
                    month_pnl[dt.month] += net
                    coin_pnl[sym] += net
                    short_pnl_total += net

        # Long signals
        for sym in _SYMBOLS:
            sig = signals.get(sym,{}).get(ts,{})
            if not sig: continue
            c = sig["close"]
            n_long = sum([sig.get("ema_long",False), sig.get("macd_long",False), sig.get("h4_long",False)])
            any_long = n_long > 0
            can_long = True
            if not long_allowed:
                can_long = False
            in_pos = sym in positions and positions[sym].get("side")=="LONG"

            if in_pos and (not any_long or not can_long):
                pos = positions.pop(sym)
                raw = (c-pos["entry"])/pos["entry"]*pos["notional"]
                net = raw - _TAKER_FEE*pos["notional"]
                equity += net
                if in_2025:
                    trades_2025.append({"ts":ts,"date":dt.strftime("%Y-%m-%d"),"sym":sym,"net":net,"side":"LONG","reason":"signal_exit"})
                    month_pnl[dt.month] += net
                    coin_pnl[sym] += net
                    long_pnl_total += net
                in_pos = False

            if not in_pos and any_long and can_long:
                m30 = mom30_series.get(sym,{}).get(ts, 0.0)
                if m30 <= 0:
                    if in_2025: mom_gate_blocked[sym] += 1
                    continue
                if in_2025: mom_gate_allowed[sym] += 1
                mult = {1:1.0,2:1.5,3:2.0}.get(n_long,1.0)
                n = _position_size(sym, ts, mult)
                equity -= _TAKER_FEE*n
                positions[sym] = {"entry":c,"notional":n,"side":"LONG","peak_price":c}

    # Close any remaining positions at 2025 year-end
    for sym, pos in list(positions.items()):
        ts_candidates = sorted(t for t in signals.get(sym,{}) if t <= ts_2025_end)
        if not ts_candidates: continue
        last_ts = ts_candidates[-1]
        c = signals[sym][last_ts]["close"]
        in_2025 = ts_2025_start <= last_ts < ts_2025_end
        if pos["side"] == "LONG":
            mtm = (c-pos["entry"])/pos["entry"]*pos["notional"]
            if in_2025:
                dt2 = datetime.fromtimestamp(last_ts/1000, tz=timezone.utc)
                trades_2025.append({"ts":last_ts,"date":dt2.strftime("%Y-%m-%d"),"sym":sym,"net":mtm,"side":"LONG","reason":"eoy_mtm"})
                coin_pnl[sym] += mtm
                month_pnl[dt2.month] += mtm
                long_pnl_total += mtm
        else:
            mtm = (pos["entry"]-c)/pos["entry"]*pos["notional"]
            if in_2025:
                dt2 = datetime.fromtimestamp(last_ts/1000, tz=timezone.utc)
                trades_2025.append({"ts":last_ts,"date":dt2.strftime("%Y-%m-%d"),"sym":sym,"net":mtm,"side":"SHORT","reason":"eoy_mtm"})
                coin_pnl[sym] += mtm
                month_pnl[dt2.month] += mtm
                short_pnl_total += mtm

    total_2025_pnl = long_pnl_total + short_pnl_total
    print(f"\nTotal 2025 P&L: ${total_2025_pnl:+,.2f}")
    print(f"  Long  P&L: ${long_pnl_total:+,.2f}")
    print(f"  Short P&L: ${short_pnl_total:+,.2f}")
    print(f"  Total trades closed in 2025: {len(trades_2025)}")

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 3: MONTHLY BREAKDOWN
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SECTION 3: MONTHLY P&L IN 2025")
    print("=" * 70)
    month_names = ["","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    print(f"\n{'Month':<8} {'P&L':>10}  {'Regime':>15}")
    for m in range(1, 13):
        if m not in month_pnl and m not in month_bull: continue
        pnl = month_pnl.get(m, 0)
        bd = month_bear.get(m, 0)
        bud = month_bull.get(m, 0)
        regime_str = f"bull={bud}d bear={bd}d"
        flag = " ✗" if pnl < 0 else " ✓"
        print(f"  {month_names[m]:<6} {pnl:>+10.2f}{flag}  {regime_str}")

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 4: PER-COIN P&L
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SECTION 4: PER-COIN P&L IN 2025")
    print("=" * 70)
    sorted_coins = sorted(coin_pnl.items(), key=lambda x: x[1])
    print(f"\n{'Coin':<12} {'P&L':>10}  {'# trades':>9}")
    coin_trades = defaultdict(int)
    for t in trades_2025:
        coin_trades[t["sym"]] += 1
    for sym, pnl in sorted_coins:
        flag = " ✗" if pnl < 0 else " ✓"
        print(f"  {sym:<12} {pnl:>+10.2f}{flag}  {coin_trades[sym]:>6}")

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 5: HARD STOP ANALYSIS
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SECTION 5: HARD STOP (15%) FIRES IN 2025")
    print("=" * 70)
    print(f"\nTotal hard stop fires: {len(hard_stop_fires)}")
    total_hs_loss = sum(h["net"] for h in hard_stop_fires)
    print(f"Total loss from hard stops: ${total_hs_loss:+,.2f}")
    for h in hard_stop_fires:
        print(f"  {h['date']} {h['sym']:<12} loss_pct={h['loss_pct']*100:.1f}%  net=${h['net']:+.2f}")

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 6: MOMENTUM GATE IMPACT
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SECTION 6: MOMENTUM GATE IMPACT IN 2025")
    print("=" * 70)
    total_blocked = sum(mom_gate_blocked.values())
    total_allowed = sum(mom_gate_allowed.values())
    print(f"\nMomentum gate blocked entries: {total_blocked}")
    print(f"Momentum gate allowed entries: {total_allowed}")
    if total_blocked + total_allowed > 0:
        print(f"Block rate: {total_blocked/(total_blocked+total_allowed)*100:.1f}%")
    if mom_gate_blocked:
        print("\nPer-coin blocked entries:")
        for sym, cnt in sorted(mom_gate_blocked.items(), key=lambda x:-x[1]):
            print(f"  {sym:<12} blocked={cnt}  allowed={mom_gate_allowed.get(sym,0)}")

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 7: SHORT TRADE BREAKDOWN
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SECTION 7: SHORT TRADES IN 2025 — DETAIL")
    print("=" * 70)
    short_trades = [t for t in trades_2025 if t["side"]=="SHORT"]
    long_trades  = [t for t in trades_2025 if t["side"]=="LONG"]
    print(f"\nShort trades closed: {len(short_trades)}  P&L: ${sum(t['net'] for t in short_trades):+,.2f}")
    print(f"Long  trades closed: {len(long_trades)}   P&L: ${sum(t['net'] for t in long_trades):+,.2f}")

    if short_trades:
        print("\nAll short trade exits in 2025:")
        for t in sorted(short_trades, key=lambda x: x["net"]):
            print(f"  {t['date']} {t['sym']:<12} {t['reason']:<14} net=${t['net']:+.2f}")

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 8: WHAT IF NO SHORTS — LONGS-ONLY IN 2025
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SECTION 8: COUNTERFACTUAL — LONGS-ONLY 2025 P&L")
    print("=" * 70)
    longs_only_pnl = sum(t["net"] for t in trades_2025 if t["side"]=="LONG")
    shorts_only_pnl = sum(t["net"] for t in trades_2025 if t["side"]=="SHORT")
    print(f"\nIf longs only: ${longs_only_pnl:+,.2f}")
    print(f"Shorts contributed: ${shorts_only_pnl:+,.2f}")

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 9: BTC PRICE CONTEXT — BIG CRASHES & CHOPPY PERIODS
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SECTION 9: BTC PRICE CONTEXT — CRASHES & VOLATILITY IN 2025")
    print("=" * 70)
    sorted_2025_ts = sorted(regime_2025.keys())
    prev_c = None
    big_moves = []
    for ts in sorted_2025_ts:
        c = regime_2025[ts]["btc_close"]
        if prev_c:
            pct = (c-prev_c)/prev_c
            if abs(pct) >= 0.04:
                big_moves.append({"date": regime_2025[ts]["date"], "pct": pct, "price": c})
        prev_c = c

    print(f"\nBig single-day moves (>=4%) in BTC during 2025: {len(big_moves)}")
    neg_moves = [m for m in big_moves if m["pct"] < 0]
    pos_moves = [m for m in big_moves if m["pct"] > 0]
    print(f"  Big drops: {len(neg_moves)}")
    print(f"  Big pumps: {len(pos_moves)}")
    print("\nTop 10 worst BTC days in 2025:")
    for m in sorted(neg_moves, key=lambda x: x["pct"])[:10]:
        print(f"  {m['date']}  {m['pct']*100:+.1f}%  BTC=${m['price']:,.0f}")

    print("\nTop 10 best BTC days in 2025:")
    for m in sorted(pos_moves, key=lambda x: -x["pct"])[:10]:
        print(f"  {m['date']}  {m['pct']*100:+.1f}%  BTC=${m['price']:,.0f}")

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 10: KEY TRADES DETAIL — BIGGEST LOSERS
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SECTION 10: BIGGEST LOSING TRADES IN 2025")
    print("=" * 70)
    sorted_trades = sorted(trades_2025, key=lambda x: x["net"])
    print(f"\nTop 15 worst individual trade exits:")
    for t in sorted_trades[:15]:
        print(f"  {t['date']} {t['sym']:<12} {t['side']:<6} {t['reason']:<14} net=${t['net']:+.2f}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTotal 2025 P&L on $5K capital: ${total_2025_pnl:+,.2f}")
    print(f"  From longs:  ${long_pnl_total:+,.2f}")
    print(f"  From shorts: ${short_pnl_total:+,.2f}")
    print(f"  Hard stop losses: ${total_hs_loss:+,.2f}")
    print(f"  Regime: {bull_days}d bull / {bear_days}d bear, {len(flips)} flips")

if __name__ == "__main__":
    main()
