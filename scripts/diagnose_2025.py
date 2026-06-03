#!/usr/bin/env python3
"""Diagnostic script: Why does 2025 lose in NexFlow V8?"""

from __future__ import annotations
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import pyarrow.parquet as pq

_CANDLE_DIR = _REPO_ROOT / "data" / "candles"
_TAKER_FEE  = 0.0006
_CAPITAL    = 5_000.0   # $5K as stated
_SYMBOLS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT",
    "XRPUSDT","ADAUSDT","DOGEUSDT","AVAXUSDT",
    "LINKUSDT","LTCUSDT","DOTUSDT","TRXUSDT",
]
_DAY_MS = 86_400_000

_2025_START = int(datetime(2025,1,1,tzinfo=timezone.utc).timestamp()*1000)
_2026_START = int(datetime(2026,1,1,tzinfo=timezone.utc).timestamp()*1000)


def _load_daily(symbol: str) -> list[dict]:
    path = _CANDLE_DIR / f"{symbol}_1D.parquet"
    if not path.exists(): return []
    tbl = pq.read_table(path, columns=["open_time","close","high","low"])
    rows = []
    for ts, c, h, l in zip(
        tbl.column("open_time").to_pylist(),
        tbl.column("close").to_pylist(),
        tbl.column("high").to_pylist(),
        tbl.column("low").to_pylist(),
    ):
        rows.append({"ts": int(ts), "close": float(c), "high": float(h), "low": float(l)})
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
                "close": closes[i],
                "ema_long": ema_long_state,
                "macd_long": macd_long[i],
                "h4_long": h4_long_state,
                "sma200_above": sma200[i] is not None and closes[i] > sma200[i],
                "sma200": sma200[i],
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
        rets = [(closes[j]-closes[j-1])/closes[j-1] for j in range(i-window+1,i+1)]
        mean = sum(rets)/len(rets)
        result[ts] = (sum((r-mean)**2 for r in rets)/len(rets))**0.5
    return result


def run_v8_diagnostic(signals):
    """Run V8 with full per-trade diagnostics for 2025."""
    from_ts = int(datetime(2021,1,1,tzinfo=timezone.utc).timestamp()*1000)
    to_ts   = int(datetime.now(timezone.utc).timestamp()*1000)

    # V8 params
    use_sma200_long_filter = True
    use_tsmom_short = True
    confluence = True
    hard_stop_pct = 0.15
    use_atr_sizing = True
    asymmetric_regime = True
    and_entry = True
    bear_drop_pct = -0.20
    confirm_days = 5
    momentum_gate = True
    momentum_gate_days = 30
    target_risk = 0.01

    base_notional = _CAPITAL / len(_SYMBOLS)

    vol_series = {sym: _vol_series(signals, sym) for sym in _SYMBOLS}

    # Precompute momentum gate
    mom30_series = {}
    for sym in _SYMBOLS:
        ts_list = sorted(signals.get(sym, {}).keys())
        closes = [signals[sym][t]["close"] for t in ts_list]
        m = {}
        for i, t in enumerate(ts_list):
            m[t] = (closes[i]-closes[i-momentum_gate_days])/closes[i-momentum_gate_days] if i >= momentum_gate_days else 0.0
        mom30_series[sym] = m

    # BTC 30d return + SMA50 for asymmetric regime
    btc_mom30 = {}
    btc_ts_list = sorted(signals.get("BTCUSDT", {}).keys())
    btc_closes = [signals["BTCUSDT"][t]["close"] for t in btc_ts_list]
    for i, t in enumerate(btc_ts_list):
        btc_mom30[t] = (btc_closes[i]-btc_closes[i-30])/btc_closes[i-30] if i >= 30 else 0.0

    all_ts = sorted(set(ts for sym in _SYMBOLS for ts in signals.get(sym,{}) if from_ts<=ts<=to_ts))

    equity = _CAPITAL
    positions = {}
    trades = []  # full trade log
    prev_btc_bear_mode = False
    btc_above_sma200_streak = 0
    regime_log = []  # track daily regime for 2025

    # Track hard stop events
    hard_stops = []
    month_pnl = {}  # (year,month) -> pnl

    def _position_size(sym, ts, mult=1.0):
        n = base_notional * mult
        if not use_atr_sizing: return n
        vol = vol_series.get(sym, {}).get(ts, 0.0)
        if vol <= 0: return n
        vol_sized = (target_risk * _CAPITAL) / vol
        return min(vol_sized * mult, base_notional * 2 * mult)

    for ts in all_ts:
        btc_sig = signals.get("BTCUSDT",{}).get(ts,{})
        btc_sma200_above = btc_sig.get("sma200_above", True)
        btc_30d_ret = btc_mom30.get(ts, 0.0)
        drop_triggered = btc_30d_ret < bear_drop_pct

        enter_bear = (not btc_sma200_above) and drop_triggered  # AND mode

        if btc_sma200_above:
            btc_above_sma200_streak += 1
        else:
            btc_above_sma200_streak = 0
        confirmed_bull = btc_above_sma200_streak >= max(1, confirm_days)

        if prev_btc_bear_mode:
            btc_bear_mode = not confirmed_bull
        else:
            btc_bear_mode = enter_bear
        prev_btc_bear_mode = btc_bear_mode
        btc_bull = not btc_bear_mode

        long_allowed = btc_bull

        yr = datetime.fromtimestamp(ts/1000, tz=timezone.utc).year
        mo = datetime.fromtimestamp(ts/1000, tz=timezone.utc).month

        # Log regime for 2025
        if _2025_START <= ts < _2026_START:
            regime_log.append({
                "ts": ts,
                "date": datetime.fromtimestamp(ts/1000, tz=timezone.utc).strftime("%Y-%m-%d"),
                "btc_bull": btc_bull,
                "btc_sma200_above": btc_sma200_above,
                "btc_30d_ret": btc_30d_ret,
                "drop_triggered": drop_triggered,
                "btc_close": btc_sig.get("close", 0),
            })

        last_rebal_ts = getattr(run_v8_diagnostic, '_last_rebal', 0)

        # TSMOM short rebalance
        if use_tsmom_short and not btc_bull and (ts - last_rebal_ts) >= 7*_DAY_MS:
            run_v8_diagnostic._last_rebal = ts
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
            desired_shorts = {sym for ret,sym in scores if ret < -0.05}

            for sym in [s for s in list(positions) if positions[s].get("side")=="SHORT"]:
                if sym not in desired_shorts:
                    c = signals.get(sym,{}).get(ts,{}).get("close")
                    if c is None: continue
                    pos = positions.pop(sym)
                    raw = (pos["entry"]-c)/pos["entry"]*pos["notional"]
                    net = raw - _TAKER_FEE*pos["notional"]
                    equity += net
                    trades.append({"ts":ts,"sym":sym,"net":net,"side":"SHORT","exit_reason":"rebalance"})
                    key = (yr,mo)
                    month_pnl[key] = month_pnl.get(key,0)+net

            for sym in desired_shorts:
                if sym not in positions:
                    c = signals.get(sym,{}).get(ts,{}).get("close")
                    if c is None: continue
                    n = _position_size(sym, ts)
                    equity -= _TAKER_FEE * n
                    positions[sym] = {"entry":c,"notional":n,"side":"SHORT"}

        # Hard stop on shorts
        if hard_stop_pct > 0 and use_tsmom_short:
            for sym in [s for s in list(positions) if positions[s].get("side")=="SHORT"]:
                c = signals.get(sym,{}).get(ts,{}).get("close")
                if c is None: continue
                pos = positions[sym]
                loss_pct = (c - pos["entry"]) / pos["entry"]
                if loss_pct >= hard_stop_pct:
                    positions.pop(sym)
                    raw = (pos["entry"]-c)/pos["entry"]*pos["notional"]
                    net = raw - _TAKER_FEE*pos["notional"]
                    equity += net
                    trades.append({"ts":ts,"sym":sym,"net":net,"side":"SHORT","exit_reason":"hard_stop"})
                    key = (yr,mo)
                    month_pnl[key] = month_pnl.get(key,0)+net
                    if _2025_START <= ts < _2026_START:
                        hard_stops.append({
                            "date": datetime.fromtimestamp(ts/1000,tz=timezone.utc).strftime("%Y-%m-%d"),
                            "sym": sym,
                            "entry": pos["entry"],
                            "exit": c,
                            "loss_pct": loss_pct*100,
                            "net": net,
                        })

        # Close shorts on bull regime
        if btc_bull and use_tsmom_short:
            for sym in [s for s in list(positions) if positions[s].get("side")=="SHORT"]:
                c = signals.get(sym,{}).get(ts,{}).get("close")
                if c is None: continue
                pos = positions.pop(sym)
                raw = (pos["entry"]-c)/pos["entry"]*pos["notional"]
                net = raw - _TAKER_FEE*pos["notional"]
                equity += net
                trades.append({"ts":ts,"sym":sym,"net":net,"side":"SHORT","exit_reason":"regime_flip"})
                key = (yr,mo)
                month_pnl[key] = month_pnl.get(key,0)+net

        # Long signals
        for sym in _SYMBOLS:
            sig = signals.get(sym,{}).get(ts,{})
            if not sig: continue
            c = sig["close"]
            n_long = sum([sig.get("ema_long",False), sig.get("macd_long",False), sig.get("h4_long",False)])
            any_long = n_long > 0

            can_long = True
            if use_sma200_long_filter and not long_allowed:
                can_long = False
            if momentum_gate and not sig.get("sma200_above", True):
                m30 = mom30_series.get(sym, {}).get(ts, 0.0)
                if m30 <= 0:
                    can_long = False

            in_pos = sym in positions and positions[sym].get("side") == "LONG"

            if in_pos and (not any_long or not can_long):
                pos = positions.pop(sym)
                raw = (c-pos["entry"])/pos["entry"]*pos["notional"]
                net = raw - _TAKER_FEE*pos["notional"]
                equity += net
                trades.append({"ts":ts,"sym":sym,"net":net,"side":"LONG","exit_reason":"signal"})
                key = (yr,mo)
                month_pnl[key] = month_pnl.get(key,0)+net
                in_pos = False

            if not in_pos and any_long and can_long:
                if momentum_gate:
                    m30 = mom30_series.get(sym, {}).get(ts, 0.0)
                    if m30 <= 0:
                        continue
                mult = {1:1.0,2:1.5,3:2.0}.get(n_long,1.0) if confluence else 1.0
                n = _position_size(sym, ts, mult)
                equity -= _TAKER_FEE * n
                positions[sym] = {"entry":c,"notional":n,"side":"LONG"}

    # Close open positions at last price
    for sym, pos in list(positions.items()):
        ts_list = sorted(t for t in signals.get(sym,{}) if t <= to_ts)
        if not ts_list: continue
        last_ts = ts_list[-1]
        p = signals[sym][last_ts]["close"]
        yr2 = datetime.fromtimestamp(last_ts/1000, tz=timezone.utc).year
        mo2 = datetime.fromtimestamp(last_ts/1000, tz=timezone.utc).month
        if pos["side"] == "LONG":
            mtm = (p-pos["entry"])/pos["entry"]*pos["notional"]
        else:
            mtm = (pos["entry"]-p)/pos["entry"]*pos["notional"]
        equity += mtm
        trades.append({"ts":last_ts,"sym":sym,"net":mtm,"side":pos["side"],"exit_reason":"eod"})
        key = (yr2, mo2)
        month_pnl[key] = month_pnl.get(key,0)+mtm

    return trades, regime_log, hard_stops, month_pnl


def main():
    print("Building signals...")
    signals = _build_signals(_SYMBOLS)
    print("Done.\n")

    trades, regime_log, hard_stops, month_pnl = run_v8_diagnostic(signals)

    # Filter trades to 2025
    trades_2025 = [t for t in trades if _2025_START <= t["ts"] < _2026_START]

    print("="*70)
    print("  NexFlow V8 — 2025 ROOT CAUSE DIAGNOSTICS  ($5K capital)")
    print("="*70)

    # ─── Q1: BTC regime in 2025 ───────────────────────────────────────────
    print("\n── Q1: BTC Regime in 2025 ──")
    bull_days = sum(1 for r in regime_log if r["btc_bull"])
    bear_days = sum(1 for r in regime_log if not r["btc_bull"])
    total_days = len(regime_log)
    print(f"  Total trading days: {total_days}")
    print(f"  BULL days: {bull_days}  ({100*bull_days/total_days:.1f}%)")
    print(f"  BEAR days: {bear_days}  ({100*bear_days/total_days:.1f}%)")

    # Detect regime flips
    flips = []
    for i in range(1, len(regime_log)):
        prev = regime_log[i-1]["btc_bull"]
        curr = regime_log[i]["btc_bull"]
        if prev != curr:
            flips.append({"date": regime_log[i]["date"], "to": "BULL" if curr else "BEAR",
                          "btc_close": regime_log[i]["btc_close"],
                          "btc_30d_ret": regime_log[i]["btc_30d_ret"]*100})
    print(f"\n  Regime flips in 2025: {len(flips)}")
    for f in flips:
        print(f"    {f['date']} → {f['to']}  BTC=${f['btc_close']:,.0f}  30d_ret={f['btc_30d_ret']:+.1f}%")

    # Whipsawing: count flip pairs within 30 days
    whips = 0
    for i in range(1, len(flips)):
        d1 = datetime.strptime(flips[i-1]["date"], "%Y-%m-%d")
        d2 = datetime.strptime(flips[i]["date"], "%Y-%m-%d")
        if (d2-d1).days <= 30:
            whips += 1
    print(f"  Whipsaw pairs (regime flip within 30d): {whips}")

    # Monthly regime breakdown
    print("\n  Monthly regime (BULL/BEAR days per month):")
    from collections import defaultdict
    monthly_regime = defaultdict(lambda: {"bull":0,"bear":0})
    for r in regime_log:
        mo = datetime.fromtimestamp(r["ts"]/1000, tz=timezone.utc).month
        if r["btc_bull"]: monthly_regime[mo]["bull"] += 1
        else: monthly_regime[mo]["bear"] += 1
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    for mo in sorted(monthly_regime.keys()):
        b = monthly_regime[mo]["bull"]
        be = monthly_regime[mo]["bear"]
        print(f"    {month_names[mo-1]}: {b} bull / {be} bear")

    # ─── Q2: Losses from longs or shorts? ────────────────────────────────
    print("\n── Q2: Long vs Short P&L in 2025 ──")
    long_pnl = sum(t["net"] for t in trades_2025 if t["side"]=="LONG")
    short_pnl = sum(t["net"] for t in trades_2025 if t["side"]=="SHORT")
    long_trades = [t for t in trades_2025 if t["side"]=="LONG"]
    short_trades = [t for t in trades_2025 if t["side"]=="SHORT"]
    print(f"  LONG  P&L: ${long_pnl:+,.2f}  ({len(long_trades)} trades)")
    print(f"  SHORT P&L: ${short_pnl:+,.2f}  ({len(short_trades)} trades)")
    total_2025 = long_pnl + short_pnl
    print(f"  TOTAL 2025: ${total_2025:+,.2f}")

    # ─── Q3: Worst months ─────────────────────────────────────────────────
    print("\n── Q3: Monthly P&L in 2025 ──")
    months_2025 = [(mo, month_pnl.get((2025, mo), 0.0)) for mo in range(1,13)]
    for mo, pnl in months_2025:
        bar = "█"*int(abs(pnl)/5) if abs(pnl) > 0 else ""
        flag = "▼ WORST" if pnl == min(p for _,p in months_2025) else ""
        print(f"  {month_names[mo-1]:3s}: ${pnl:>+9,.2f}  {flag}")

    worst_mo = min(months_2025, key=lambda x: x[1])
    print(f"\n  Worst month: {month_names[worst_mo[0]-1]} 2025  (${worst_mo[1]:+,.2f})")

    # ─── Q4: BTC price action in 2025 ────────────────────────────────────
    print("\n── Q4: BTC Price Action in 2025 ──")
    btc_2025 = [(r["date"], r["btc_close"]) for r in regime_log]
    if btc_2025:
        print(f"  Jan 1 open: ${btc_2025[0][1]:,.0f}")
        print(f"  Latest:     ${btc_2025[-1][1]:,.0f}  ({btc_2025[-1][0]})")
        max_p = max(btc_2025, key=lambda x: x[1])
        min_p = min(btc_2025, key=lambda x: x[1])
        print(f"  YTD high:   ${max_p[1]:,.0f}  on {max_p[0]}")
        print(f"  YTD low:    ${min_p[1]:,.0f}  on {min_p[0]}")
        ath_to_low = (max_p[1]-min_p[1])/max_p[1]*100
        print(f"  Max drawdown from YTD high: {ath_to_low:.1f}%")

    # Monthly BTC return
    print("\n  Monthly BTC returns:")
    for mo in range(1,13):
        days_in_mo = [(r["date"], r["btc_close"]) for r in regime_log
                      if datetime.fromtimestamp(int([rr["ts"] for rr in regime_log if rr["date"]==r["date"]][0])/1000,tz=timezone.utc).month == mo]
        if len(days_in_mo) >= 2:
            ret = (days_in_mo[-1][1]-days_in_mo[0][1])/days_in_mo[0][1]*100
            print(f"    {month_names[mo-1]}: {ret:+.1f}%  ({days_in_mo[0][1]:,.0f} → {days_in_mo[-1][1]:,.0f})")

    # ─── Q5: Hard stop events in 2025 ─────────────────────────────────────
    print("\n── Q5: Hard Stop (15%) Triggers in 2025 ──")
    print(f"  Total hard stops: {len(hard_stops)}")
    total_hard_stop_loss = sum(h["net"] for h in hard_stops)
    print(f"  Total loss from hard stops: ${total_hard_stop_loss:+,.2f}")
    for h in sorted(hard_stops, key=lambda x: x["net"]):
        print(f"    {h['date']}  {h['sym']:12s}  entry=${h['entry']:>10,.2f}  exit=${h['exit']:>10,.2f}  "
              f"loss={h['loss_pct']:+.1f}%  net=${h['net']:+,.2f}")

    # ─── Q6: Which coins lost most in 2025 ────────────────────────────────
    print("\n── Q6: Per-Coin P&L in 2025 ──")
    coin_pnl = {}
    coin_trades = {}
    for t in trades_2025:
        sym = t["sym"]
        coin_pnl[sym] = coin_pnl.get(sym, 0) + t["net"]
        coin_trades[sym] = coin_trades.get(sym, 0) + 1
    for sym, pnl in sorted(coin_pnl.items(), key=lambda x: x[1]):
        flag = "  ← WORST" if pnl == min(coin_pnl.values()) else ""
        print(f"  {sym:12s}: ${pnl:>+9,.2f}  ({coin_trades.get(sym,0)} trades){flag}")

    # ─── Q7: Momentum gate impact ─────────────────────────────────────────
    print("\n── Q7: Momentum Gate Analysis ──")
    # Count how many long entries were BLOCKED by momentum gate in 2025
    # We need to re-scan and count gate blocks
    mom_gate_blocks = 0
    mom_gate_block_details = []
    for sym in _SYMBOLS:
        ts_list = sorted(signals.get(sym, {}).keys())
        closes = [signals[sym][t]["close"] for t in ts_list]
        mom30 = {}
        for i, t in enumerate(ts_list):
            mom30[t] = (closes[i]-closes[i-30])/closes[i-30] if i >= 30 else 0.0

        for t in ts_list:
            if not (_2025_START <= t < _2026_START): continue
            sig = signals[sym][t]
            n_long = sum([sig.get("ema_long",False), sig.get("macd_long",False), sig.get("h4_long",False)])
            if n_long > 0:  # would have entered
                m = mom30.get(t, 0.0)
                if m <= 0:
                    mom_gate_blocks += 1
                    if len(mom_gate_block_details) < 5:
                        mom_gate_block_details.append({
                            "date": datetime.fromtimestamp(t/1000,tz=timezone.utc).strftime("%Y-%m-%d"),
                            "sym": sym, "mom30": m*100
                        })

    print(f"  Long entries BLOCKED by momentum gate in 2025: {mom_gate_blocks}")
    print(f"  (These had n_long>0 signal but 30d return <= 0)")
    for d in mom_gate_block_details:
        print(f"    {d['date']}  {d['sym']:12s}  30d_ret={d['mom30']:+.1f}%")

    # Also check: what happened to SHORT trades by exit reason
    print("\n  Short trade exit reasons in 2025:")
    from collections import Counter
    exit_reasons = Counter(t.get("exit_reason","?") for t in trades_2025 if t["side"]=="SHORT")
    for reason, cnt in exit_reasons.most_common():
        pnl_by_reason = sum(t["net"] for t in trades_2025 if t["side"]=="SHORT" and t.get("exit_reason")==reason)
        print(f"    {reason:20s}: {cnt:3d} trades  ${pnl_by_reason:+,.2f}")

    # ─── Summary ──────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  SUMMARY OF ROOT CAUSES")
    print("="*70)
    print(f"  Total 2025 P&L: ${total_2025:+,.2f}")
    print(f"  Long contribution:  ${long_pnl:+,.2f}")
    print(f"  Short contribution: ${short_pnl:+,.2f}")
    print(f"  Hard stop losses:   ${total_hard_stop_loss:+,.2f}  ({len(hard_stops)} events)")
    print(f"  Regime whipsaws:    {whips} pairs  ({len(flips)} total flips)")
    print()


if __name__ == "__main__":
    main()
