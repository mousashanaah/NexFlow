#!/usr/bin/env python3
"""Diagnostic script: why does V8 lose in 2023?"""
from __future__ import annotations
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import pyarrow.parquet as pq

_CANDLE_DIR = _REPO_ROOT / "data" / "candles"
_DAY_MS = 86_400_000
_TAKER_FEE = 0.0006
_CAPITAL = 5_000.0
_SYMBOLS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT",
    "XRPUSDT","ADAUSDT","DOGEUSDT","AVAXUSDT",
    "LINKUSDT","LTCUSDT","DOTUSDT","TRXUSDT",
]

_2023_START = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
_2023_END   = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)


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


def main():
    print("=" * 70)
    print("  V8 2023 LOSS DIAGNOSTIC  (capital=$5,000)")
    print("=" * 70)

    # ── 1. BTC regime ──
    btc_bars = _load_daily("BTCUSDT")
    btc_ts   = [b["ts"] for b in btc_bars]
    btc_cl   = [b["close"] for b in btc_bars]
    btc_sma200 = _sma_series(btc_cl, 200)

    btc_mom30: dict[int, float] = {}
    for i, t in enumerate(btc_ts):
        btc_mom30[t] = (btc_cl[i] - btc_cl[i-30]) / btc_cl[i-30] if i >= 30 else 0.0

    bear_drop_pct = -0.20
    confirm_days  = 5
    prev_bear = False
    streak = 0
    regime_by_ts: dict[int, bool] = {}

    for i, t in enumerate(btc_ts):
        sma200_v = btc_sma200[i]
        btc_above_sma200 = sma200_v is not None and btc_cl[i] > sma200_v
        drop_triggered = btc_mom30.get(t, 0.0) < bear_drop_pct
        enter_bear = (not btc_above_sma200) and drop_triggered
        if btc_above_sma200:
            streak += 1
        else:
            streak = 0
        confirmed_bull = streak >= max(1, confirm_days)
        if prev_bear:
            bear_mode = not confirmed_bull
        else:
            bear_mode = enter_bear
        prev_bear = bear_mode
        regime_by_ts[t] = bear_mode

    bear_days_2023 = 0; bull_days_2023 = 0
    bear_dates = []
    for t, is_bear in regime_by_ts.items():
        if _2023_START <= t < _2023_END:
            if is_bear:
                bear_days_2023 += 1
                bear_dates.append(datetime.fromtimestamp(t/1000, tz=timezone.utc).strftime("%Y-%m-%d"))
            else:
                bull_days_2023 += 1

    print(f"\n── 1. BTC Regime in 2023 (V8 AND-entry asymmetric) ──")
    print(f"  Bull days: {bull_days_2023}")
    print(f"  Bear days: {bear_days_2023}")
    btc_jan1  = btc_cl[btc_ts.index(min(t for t in btc_ts if t >= _2023_START))]
    btc_dec31 = btc_cl[btc_ts.index(max(t for t in btc_ts if t < _2023_END))]
    print(f"  BTC Jan-01-2023: ${btc_jan1:,.0f}")
    print(f"  BTC Dec-31-2023: ${btc_dec31:,.0f}")
    print(f"  BTC 2023 return: {(btc_dec31/btc_jan1 - 1)*100:.1f}%")

    # ── 2. Regime flips ──
    print(f"\n── 2. AND-entry regime flip analysis in 2023 ──")
    flips = []
    prev_state = None
    for t in sorted(regime_by_ts):
        if _2023_START <= t < _2023_END:
            s = regime_by_ts[t]
            if prev_state is not None and s != prev_state:
                d = datetime.fromtimestamp(t/1000, tz=timezone.utc).strftime("%Y-%m-%d")
                flips.append((d, "BEAR" if s else "BULL"))
            prev_state = s
    print(f"  Regime flips in 2023: {len(flips)}")
    for d, label in flips:
        print(f"    {d}  -> {label}")
    if bear_dates:
        print(f"  First bear day: {bear_dates[0]}")
        print(f"  Last bear day:  {bear_dates[-1]}")

    # ── Build signals ──
    signals: dict[str, dict] = {}
    mom30_series: dict[str, dict[int, float]] = {}

    for sym in _SYMBOLS:
        bars = _load_daily(sym)
        if not bars: continue
        closes = [b["close"] for b in bars]
        ts_list = [b["ts"] for b in bars]
        ef8  = _ema_series(closes, 8)
        ef21 = _ema_series(closes, 21)
        sma200 = _sma_series(closes, 200)
        macd_long = _macd_long_series(closes)
        ef5  = _ema_series(closes, 5)
        ef13 = _ema_series(closes, 13)

        by_ts = {}
        ema_long_state = False; h4_long_state = False
        prev_ema_above = None; prev_h4_above = None

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
                "sma200":      sma200[i],
            }
        signals[sym] = by_ts

        m30: dict[int, float] = {}
        for i, t in enumerate(ts_list):
            m30[t] = (closes[i] - closes[i-30]) / closes[i-30] if i >= 30 else 0.0
        mom30_series[sym] = m30

    # Vol series
    vol_series: dict[str, dict[int, float]] = {}
    for sym in _SYMBOLS:
        ts_list2 = sorted(signals.get(sym, {}).keys())
        closes2 = [signals[sym][t]["close"] for t in ts_list2]
        result2: dict[int, float] = {}
        for i, t in enumerate(ts_list2):
            if i < 14:
                result2[t] = 0.0
                continue
            rets = [(closes2[j] - closes2[j-1]) / closes2[j-1] for j in range(i-14+1, i+1)]
            mean = sum(rets) / len(rets)
            result2[t] = (sum((r-mean)**2 for r in rets) / len(rets))**0.5
        vol_series[sym] = result2

    base_notional = _CAPITAL / len(_SYMBOLS)
    target_risk = 0.01

    def pos_size(sym, ts, mult=1.0):
        n = base_notional * mult
        vol = vol_series.get(sym, {}).get(ts, 0.0)
        if vol <= 0: return n
        vol_sized = (target_risk * _CAPITAL) / vol
        return min(vol_sized * mult, base_notional * 2 * mult)

    all_ts_global = sorted(set(ts for sym in _SYMBOLS for ts in signals.get(sym, {})))

    equity = _CAPITAL
    positions: dict[str, dict] = {}
    trades_2023: list[dict] = []
    last_rebal_ts = 0
    prev_bear2 = False
    streak2 = 0
    gate_blocked: dict[str, int] = {s: 0 for s in _SYMBOLS}

    # Track when BTC first crosses above SMA200 in 2023
    btc_sma200_crossovers = []
    prev_btc_above = None

    for ts in all_ts_global:
        btc_sig = signals.get("BTCUSDT", {}).get(ts, {})
        btc_sma200_above = btc_sig.get("sma200_above", True)
        drop_triggered = btc_mom30.get(ts, 0.0) < -0.20

        # Track BTC SMA200 crossovers in 2023
        if _2023_START <= ts < _2023_END:
            if prev_btc_above is not None and btc_sma200_above != prev_btc_above:
                d = datetime.fromtimestamp(ts/1000, tz=timezone.utc).strftime("%Y-%m-%d")
                btc_price = btc_sig.get("close", 0)
                btc_sma200_crossovers.append((d, "above" if btc_sma200_above else "below", btc_price))
        prev_btc_above = btc_sma200_above

        enter_bear = (not btc_sma200_above) and drop_triggered
        if btc_sma200_above:
            streak2 += 1
        else:
            streak2 = 0
        confirmed_bull = streak2 >= max(1, confirm_days)
        if prev_bear2:
            bear_mode = not confirmed_bull
        else:
            bear_mode = enter_bear
        prev_bear2 = bear_mode
        btc_bull = not bear_mode

        # TSMOM short rebalance
        if not btc_bull and (ts - last_rebal_ts) >= 7 * _DAY_MS:
            last_rebal_ts = ts
            scores = []
            for sym in _SYMBOLS:
                sym_ts_list = sorted(signals.get(sym, {}).keys())
                past_ts = [t for t in sym_ts_list if t <= ts]
                if len(past_ts) < 127: continue
                c_now  = signals[sym][past_ts[-1]]["close"]
                c_past = signals[sym][past_ts[-127]]["close"]
                ret = (c_now - c_past) / c_past
                scores.append((ret, sym))
            scores.sort()
            desired_shorts = {sym for ret, sym in scores if ret < -0.05}

            for sym in [s for s in list(positions) if positions[s].get("side") == "SHORT"]:
                if sym not in desired_shorts:
                    c = signals.get(sym, {}).get(ts, {}).get("close")
                    if c is None: continue
                    pos = positions.pop(sym)
                    raw = (pos["entry"] - c) / pos["entry"] * pos["notional"]
                    net = raw - _TAKER_FEE * pos["notional"]
                    equity += net
                    if _2023_START <= ts < _2023_END:
                        trades_2023.append({"ts": ts, "sym": sym, "net": net, "side": "SHORT",
                                            "entry": pos["entry"], "exit": c})

            for sym in desired_shorts:
                if sym not in positions:
                    c = signals.get(sym, {}).get(ts, {}).get("close")
                    if c is None: continue
                    n = pos_size(sym, ts)
                    equity -= _TAKER_FEE * n
                    positions[sym] = {"entry": c, "notional": n, "side": "SHORT"}

        # Hard stop shorts
        for sym in [s for s in list(positions) if positions[s].get("side") == "SHORT"]:
            c = signals.get(sym, {}).get(ts, {}).get("close")
            if c is None: continue
            pos = positions[sym]
            loss_pct = (c - pos["entry"]) / pos["entry"]
            if loss_pct >= 0.15:
                positions.pop(sym)
                raw = (pos["entry"] - c) / pos["entry"] * pos["notional"]
                net = raw - _TAKER_FEE * pos["notional"]
                equity += net
                if _2023_START <= ts < _2023_END:
                    trades_2023.append({"ts": ts, "sym": sym, "net": net, "side": "SHORT_STOP",
                                        "entry": pos["entry"], "exit": c})

        # Close shorts on bull flip
        if btc_bull:
            for sym in [s for s in list(positions) if positions[s].get("side") == "SHORT"]:
                c = signals.get(sym, {}).get(ts, {}).get("close")
                if c is None: continue
                pos = positions.pop(sym)
                raw = (pos["entry"] - c) / pos["entry"] * pos["notional"]
                net = raw - _TAKER_FEE * pos["notional"]
                equity += net
                if _2023_START <= ts < _2023_END:
                    trades_2023.append({"ts": ts, "sym": sym, "net": net, "side": "SHORT_CLOSED_BULL",
                                        "entry": pos["entry"], "exit": c})

        # Long signals
        for sym in _SYMBOLS:
            sig = signals.get(sym, {}).get(ts, {})
            if not sig: continue
            c = sig["close"]
            n_long = sum([sig.get("ema_long", False), sig.get("macd_long", False), sig.get("h4_long", False)])
            any_long = n_long > 0
            can_long = btc_bull
            in_pos = sym in positions and positions[sym].get("side") == "LONG"

            if in_pos and (not any_long or not can_long):
                pos = positions.pop(sym)
                raw = (c - pos["entry"]) / pos["entry"] * pos["notional"]
                net = raw - _TAKER_FEE * pos["notional"]
                equity += net
                if _2023_START <= ts < _2023_END:
                    trades_2023.append({"ts": ts, "sym": sym, "net": net, "side": "LONG",
                                        "entry": pos["entry"], "exit": c})
                in_pos = False

            if not in_pos and any_long and can_long:
                m30 = mom30_series.get(sym, {}).get(ts, 0.0)
                if m30 <= 0:
                    if _2023_START <= ts < _2023_END:
                        gate_blocked[sym] += 1
                    continue
                mult = {1: 1.0, 2: 1.5, 3: 2.0}.get(n_long, 1.0)
                n = pos_size(sym, ts, mult)
                equity -= _TAKER_FEE * n
                positions[sym] = {"entry": c, "notional": n, "side": "LONG"}

    # ── 3 & 4. Summary ──
    long_pnl  = sum(t["net"] for t in trades_2023 if "LONG" in t["side"])
    short_pnl = sum(t["net"] for t in trades_2023 if "SHORT" in t["side"])
    total_pnl = long_pnl + short_pnl
    long_count  = len([t for t in trades_2023 if "LONG" in t["side"]])
    short_count = len([t for t in trades_2023 if "SHORT" in t["side"]])

    print(f"\n── 3. Long vs Short PnL in 2023 ──")
    print(f"  Total 2023 realized PnL:  ${total_pnl:>+,.2f}")
    print(f"  Long trades PnL:          ${long_pnl:>+,.2f}  ({long_count} trades)")
    print(f"  Short trades PnL:         ${short_pnl:>+,.2f}  ({short_count} trades)")

    print(f"\n── 4. Per-coin 2023 PnL (worst to best) ──")
    coin_pnl: dict[str, float] = {}
    for t in trades_2023:
        coin_pnl[t["sym"]] = coin_pnl.get(t["sym"], 0.0) + t["net"]
    for sym, pnl in sorted(coin_pnl.items(), key=lambda x: x[1]):
        print(f"    {sym:<12}  ${pnl:>+,.2f}")

    # ── 5. BTC monthly ──
    print(f"\n── 5. BTC monthly closes in 2023 (context: recovery year?) ──")
    monthly: dict[str, float] = {}
    for b in btc_bars:
        if _2023_START <= b["ts"] < _2023_END:
            m = datetime.fromtimestamp(b["ts"]/1000, tz=timezone.utc).strftime("%Y-%m")
            monthly[m] = b["close"]
    prev_price = btc_jan1
    for m in sorted(monthly):
        p = monthly[m]
        chg = (p / prev_price - 1) * 100
        print(f"    {m}  ${p:>10,.0f}  ({chg:>+.1f}% from prev)")
        prev_price = p

    print(f"\n  BTC SMA200 crossovers in 2023:")
    for d, direction, price in btc_sma200_crossovers:
        print(f"    {d}  crossed {direction} SMA200  @ ${price:,.0f}")

    # ── 6. Momentum gate ──
    print(f"\n── 6. Momentum gate blocks in 2023 (days entry was prevented) ──")
    total_blocked = 0
    for sym in _SYMBOLS:
        n = gate_blocked.get(sym, 0)
        total_blocked += n
        if n > 0:
            print(f"    {sym:<12}  blocked {n} entry days")
    print(f"  Total blocked entry days: {total_blocked}")

    # ── Short trade detail ──
    print(f"\n── Short trade details in 2023 (closed trades) ──")
    for t in sorted(trades_2023, key=lambda x: x["ts"]):
        if "SHORT" in t["side"]:
            d = datetime.fromtimestamp(t["ts"]/1000, tz=timezone.utc).strftime("%Y-%m-%d")
            print(f"    {d}  {t['sym']:<12}  {t['side']:<25}  entry=${t['entry']:>10,.3f}  exit=${t['exit']:>10,.3f}  net=${t['net']:>+,.2f}")

    # ── Long trade detail ──
    print(f"\n── Long trade details in 2023 (closed trades) ──")
    for t in sorted(trades_2023, key=lambda x: x["ts"]):
        if t["side"] == "LONG":
            d = datetime.fromtimestamp(t["ts"]/1000, tz=timezone.utc).strftime("%Y-%m-%d")
            print(f"    {d}  {t['sym']:<12}  entry=${t['entry']:>10,.3f}  exit=${t['exit']:>10,.3f}  net=${t['net']:>+,.2f}")

    print(f"\n── Open positions at end of 2023 ──")
    open_longs  = {s: p for s, p in positions.items() if p["side"] == "LONG"}
    open_shorts = {s: p for s, p in positions.items() if p["side"] == "SHORT"}
    print(f"  Open longs:  {list(open_longs.keys())}")
    print(f"  Open shorts: {list(open_shorts.keys())}")
    if open_shorts:
        print(f"  (Shorts carried from 2022 bear regime being unwound at year-end)")

    print("\n" + "="*70)
    print("  DIAGNOSTIC COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
