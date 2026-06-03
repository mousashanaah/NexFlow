#!/usr/bin/env python3
"""Diagnostic script: Why does V8 lose money in 2023?"""

from __future__ import annotations
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    import pyarrow.parquet as pq
except ImportError:
    print("[ERROR] pip install pyarrow"); sys.exit(1)

_CANDLE_DIR = _REPO_ROOT / "data" / "candles"
_TAKER_FEE  = 0.0006
_CAPITAL    = 5_000.0  # as used in the question ($5K)
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


def _ema_series(closes: list[float], period: int) -> list[Optional[float]]:
    alpha = 2.0/(period+1)
    result: list[Optional[float]] = [None]*len(closes)
    ema = None
    for i, c in enumerate(closes):
        ema = alpha*c + (1-alpha)*ema if ema is not None else c
        if i >= period-1: result[i] = ema
    return result


def _sma_series(closes: list[float], period: int) -> list[Optional[float]]:
    result: list[Optional[float]] = [None]*len(closes)
    for i in range(period-1, len(closes)):
        result[i] = sum(closes[i-period+1:i+1])/period
    return result


def _macd_long_series(closes: list[float]) -> list[bool]:
    ef = _ema_series(closes, 12)
    es = _ema_series(closes, 26)
    macds = [ef[i]-es[i] if ef[i] and es[i] else None for i in range(len(closes))]
    sig_vals = [m for m in macds if m is not None]
    sig_raw = _ema_series(sig_vals, 9)
    sig: list[Optional[float]] = []
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


def _build_signals(symbols: list[str]) -> dict:
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


def _vol_series(signals: dict, sym: str, window: int = 14) -> dict[int, float]:
    ts_list = sorted(signals.get(sym, {}).keys())
    closes = [signals[sym][t]["close"] for t in ts_list]
    result: dict[int, float] = {}
    for i, ts in enumerate(ts_list):
        if i < window:
            result[ts] = 0.0
            continue
        rets = [(closes[j] - closes[j-1]) / closes[j-1] for j in range(i - window + 1, i + 1)]
        mean = sum(rets) / len(rets)
        variance = sum((r - mean)**2 for r in rets) / len(rets)
        result[ts] = variance**0.5
    return result


def main():
    Y2023_START = int(datetime(2023,1,1,tzinfo=timezone.utc).timestamp()*1000)
    Y2023_END   = int(datetime(2024,1,1,tzinfo=timezone.utc).timestamp()*1000)

    # Use a wider range so indicators can warm up
    from_ts = int(datetime(2021,1,1,tzinfo=timezone.utc).timestamp()*1000)
    to_ts   = int(datetime(2024,6,1,tzinfo=timezone.utc).timestamp()*1000)

    print("Building signals...")
    signals = _build_signals(_SYMBOLS)
    print("Done.\n")

    # ── Q1 & Q2: BTC regime analysis in 2023 ──────────────────────────────────
    print("=" * 65)
    print("Q1 & Q2: BTC Regime Analysis in 2023")
    print("=" * 65)

    btc_ts = sorted(t for t in signals.get("BTCUSDT", {}) if Y2023_START <= t < Y2023_END)

    # Precompute btc_mom30 and asymmetric regime state
    all_btc_ts = sorted(signals.get("BTCUSDT", {}).keys())
    btc_closes_all = [signals["BTCUSDT"][t]["close"] for t in all_btc_ts]
    btc_sma50_vals = _sma_series(btc_closes_all, 50)
    btc_mom30: dict[int, float] = {}
    for i, t in enumerate(all_btc_ts):
        if i >= 30:
            btc_mom30[t] = (btc_closes_all[i] - btc_closes_all[i-30]) / btc_closes_all[i-30]
        else:
            btc_mom30[t] = 0.0

    # Replay V8 asymmetric regime logic
    prev_btc_bear_mode = False
    btc_above_sma200_streak = 0
    bear_drop_pct = -0.20
    confirm_days = 5

    regime_log = []  # (date, bear_mode, btc_close, sma200, mom30)
    regime_flips = []

    for i, t in enumerate(all_btc_ts):
        sig = signals["BTCUSDT"][t]
        btc_sma200_above = sig.get("sma200_above", True)
        btc_30d_ret = btc_mom30.get(t, 0.0)
        drop_triggered = btc_30d_ret < bear_drop_pct

        # AND mode entry
        enter_bear = (not btc_sma200_above) and drop_triggered

        if btc_sma200_above:
            btc_above_sma200_streak += 1
        else:
            btc_above_sma200_streak = 0

        confirmed_bull = btc_above_sma200_streak >= max(1, confirm_days)
        if prev_btc_bear_mode:
            btc_bear_mode = not confirmed_bull
        else:
            btc_bear_mode = enter_bear
        if btc_bear_mode != prev_btc_bear_mode:
            regime_flips.append({
                "date": datetime.fromtimestamp(t/1000, tz=timezone.utc).strftime("%Y-%m-%d"),
                "to_bear": btc_bear_mode,
                "btc_close": sig["close"],
                "sma200": sig.get("sma200"),
                "mom30": btc_30d_ret,
                "sma200_above": btc_sma200_above,
            })

        if Y2023_START <= t < Y2023_END:
            regime_log.append({
                "ts": t,
                "date": datetime.fromtimestamp(t/1000, tz=timezone.utc).strftime("%Y-%m-%d"),
                "bear_mode": btc_bear_mode,
                "btc_close": sig["close"],
                "sma200": sig.get("sma200"),
                "mom30": btc_30d_ret,
                "sma200_above": btc_sma200_above,
                "drop_triggered": drop_triggered,
            })

        prev_btc_bear_mode = btc_bear_mode

    bear_days = sum(1 for r in regime_log if r["bear_mode"])
    bull_days = sum(1 for r in regime_log if not r["bear_mode"])
    print(f"  2023 BULL days: {bull_days}")
    print(f"  2023 BEAR days: {bear_days}")
    print(f"  Total days: {len(regime_log)}")

    btc_jan1 = signals["BTCUSDT"][min(btc_ts)]["close"] if btc_ts else None
    btc_dec31 = signals["BTCUSDT"][max(btc_ts)]["close"] if btc_ts else None
    if btc_jan1 and btc_dec31:
        btc_return = (btc_dec31 - btc_jan1) / btc_jan1 * 100
        print(f"\n  Q5 BTC Price in 2023:")
        print(f"    Jan 1 2023: ${btc_jan1:,.0f}")
        print(f"    Dec 31 2023: ${btc_dec31:,.0f}")
        print(f"    Full-year return: +{btc_return:.1f}%")

    print(f"\n  Q2: Regime flips in 2023 (AND-entry V8 logic):")
    flips_2023 = [f for f in regime_flips if f["date"].startswith("2023")]
    print(f"  Total flips in 2023: {len(flips_2023)}")
    for f in flips_2023:
        direction = "BULL→BEAR" if f["to_bear"] else "BEAR→BULL"
        sma200_str = f"${f['sma200']:,.0f}" if f['sma200'] else "N/A"
        print(f"    {f['date']}  {direction}  BTC=${f['btc_close']:,.0f}  "
              f"SMA200={sma200_str}  "
              f"mom30={f['mom30']:.3f}  above_sma200={f['sma200_above']}")

    # Show BTC regime trajectory — monthly summary
    print(f"\n  Monthly bear/bull breakdown 2023:")
    months = {}
    for r in regime_log:
        m = r["date"][:7]
        if m not in months:
            months[m] = {"bear": 0, "bull": 0, "btc_close": []}
        if r["bear_mode"]: months[m]["bear"] += 1
        else: months[m]["bull"] += 1
        months[m]["btc_close"].append(r["btc_close"])
    for m in sorted(months):
        b = months[m]
        avg_c = sum(b["btc_close"]) / len(b["btc_close"])
        print(f"    {m}  bear={b['bear']:3d}d  bull={b['bull']:3d}d  avg_BTC=${avg_c:,.0f}")

    # ── Full V8 backtest with per-trade logging ──────────────────────────────
    print("\n" + "="*65)
    print("Q3, Q4, Q6: Per-trade analysis for 2023")
    print("="*65)

    # Rerun V8 with full trade capture
    base_notional = _CAPITAL / len(_SYMBOLS)

    vol_series_all: dict[str, dict[int, float]] = {}
    for sym in _SYMBOLS:
        vol_series_all[sym] = _vol_series(signals, sym)

    # Momentum gate
    mom30_series: dict[str, dict[int, float]] = {}
    for sym in _SYMBOLS:
        ts_list = sorted(signals.get(sym, {}).keys())
        closes = [signals[sym][t]["close"] for t in ts_list]
        m30: dict[int, float] = {}
        for i, t in enumerate(ts_list):
            if i < 30:
                m30[t] = 0.0
            else:
                m30[t] = (closes[i] - closes[i-30]) / closes[i-30]
        mom30_series[sym] = m30

    def _position_size(sym: str, ts: int, mult: float = 1.0) -> float:
        n = base_notional * mult
        vol = vol_series_all.get(sym, {}).get(ts, 0.0)
        if vol <= 0:
            return n
        vol_sized = (0.01 * _CAPITAL) / vol
        return min(vol_sized * mult, base_notional * 2 * mult)

    all_ts = sorted(set(ts for sym in _SYMBOLS for ts in signals.get(sym,{}) if from_ts<=ts<=to_ts))
    equity = _CAPITAL
    positions: dict[str,dict] = {}
    trades_2023: list[dict] = []
    all_trades: list[dict] = []

    # Reset regime state
    prev_btc_bear_mode = False
    btc_above_sma200_streak = 0
    last_rebal_ts = 0

    # Track momentum gate blocks in 2023
    mom_gate_blocks = {sym: 0 for sym in _SYMBOLS}
    mom_gate_would_win = {sym: 0.0 for sym in _SYMBOLS}  # estimated PnL if gate was off

    for ts in all_ts:
        btc_sig = signals.get("BTCUSDT",{}).get(ts,{})
        btc_sma200_above = btc_sig.get("sma200_above", True)
        btc_30d_ret = btc_mom30.get(ts, 0.0)
        drop_triggered = btc_30d_ret < bear_drop_pct

        # AND mode
        enter_bear = (not btc_sma200_above) and drop_triggered
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

        in_2023 = Y2023_START <= ts < Y2023_END

        # TSMOM short rebalance
        if not btc_bull and (ts - last_rebal_ts) >= 7*_DAY_MS:
            last_rebal_ts = ts
            scores = []
            for sym in _SYMBOLS:
                sym_ts_list = sorted(signals.get(sym,{}).keys())
                past_ts_candidates = [t for t in sym_ts_list if t <= ts]
                if len(past_ts_candidates) < 127: continue
                c_now  = signals[sym][past_ts_candidates[-1]]["close"]
                c_past = signals[sym][past_ts_candidates[-127]]["close"]
                ret = (c_now - c_past) / c_past
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
                    yr = datetime.fromtimestamp(ts/1000,tz=timezone.utc).year
                    trade = {"ts":ts,"sym":sym,"net":net,"side":"SHORT",
                             "entry":pos["entry"],"exit":c,"notional":pos["notional"],"yr":yr}
                    all_trades.append(trade)
                    if in_2023: trades_2023.append(trade)

            for sym in desired_shorts:
                if sym not in positions:
                    c = signals.get(sym,{}).get(ts,{}).get("close")
                    if c is None: continue
                    n = _position_size(sym, ts)
                    equity -= _TAKER_FEE * n
                    positions[sym] = {"entry":c,"notional":n,"side":"SHORT"}

        # Hard stop on shorts
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
                yr = datetime.fromtimestamp(ts/1000,tz=timezone.utc).year
                trade = {"ts":ts,"sym":sym,"net":net,"side":"SHORT",
                         "entry":pos["entry"],"exit":c,"notional":pos["notional"],"yr":yr}
                all_trades.append(trade)
                if in_2023: trades_2023.append(trade)

        # Close shorts if bull
        if btc_bull:
            for sym in [s for s in list(positions) if positions[s].get("side")=="SHORT"]:
                c = signals.get(sym,{}).get(ts,{}).get("close")
                if c is None: continue
                pos = positions.pop(sym)
                raw = (pos["entry"]-c)/pos["entry"]*pos["notional"]
                net = raw - _TAKER_FEE*pos["notional"]
                equity += net
                yr = datetime.fromtimestamp(ts/1000,tz=timezone.utc).year
                trade = {"ts":ts,"sym":sym,"net":net,"side":"SHORT",
                         "entry":pos["entry"],"exit":c,"notional":pos["notional"],"yr":yr}
                all_trades.append(trade)
                if in_2023: trades_2023.append(trade)

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

            # Momentum gate check
            mom_blocked = False
            if can_long and any_long:
                m30 = mom30_series.get(sym, {}).get(ts, 0.0)
                if m30 <= 0:
                    mom_blocked = True
                    can_long = False
                    if in_2023:
                        mom_gate_blocks[sym] += 1

            in_pos = sym in positions and positions[sym].get("side") == "LONG"

            if in_pos and (not any_long or not can_long):
                pos = positions.pop(sym)
                raw = (c-pos["entry"])/pos["entry"]*pos["notional"]
                net = raw - _TAKER_FEE*pos["notional"]
                equity += net
                yr = datetime.fromtimestamp(ts/1000,tz=timezone.utc).year
                trade = {"ts":ts,"sym":sym,"net":net,"side":"LONG",
                         "entry":pos["entry"],"exit":c,"notional":pos["notional"],"yr":yr}
                all_trades.append(trade)
                if in_2023: trades_2023.append(trade)
                in_pos = False

            if not in_pos and any_long and can_long:
                mult = {1:1.0,2:1.5,3:2.0}.get(n_long,1.0)
                n = _position_size(sym, ts, mult)
                equity -= _TAKER_FEE * n
                positions[sym] = {"entry":c,"notional":n,"side":"LONG","peak_price":c}

    # Mark remaining open positions at end of 2023 boundary
    open_at_year_end = {}
    for sym, pos in positions.items():
        ts_list = sorted(t for t in signals.get(sym,{}) if t <= Y2023_END)
        if ts_list:
            open_at_year_end[sym] = {"pos": pos, "last_ts": ts_list[-1],
                                      "last_close": signals[sym][ts_list[-1]]["close"]}

    print(f"\n  V8 2023 trades: {len(trades_2023)}")

    # ── Q3: Long vs Short PnL breakdown ──
    long_pnl_2023 = sum(t["net"] for t in trades_2023 if t["side"] == "LONG")
    short_pnl_2023 = sum(t["net"] for t in trades_2023 if t["side"] == "SHORT")
    long_wins = sum(1 for t in trades_2023 if t["side"] == "LONG" and t["net"] > 0)
    long_losses = sum(1 for t in trades_2023 if t["side"] == "LONG" and t["net"] <= 0)
    short_wins = sum(1 for t in trades_2023 if t["side"] == "SHORT" and t["net"] > 0)
    short_losses = sum(1 for t in trades_2023 if t["side"] == "SHORT" and t["net"] <= 0)

    print(f"\n  Q3: Loss breakdown by side (2023):")
    print(f"    LONG  PnL: ${long_pnl_2023:>+,.2f}  (wins={long_wins}, losses={long_losses})")
    print(f"    SHORT PnL: ${short_pnl_2023:>+,.2f}  (wins={short_wins}, losses={short_losses})")

    # ── Q4: Per-coin PnL ──
    print(f"\n  Q4: Per-coin PnL in 2023:")
    coin_pnl: dict[str, float] = {sym: 0.0 for sym in _SYMBOLS}
    coin_trades: dict[str, int] = {sym: 0 for sym in _SYMBOLS}
    for t in trades_2023:
        coin_pnl[t["sym"]] = coin_pnl.get(t["sym"], 0.0) + t["net"]
        coin_trades[t["sym"]] = coin_trades.get(t["sym"], 0) + 1

    for sym in sorted(coin_pnl, key=lambda s: coin_pnl[s]):
        pnl = coin_pnl[sym]
        n_trades = coin_trades[sym]
        sides = set(t["side"] for t in trades_2023 if t["sym"] == sym)
        print(f"    {sym:<12} ${pnl:>+8,.2f}  trades={n_trades}  sides={sides}")

    total_trade_pnl = sum(t["net"] for t in trades_2023)
    print(f"\n    Total closed PnL 2023: ${total_trade_pnl:>+,.2f}")

    # Unrealised P&L on positions open through year-end
    print(f"\n  Open positions spanning 2023 year-end boundary:")
    for sym, info in open_at_year_end.items():
        pos = info["pos"]
        lc = info["last_close"]
        if pos["side"] == "LONG":
            mtm = (lc - pos["entry"]) / pos["entry"] * pos["notional"]
        else:
            mtm = (pos["entry"] - lc) / pos["entry"] * pos["notional"]
        print(f"    {sym:<12} {pos['side']}  entry=${pos['entry']:,.2f}  "
              f"last_close=${lc:,.2f}  unrealised_mtm=${mtm:>+,.2f}  notional=${pos['notional']:,.2f}")

    # ── Q6: Momentum gate analysis ──
    print(f"\n  Q6: Momentum gate blocks in 2023 (days blocked from entry):")
    total_blocked = sum(mom_gate_blocks.values())
    for sym in sorted(mom_gate_blocks, key=lambda s: -mom_gate_blocks[s]):
        if mom_gate_blocks[sym] > 0:
            print(f"    {sym:<12} blocked {mom_gate_blocks[sym]:3d} days")
    print(f"    Total signal-days blocked by momentum gate: {total_blocked}")

    # ── What BTC actually did vs regime assignment ──
    print(f"\n" + "="*65)
    print("Summary: BTC recovery year vs bear-mode assignment")
    print("="*65)
    # Find when BTC crossed SMA200 upward in 2023
    prev_above = None
    crossings = []
    for i, t in enumerate(all_btc_ts):
        sig = signals["BTCUSDT"][t]
        above = sig.get("sma200_above", False)
        if prev_above is not None and above != prev_above:
            crossings.append({
                "date": datetime.fromtimestamp(t/1000, tz=timezone.utc).strftime("%Y-%m-%d"),
                "crossed": "ABOVE" if above else "BELOW",
                "btc": sig["close"],
                "sma200": sig.get("sma200"),
            })
        prev_above = above
    crossings_2023 = [c for c in crossings if c["date"].startswith("2023")]
    print(f"\n  BTC SMA200 raw crossings in 2023: {len(crossings_2023)}")
    for c in crossings_2023:
        print(f"    {c['date']}  {c['crossed']}  BTC=${c['btc']:,.0f}  SMA200=${c['sma200']:,.0f}")

    print(f"\n  Key finding: BTC rose {btc_return:.1f}% in 2023 but was in BEAR mode {bear_days}/{len(regime_log)} days.")
    print(f"  When in BEAR mode, system shorts coins & blocks longs.")
    print(f"  BTC crossing SMA200 upward still requires {confirm_days} consecutive days to confirm bull.")

    # ── Short trades detail in 2023 ──
    short_trades_2023 = [t for t in trades_2023 if t["side"] == "SHORT"]
    if short_trades_2023:
        print(f"\n  Short trades in 2023 detail (sorted by PnL):")
        short_trades_2023.sort(key=lambda t: t["net"])
        for t in short_trades_2023[:20]:
            dt = datetime.fromtimestamp(t["ts"]/1000, tz=timezone.utc).strftime("%Y-%m-%d")
            pct = (t["exit"] - t["entry"]) / t["entry"] * 100 if t["exit"] and t["entry"] else 0
            print(f"    {dt}  {t['sym']:<12}  entry=${t['entry']:>10,.3f}  exit=${t['exit']:>10,.3f}  "
                  f"price_chg={pct:>+6.1f}%  net=${t['net']:>+8,.2f}")

    # ── Long trades detail in 2023 ──
    long_trades_2023 = [t for t in trades_2023 if t["side"] == "LONG"]
    if long_trades_2023:
        print(f"\n  Long trades in 2023 detail (sorted by PnL):")
        long_trades_2023.sort(key=lambda t: t["net"])
        for t in long_trades_2023:
            dt = datetime.fromtimestamp(t["ts"]/1000, tz=timezone.utc).strftime("%Y-%m-%d")
            pct = (t["exit"] - t["entry"]) / t["entry"] * 100 if t["exit"] and t["entry"] else 0
            print(f"    {dt}  {t['sym']:<12}  entry=${t['entry']:>10,.3f}  exit=${t['exit']:>10,.3f}  "
                  f"price_chg={pct:>+6.1f}%  net=${t['net']:>+8,.2f}")

    print(f"\n  DONE.")


if __name__ == "__main__":
    main()
