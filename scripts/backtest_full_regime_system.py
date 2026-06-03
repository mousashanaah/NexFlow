#!/usr/bin/env python3
"""Full regime-aware system backtest.

Tests the complete combined system with BTC SMA200 as master regime switch:

  BTC > SMA200 (BULL regime):
    - Long trio runs freely (EMA 8/21 + MACD + 4H EMA 5/13)
    - Confluence sizing: 1×/1.5×/2× based on strategy agreement
    - No shorts

  BTC < SMA200 (BEAR regime):
    - No new long entries (existing longs close on their EMA/MACD signals)
    - TSMOM short: short coins with 126d return < -5%
    - Rebalance shorts weekly

Compares 4 variants:
  V1: Long trio only (baseline, no filter, no shorts)
  V2: Long trio + BTC SMA200 long filter (no new longs in bear)
  V3: V2 + TSMOM short in bear regime
  V4: V3 + per-coin SMA200 filter on longs (each coin must be above its own SMA200)

Capital: $100,000 | 12 coins | 2021-2026
"""

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
_CAPITAL    = 100_000.0
_SYMBOLS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT",
    "XRPUSDT","ADAUSDT","DOGEUSDT","AVAXUSDT",
    "LINKUSDT","LTCUSDT","DOTUSDT","TRXUSDT",
]
_DAY_MS = 86_400_000
_IS_TS  = int(datetime(2023,1,1,tzinfo=timezone.utc).timestamp()*1000)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _load_daily(symbol: str) -> list[dict]:
    path = _CANDLE_DIR / f"{symbol}_1D.parquet"
    if not path.exists(): return []
    tbl = pq.read_table(path, columns=["open_time","close"])
    rows = [{"ts":int(ts),"close":float(c)}
            for ts,c in zip(tbl.column("open_time").to_pylist(),
                            tbl.column("close").to_pylist())]
    return sorted(rows, key=lambda x: x["ts"])


def _load_4h_as_daily_proxy(symbol: str) -> dict[int, float]:
    """Return dict of day_ts → last 4H close of that day."""
    path = _CANDLE_DIR / f"{symbol}_1H.parquet"
    if not path.exists(): return {}
    tbl = pq.read_table(path, columns=["open_time","close"])
    rows = sorted(zip(tbl.column("open_time").to_pylist(),
                      tbl.column("close").to_pylist()))
    # group by 4H bucket
    buckets: dict[int,float] = {}
    for ts, c in rows:
        hour = (int(ts) % _DAY_MS) // 3_600_000
        bts = (int(ts) // _DAY_MS)*_DAY_MS + (hour//4)*4*3_600_000
        buckets[bts] = float(c)
    # map 4H bucket → daily: last 4H of the day
    day_close: dict[int,float] = {}
    for bts, c in buckets.items():
        day = (bts // _DAY_MS) * _DAY_MS
        if day not in day_close or bts > day_close.get(day+"_ts", 0):
            day_close[day] = c
            day_close[day+"_ts"] = bts  # type: ignore
    return {k:v for k,v in day_close.items() if isinstance(k, int) and k % _DAY_MS == 0}


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Build per-symbol signal arrays
# ---------------------------------------------------------------------------
def _build_signals(symbols: list[str]) -> dict:
    """
    Returns for each symbol, indexed by timestamp:
      ema_long, macd_long, h4_long, sma200_above, close
    """
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

        # 4H EMA 5/13 — approximate using daily with shorter EMAs as proxy
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
                h4_long_state = h4_above and ema_above  # h4 filtered by daily trend

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


# ---------------------------------------------------------------------------
# Portfolio backtest
# ---------------------------------------------------------------------------
def _run(
    signals: dict,
    use_sma200_long_filter: bool,   # block new longs when BTC < SMA200
    use_tsmom_short: bool,          # short in bear regime
    use_per_coin_sma200: bool,      # each coin must be above its own SMA200
    confluence: bool,               # scale by strategy agreement
    from_ts: int,
    to_ts: int,
    use_btc_ema_long_filter: bool = False,  # also require BTC EMA8 > EMA21 for new longs
    use_coin_sma50: bool = False,           # each coin must be above its own SMA50
) -> dict:
    all_ts = sorted(set(ts for sym in _SYMBOLS for ts in signals.get(sym,{}) if from_ts<=ts<=to_ts))
    base_notional = _CAPITAL / len(_SYMBOLS)

    equity = _CAPITAL; peak = _CAPITAL; max_dd = 0.0
    positions: dict[str,dict] = {}  # {sym: {entry, notional}}
    trades: list[dict] = []
    year_pnl: dict[int,float] = {}
    last_rebal_ts = 0

    for ts in all_ts:
        # BTC regime
        btc_sig = signals.get("BTCUSDT",{}).get(ts,{})
        btc_bull = btc_sig.get("sma200_above", True)
        # Optionally also require BTC short-term uptrend (EMA8 > EMA21) for new longs
        btc_ema_bull = btc_sig.get("ema_long", True) if use_btc_ema_long_filter else True
        long_allowed = btc_bull and btc_ema_bull

        # ── TSMOM short rebalance (weekly) ──
        if use_tsmom_short and not btc_bull and (ts - last_rebal_ts) >= 7*_DAY_MS:
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

            # Close shorts no longer desired
            for sym in [s for s in list(positions) if positions[s].get("side")=="SHORT"]:
                if sym not in desired_shorts:
                    c = signals.get(sym,{}).get(ts,{}).get("close")
                    if c is None: continue
                    pos = positions.pop(sym)
                    raw = (pos["entry"]-c)/pos["entry"]*pos["notional"]
                    net = raw - _TAKER_FEE*pos["notional"]
                    equity += net
                    trades.append({"ts":ts,"sym":sym,"net":net,"side":"SHORT"})
                    yr = datetime.fromtimestamp(ts/1000,tz=timezone.utc).year
                    year_pnl[yr] = year_pnl.get(yr,0)+net

            # Open new shorts
            for sym in desired_shorts:
                if sym not in positions:
                    c = signals.get(sym,{}).get(ts,{}).get("close")
                    if c is None: continue
                    equity -= _TAKER_FEE*base_notional
                    positions[sym] = {"entry":c,"notional":base_notional,"side":"SHORT"}

        # ── Close any shorts if we're back in bull ──
        if btc_bull and use_tsmom_short:
            for sym in [s for s in list(positions) if positions[s].get("side")=="SHORT"]:
                c = signals.get(sym,{}).get(ts,{}).get("close")
                if c is None: continue
                pos = positions.pop(sym)
                raw = (pos["entry"]-c)/pos["entry"]*pos["notional"]
                net = raw - _TAKER_FEE*pos["notional"]
                equity += net
                trades.append({"ts":ts,"sym":sym,"net":net,"side":"SHORT"})
                yr = datetime.fromtimestamp(ts/1000,tz=timezone.utc).year
                year_pnl[yr] = year_pnl.get(yr,0)+net

        # ── Long signals ──
        for sym in _SYMBOLS:
            sig = signals.get(sym,{}).get(ts,{})
            if not sig: continue
            c = sig["close"]
            n_long = sum([sig.get("ema_long",False), sig.get("macd_long",False), sig.get("h4_long",False)])
            any_long = n_long > 0

            # Determine if we should be long this coin
            can_long = True
            if use_sma200_long_filter and not long_allowed:
                can_long = False  # bear regime or BTC EMA bearish: no new longs
            if use_per_coin_sma200 and not sig.get("sma200_above", True):
                can_long = False  # coin below own SMA200
            if use_coin_sma50 and not sig.get("sma50_above", True):
                can_long = False  # coin below own SMA50 — in downtrend, skip bounces

            in_pos = sym in positions and positions[sym].get("side") == "LONG"

            # Close long if signal gone or regime changed
            if in_pos and (not any_long or not can_long):
                pos = positions.pop(sym)
                raw = (c-pos["entry"])/pos["entry"]*pos["notional"]
                net = raw - _TAKER_FEE*pos["notional"]
                equity += net
                trades.append({"ts":ts,"sym":sym,"net":net,"side":"LONG"})
                yr = datetime.fromtimestamp(ts/1000,tz=timezone.utc).year
                year_pnl[yr] = year_pnl.get(yr,0)+net
                in_pos = False

            # Open long if signal present and regime allows
            if not in_pos and any_long and can_long:
                if confluence:
                    mult = {1:1.0,2:1.5,3:2.0}.get(n_long,1.0)
                else:
                    mult = 1.0
                n = base_notional * mult
                equity -= _TAKER_FEE*n
                positions[sym] = {"entry":c,"notional":n,"side":"LONG"}

        if equity > peak: peak = equity
        dd = (peak-equity)/peak
        if dd > max_dd: max_dd = dd

    # Close all remaining positions at last available price (mark to market)
    last_prices = {}
    last_ts_by_sym: dict[str,int] = {}
    for sym in _SYMBOLS:
        ts_list = sorted(t for t in signals.get(sym,{}) if t <= to_ts)
        if ts_list:
            last_prices[sym] = signals[sym][ts_list[-1]]["close"]
            last_ts_by_sym[sym] = ts_list[-1]

    unrealised = 0.0
    for sym, pos in positions.items():
        p = last_prices.get(sym, pos["entry"])
        last_ts = last_ts_by_sym.get(sym, to_ts)
        if pos["side"] == "LONG":
            mtm = (p-pos["entry"])/pos["entry"]*pos["notional"]
        else:
            mtm = (pos["entry"]-p)/pos["entry"]*pos["notional"]
        unrealised += mtm
        yr = datetime.fromtimestamp(last_ts/1000, tz=timezone.utc).year
        year_pnl[yr] = year_pnl.get(yr, 0) + mtm

    total_eq = equity + unrealised
    net = total_eq - _CAPITAL
    years = (to_ts-from_ts)/(1_000*86_400*365.25)
    cagr = (total_eq/_CAPITAL)**(1/years)-1 if years>0 and total_eq>0 else -1.0

    gw = sum(t["net"] for t in trades if t["net"]>0)
    gl = sum(abs(t["net"]) for t in trades if t["net"]<0)
    pf = gw/gl if gl>0 else float("inf")

    is_t  = [t for t in trades if t["ts"] <  _IS_TS]
    oos_t = [t for t in trades if t["ts"] >= _IS_TS]
    def _pf(ts): gw=sum(t["net"] for t in ts if t["net"]>0); gl=sum(abs(t["net"]) for t in ts if t["net"]<0); return gw/gl if gl>0 else float("inf")

    return {
        "equity":total_eq,"net":net,"cagr":cagr,"max_dd":max_dd,
        "pf":pf,"n":len(trades),"is_pf":_pf(is_t),"oos_pf":_pf(oos_t),
        "year_pnl":year_pnl,
    }


def _print(label: str, r: dict) -> None:
    print(f"\n{'='*58}")
    print(f"  {label}")
    print(f"{'='*58}")
    print(f"  Equity : ${r['equity']:>12,.0f}  (net ${r['net']:>+,.0f})")
    print(f"  CAGR   : {r['cagr']*100:.1f}%")
    print(f"  Max DD : {r['max_dd']*100:.1f}%")
    print(f"  PF     : {r['pf']:.2f}  (IS:{r['is_pf']:.2f}  OOS:{r['oos_pf']:.2f})")
    print(f"  Trades : {r['n']}")
    print()
    losing_years = []
    for yr in sorted(r["year_pnl"]):
        p = r["year_pnl"][yr]
        tag = " <<BEAR" if yr in [2022,2025,2026] else (" <<BULL" if yr in [2021,2024] else "")
        flag = " ✓" if p > 0 else " ✗"
        print(f"    {yr}: ${p:>+10,.0f}{tag}{flag}")
        if p < 0: losing_years.append(yr)
    print(f"\n  Losing years: {losing_years if losing_years else 'NONE ✓'}")
    verdict = "✓ GO" if r['pf']>=1.20 and r['max_dd']<=0.45 and r['cagr']>=0.15 else "MARGINAL" if r['pf']>=1.10 else "KILL"
    print(f"  VERDICT: {verdict}")


def main():
    from_ts = int(datetime(2021,1,1,tzinfo=timezone.utc).timestamp()*1000)
    to_ts   = int(datetime.now(timezone.utc).timestamp()*1000)

    print("Full Regime-Aware System — 2021 to present")
    print(f"Capital: ${_CAPITAL:,.0f}  |  12 coins  |  Fee: 0.06%/side")
    print()

    print("Building signals for all 12 coins ...")
    signals = _build_signals(_SYMBOLS)
    print("Done.\n")

    results = {}

    results["V1"] = _run(signals, False, False, False, True,  from_ts, to_ts)
    _print("V1: Long Trio + Confluence (baseline, no regime filter)", results["V1"])

    results["V2"] = _run(signals, True,  False, False, True,  from_ts, to_ts)
    _print("V2: V1 + BTC SMA200 long filter (no new longs in bear)", results["V2"])

    results["V3"] = _run(signals, True,  True,  False, True,  from_ts, to_ts)
    _print("V3: V2 + TSMOM short in bear regime", results["V3"])

    results["V4"] = _run(signals, True,  True,  True,  True,  from_ts, to_ts)
    _print("V4: V3 + per-coin SMA200 filter on longs", results["V4"])

    results["V5"] = _run(signals, True,  True,  False, True,  from_ts, to_ts, use_btc_ema_long_filter=True)
    _print("V5: V3 + BTC EMA8>EMA21 required for longs (tighter bull gate)", results["V5"])

    results["V6"] = _run(signals, True,  True,  False, True,  from_ts, to_ts, use_coin_sma50=True)
    _print("V6: V3 + per-coin SMA50 filter (no longs on coins in downtrend)", results["V6"])

    # Side-by-side year comparison
    print(f"\n{'='*78}")
    print("  Year-by-year comparison (all variants, $100K)")
    print(f"{'='*78}")
    print(f"  {'Year':<6} {'V1 (base)':>12} {'V2 (+SMA200L)':>14} {'V3 (+Short)':>12} {'V5 (+EMA gate)':>14}")
    print(f"  {'-'*6} {'-'*12} {'-'*14} {'-'*12} {'-'*14}")
    all_years = sorted(set(yr for r in results.values() for yr in r["year_pnl"]))
    for yr in all_years:
        tag = " B" if yr in [2022,2025,2026] else ""
        vals = [results[v]["year_pnl"].get(yr,0) for v in ["V1","V2","V3","V5"]]
        flags = ["✓" if v>0 else "✗" for v in vals]
        print(f"  {yr}{tag:<4} "
              f"${vals[0]:>+10,.0f}{flags[0]} "
              f"${vals[1]:>+11,.0f}{flags[1]}  "
              f"${vals[2]:>+10,.0f}{flags[2]} "
              f"${vals[3]:>+12,.0f}{flags[3]}")
    print()
    for v_key, label in [("V1","V1"),("V2","V2"),("V3","V3"),("V5","V5")]:
        r = results[v_key]
        print(f"  {label}: CAGR={r['cagr']*100:.1f}%  DD={r['max_dd']*100:.1f}%  PF={r['pf']:.2f}  eq=${r['equity']:,.0f}")


if __name__ == "__main__":
    main()
