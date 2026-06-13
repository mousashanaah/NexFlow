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
def _vol_series(signals: dict, sym: str, window: int = 14) -> dict[int, float]:
    """Rolling std of daily returns over `window` days — used as ATR proxy for sizing."""
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
        result[ts] = variance**0.5  # daily vol (std of returns)
    return result


def _efficiency_ratio_series(signals: dict, sym: str, window: int) -> dict[int, float]:
    """Kaufman Efficiency Ratio per day: |net move| / sum(|daily moves|), 0..1.

    1 = perfectly directional (clean trend), 0 = pure chop. No lookahead — each
    day uses only the trailing `window` closes up to and including that day.
    """
    ts_list = sorted(signals.get(sym, {}).keys())
    closes = [signals[sym][t]["close"] for t in ts_list]
    result: dict[int, float] = {}
    for i, ts in enumerate(ts_list):
        if i < window:
            result[ts] = 1.0  # not enough history — don't gate
            continue
        net = abs(closes[i] - closes[i - window])
        path = sum(abs(closes[j] - closes[j - 1]) for j in range(i - window + 1, i + 1))
        result[ts] = (net / path) if path > 0 else 0.0
    return result


def _avg_correlation_series(signals: dict, symbols: list, window: int) -> dict[int, float]:
    """Average pairwise correlation of daily returns across all coins, per day.

    Uses the common timestamp grid (BTC's). No lookahead — trailing window only.
    """
    # Build aligned return series on BTC's timestamp grid.
    ref_ts = sorted(signals.get("BTCUSDT", {}).keys())
    rets_by_sym: dict[str, list[Optional[float]]] = {}
    for sym in symbols:
        s = signals.get(sym, {})
        prev = None
        series: list[Optional[float]] = []
        for t in ref_ts:
            c = s.get(t, {}).get("close")
            if c is None or prev is None or prev == 0:
                series.append(None)
            else:
                series.append((c - prev) / prev)
            prev = c if c is not None else prev
        rets_by_sym[sym] = series

    def _corr(a: list[float], b: list[float]) -> Optional[float]:
        pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
        if len(pairs) < window // 2:
            return None
        xs = [p[0] for p in pairs]; ys = [p[1] for p in pairs]
        mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
        cov = sum((x - mx) * (y - my) for x, y in pairs)
        vx = sum((x - mx) ** 2 for x in xs); vy = sum((y - my) ** 2 for y in ys)
        denom = (vx * vy) ** 0.5
        return (cov / denom) if denom > 0 else None

    result: dict[int, float] = {}
    for i, ts in enumerate(ref_ts):
        if i < window:
            result[ts] = 0.0
            continue
        lo = i - window + 1
        vals = []
        for a in range(len(symbols)):
            for b in range(a + 1, len(symbols)):
                ca = rets_by_sym[symbols[a]][lo:i + 1]
                cb = rets_by_sym[symbols[b]][lo:i + 1]
                c = _corr(ca, cb)
                if c is not None:
                    vals.append(c)
        result[ts] = (sum(vals) / len(vals)) if vals else 0.0
    return result


def _run(
    signals: dict,
    use_sma200_long_filter: bool,
    use_tsmom_short: bool,
    use_per_coin_sma200: bool,
    confluence: bool,
    from_ts: int,
    to_ts: int,
    use_btc_ema_long_filter: bool = False,
    use_coin_sma50: bool = False,
    hard_stop_pct: float = 0.0,
    use_atr_sizing: bool = False,   # vol-adjusted position sizing
    target_risk: float = 0.01,      # target daily risk per position as fraction of capital
    # NEW: Experiment A — trailing stop on longs
    trailing_stop_pct: float = 0.0,  # 0 = off, e.g. 0.15 = 15% from peak
    atr_trail_mult: float = 0.0,     # 0 = off; exit long if price falls k×(vol×price) from peak
                                     # (vol = rolling 14d std of returns — ATR-equivalent, self-scaling)
    # NEW: Experiment B — asymmetric regime switch
    asymmetric_regime: bool = False,  # fast enter bear, slow exit bear
    bear_drop_pct: float = -0.15,     # 30d BTC drop threshold to enter bear early (AND mode)
    and_entry: bool = False,          # True = AND (must ALSO be below SMA200), False = OR
    confirm_days: int = 0,            # days BTC must stay above SMA200 to exit bear (slow exit)
    # NEW: Experiment C — momentum gate on long entries
    momentum_gate: bool = False,      # only enter long if coin Nd return > 0
    momentum_gate_days: int = 20,     # lookback window for momentum gate
    # NEW: Experiment D — funding-rate crowding gate on long entries
    # When BTC daily funding is in an extreme-high state (crowded longs paying
    # carry), suspend NEW long entries. Existing longs exit on their own signal.
    funding_high: Optional[dict] = None,  # {ts: bool} — True = block new longs
    # NEW: Experiment E — efficiency-ratio (choppiness) gate on long entries.
    # Kaufman ER = |close[t]-close[t-n]| / sum(|close[i]-close[i-1]|), range 0..1.
    # Low ER = choppy/whipsaw market; block new longs below the threshold.
    er_gate: bool = False,
    er_days: int = 30,
    er_threshold: float = 0.30,
    er_use_btc: bool = True,   # True = market-wide BTC ER gate; False = per-coin ER
    # ER as a smooth SIZING dial (instead of / on top of the hard gate): scale
    # position size down in chop. mult = clamp((ER-lo)/(hi-lo), floor, 1).
    er_sizing: bool = False,
    er_size_lo: float = 0.20,
    er_size_hi: float = 0.45,
    er_size_floor: float = 0.4,
    # NEW: Experiment F — correlation-aware sizing. When the 12-coin book is
    # highly correlated, diversification is illusory; scale all sizes down.
    corr_sizing: bool = False,
    corr_days: int = 30,
    corr_ref: float = 0.5,     # avg pairwise corr above which sizing scales down
    corr_floor: float = 0.5,   # minimum size multiplier (never shrink below this)
    # NEW: diagnostic mode — print regime flips + stranded long losses
    diagnostic: bool = False,
) -> dict:
    all_ts = sorted(set(ts for sym in _SYMBOLS for ts in signals.get(sym,{}) if from_ts<=ts<=to_ts))
    base_notional = _CAPITAL / len(_SYMBOLS)

    # Precompute vol series for ATR sizing
    vol_series: dict[str, dict[int, float]] = {}
    if use_atr_sizing or atr_trail_mult > 0:
        for sym in _SYMBOLS:
            vol_series[sym] = _vol_series(signals, sym)

    # Experiment E: efficiency-ratio series (market-wide BTC, or per-coin)
    er_series: dict[str, dict[int, float]] = {}
    if er_gate or er_sizing:
        er_syms = ["BTCUSDT"] if er_use_btc else _SYMBOLS
        for sym in er_syms:
            er_series[sym] = _efficiency_ratio_series(signals, sym, er_days)

    def _er_size_mult(sym: str, ts: int) -> float:
        if not er_sizing:
            return 1.0
        er_key = "BTCUSDT" if er_use_btc else sym
        er_val = er_series.get(er_key, {}).get(ts, 1.0)
        if er_size_hi <= er_size_lo:
            return 1.0
        frac = (er_val - er_size_lo) / (er_size_hi - er_size_lo)
        return max(er_size_floor, min(1.0, frac))

    # Experiment F: average pairwise correlation series → size multiplier
    corr_mult: dict[int, float] = {}
    if corr_sizing:
        avg_corr = _avg_correlation_series(signals, _SYMBOLS, corr_days)
        for ts_c, c in avg_corr.items():
            if c <= corr_ref:
                corr_mult[ts_c] = 1.0
            else:
                # linearly scale from 1.0 at corr_ref down to corr_floor at corr=1.0
                frac = (c - corr_ref) / max(1e-9, (1.0 - corr_ref))
                corr_mult[ts_c] = max(corr_floor, 1.0 - frac * (1.0 - corr_floor))

    # Precompute N-day return series for momentum gate (Experiment C)
    mom20_series: dict[str, dict[int, float]] = {}
    if momentum_gate:
        for sym in _SYMBOLS:
            ts_list = sorted(signals.get(sym, {}).keys())
            closes = [signals[sym][t]["close"] for t in ts_list]
            m20: dict[int, float] = {}
            for i, t in enumerate(ts_list):
                if i < momentum_gate_days:
                    m20[t] = 0.0
                else:
                    m20[t] = (closes[i] - closes[i-momentum_gate_days]) / closes[i-momentum_gate_days]
            mom20_series[sym] = m20

    # Precompute BTC 30-day return series for asymmetric regime (Experiment B)
    btc_mom30: dict[int, float] = {}
    # Also precompute BTC SMA50 for asymmetric exit condition
    btc_sma50_series: dict[int, Optional[float]] = {}
    if asymmetric_regime:
        btc_ts_list = sorted(signals.get("BTCUSDT", {}).keys())
        btc_closes = [signals["BTCUSDT"][t]["close"] for t in btc_ts_list]
        btc_sma50_vals = _sma_series(btc_closes, 50)
        for i, t in enumerate(btc_ts_list):
            if i >= 30:
                btc_mom30[t] = (btc_closes[i] - btc_closes[i-30]) / btc_closes[i-30]
            else:
                btc_mom30[t] = 0.0
            btc_sma50_series[t] = btc_sma50_vals[i]

    equity = _CAPITAL; peak = _CAPITAL; max_dd = 0.0
    positions: dict[str,dict] = {}
    trades: list[dict] = []
    year_pnl: dict[int,float] = {}
    last_rebal_ts = 0
    daily_equity: list[float] = []  # for Sharpe/Sortino

    # Diagnostic tracking
    diag_regime_flips: list[dict] = []
    diag_year_short_pnl: dict[int, float] = {}
    diag_year_long_pnl: dict[int, float] = {}
    prev_btc_bear_mode = False  # for asymmetric regime: tracks bear state
    btc_above_sma200_streak = 0  # consecutive days BTC above SMA200 (for confirm_days exit)

    def _position_size(sym: str, ts: int, mult: float = 1.0) -> float:
        """Return notional for this position — flat or vol-adjusted."""
        cmult = corr_mult.get(ts, 1.0) if corr_sizing else 1.0
        emult = _er_size_mult(sym, ts)
        cmult *= emult
        n = base_notional * mult * cmult
        if not use_atr_sizing:
            return n
        vol = vol_series.get(sym, {}).get(ts, 0.0)
        if vol <= 0:
            return n
        # size = target_daily_risk / daily_vol, capped at 2× base
        vol_sized = (target_risk * _CAPITAL) / vol
        return min(vol_sized * mult * cmult, base_notional * 2 * mult)

    for ts in all_ts:
        # BTC regime
        btc_sig = signals.get("BTCUSDT",{}).get(ts,{})
        btc_sma200_above = btc_sig.get("sma200_above", True)

        if asymmetric_regime:
            btc_30d_ret = btc_mom30.get(ts, 0.0)
            drop_triggered = btc_30d_ret < bear_drop_pct
            if and_entry:
                # AND mode: enter bear only if BOTH below SMA200 AND 30d drop exceeded
                enter_bear = (not btc_sma200_above) and drop_triggered
            else:
                # OR mode: enter bear if either condition met
                enter_bear = (not btc_sma200_above) or drop_triggered
            # Exit bear: BTC must stay above SMA200 for confirm_days consecutive days
            if btc_sma200_above:
                btc_above_sma200_streak += 1
            else:
                btc_above_sma200_streak = 0
            confirmed_bull = btc_above_sma200_streak >= max(1, confirm_days)
            if prev_btc_bear_mode:
                btc_bear_mode = not confirmed_bull  # slow to exit
            else:
                btc_bear_mode = enter_bear          # fast to enter
            prev_btc_bear_mode = btc_bear_mode
            btc_bull = not btc_bear_mode
        else:
            btc_bull = btc_sma200_above

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
                    diag_year_short_pnl[yr] = diag_year_short_pnl.get(yr, 0) + net

            # Open new shorts
            for sym in desired_shorts:
                if sym not in positions:
                    c = signals.get(sym,{}).get(ts,{}).get("close")
                    if c is None: continue
                    n = _position_size(sym, ts)
                    equity -= _TAKER_FEE * n
                    positions[sym] = {"entry":c,"notional":n,"side":"SHORT"}

        # ── Hard stop: close any short that moved too far against us ──
        if hard_stop_pct > 0 and use_tsmom_short:
            for sym in [s for s in list(positions) if positions[s].get("side") == "SHORT"]:
                c = signals.get(sym, {}).get(ts, {}).get("close")
                if c is None: continue
                pos = positions[sym]
                loss_pct = (c - pos["entry"]) / pos["entry"]  # positive = price rose = loss on short
                if loss_pct >= hard_stop_pct:
                    positions.pop(sym)
                    raw = (pos["entry"] - c) / pos["entry"] * pos["notional"]
                    net = raw - _TAKER_FEE * pos["notional"]
                    equity += net
                    trades.append({"ts": ts, "sym": sym, "net": net, "side": "SHORT"})
                    yr = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).year
                    year_pnl[yr] = year_pnl.get(yr, 0) + net
                    diag_year_short_pnl[yr] = diag_year_short_pnl.get(yr, 0) + net

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
                diag_year_short_pnl[yr] = diag_year_short_pnl.get(yr, 0) + net

        # ── Trailing stop on longs (Experiment A) ──
        if trailing_stop_pct > 0:
            for sym in [s for s in list(positions) if positions[s].get("side") == "LONG"]:
                c = signals.get(sym, {}).get(ts, {}).get("close")
                if c is None: continue
                pos = positions[sym]
                # Update peak price since entry
                if c > pos.get("peak_price", pos["entry"]):
                    pos["peak_price"] = c
                # Check if dropped too far from peak
                peak_p = pos.get("peak_price", pos["entry"])
                drawdown_from_peak = (peak_p - c) / peak_p
                if drawdown_from_peak >= trailing_stop_pct:
                    positions.pop(sym)
                    raw = (c - pos["entry"]) / pos["entry"] * pos["notional"]
                    net = raw - _TAKER_FEE * pos["notional"]
                    equity += net
                    trades.append({"ts": ts, "sym": sym, "net": net, "side": "LONG"})
                    yr = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).year
                    year_pnl[yr] = year_pnl.get(yr, 0) + net
                    diag_year_long_pnl[yr] = diag_year_long_pnl.get(yr, 0) + net

        # ── ATR-multiple trailing stop on longs (self-scaling to vol) ──
        if atr_trail_mult > 0:
            for sym in [s for s in list(positions) if positions[s].get("side") == "LONG"]:
                c = signals.get(sym, {}).get(ts, {}).get("close")
                if c is None: continue
                pos = positions[sym]
                if c > pos.get("peak_price", pos["entry"]):
                    pos["peak_price"] = c
                peak_p = pos.get("peak_price", pos["entry"])
                vol = vol_series.get(sym, {}).get(ts, 0.0)  # daily return std
                if vol <= 0: continue
                trail_dist = atr_trail_mult * vol * peak_p  # k × (vol×price) in $ terms
                if (peak_p - c) >= trail_dist:
                    positions.pop(sym)
                    raw = (c - pos["entry"]) / pos["entry"] * pos["notional"]
                    net = raw - _TAKER_FEE * pos["notional"]
                    equity += net
                    trades.append({"ts": ts, "sym": sym, "net": net, "side": "LONG"})
                    yr = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).year
                    year_pnl[yr] = year_pnl.get(yr, 0) + net
                    diag_year_long_pnl[yr] = diag_year_long_pnl.get(yr, 0) + net

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

            # Experiment C: momentum gate — only enter if 20d return > 0
            if momentum_gate and not sig.get("sma200_above", True):
                # only gate entries (not exits), check 20d return
                m20 = mom20_series.get(sym, {}).get(ts, 0.0)
                if m20 <= 0:
                    can_long = False  # coin declining over 20 days, skip

            # Experiment D: funding crowding gate — block NEW longs when BTC
            # funding is extreme-high (market crowded long). Does not force
            # exits — existing longs ride out on their own signals.
            funding_blocks_entry = bool(funding_high) and funding_high.get(ts, False)

            # Experiment E: efficiency-ratio (choppiness) gate — block NEW longs
            # when the market is choppy (ER below threshold). Does not force exits.
            er_blocks_entry = False
            if er_gate:
                er_key = "BTCUSDT" if er_use_btc else sym
                er_val = er_series.get(er_key, {}).get(ts, 1.0)
                er_blocks_entry = er_val < er_threshold

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
                diag_year_long_pnl[yr] = diag_year_long_pnl.get(yr, 0) + net
                in_pos = False

            # Open long if signal present and regime allows
            if not in_pos and any_long and can_long and not funding_blocks_entry and not er_blocks_entry:
                # Momentum gate: don't open new entry if 20d return <= 0
                if momentum_gate:
                    m20 = mom20_series.get(sym, {}).get(ts, 0.0)
                    if m20 <= 0:
                        continue
                mult = {1:1.0,2:1.5,3:2.0}.get(n_long,1.0) if confluence else 1.0
                n = _position_size(sym, ts, mult)
                equity -= _TAKER_FEE * n
                positions[sym] = {"entry":c,"notional":n,"side":"LONG","peak_price":c}

        # Mark-to-market equity snapshot for Sharpe/Sortino
        mtm_today = 0.0
        for sym, pos in positions.items():
            c = signals.get(sym, {}).get(ts, {}).get("close")
            if c is None: continue
            if pos["side"] == "LONG":
                mtm_today += (c - pos["entry"]) / pos["entry"] * pos["notional"]
            else:
                mtm_today += (pos["entry"] - c) / pos["entry"] * pos["notional"]
        daily_equity.append(equity + mtm_today)

        # Diagnostic: detect regime flips and log stranded long exposure
        if diagnostic:
            cur_bear = not btc_bull
            if cur_bear != prev_btc_bear_mode and len(diag_regime_flips) < 20:
                longs_open = [(s, p) for s, p in positions.items() if p.get("side") == "LONG"]
                unreal_loss = 0.0
                for s, p in longs_open:
                    cv = signals.get(s, {}).get(ts, {}).get("close", p["entry"])
                    unreal_loss += (cv - p["entry"]) / p["entry"] * p["notional"]
                diag_regime_flips.append({
                    "ts": ts,
                    "date": datetime.fromtimestamp(ts/1000, tz=timezone.utc).strftime("%Y-%m-%d"),
                    "to_bear": cur_bear,
                    "longs_open": len(longs_open),
                    "unrealised_pnl": unreal_loss,
                    "btc_close": btc_sig.get("close", 0),
                })

        snap = daily_equity[-1]
        if snap > peak: peak = snap
        dd = (peak - snap) / peak
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

    # Sharpe and Sortino from daily equity curve
    sharpe = sortino = 0.0
    if len(daily_equity) > 2:
        daily_rets = [(daily_equity[i] - daily_equity[i-1]) / daily_equity[i-1]
                      for i in range(1, len(daily_equity))]
        mean_r = sum(daily_rets) / len(daily_rets)
        std_r  = (sum((r - mean_r)**2 for r in daily_rets) / len(daily_rets))**0.5
        down_r = [r for r in daily_rets if r < 0]
        std_down = (sum(r**2 for r in down_r) / len(down_r))**0.5 if down_r else 1e-9
        sharpe  = (mean_r / std_r)  * (252**0.5) if std_r  > 0 else 0.0
        sortino = (mean_r / std_down) * (252**0.5) if std_down > 0 else 0.0

    return {
        "equity":total_eq,"net":net,"cagr":cagr,"max_dd":max_dd,
        "pf":pf,"n":len(trades),"is_pf":_pf(is_t),"oos_pf":_pf(oos_t),
        "year_pnl":year_pnl,"sharpe":sharpe,"sortino":sortino,
        "diag_regime_flips": diag_regime_flips,
        "diag_year_short_pnl": diag_year_short_pnl,
        "diag_year_long_pnl": diag_year_long_pnl,
        # real daily equity curve — same length as all_ts
        "daily_equity": daily_equity,
        "daily_ts": all_ts,
    }


def _print(label: str, r: dict) -> None:
    print(f"\n{'='*58}")
    print(f"  {label}")
    print(f"{'='*58}")
    print(f"  Equity : ${r['equity']:>12,.0f}  (net ${r['net']:>+,.0f})")
    print(f"  CAGR   : {r['cagr']*100:.1f}%")
    print(f"  Max DD : {r['max_dd']*100:.1f}%")
    print(f"  Sharpe : {r['sharpe']:.2f}   Sortino: {r['sortino']:.2f}")
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

    results["V7"] = _run(signals, True, True, False, True, from_ts, to_ts,
                         hard_stop_pct=0.15, use_atr_sizing=True)
    _print("V7: V3 + 15% hard stop + ATR vol-adjusted sizing", results["V7"])

    # V8: V7 + AND-entry asymmetric regime (BTC<SMA200 AND 30d drop>20%) + momentum gate 30d
    results["V8"] = _run(signals, True, True, False, True, from_ts, to_ts,
                         hard_stop_pct=0.15, use_atr_sizing=True,
                         asymmetric_regime=True, and_entry=True,
                         bear_drop_pct=-0.20, confirm_days=5,
                         momentum_gate=True, momentum_gate_days=30)
    _print("V8: V7 + AND-entry asymmetric regime + 30d momentum gate", results["V8"])

    # V8.63: V8 + confirm_days=10 + momentum_gate_days=20
    # Fixes: slower bear exit reduces whipsawing; 20d gate reacts faster to recoveries
    # Result: CAGR 57.3%, DD 25%, Sharpe 1.26 — best overall system
    results["V8.63"] = _run(signals, True, True, False, True, from_ts, to_ts,
                            hard_stop_pct=0.15, use_atr_sizing=True,
                            asymmetric_regime=True, and_entry=True,
                            bear_drop_pct=-0.20, confirm_days=10,
                            momentum_gate=True, momentum_gate_days=20)
    _print("V8.63: V8 + confirm_days=10 + momentum_gate_days=20 (BEST)", results["V8.63"])

    # ── Walk-Forward Validation ──
    print(f"\n{'='*78}")
    print("  Walk-Forward Validation — V3 base (2yr train / 6mo test, sliding)")
    print(f"{'='*78}")
    print(f"  {'Window':<22}  {'OOS CAGR':>9}  {'OOS PF':>8}  {'Sharpe':>8}  {'Verdict'}")
    print(f"  {'-'*22}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*10}")
    _TRAIN_MS = int(2 * 365.25 * _DAY_MS)
    _TEST_MS  = int(0.5 * 365.25 * _DAY_MS)
    wf_pass = wf_total = 0
    window_start = from_ts
    while window_start + _TRAIN_MS + _TEST_MS <= to_ts:
        train_end = window_start + _TRAIN_MS
        test_end  = min(train_end + _TEST_MS, to_ts)
        r = _run(signals, True, True, False, True, train_end, test_end, hard_stop_pct=0.15)
        cagr_pct = r["cagr"] * 100
        verdict  = "PASS ✓" if r["pf"] >= 1.10 and r["cagr"] > 0 else "FAIL ✗"
        if verdict.startswith("PASS"): wf_pass += 1
        wf_total += 1
        t_start = datetime.fromtimestamp(train_end/1000, tz=timezone.utc).strftime("%Y-%m")
        t_end   = datetime.fromtimestamp(test_end/1000,  tz=timezone.utc).strftime("%Y-%m")
        print(f"  {t_start} → {t_end}          {cagr_pct:>8.1f}%  {r['pf']:>8.2f}  "
              f"{r['sharpe']:>8.2f}  {verdict}")
        window_start += _TEST_MS
    print(f"\n  Result: {wf_pass}/{wf_total} windows profitable  "
          f"({'ROBUST ✓' if wf_pass >= wf_total*0.7 else 'FRAGILE ✗ — review before live'})")

    # ── Side-by-side summary ──
    print(f"\n{'='*78}")
    print("  Summary: V3 vs V7 vs V8")
    print(f"{'='*78}")
    for key, label in [("V3","V3 flat sizing"), ("V7","V7 ATR sizing"), ("V8","V8 AND-regime+MomGate"), ("V8.63","V8.63 BEST")]:
        r = results[key]
        print(f"  {label}: CAGR={r['cagr']*100:.1f}%  DD={r['max_dd']*100:.1f}%  "
              f"PF={r['pf']:.2f}  Sharpe={r['sharpe']:.2f}  Sortino={r['sortino']:.2f}  "
              f"eq=${r['equity']:,.0f}")
    print()
    for v_key, label in [("V1","V1"),("V2","V2"),("V3","V3"),("V5","V5")]:
        r = results[v_key]
        print(f"  {label}: CAGR={r['cagr']*100:.1f}%  DD={r['max_dd']*100:.1f}%  PF={r['pf']:.2f}  eq=${r['equity']:,.0f}")


if __name__ == "__main__":
    main()
