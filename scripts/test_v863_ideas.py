#!/usr/bin/env python3
"""Standalone test of two enhancement ideas for V8.63.

Idea 1: Volatility regime sizing
  - Compute BTC 20-day realized vol (std of daily returns)
  - vol > 4%/day  → scale all positions to 0.5x
  - vol < 2%/day  → scale all positions to 1.5x
  - otherwise     → normal 1.0x

Idea 2: Bear market short quality filter
  - Only short coins that are ALSO below their own SMA200
  - Avoids shorting coins that are oversold and about to bounce

Runs three configs vs V8.63 base and prints year-by-year PnL at $5K.
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
_IS_TS  = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
SCALE = 5_000 / _CAPITAL  # 0.05 — scale $100k results to $5k


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_daily(symbol: str) -> list[dict]:
    path = _CANDLE_DIR / f"{symbol}_1D.parquet"
    if not path.exists():
        return []
    tbl = pq.read_table(path, columns=["open_time", "close"])
    rows = [
        {"ts": int(ts), "close": float(c)}
        for ts, c in zip(
            tbl.column("open_time").to_pylist(),
            tbl.column("close").to_pylist(),
        )
    ]
    return sorted(rows, key=lambda x: x["ts"])


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def _ema_series(closes: list[float], period: int) -> list[Optional[float]]:
    alpha = 2.0 / (period + 1)
    result: list[Optional[float]] = [None] * len(closes)
    ema = None
    for i, c in enumerate(closes):
        ema = alpha * c + (1 - alpha) * ema if ema is not None else c
        if i >= period - 1:
            result[i] = ema
    return result


def _sma_series(closes: list[float], period: int) -> list[Optional[float]]:
    result: list[Optional[float]] = [None] * len(closes)
    for i in range(period - 1, len(closes)):
        result[i] = sum(closes[i - period + 1 : i + 1]) / period
    return result


def _macd_long_series(closes: list[float]) -> list[bool]:
    ef = _ema_series(closes, 12)
    es = _ema_series(closes, 26)
    macds = [ef[i] - es[i] if ef[i] and es[i] else None for i in range(len(closes))]
    sig_vals = [m for m in macds if m is not None]
    sig_raw = _ema_series(sig_vals, 9)
    sig: list[Optional[float]] = []
    idx = 0
    for m in macds:
        if m is None:
            sig.append(None)
        else:
            sig.append(sig_raw[idx])
            idx += 1
    hist = [
        macds[i] - sig[i] if macds[i] is not None and sig[i] is not None else None
        for i in range(len(closes))
    ]
    result = [False] * len(closes)
    in_long = False
    for i in range(1, len(closes)):
        if hist[i - 1] is not None and hist[i] is not None:
            if hist[i - 1] <= 0 < hist[i]:
                in_long = True
            elif hist[i - 1] >= 0 > hist[i]:
                in_long = False
        result[i] = in_long
    return result


# ---------------------------------------------------------------------------
# Build per-symbol signal arrays (identical to backtest_full_regime_system.py)
# ---------------------------------------------------------------------------
def _build_signals(symbols: list[str]) -> dict:
    out = {}
    for sym in symbols:
        bars = _load_daily(sym)
        if not bars:
            continue
        closes = [b["close"] for b in bars]
        ts_list = [b["ts"] for b in bars]

        ef8    = _ema_series(closes, 8)
        ef21   = _ema_series(closes, 21)
        sma200 = _sma_series(closes, 200)
        sma50  = _sma_series(closes, 50)
        macd_long = _macd_long_series(closes)
        ef5   = _ema_series(closes, 5)
        ef13  = _ema_series(closes, 13)

        by_ts: dict[int, dict] = {}
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
                "close":        closes[i],
                "ema_long":     ema_long_state,
                "macd_long":    macd_long[i],
                "h4_long":      h4_long_state,
                "sma200_above": sma200[i] is not None and closes[i] > sma200[i],
                "sma50_above":  sma50[i]  is not None and closes[i] > sma50[i],
                "sma200":       sma200[i],
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
        rets = [(closes[j] - closes[j - 1]) / closes[j - 1] for j in range(i - window + 1, i + 1)]
        mean = sum(rets) / len(rets)
        variance = sum((r - mean) ** 2 for r in rets) / len(rets)
        result[ts] = variance ** 0.5
    return result


# ---------------------------------------------------------------------------
# Core simulation — V8.63 params + optional idea flags
# ---------------------------------------------------------------------------
def _run(
    signals: dict,
    from_ts: int,
    to_ts: int,
    # V8.63 fixed params
    use_sma200_long_filter: bool = True,
    use_tsmom_short: bool = True,
    confluence: bool = True,
    hard_stop_pct: float = 0.15,
    use_atr_sizing: bool = True,
    asymmetric_regime: bool = True,
    and_entry: bool = True,
    bear_drop_pct: float = -0.20,
    confirm_days: int = 10,
    momentum_gate: bool = True,
    momentum_gate_days: int = 20,
    target_risk: float = 0.01,
    # --- IDEA 1: volatility regime sizing ---
    vol_regime_sizing: bool = False,
    vol_window: int = 20,         # lookback days for BTC vol
    vol_high_thresh: float = 0.04,  # > 4%/day → scale 0.5x
    vol_low_thresh:  float = 0.02,  # < 2%/day → scale 1.5x
    # --- IDEA 2: short quality filter (coin below SMA200) ---
    short_sma200_filter: bool = False,
) -> dict:
    all_ts = sorted(
        set(ts for sym in _SYMBOLS for ts in signals.get(sym, {}) if from_ts <= ts <= to_ts)
    )
    base_notional = _CAPITAL / len(_SYMBOLS)

    # ATR vol sizing (same as original)
    vol_series_cache: dict[str, dict[int, float]] = {}
    if use_atr_sizing:
        for sym in _SYMBOLS:
            vol_series_cache[sym] = _vol_series(signals, sym)

    # Momentum gate
    mom_series: dict[str, dict[int, float]] = {}
    if momentum_gate:
        for sym in _SYMBOLS:
            ts_list = sorted(signals.get(sym, {}).keys())
            closes = [signals[sym][t]["close"] for t in ts_list]
            m: dict[int, float] = {}
            for i, t in enumerate(ts_list):
                m[t] = (
                    (closes[i] - closes[i - momentum_gate_days]) / closes[i - momentum_gate_days]
                    if i >= momentum_gate_days
                    else 0.0
                )
            mom_series[sym] = m

    # Asymmetric regime: BTC 30d return + SMA50
    btc_mom30: dict[int, float] = {}
    if asymmetric_regime:
        btc_ts = sorted(signals.get("BTCUSDT", {}).keys())
        btc_cls = [signals["BTCUSDT"][t]["close"] for t in btc_ts]
        for i, t in enumerate(btc_ts):
            btc_mom30[t] = (
                (btc_cls[i] - btc_cls[i - 30]) / btc_cls[i - 30] if i >= 30 else 0.0
            )

    # Idea 1: BTC 20-day realized vol series
    btc_20d_vol: dict[int, float] = {}
    if vol_regime_sizing:
        btc_ts = sorted(signals.get("BTCUSDT", {}).keys())
        btc_cls = [signals["BTCUSDT"][t]["close"] for t in btc_ts]
        for i, t in enumerate(btc_ts):
            if i < vol_window:
                btc_20d_vol[t] = 0.0
            else:
                rets = [
                    (btc_cls[j] - btc_cls[j - 1]) / btc_cls[j - 1]
                    for j in range(i - vol_window + 1, i + 1)
                ]
                mean = sum(rets) / len(rets)
                variance = sum((r - mean) ** 2 for r in rets) / len(rets)
                btc_20d_vol[t] = variance ** 0.5

    equity = _CAPITAL
    peak = _CAPITAL
    max_dd = 0.0
    positions: dict[str, dict] = {}
    trades: list[dict] = []
    year_pnl: dict[int, float] = {}
    last_rebal_ts = 0
    daily_equity: list[float] = []

    prev_btc_bear_mode = False
    btc_above_sma200_streak = 0

    def _position_size(sym: str, ts: int, mult: float = 1.0, vol_mult: float = 1.0) -> float:
        """Return notional for this position."""
        n = base_notional * mult * vol_mult
        if not use_atr_sizing:
            return n
        vol = vol_series_cache.get(sym, {}).get(ts, 0.0)
        if vol <= 0:
            return n
        vol_sized = (target_risk * _CAPITAL) / vol
        return min(vol_sized * mult * vol_mult, base_notional * 2 * mult * vol_mult)

    for ts in all_ts:
        btc_sig = signals.get("BTCUSDT", {}).get(ts, {})
        btc_sma200_above = btc_sig.get("sma200_above", True)

        # --- Asymmetric regime logic ---
        if asymmetric_regime:
            btc_30d_ret = btc_mom30.get(ts, 0.0)
            drop_triggered = btc_30d_ret < bear_drop_pct
            enter_bear = (not btc_sma200_above) and drop_triggered  # AND mode
            if btc_sma200_above:
                btc_above_sma200_streak += 1
            else:
                btc_above_sma200_streak = 0
            confirmed_bull = btc_above_sma200_streak >= max(1, confirm_days)
            btc_bear_mode = (not confirmed_bull) if prev_btc_bear_mode else enter_bear
            prev_btc_bear_mode = btc_bear_mode
            btc_bull = not btc_bear_mode
        else:
            btc_bull = btc_sma200_above

        long_allowed = btc_bull

        # --- Idea 1: determine vol multiplier for this day ---
        vol_mult = 1.0
        if vol_regime_sizing:
            btc_vol = btc_20d_vol.get(ts, 0.0)
            if btc_vol > vol_high_thresh:
                vol_mult = 0.5
            elif btc_vol > 0 and btc_vol < vol_low_thresh:
                vol_mult = 1.5

        # --- TSMOM short rebalance (weekly) ---
        if use_tsmom_short and not btc_bull and (ts - last_rebal_ts) >= 7 * _DAY_MS:
            last_rebal_ts = ts
            scores = []
            for sym in _SYMBOLS:
                sym_ts_list = sorted(signals.get(sym, {}).keys())
                past = [t for t in sym_ts_list if t <= ts]
                if len(past) < 127:
                    continue
                c_now  = signals[sym][past[-1]]["close"]
                c_past = signals[sym][past[-127]]["close"]
                ret = (c_now - c_past) / c_past
                scores.append((ret, sym))
            scores.sort()

            # Idea 2: apply quality filter — only short coins also below SMA200
            if short_sma200_filter:
                desired_shorts = {
                    sym for ret, sym in scores
                    if ret < -0.05 and not signals.get(sym, {}).get(ts, {}).get("sma200_above", True)
                }
            else:
                desired_shorts = {sym for ret, sym in scores if ret < -0.05}

            # Close shorts no longer desired
            for sym in [s for s in list(positions) if positions[s].get("side") == "SHORT"]:
                if sym not in desired_shorts:
                    c = signals.get(sym, {}).get(ts, {}).get("close")
                    if c is None:
                        continue
                    pos = positions.pop(sym)
                    raw = (pos["entry"] - c) / pos["entry"] * pos["notional"]
                    net = raw - _TAKER_FEE * pos["notional"]
                    equity += net
                    trades.append({"ts": ts, "sym": sym, "net": net, "side": "SHORT"})
                    yr = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).year
                    year_pnl[yr] = year_pnl.get(yr, 0) + net

            # Open new shorts
            for sym in desired_shorts:
                if sym not in positions:
                    c = signals.get(sym, {}).get(ts, {}).get("close")
                    if c is None:
                        continue
                    n = _position_size(sym, ts, vol_mult=vol_mult)
                    equity -= _TAKER_FEE * n
                    positions[sym] = {"entry": c, "notional": n, "side": "SHORT"}

        # --- Hard stop on shorts ---
        if hard_stop_pct > 0 and use_tsmom_short:
            for sym in [s for s in list(positions) if positions[s].get("side") == "SHORT"]:
                c = signals.get(sym, {}).get(ts, {}).get("close")
                if c is None:
                    continue
                pos = positions[sym]
                loss_pct = (c - pos["entry"]) / pos["entry"]
                if loss_pct >= hard_stop_pct:
                    positions.pop(sym)
                    raw = (pos["entry"] - c) / pos["entry"] * pos["notional"]
                    net = raw - _TAKER_FEE * pos["notional"]
                    equity += net
                    trades.append({"ts": ts, "sym": sym, "net": net, "side": "SHORT"})
                    yr = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).year
                    year_pnl[yr] = year_pnl.get(yr, 0) + net

        # --- Close shorts if back in bull ---
        if btc_bull and use_tsmom_short:
            for sym in [s for s in list(positions) if positions[s].get("side") == "SHORT"]:
                c = signals.get(sym, {}).get(ts, {}).get("close")
                if c is None:
                    continue
                pos = positions.pop(sym)
                raw = (pos["entry"] - c) / pos["entry"] * pos["notional"]
                net = raw - _TAKER_FEE * pos["notional"]
                equity += net
                trades.append({"ts": ts, "sym": sym, "net": net, "side": "SHORT"})
                yr = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).year
                year_pnl[yr] = year_pnl.get(yr, 0) + net

        # --- Long signals ---
        for sym in _SYMBOLS:
            sig = signals.get(sym, {}).get(ts, {})
            if not sig:
                continue
            c = sig["close"]
            n_long = sum([sig.get("ema_long", False), sig.get("macd_long", False), sig.get("h4_long", False)])
            any_long = n_long > 0

            can_long = True
            if use_sma200_long_filter and not long_allowed:
                can_long = False
            if momentum_gate and not sig.get("sma200_above", True):
                m20 = mom_series.get(sym, {}).get(ts, 0.0)
                if m20 <= 0:
                    can_long = False

            in_pos = sym in positions and positions[sym].get("side") == "LONG"

            if in_pos and (not any_long or not can_long):
                pos = positions.pop(sym)
                raw = (c - pos["entry"]) / pos["entry"] * pos["notional"]
                net = raw - _TAKER_FEE * pos["notional"]
                equity += net
                trades.append({"ts": ts, "sym": sym, "net": net, "side": "LONG"})
                yr = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).year
                year_pnl[yr] = year_pnl.get(yr, 0) + net
                in_pos = False

            if not in_pos and any_long and can_long:
                if momentum_gate:
                    m20 = mom_series.get(sym, {}).get(ts, 0.0)
                    if m20 <= 0:
                        continue
                mult = {1: 1.0, 2: 1.5, 3: 2.0}.get(n_long, 1.0) if confluence else 1.0
                n = _position_size(sym, ts, mult, vol_mult=vol_mult)
                equity -= _TAKER_FEE * n
                positions[sym] = {"entry": c, "notional": n, "side": "LONG", "peak_price": c}

        # Mark-to-market for Sharpe
        mtm = 0.0
        for sym, pos in positions.items():
            c = signals.get(sym, {}).get(ts, {}).get("close")
            if c is None:
                continue
            if pos["side"] == "LONG":
                mtm += (c - pos["entry"]) / pos["entry"] * pos["notional"]
            else:
                mtm += (pos["entry"] - c) / pos["entry"] * pos["notional"]
        daily_equity.append(equity + mtm)

        snap = daily_equity[-1]
        if snap > peak:
            peak = snap
        dd = (peak - snap) / peak
        if dd > max_dd:
            max_dd = dd

    # Force-close remaining positions
    last_prices: dict[str, float] = {}
    last_ts_by_sym: dict[str, int] = {}
    for sym in _SYMBOLS:
        ts_list = sorted(t for t in signals.get(sym, {}) if t <= to_ts)
        if ts_list:
            last_prices[sym] = signals[sym][ts_list[-1]]["close"]
            last_ts_by_sym[sym] = ts_list[-1]

    unrealised = 0.0
    for sym, pos in positions.items():
        p = last_prices.get(sym, pos["entry"])
        last_t = last_ts_by_sym.get(sym, to_ts)
        if pos["side"] == "LONG":
            mtm = (p - pos["entry"]) / pos["entry"] * pos["notional"]
        else:
            mtm = (pos["entry"] - p) / pos["entry"] * pos["notional"]
        unrealised += mtm
        yr = datetime.fromtimestamp(last_t / 1000, tz=timezone.utc).year
        year_pnl[yr] = year_pnl.get(yr, 0) + mtm

    total_eq = equity + unrealised
    years = (to_ts - from_ts) / (1_000 * 86_400 * 365.25)
    cagr = (total_eq / _CAPITAL) ** (1 / years) - 1 if years > 0 and total_eq > 0 else -1.0

    gw = sum(t["net"] for t in trades if t["net"] > 0)
    gl = sum(abs(t["net"]) for t in trades if t["net"] < 0)
    pf = gw / gl if gl > 0 else float("inf")

    is_t  = [t for t in trades if t["ts"] <  _IS_TS]
    oos_t = [t for t in trades if t["ts"] >= _IS_TS]

    def _pf(ts_):
        gw_ = sum(t["net"] for t in ts_ if t["net"] > 0)
        gl_ = sum(abs(t["net"]) for t in ts_ if t["net"] < 0)
        return gw_ / gl_ if gl_ > 0 else float("inf")

    sharpe = 0.0
    if len(daily_equity) > 2:
        daily_rets = [
            (daily_equity[i] - daily_equity[i - 1]) / daily_equity[i - 1]
            for i in range(1, len(daily_equity))
        ]
        mean_r = sum(daily_rets) / len(daily_rets)
        std_r  = (sum((r - mean_r) ** 2 for r in daily_rets) / len(daily_rets)) ** 0.5
        sharpe = (mean_r / std_r) * (252 ** 0.5) if std_r > 0 else 0.0

    return {
        "equity": total_eq,
        "cagr":   cagr,
        "max_dd": max_dd,
        "pf":     pf,
        "n":      len(trades),
        "is_pf":  _pf(is_t),
        "oos_pf": _pf(oos_t),
        "year_pnl": year_pnl,
        "sharpe": sharpe,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
YEARS = [2021, 2022, 2023, 2024, 2025, 2026]

# V8.63 known reference values (scaled to $5K) — from backtest_full_regime_system.py comments
# We run V8.63 fresh in this script so all comparisons are internally consistent.
_BASE: dict = {}


def _print_result(label: str, r: dict, base: dict | None = None) -> None:
    yp = r["year_pnl"]
    print(f"\n{'='*68}")
    print(f"  {label}")
    print(f"{'='*68}")
    print(f"  CAGR: {r['cagr']*100:.1f}%   MaxDD: {r['max_dd']*100:.1f}%   "
          f"Sharpe: {r['sharpe']:.2f}   PF: {r['pf']:.2f}")
    print()
    print(f"  {'Year':<6} {'PnL @$5K':>12}  {'vs Base':>10}")
    for yr in YEARS:
        p = yp.get(yr, 0.0) * SCALE
        if base:
            bp = base["year_pnl"].get(yr, 0.0) * SCALE
            delta = p - bp
            delta_str = f"  ({delta:>+8,.0f})"
        else:
            delta_str = ""
        print(f"  {yr:<6} ${p:>+10,.0f}{delta_str}")
    print()
    if base:
        b_cagr = base["cagr"] * 100
        b_dd   = base["max_dd"] * 100
        b_sh   = base["sharpe"]
        d_cagr = r["cagr"] * 100 - b_cagr
        d_dd   = r["max_dd"] * 100 - b_dd
        d_sh   = r["sharpe"] - b_sh
        print(f"  vs V8.63 base:  CAGR {d_cagr:>+.1f}pp   MaxDD {d_dd:>+.1f}pp   Sharpe {d_sh:>+.2f}")


def _is_improvement(r: dict, base: dict) -> bool:
    """True if this variant is a genuine overall improvement."""
    # Better or equal CAGR, not significantly worse DD, better or equal Sharpe,
    # and fewer losing years or higher total PnL.
    if r["cagr"] <= base["cagr"] * 0.98:
        return False
    if r["max_dd"] > base["max_dd"] * 1.10:
        return False
    if r["sharpe"] < base["sharpe"] * 0.95:
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    from_ts = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    to_ts   = int(datetime.now(timezone.utc).timestamp() * 1000)

    print("Building signals for all 12 coins...")
    signals = _build_signals(_SYMBOLS)
    print("Done.\n")

    # --- V8.63 base ---
    print("Running V8.63 base...")
    base = _run(signals, from_ts, to_ts)
    _print_result("V8.63 BASE (reference)", base)

    results: list[tuple[str, dict]] = []

    # --- Idea 1: vol regime sizing ---
    print("\nRunning Idea 1: volatility regime sizing...")
    r1 = _run(signals, from_ts, to_ts, vol_regime_sizing=True)
    _print_result("IDEA 1: Volatility Regime Sizing (BTC 20d vol)", r1, base)
    results.append(("Idea 1: Vol Regime Sizing", r1))

    # --- Idea 2: short quality filter ---
    print("\nRunning Idea 2: bear market short SMA200 quality filter...")
    r2 = _run(signals, from_ts, to_ts, short_sma200_filter=True)
    _print_result("IDEA 2: Short Quality Filter (coin below SMA200 only)", r2, base)
    results.append(("Idea 2: Short SMA200 Filter", r2))

    # --- Idea 1 + 2 combined ---
    print("\nRunning Ideas 1+2 combined...")
    r3 = _run(signals, from_ts, to_ts, vol_regime_sizing=True, short_sma200_filter=True)
    _print_result("IDEA 1+2 COMBINED: Vol Sizing + Short SMA200 Filter", r3, base)
    results.append(("Idea 1+2 Combined", r3))

    # --- Summary ---
    print(f"\n\n{'='*80}")
    print("  SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Variant':<42}  {'CAGR':>6}  {'MaxDD':>6}  {'Sharpe':>6}  {'PF':>5}  {'Improvement?'}")
    print(f"  {'-'*42}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*12}")
    print(f"  {'V8.63 BASE':<42}  {base['cagr']*100:>5.1f}%  {base['max_dd']*100:>5.1f}%  "
          f"{base['sharpe']:>6.2f}  {base['pf']:>5.2f}  ---")
    for label, r in results:
        improved = _is_improvement(r, base)
        verdict  = "YES - GENUINE" if improved else "no"
        print(f"  {label:<42}  {r['cagr']*100:>5.1f}%  {r['max_dd']*100:>5.1f}%  "
              f"{r['sharpe']:>6.2f}  {r['pf']:>5.2f}  {verdict}")

    # Detailed report for genuine improvements
    improvements = [(label, r) for label, r in results if _is_improvement(r, base)]
    if improvements:
        print(f"\n\n{'='*80}")
        print("  GENUINE IMPROVEMENTS FOUND")
        print(f"{'='*80}")
        for label, r in improvements:
            print(f"\n  {label}")
            print(f"  CAGR  : {r['cagr']*100:.1f}%  (base {base['cagr']*100:.1f}%,"
                  f" +{(r['cagr']-base['cagr'])*100:.1f}pp)")
            print(f"  MaxDD : {r['max_dd']*100:.1f}%  (base {base['max_dd']*100:.1f}%)")
            print(f"  Sharpe: {r['sharpe']:.2f}  (base {base['sharpe']:.2f})")
            print(f"  PF    : {r['pf']:.2f}  IS: {r['is_pf']:.2f}  OOS: {r['oos_pf']:.2f}")
            print()
            print(f"  Year-by-year at $5K:")
            for yr in YEARS:
                p  = r["year_pnl"].get(yr, 0.0) * SCALE
                bp = base["year_pnl"].get(yr, 0.0) * SCALE
                arrow = "^" if p > bp else ("v" if p < bp else "=")
                print(f"    {yr}: ${p:>+9,.0f}  (base ${bp:>+9,.0f})  {arrow} ({p-bp:>+8,.0f})")
    else:
        print("\n  No genuine improvement found. Neither idea beats V8.63 on all metrics.")


if __name__ == "__main__":
    main()
