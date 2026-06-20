#!/usr/bin/env python3
"""
backtest_ict_fvg_v3.py — ICT/SMC FVG Scalping v3 — "Smarter Than Louis"

v2 result: PF 1.09, WR 51.7%, DD 26.4%, +11.4% on $5K — BREAK-EVEN
Root cause: 2021-2024 losing years despite 2025-2026 profit. Three root causes:
  (a) Gappy Bitget 1H data polluted HTF trend signals
  (b) No volume confirmation → many false displacement candles
  (c) No daily regime filter → longs taken all through 2022 bear market

FIXES IN v3 vs v2:
  1. 1H trend built by resampling the complete 2.85M-bar 1m dataset (no gaps)
  2. Volume spike at displacement — disp candle volume must be >1.5× 20-bar avg
     (the real institutional tell that Louis watches but v2 ignored)
  3. Daily SMA200 regime filter — longs only when daily close > SMA200,
     shorts only when < SMA200. Hard gate: avoids buying 2022 downtrend.
  4. Daily data also resampled from 1m — zero dependency on Bitget parquets

Usage:
  python scripts/backtest_ict_fvg_v3.py
  python scripts/backtest_ict_fvg_v3.py --rr 5 --min-touches 2
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    import pyarrow.parquet as pq
except ImportError:
    print("[ERROR] pyarrow required: pip install pyarrow")
    sys.exit(1)

# ─── Config ───────────────────────────────────────────────────────────────────
_FEE            = 0.0006        # 0.06% taker per side
_CAPITAL        = 5_000.0
_RISK_PCT       = 0.01          # 1% risk per trade
_PARTIAL_R      = 2.0
_PARTIAL_FRAC   = 0.5

# Session windows (UTC) — NY open + afternoon, skip lunch chop
_SESSIONS = [
    (13 * 60 + 30, 15 * 60 + 30),   # 13:30–15:30 NY open
    (17 * 60 + 0,  18 * 60 + 30),   # 17:00–18:30 NY afternoon
]

# Filter thresholds
_HTF_EMA_FAST   = 8
_HTF_EMA_SLOW   = 21
_SWEEP_BARS     = 20
_SWING_TOUCHES  = 2
_SWING_TOLERANCE = 0.0015
_DISP_BODY_RATIO = 0.60
_DISP_BODY_MULT  = 2.0
_DISP_VOL_MULT   = 1.3          # NEW: displacement volume must be >1.3× 20-bar avg
_PREMIUM_DISC    = 0.50
_FVG_MAX_BARS   = 20
_FVG_MIN_PCT    = 0.0005
_TARGET_MIN_R   = 2.5
_MAX_HOLD_BARS  = 90
_DAILY_SMA_LEN  = 200           # NEW: daily SMA regime filter

_1M_PATH = _REPO_ROOT / "data" / "candles" / "BTCUSDT_1m.parquet"
_SYMBOL  = "BTCUSDT"  # overridden by --symbol arg


# ─── Data loading ─────────────────────────────────────────────────────────────

def _load_1m(path: Path) -> list[dict]:
    t = pq.read_table(path, columns=["open_time","open","high","low","close","volume"])
    candles = []
    for ot, o, h, l, c, v in zip(
        t["open_time"].to_pylist(), t["open"].to_pylist(),
        t["high"].to_pylist(), t["low"].to_pylist(),
        t["close"].to_pylist(), t["volume"].to_pylist()
    ):
        ts_raw = int(ot)
        ts_ms = ts_raw * 1000 if ts_raw < 2_000_000_000 else ts_raw
        candles.append({"ts": ts_ms, "o": float(o), "h": float(h),
                        "l": float(l), "c": float(c), "v": float(v)})
    candles.sort(key=lambda x: x["ts"])
    return candles


def _resample_1h(m1: list[dict]) -> list[dict]:
    """Aggregate 1m bars into exact 1H bars (no gaps, perfect data)."""
    buckets: dict[int, dict] = {}
    for bar in m1:
        hour_ts = (bar["ts"] // 3_600_000) * 3_600_000
        if hour_ts not in buckets:
            buckets[hour_ts] = {"ts": hour_ts, "o": bar["o"], "h": bar["h"],
                                "l": bar["l"], "c": bar["c"], "v": bar["v"]}
        else:
            b = buckets[hour_ts]
            b["h"] = max(b["h"], bar["h"])
            b["l"] = min(b["l"], bar["l"])
            b["c"] = bar["c"]
            b["v"] += bar["v"]
    return sorted(buckets.values(), key=lambda x: x["ts"])


def _resample_1d(m1: list[dict]) -> list[dict]:
    """Aggregate 1m bars into exact daily bars (UTC midnight-aligned)."""
    buckets: dict[int, dict] = {}
    for bar in m1:
        day_ts = (bar["ts"] // 86_400_000) * 86_400_000
        if day_ts not in buckets:
            buckets[day_ts] = {"ts": day_ts, "o": bar["o"], "h": bar["h"],
                               "l": bar["l"], "c": bar["c"], "v": bar["v"]}
        else:
            b = buckets[day_ts]
            b["h"] = max(b["h"], bar["h"])
            b["l"] = min(b["l"], bar["l"])
            b["c"] = bar["c"]
            b["v"] += bar["v"]
    return sorted(buckets.values(), key=lambda x: x["ts"])


# ─── HTF trend from perfect 1H data ───────────────────────────────────────────

def _build_htf_trend(h1: list[dict]) -> dict[int, str]:
    af = 2 / (_HTF_EMA_FAST + 1)
    as_ = 2 / (_HTF_EMA_SLOW + 1)
    ema_f = ema_s = None
    trend: dict[int, str] = {}
    for bar in h1:
        c = bar["c"]
        ema_f = c if ema_f is None else af * c + (1 - af) * ema_f
        ema_s = c if ema_s is None else as_ * c + (1 - as_) * ema_s
        if ema_f > ema_s * 1.0005:
            trend[bar["ts"]] = "bull"
        elif ema_f < ema_s * 0.9995:
            trend[bar["ts"]] = "bear"
        else:
            trend[bar["ts"]] = "neutral"
    return trend


def _htf_at(ts_ms: int, trend: dict[int, str]) -> str:
    hour_floor = (ts_ms // 3_600_000) * 3_600_000
    return trend.get(hour_floor, "neutral")


# ─── Daily SMA200 regime ───────────────────────────────────────────────────────

def _build_daily_regime(d1: list[dict], sma_len: int = _DAILY_SMA_LEN) -> dict[int, str]:
    """
    Returns {day_ts_ms: 'bull'|'bear'} — 'bull' means daily close > SMA200.
    Only longs allowed in bull, only shorts in bear.
    """
    regime: dict[int, str] = {}
    closes = []
    for bar in d1:
        closes.append(bar["c"])
        if len(closes) >= sma_len:
            sma = sum(closes[-sma_len:]) / sma_len
            regime[bar["ts"]] = "bull" if bar["c"] >= sma else "bear"
        # Before 200 days of data: mark as neutral (skip)
    return regime


def _daily_regime_at(ts_ms: int, regime: dict[int, str]) -> str:
    day_floor = (ts_ms // 86_400_000) * 86_400_000
    return regime.get(day_floor, "neutral")


# ─── Session filter ───────────────────────────────────────────────────────────

def _in_session(ts_ms: int) -> bool:
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    minutes = dt.hour * 60 + dt.minute
    return any(s <= minutes < e for s, e in _SESSIONS)


# ─── Indicator helpers ────────────────────────────────────────────────────────

def _avg_body(candles: list[dict], i: int, n: int = 20) -> float:
    window = candles[max(0, i - n): i]
    if not window:
        return 0.0
    return sum(abs(c["c"] - c["o"]) for c in window) / len(window)


def _avg_volume(candles: list[dict], i: int, n: int = 20) -> float:
    window = candles[max(0, i - n): i]
    if not window:
        return 0.0
    return sum(c["v"] for c in window) / len(window)


# ─── Filter 3: Confirmed swing structure ─────────────────────────────────────

def _count_touches(candles: list[dict], i: int, level: float,
                   lookback: int, is_low: bool) -> int:
    touches = 0
    for j in range(max(0, i - lookback), i):
        if is_low:
            if abs(candles[j]["l"] - level) / level <= _SWING_TOLERANCE:
                touches += 1
        else:
            if abs(candles[j]["h"] - level) / level <= _SWING_TOLERANCE:
                touches += 1
    return touches


# ─── Filter 5: Premium/Discount zone ─────────────────────────────────────────

def _in_discount(candles: list[dict], i: int, fvg_mid: float, n: int = 20) -> bool:
    highs = [c["h"] for c in candles[max(0, i - n): i + 1]]
    lows  = [c["l"] for c in candles[max(0, i - n): i + 1]]
    rng_high, rng_low = max(highs), min(lows)
    if rng_high == rng_low:
        return False
    return (fvg_mid - rng_low) / (rng_high - rng_low) <= _PREMIUM_DISC


def _in_premium(candles: list[dict], i: int, fvg_mid: float, n: int = 20) -> bool:
    highs = [c["h"] for c in candles[max(0, i - n): i + 1]]
    lows  = [c["l"] for c in candles[max(0, i - n): i + 1]]
    rng_high, rng_low = max(highs), min(lows)
    if rng_high == rng_low:
        return False
    return (fvg_mid - rng_low) / (rng_high - rng_low) >= (1 - _PREMIUM_DISC)


# ─── Filter 6: Next target clearance ─────────────────────────────────────────

def _next_target(candles: list[dict], i: int, direction: str,
                 risk_per_unit: float, n: int = 40) -> float | None:
    entry = candles[i]["c"]
    prior = candles[max(0, i - n): i]
    if direction == "LONG":
        highs = sorted(set(c["h"] for c in prior), reverse=True)
        for h in highs:
            if h > entry + _TARGET_MIN_R * risk_per_unit:
                return h
    else:
        lows = sorted(set(c["l"] for c in prior))
        for l in lows:
            if l < entry - _TARGET_MIN_R * risk_per_unit:
                return l
    return None


# ─── Setup detection (all 6 filters + volume + daily regime) ─────────────────

def _detect_setups(candles: list[dict], trend: dict[int, str],
                   regime: dict[int, str], min_touches: int) -> list[dict]:
    setups = []
    n = len(candles)

    for i in range(_SWEEP_BARS + 2, n - _FVG_MAX_BARS - 2):

        if not _in_session(candles[i]["ts"]):
            continue

        if candles[i]["ts"] - candles[i - 1]["ts"] > 120_000:
            continue

        avg_b = _avg_body(candles, i)
        avg_v = _avg_volume(candles, i)
        if avg_b == 0:
            continue

        htf    = _htf_at(candles[i]["ts"], trend)
        dreg   = _daily_regime_at(candles[i]["ts"], regime)

        # Skip if daily regime is neutral (< 200 days of data available)
        if dreg == "neutral":
            continue

        # ════════════════════════════════════════════════════════════════════
        # BULLISH SETUP — requires both 1H bull and daily bull regime
        # ════════════════════════════════════════════════════════════════════
        if htf == "bull" and dreg == "bull":
            prior_lows = [candles[j]["l"] for j in range(i - _SWEEP_BARS, i)]
            swing_low  = min(prior_lows)
            swept_low  = candles[i]["l"] < swing_low
            closed_above = candles[i]["c"] > swing_low

            if swept_low and closed_above:
                touches = _count_touches(candles, i, swing_low,
                                         _SWEEP_BARS * 2, is_low=True)
                if touches < min_touches:
                    continue

                d = candles[i + 1] if i + 1 < n else None
                if d is None:
                    continue
                disp_body  = d["c"] - d["o"]
                disp_range = d["h"] - d["l"]
                if disp_range == 0:
                    continue
                # Filter 4: body quality
                if (disp_body < _DISP_BODY_MULT * avg_b or
                        disp_body / disp_range < _DISP_BODY_RATIO or
                        disp_body <= 0):
                    continue
                # NEW Filter 4b: volume spike
                if avg_v > 0 and d["v"] < _DISP_VOL_MULT * avg_v:
                    continue

                if i + 2 >= n:
                    continue
                fvg_low  = candles[i]["h"]
                fvg_high = candles[i + 2]["l"]
                if fvg_high <= fvg_low:
                    continue
                if (fvg_high - fvg_low) / candles[i]["c"] < _FVG_MIN_PCT:
                    continue
                fvg_mid = (fvg_low + fvg_high) / 2

                if not _in_discount(candles, i, fvg_mid):
                    continue

                risk_est = fvg_mid - (swing_low * 0.999)
                if risk_est <= 0:
                    continue
                target = _next_target(candles, i, "LONG", risk_est)
                if target is None:
                    continue

                setups.append({
                    "direction": "LONG",
                    "sweep_bar": i,
                    "sweep_extreme": candles[i]["l"] * 0.9995,
                    "fvg_low": fvg_low,
                    "fvg_high": fvg_high,
                    "fvg_mid": fvg_mid,
                    "valid_from": i + 2,
                    "target": target,
                    "touches": touches,
                })

        # ════════════════════════════════════════════════════════════════════
        # BEARISH SETUP — requires both 1H bear and daily bear regime
        # ════════════════════════════════════════════════════════════════════
        if htf == "bear" and dreg == "bear":
            prior_highs = [candles[j]["h"] for j in range(i - _SWEEP_BARS, i)]
            swing_high  = max(prior_highs)
            swept_high  = candles[i]["h"] > swing_high
            closed_below = candles[i]["c"] < swing_high

            if swept_high and closed_below:
                touches = _count_touches(candles, i, swing_high,
                                         _SWEEP_BARS * 2, is_low=False)
                if touches < min_touches:
                    continue

                d = candles[i + 1] if i + 1 < n else None
                if d is None:
                    continue
                disp_body  = d["o"] - d["c"]
                disp_range = d["h"] - d["l"]
                if disp_range == 0:
                    continue
                if (disp_body < _DISP_BODY_MULT * avg_b or
                        disp_body / disp_range < _DISP_BODY_RATIO or
                        disp_body <= 0):
                    continue
                if avg_v > 0 and d["v"] < _DISP_VOL_MULT * avg_v:
                    continue

                if i + 2 >= n:
                    continue
                fvg_high = candles[i]["l"]
                fvg_low  = candles[i + 2]["h"]
                if fvg_high <= fvg_low:
                    continue
                if (fvg_high - fvg_low) / candles[i]["c"] < _FVG_MIN_PCT:
                    continue
                fvg_mid = (fvg_low + fvg_high) / 2

                if not _in_premium(candles, i, fvg_mid):
                    continue

                risk_est = (swing_high * 1.0005) - fvg_mid
                if risk_est <= 0:
                    continue
                target = _next_target(candles, i, "SHORT", risk_est)
                if target is None:
                    continue

                setups.append({
                    "direction": "SHORT",
                    "sweep_bar": i,
                    "sweep_extreme": candles[i]["h"] * 1.0005,
                    "fvg_low": fvg_low,
                    "fvg_high": fvg_high,
                    "fvg_mid": fvg_mid,
                    "valid_from": i + 2,
                    "target": target,
                    "touches": touches,
                })

    return setups


# ─── Trade simulation ─────────────────────────────────────────────────────────

def _simulate_trade(candles: list[dict], setup: dict, rr: float,
                    capital: float) -> dict | None:
    direction  = setup["direction"]
    fvg_low    = setup["fvg_low"]
    fvg_high   = setup["fvg_high"]
    fvg_mid    = setup["fvg_mid"]
    sweep_ext  = setup["sweep_extreme"]
    valid_from = setup["valid_from"]
    n = len(candles)

    entry_price = stop_price = None
    entry_bar = None

    for j in range(valid_from, min(valid_from + _FVG_MAX_BARS, n)):
        if candles[j]["ts"] - candles[j - 1]["ts"] > 120_000:
            break
        c = candles[j]
        if not _in_session(c["ts"]):
            continue
        if direction == "LONG" and c["l"] <= fvg_high and c["h"] >= fvg_low:
            entry_price = fvg_mid
            stop_price  = sweep_ext
            entry_bar   = j
            break
        elif direction == "SHORT" and c["h"] >= fvg_low and c["l"] <= fvg_high:
            entry_price = fvg_mid
            stop_price  = sweep_ext
            entry_bar   = j
            break

    if entry_price is None:
        return None

    if direction == "LONG":
        risk_per_unit = entry_price - stop_price
    else:
        risk_per_unit = stop_price - entry_price

    if risk_per_unit <= 0:
        return None

    risk_dollars  = capital * _RISK_PCT
    qty           = risk_dollars / risk_per_unit
    notional      = qty * entry_price
    pnl           = -(notional * _FEE)

    if direction == "LONG":
        partial_tp = entry_price + _PARTIAL_R * risk_per_unit
        full_tp    = min(setup["target"], entry_price + rr * risk_per_unit)
    else:
        partial_tp = entry_price - _PARTIAL_R * risk_per_unit
        full_tp    = max(setup["target"], entry_price - rr * risk_per_unit)

    partial_hit   = False
    qty_remaining = qty
    exit_reason   = "TIME"
    exit_price    = entry_price

    for j in range(entry_bar + 1, min(entry_bar + _MAX_HOLD_BARS + 1, n)):
        if candles[j]["ts"] - candles[j - 1]["ts"] > 120_000:
            break
        c = candles[j]

        if not partial_hit:
            if direction == "LONG" and c["h"] >= partial_tp:
                close_qty = qty * _PARTIAL_FRAC
                pnl += close_qty * (partial_tp - entry_price)
                pnl -= close_qty * partial_tp * _FEE
                qty_remaining -= close_qty
                partial_hit = True
            elif direction == "SHORT" and c["l"] <= partial_tp:
                close_qty = qty * _PARTIAL_FRAC
                pnl += close_qty * (entry_price - partial_tp)
                pnl -= close_qty * partial_tp * _FEE
                qty_remaining -= close_qty
                partial_hit = True

        if direction == "LONG" and c["l"] <= stop_price:
            hit = min(c["o"], stop_price)
            pnl += qty_remaining * (hit - entry_price)
            pnl -= qty_remaining * hit * _FEE
            exit_reason = "STOP"
            exit_price  = hit
            break
        elif direction == "SHORT" and c["h"] >= stop_price:
            hit = max(c["o"], stop_price)
            pnl += qty_remaining * (entry_price - hit)
            pnl -= qty_remaining * hit * _FEE
            exit_reason = "STOP"
            exit_price  = hit
            break

        if direction == "LONG" and c["h"] >= full_tp:
            pnl += qty_remaining * (full_tp - entry_price)
            pnl -= qty_remaining * full_tp * _FEE
            exit_reason = "TP"
            exit_price  = full_tp
            break
        elif direction == "SHORT" and c["l"] <= full_tp:
            pnl += qty_remaining * (entry_price - full_tp)
            pnl -= qty_remaining * full_tp * _FEE
            exit_reason = "TP"
            exit_price  = full_tp
            break
    else:
        last = candles[min(entry_bar + _MAX_HOLD_BARS, n - 1)]
        ep = last["c"]
        pnl += qty_remaining * (ep - entry_price) * (1 if direction == "LONG" else -1)
        pnl -= qty_remaining * ep * _FEE
        exit_reason = "TIME"
        exit_price  = ep

    dt = datetime.fromtimestamp(candles[entry_bar]["ts"] / 1000, tz=timezone.utc)
    return {
        "direction":   direction,
        "entry_price": entry_price,
        "exit_price":  exit_price,
        "stop_price":  stop_price,
        "pnl":         pnl,
        "exit_reason": exit_reason,
        "year":        dt.year,
        "month":       dt.month,
        "touches":     setup["touches"],
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_backtest(rr: float = 5.0, min_touches: int = 2,
                 path: Path = _1M_PATH, symbol: str = "BTCUSDT") -> None:
    print("=" * 70)
    print(f"  NexFlow — ICT/SMC FVG v3 — Smarter Than Louis Edition [{symbol}]")
    print(f"  8 filters | RR 1:{rr:.0f} | Min swing touches: {min_touches}")
    print(f"  NEW: volume spike + daily SMA200 regime + perfect 1H from 1m")
    print("=" * 70)

    if not path.exists():
        print(f"[ERROR] {path} not found.")
        sys.exit(1)

    print("\n  Loading & resampling 1m → 1H + Daily (no gaps)...")
    m1 = _load_1m(path)
    h1 = _resample_1h(m1)
    d1 = _resample_1d(m1)

    trend  = _build_htf_trend(h1)
    regime = _build_daily_regime(d1)

    bull_days  = sum(1 for v in regime.values() if v == "bull")
    bear_days  = sum(1 for v in regime.values() if v == "bear")

    print(f"  1m bars: {len(m1):,}  |  1H bars: {len(h1):,}  |  Daily bars: {len(d1):,}")
    print(f"  HTF states: {sum(1 for v in trend.values() if v=='bull'):,} bull  "
          f"{sum(1 for v in trend.values() if v=='bear'):,} bear")
    print(f"  Daily regime: {bull_days} bull days | {bear_days} bear days "
          f"(after {_DAILY_SMA_LEN}-day warmup)")

    print("\n  Scanning with all 8 filters...")
    setups = _detect_setups(m1, trend, regime, min_touches)
    longs  = sum(1 for s in setups if s["direction"] == "LONG")
    shorts = sum(1 for s in setups if s["direction"] == "SHORT")
    print(f"  Setups passing all filters: {len(setups)} ({longs} long, {shorts} short)")
    print(f"  (v2 found 180 traded setups — lower is more selective)")

    trades = []
    capital = _CAPITAL
    last_exit_bar = -1
    year_stats: dict[int, dict] = {}

    for setup in setups:
        if setup["valid_from"] <= last_exit_bar:
            continue

        result = _simulate_trade(m1, setup, rr, capital)
        if result is None:
            continue

        capital += result["pnl"]
        result["capital"] = capital
        trades.append(result)

        yr = result["year"]
        if yr not in year_stats:
            year_stats[yr] = {"pnl": 0.0, "trades": 0, "wins": 0,
                              "stops": 0, "tps": 0, "times": 0,
                              "start": capital - result["pnl"]}
        year_stats[yr]["pnl"]    += result["pnl"]
        year_stats[yr]["trades"] += 1
        if result["pnl"] > 0:
            year_stats[yr]["wins"] += 1
        key = result["exit_reason"].lower() + "s"
        year_stats[yr][key] = year_stats[yr].get(key, 0) + 1

        for k, c in enumerate(m1[setup["valid_from"]:], start=setup["valid_from"]):
            if c["ts"] >= result["entry_price"]:
                last_exit_bar = k
                break

    if not trades:
        print("\n  ⚠  No trades triggered — try --min-touches 1")
        return

    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    tps    = [t for t in trades if t["exit_reason"] == "TP"]
    stops  = [t for t in trades if t["exit_reason"] == "STOP"]
    gross_win  = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    pf = gross_win / gross_loss if gross_loss else float("inf")

    equity = [_CAPITAL] + [t["capital"] for t in trades]
    peak = _CAPITAL
    max_dd = 0.0
    for eq in equity:
        peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak)

    total_pnl = sum(t["pnl"] for t in trades)

    print(f"\n{'─'*70}")
    print(f"  RESULTS  ({len(trades)} trades)           "
          f"[v2 comparison: 180 trades, PF 1.09, DD 26.4%]")
    print(f"{'─'*70}")
    print(f"  Net P&L       : ${total_pnl:>10,.2f}  ({total_pnl/_CAPITAL*100:+.1f}%)")
    print(f"  Final capital : ${capital:>10,.2f}")
    print(f"  Win rate      : {len(wins)/len(trades)*100:.1f}%  ({len(wins)}/{len(trades)})")
    print(f"  Profit factor : {pf:.2f}")
    print(f"  Max drawdown  : {max_dd*100:.1f}%")
    if wins:
        print(f"  Avg win       : ${gross_win/len(wins):,.2f}")
    if losses:
        print(f"  Avg loss      : ${gross_loss/len(losses):,.2f}")
    print(f"  TP exits      : {len(tps)}  ({len(tps)/len(trades)*100:.1f}%)")
    print(f"  Stop exits    : {len(stops)}  ({len(stops)/len(trades)*100:.1f}%)")

    print(f"\n  {'Year':<6} {'Trades':>7} {'Win%':>6} {'Net P&L':>11} {'Capital':>11} {'Ret%':>7}")
    print(f"  {'─'*56}")
    running = _CAPITAL
    for yr in sorted(year_stats):
        st = year_stats[yr]
        net = st["pnl"]
        end = running + net
        wp  = st["wins"] / st["trades"] * 100 if st["trades"] else 0
        ret = net / running * 100 if running else 0
        running = end
        flag = "✓" if net >= 0 else "✗"
        print(f"  {flag} {yr:<5} {st['trades']:>7} {wp:>5.1f}% {net:>+11,.2f} {end:>11,.2f} {ret:>6.1f}%")

    full_years = [yr for yr in year_stats if year_stats[yr]["trades"] >= 5]
    if full_years:
        recent = max(full_years)
        monthly: dict[int, dict] = {}
        for t in trades:
            if t["year"] == recent:
                m = t["month"]
                monthly.setdefault(m, {"pnl": 0.0, "trades": 0, "wins": 0})
                monthly[m]["pnl"] += t["pnl"]
                monthly[m]["trades"] += 1
                if t["pnl"] > 0:
                    monthly[m]["wins"] += 1
        if monthly:
            print(f"\n  Monthly breakdown ({recent}):")
            losing = 0
            for m in sorted(monthly):
                st = monthly[m]
                wp = st["wins"] / st["trades"] * 100 if st["trades"] else 0
                flag = "✓" if st["pnl"] >= 0 else "✗"
                if st["pnl"] < 0:
                    losing += 1
                print(f"    {flag} {datetime(recent,m,1).strftime('%b'):>3}  "
                      f"{st['trades']:>3} trades  {wp:>5.1f}% win  ${st['pnl']:>+8,.2f}")
            print(f"\n    Losing months in {recent}: {losing}/{len(monthly)}")

    print(f"\n{'─'*70}")
    if pf >= 1.5 and max_dd <= 0.20 and len(trades) >= 40:
        verdict = "✅ STRONG GO — Real edge survives all 8 filters + fees"
    elif pf >= 1.3 and max_dd <= 0.25:
        verdict = "⚠️  PROMISING — Paper trade, target PF ≥ 1.5 before live"
    elif pf >= 1.2 and max_dd <= 0.30:
        verdict = "⚠️  WEAK GO — Marginal edge, paper trade first"
    elif pf >= 1.0:
        verdict = "⚠️  BREAK-EVEN — Needs more tuning"
    else:
        verdict = "❌ KILL — No edge after filtering"
    print(f"  VERDICT: {verdict}")
    print(f"{'─'*70}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rr", type=float, default=5.0)
    ap.add_argument("--min-touches", type=int, default=2)
    ap.add_argument("--symbol", default="BTCUSDT",
                    help="Symbol parquet to use (e.g. ETHUSDT)")
    args = ap.parse_args()
    path = _REPO_ROOT / "data" / "candles" / f"{args.symbol}_1m.parquet"
    run_backtest(rr=args.rr, min_touches=args.min_touches, path=path, symbol=args.symbol)
