#!/usr/bin/env python3
"""
backtest_ict_fvg_v2.py — ICT/SMC FVG Scalping — "Better Than Louis" Edition

Version 1 (mechanical) result: PF 0.46, wiped $5K. Every parameter set failed.
Root cause: taking every mechanical signal. Louis takes ~10% of signals — the
ones where 6 context filters all agree. This version encodes those filters.

FILTERS ADDED vs v1:
  1. HTF Trend Alignment  — 1H EMA(8) > EMA(21) for longs, < for shorts
                            (kills fighting the trend — biggest edge killer)
  2. Session precision    — Only NY open (13:30–15:30 UTC) and afternoon
                            (17:00–18:30 UTC). Skip lunch chop (15:30–17:00)
  3. Confirmed swing      — Swept level must have been respected (bounced)
                            at least TWICE in prior 40 bars (real structure)
  4. Displacement quality — Post-sweep candle: body/range > 0.60 AND body
                            > 2× 20-bar avg body (real institutional move)
  5. Premium/Discount     — Longs only in lower 40% of 20-bar range;
                            Shorts only in upper 40% (never buy expensive)
  6. Next target clearance— Must be a clear unswept swing high/low at least
                            2.5R away before taking the trade

Usage:
  python scripts/backtest_ict_fvg_v2.py
  python scripts/backtest_ict_fvg_v2.py --rr 5 --min-touches 2
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

# Session windows (UTC) — NY open + afternoon only, skip lunch chop
_SESSIONS = [
    (13 * 60 + 30, 15 * 60 + 30),   # 13:30–15:30 NY open
    (17 * 60 + 0,  18 * 60 + 30),   # 17:00–18:30 NY afternoon
]

# Filter thresholds
_HTF_EMA_FAST   = 8
_HTF_EMA_SLOW   = 21
_SWEEP_BARS     = 20            # look back for structure level
_SWING_TOUCHES  = 2             # minimum prior touches before the sweep
_SWING_TOLERANCE = 0.0015       # 0.15% tolerance for "touching" a level
_DISP_BODY_RATIO = 0.60         # displacement body/range >= this
_DISP_BODY_MULT  = 2.0          # body >= this × 20-bar avg body
_PREMIUM_DISC    = 0.40         # longs in bottom 40%, shorts in top 40%
_FVG_MAX_BARS   = 20            # bars to wait for retrace into FVG
_FVG_MIN_PCT    = 0.0005        # minimum FVG size (0.05% of price)
_TARGET_MIN_R   = 2.5           # next target must be >= 2.5R away
_MAX_HOLD_BARS  = 90            # time stop: 1.5 hours

_1M_PATH = _REPO_ROOT / "data" / "candles" / "BTCUSDT_1m.parquet"
_1H_PATH = _REPO_ROOT / "data" / "candles" / "BTCUSDT_1H.parquet"


# ─── Data loading ─────────────────────────────────────────────────────────────

def _load_candles(path: Path) -> list[dict]:
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


# ─── HTF EMA builder ──────────────────────────────────────────────────────────

def _build_htf_trend(h1_candles: list[dict]) -> dict[int, str]:
    """
    Returns {hour_ts_ms: 'bull'|'bear'|'neutral'} for every 1H bar.
    Uses EMA(8) vs EMA(21) of 1H closes.
    """
    af = 2 / (_HTF_EMA_FAST + 1)
    as_ = 2 / (_HTF_EMA_SLOW + 1)
    ema_f = ema_s = None
    trend: dict[int, str] = {}

    for bar in h1_candles:
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
    """Return the HTF trend for the 1H bar that contains this 1m timestamp."""
    hour_floor = (ts_ms // 3_600_000) * 3_600_000
    return trend.get(hour_floor, "neutral")


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
    """Count how many bars in [i-lookback, i-1] touched the given level."""
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

def _in_discount(candles: list[dict], i: int, fvg_mid: float,
                 n: int = 20) -> bool:
    """Longs: FVG midpoint in lower 40% of recent range."""
    highs = [c["h"] for c in candles[max(0, i - n): i + 1]]
    lows  = [c["l"] for c in candles[max(0, i - n): i + 1]]
    rng_high, rng_low = max(highs), min(lows)
    if rng_high == rng_low:
        return False
    pct = (fvg_mid - rng_low) / (rng_high - rng_low)
    return pct <= _PREMIUM_DISC


def _in_premium(candles: list[dict], i: int, fvg_mid: float,
                n: int = 20) -> bool:
    """Shorts: FVG midpoint in upper 40% of recent range."""
    highs = [c["h"] for c in candles[max(0, i - n): i + 1]]
    lows  = [c["l"] for c in candles[max(0, i - n): i + 1]]
    rng_high, rng_low = max(highs), min(lows)
    if rng_high == rng_low:
        return False
    pct = (fvg_mid - rng_low) / (rng_high - rng_low)
    return pct >= (1 - _PREMIUM_DISC)


# ─── Filter 6: Next target clearance ─────────────────────────────────────────

def _next_target(candles: list[dict], i: int, direction: str,
                 risk_per_unit: float, n: int = 40) -> float | None:
    """
    Find the nearest unswept swing high (long) or swing low (short)
    that is at least _TARGET_MIN_R × risk_per_unit away.
    Returns the target price or None if no valid target found.
    """
    entry = candles[i]["c"]
    future = candles[i + 1: i + n + 1]
    if direction == "LONG":
        # Look at prior swing highs in the next N bars of history
        # (use prior bars as proxy for where liquidity sits)
        prior = candles[max(0, i - n): i]
        highs = sorted(set(c["h"] for c in prior), reverse=True)
        for h in highs:
            if h > entry + _TARGET_MIN_R * risk_per_unit:
                return h
    else:
        prior = candles[max(0, i - n): i]
        lows = sorted(set(c["l"] for c in prior))
        for l in lows:
            if l < entry - _TARGET_MIN_R * risk_per_unit:
                return l
    return None


# ─── Setup detection (with all 6 filters) ────────────────────────────────────

def _detect_setups(candles: list[dict], trend: dict[int, str],
                   min_touches: int) -> list[dict]:
    setups = []
    n = len(candles)

    for i in range(_SWEEP_BARS + 2, n - _FVG_MAX_BARS - 2):

        # ── Filter 1: Session precision ──────────────────────────────────────
        if not _in_session(candles[i]["ts"]):
            continue

        # ── Filter 2: No gap before this bar (inside a real data batch) ──────
        if candles[i]["ts"] - candles[i - 1]["ts"] > 120_000:
            continue   # batch boundary — structure is artifact, not real

        avg_b = _avg_body(candles, i)
        if avg_b == 0:
            continue

        htf = _htf_at(candles[i]["ts"], trend)

        # ════════════════════════════════════════════════════════════════════
        # BULLISH SETUP
        # ════════════════════════════════════════════════════════════════════
        if htf == "bull":
            prior_lows = [candles[j]["l"] for j in range(i - _SWEEP_BARS, i)]
            swing_low  = min(prior_lows)
            swept_low  = candles[i]["l"] < swing_low
            closed_above = candles[i]["c"] > swing_low

            if swept_low and closed_above:
                # Filter 3: Confirmed structure (swing_low had real touches)
                touches = _count_touches(candles, i, swing_low,
                                         _SWEEP_BARS * 2, is_low=True)
                if touches < min_touches:
                    continue

                # Filter 4: Displacement quality
                d = candles[i + 1] if i + 1 < n else None
                if d is None:
                    continue
                disp_body = d["c"] - d["o"]
                disp_range = d["h"] - d["l"]
                if disp_range == 0:
                    continue
                if (disp_body < _DISP_BODY_MULT * avg_b or
                        disp_body / disp_range < _DISP_BODY_RATIO or
                        disp_body <= 0):
                    continue

                # FVG zone
                if i + 2 >= n:
                    continue
                fvg_low  = candles[i]["h"]
                fvg_high = candles[i + 2]["l"]
                if fvg_high <= fvg_low:
                    continue
                if (fvg_high - fvg_low) / candles[i]["c"] < _FVG_MIN_PCT:
                    continue
                fvg_mid = (fvg_low + fvg_high) / 2

                # Filter 5: Discount zone
                if not _in_discount(candles, i, fvg_mid):
                    continue

                # Filter 6: Next target
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
        # BEARISH SETUP
        # ════════════════════════════════════════════════════════════════════
        if htf == "bear":
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
                disp_body = d["o"] - d["c"]
                disp_range = d["h"] - d["l"]
                if disp_range == 0:
                    continue
                if (disp_body < _DISP_BODY_MULT * avg_b or
                        disp_body / disp_range < _DISP_BODY_RATIO or
                        disp_body <= 0):
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
            break   # don't wait across a data gap
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
    pnl           = -(notional * _FEE)   # entry fee

    # Targets — use actual structure target for the runner if achievable,
    # otherwise fall back to fixed RR
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
            break   # data gap = time stop
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

        # Stop loss
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

        # Full TP
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

def run_backtest(rr: float = 5.0, min_touches: int = 2) -> None:
    print("=" * 70)
    print("  NexFlow — ICT/SMC FVG v2 — Better Than Louis Edition")
    print(f"  6 context filters | RR 1:{rr:.0f} | Min swing touches: {min_touches}")
    print(f"  Sessions: NY open (13:30–15:30) + Afternoon (17:00–18:30 UTC)")
    print("=" * 70)

    if not _1M_PATH.exists():
        print(f"[ERROR] {_1M_PATH} not found. Download first:")
        print("  python scripts/download_candles.py --symbol BTCUSDT --start 2021-01-01")
        sys.exit(1)

    m1 = _load_candles(_1M_PATH)
    h1 = _load_candles(_1H_PATH)
    trend = _build_htf_trend(h1)

    print(f"\n  1m bars: {len(m1):,}  |  1H bars: {len(h1):,}")
    print(f"  HTF states: {sum(1 for v in trend.values() if v=='bull'):,} bull  "
          f"{sum(1 for v in trend.values() if v=='bear'):,} bear  "
          f"{sum(1 for v in trend.values() if v=='neutral'):,} neutral")

    print("\n  Scanning with all 6 filters...")
    setups = _detect_setups(m1, trend, min_touches)
    longs  = sum(1 for s in setups if s["direction"] == "LONG")
    shorts = sum(1 for s in setups if s["direction"] == "SHORT")
    print(f"  Setups passing all filters: {len(setups)} ({longs} long, {shorts} short)")
    print(f"  (v1 found 479 raw setups — reduction shows filter strictness)")

    trades = []
    capital = _CAPITAL
    last_exit_bar = -1
    year_stats: dict[int, dict] = {}
    filters_rejected = {"htf": 0, "session": 0, "gap": 0, "touches": 0,
                        "disp": 0, "zone": 0, "target": 0}

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
        print("\n  ⚠  No trades triggered — filters may be too strict.")
        print("     Try: --min-touches 1")
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
          f"[v1 comparison: 472 trades, PF 0.46]")
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
        print(f"  {yr:<6} {st['trades']:>7} {wp:>5.1f}% {net:>+11,.2f} {end:>11,.2f} {ret:>6.1f}%")

    # Monthly breakdown for most recent full year
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
    if pf >= 1.5 and max_dd <= 0.20 and len(trades) >= 50:
        verdict = "✅ STRONG GO — Real edge survives all 6 filters + fees"
    elif pf >= 1.2 and max_dd <= 0.30:
        verdict = "⚠️  WEAK GO — Marginal edge, paper trade first"
    elif pf >= 1.0:
        verdict = "⚠️  BREAK-EVEN — Needs more data / tuning"
    else:
        verdict = "❌ KILL — Still no edge after filtering"
    print(f"  VERDICT: {verdict}")
    print(f"{'─'*70}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rr", type=float, default=5.0)
    ap.add_argument("--min-touches", type=int, default=2)
    args = ap.parse_args()
    run_backtest(rr=args.rr, min_touches=args.min_touches)
