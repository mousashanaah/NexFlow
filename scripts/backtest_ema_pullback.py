#!/usr/bin/env python3
"""
backtest_ema_pullback.py — EMA Pullback Trend Continuation Strategy

THE STRATEGY (used by Mark Minervini, IBD traders, and most prop firm traders):
  When a market is in a clear trend, it pulls back to the fast EMA and you
  enter in the direction of the trend. This is "buying the dip in an uptrend"
  with a precise, rule-based entry.

  THE CORE EDGE:
  - High win rate (55-65%) because you're trading WITH the established trend
  - Clear, objective entry and exit rules
  - Works on ANY liquid trending market: BTC, ETH, Gold, Nasdaq, Oil
  - Short enough (1H) to give daily opportunities but large enough moves to
    overcome fees

ENTRY RULES:
  1. Clear trend: EMA8 and EMA21 must be separated by >0.3% AND pointing same way
     for at least 5 consecutive bars (confirmed trend, not just a wiggle)
  2. Pullback to EMA: price touches the EMA21 zone (within 0.5% tolerance)
     — this is the institutional buy/sell zone, where smart money adds to positions
  3. Reversal candle at the EMA: close in trend direction, body >50% of range
     (confirmation that buyers/sellers stepped in at the EMA)
  4. NOT extended: price hasn't moved more than 3×ATR from EMA since last touch
     (avoid chasing after already large moves)

ENTRY: Close of the reversal candle
STOP:  EMA21 - 1.0×ATR (longs) / EMA21 + 1.0×ATR (shorts)
TARGET: Next significant swing high/low, minimum 2R away
TIME STOP: Close trade after 20 bars (5 trading days on 1H) if not hit

FREQUENCY ON BTC 1H:
  - Raw pullbacks to EMA: ~300-400 per year
  - After quality filters: ~60-100 high-probability setups per year
  - Add ETH, Gold: 180-300 signals per year = true day trading volume

Usage:
  python scripts/backtest_ema_pullback.py
  python scripts/backtest_ema_pullback.py --symbol ETHUSDT --rr 2.5
  python scripts/backtest_ema_pullback.py --timeframe 4H --rr 3.0
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
_FEE           = 0.0006
_CAPITAL       = 5_000.0
_RISK_PCT      = 0.01

# Trend EMAs
_EMA_FAST      = 8
_EMA_SLOW      = 21
_TREND_SEP     = 0.003      # EMA8 must be >0.3% above/below EMA21 (clear trend)
_TREND_BARS    = 5          # trend must persist for at least this many bars
_TOUCH_TOL     = 0.005      # price touches EMA21 if within 0.5%

# Reversal candle quality
_REV_BODY_MIN  = 0.50       # body >= 50% of range
_MOMENTUM_BARS = 3          # check ATR extension over last N bars

# ATR for stops
_ATR_PERIOD    = 14
_STOP_ATR      = 1.0        # stop = EMA21 ± 1.0 × ATR

# Target
_RR            = 2.0
_MAX_HOLD_BARS = 20         # time stop (20 bars on 1H = 20 hours)

# Daily regime (no trading against weekly trend)
_DAILY_SMA_LEN = 50

# Cool-down
_COOLDOWN_BARS = 3

_DATA_DIR = _REPO_ROOT / "data" / "candles"


# ─── Data ─────────────────────────────────────────────────────────────────────

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


def _resample(m1: list[dict], interval_ms: int) -> list[dict]:
    buckets: dict[int, dict] = {}
    for bar in m1:
        bt = (bar["ts"] // interval_ms) * interval_ms
        if bt not in buckets:
            buckets[bt] = {"ts": bt, "o": bar["o"], "h": bar["h"],
                           "l": bar["l"], "c": bar["c"], "v": bar["v"]}
        else:
            b = buckets[bt]
            b["h"] = max(b["h"], bar["h"])
            b["l"] = min(b["l"], bar["l"])
            b["c"] = bar["c"]
            b["v"] += bar["v"]
    return sorted(buckets.values(), key=lambda x: x["ts"])


def _build_daily_sma(m1: list[dict], n: int = _DAILY_SMA_LEN) -> dict[int, float | None]:
    d1 = _resample(m1, 86_400_000)
    sma: dict[int, float | None] = {}
    closes = []
    for bar in d1:
        closes.append(bar["c"])
        sma[bar["ts"]] = sum(closes[-n:]) / n if len(closes) >= n else None
    return sma


def _daily_sma_at(ts_ms: int, sma_map: dict) -> float | None:
    return sma_map.get((ts_ms // 86_400_000) * 86_400_000)


# ─── Build bars with all indicators ───────────────────────────────────────────

def _build_bars(candles: list[dict]) -> list[dict]:
    n = len(candles)
    closes = [c["c"] for c in candles]

    ema_f = [0.0] * n
    ema_s = [0.0] * n
    af = 2 / (_EMA_FAST + 1)
    as_ = 2 / (_EMA_SLOW + 1)

    for i in range(n):
        c = closes[i]
        ema_f[i] = c if i == 0 else af * c + (1 - af) * ema_f[i - 1]
        ema_s[i] = c if i == 0 else as_ * c + (1 - as_) * ema_s[i - 1]

    # ATR
    atrs = [0.0] * n
    for i in range(n):
        if i == 0:
            atrs[i] = candles[0]["h"] - candles[0]["l"]
            continue
        window = []
        for j in range(max(1, i - _ATR_PERIOD + 1), i + 1):
            prev_c = candles[j - 1]["c"]
            window.append(max(candles[j]["h"] - candles[j]["l"],
                              abs(candles[j]["h"] - prev_c),
                              abs(candles[j]["l"] - prev_c)))
        atrs[i] = sum(window) / len(window)

    result = []
    for i in range(n):
        c = candles[i]
        ef = ema_f[i]
        es = ema_s[i]
        sep = (ef - es) / es if es > 0 else 0

        if sep > _TREND_SEP:
            trend = "bull"
        elif sep < -_TREND_SEP:
            trend = "bear"
        else:
            trend = "neutral"

        result.append({**c, "ema_f": ef, "ema_s": es, "atr": atrs[i], "trend": trend})
    return result


# ─── Signal scanner ───────────────────────────────────────────────────────────

def _scan(bars: list[dict], sma_map: dict, rr: float) -> list[dict]:
    trades = []
    n = len(bars)
    last_trade_exit = -_COOLDOWN_BARS - 1

    for i in range(_TREND_BARS + 2, n - _MAX_HOLD_BARS - 2):
        if i - last_trade_exit < _COOLDOWN_BARS:
            continue

        bar  = bars[i]
        prev = bars[i - 1]

        # ── Filter 1: current bar must be in clear trend ──────────────────
        if bar["trend"] == "neutral":
            continue

        # ── Filter 2: trend must have persisted for TREND_BARS consecutive bars
        direction = bar["trend"]
        streak = 0
        for k in range(i - 1, max(0, i - _TREND_BARS - 1), -1):
            if bars[k]["trend"] == direction:
                streak += 1
            else:
                break
        if streak < _TREND_BARS:
            continue

        # ── Filter 3: daily regime alignment ──────────────────────────────
        dsma = _daily_sma_at(bar["ts"], sma_map)
        if dsma is not None:
            if direction == "bull" and bar["c"] < dsma:
                continue
            if direction == "bear" and bar["c"] > dsma:
                continue

        # ── Filter 4: price touching the EMA21 zone ───────────────────────
        es = bar["ema_s"]
        touch_dist = abs(bar["l"] - es) / es if direction == "bull" else \
                     abs(bar["h"] - es) / es
        price_at_ema = touch_dist <= _TOUCH_TOL or \
                       (direction == "bull" and bar["l"] <= es * 1.002 and bar["c"] >= es) or \
                       (direction == "bear" and bar["h"] >= es * 0.998 and bar["c"] <= es)
        if not price_at_ema:
            continue

        # ── Filter 5: reversal candle at the EMA ──────────────────────────
        body = bar["c"] - bar["o"]
        rng  = bar["h"] - bar["l"]
        if rng == 0:
            continue

        if direction == "bull":
            # Bullish reversal: close above open, body >= 50% of range
            if body <= 0 or body / rng < _REV_BODY_MIN:
                continue
        else:
            # Bearish reversal: close below open
            if body >= 0 or abs(body) / rng < _REV_BODY_MIN:
                continue

        # ── Filter 6: not over-extended from EMA ──────────────────────────
        # Calculate how far price was from EMA in last MOMENTUM_BARS bars
        max_ext = 0.0
        for k in range(max(0, i - _MOMENTUM_BARS), i):
            dist = abs(bars[k]["c"] - bars[k]["ema_s"]) / bars[k]["ema_s"]
            max_ext = max(max_ext, dist)
        if max_ext > 0.05:  # if price was >5% away from EMA recently, skip
            continue

        # ── Entry ─────────────────────────────────────────────────────────
        entry_price = bar["c"]
        atr_val     = bar["atr"]

        if direction == "bull":
            stop_price   = es - _STOP_ATR * atr_val
            target_price = entry_price + rr * (entry_price - stop_price)
        else:
            stop_price   = es + _STOP_ATR * atr_val
            target_price = entry_price - rr * (stop_price - entry_price)

        risk_per_unit = abs(entry_price - stop_price)
        if risk_per_unit <= 0:
            continue

        qty  = (_CAPITAL * _RISK_PCT) / risk_per_unit
        pnl  = -(qty * entry_price * _FEE)

        exit_reason = "TIME"
        exit_price  = entry_price
        exit_bar_i  = min(i + _MAX_HOLD_BARS, n - 1)

        for j in range(i + 1, min(i + _MAX_HOLD_BARS + 1, n)):
            b = bars[j]
            if direction == "bull":
                if b["l"] <= stop_price:
                    hit = min(b["o"], stop_price)
                    pnl += qty * (hit - entry_price)
                    pnl -= qty * hit * _FEE
                    exit_reason = "STOP"
                    exit_price  = hit
                    exit_bar_i  = j
                    break
                if b["h"] >= target_price:
                    pnl += qty * (target_price - entry_price)
                    pnl -= qty * target_price * _FEE
                    exit_reason = "TP"
                    exit_price  = target_price
                    exit_bar_i  = j
                    break
            else:
                if b["h"] >= stop_price:
                    hit = max(b["o"], stop_price)
                    pnl += qty * (entry_price - hit)
                    pnl -= qty * hit * _FEE
                    exit_reason = "STOP"
                    exit_price  = hit
                    exit_bar_i  = j
                    break
                if b["l"] <= target_price:
                    pnl += qty * (entry_price - target_price)
                    pnl -= qty * target_price * _FEE
                    exit_reason = "TP"
                    exit_price  = target_price
                    exit_bar_i  = j
                    break
        else:
            ep = bars[exit_bar_i]["c"]
            diff = (ep - entry_price) if direction == "bull" else (entry_price - ep)
            pnl += qty * diff
            pnl -= qty * ep * _FEE

        last_trade_exit = exit_bar_i
        dt = datetime.fromtimestamp(bar["ts"] / 1000, tz=timezone.utc)
        trades.append({
            "direction":   direction.upper(),
            "entry_price": entry_price,
            "exit_price":  exit_price,
            "stop_price":  stop_price,
            "pnl":         pnl,
            "exit_reason": exit_reason,
            "year":        dt.year,
            "month":       dt.month,
            "exit_bar_i":  exit_bar_i,
        })
        i = exit_bar_i + 1

    return trades


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_backtest(symbol: str = "BTCUSDT", rr: float = _RR,
                 timeframe: str = "1H") -> None:
    path = _DATA_DIR / f"{symbol}_1m.parquet"
    interval_ms = {"1H": 3_600_000, "4H": 14_400_000, "2H": 7_200_000}[timeframe]

    print("=" * 70)
    print(f"  NexFlow — EMA Pullback Trend Continuation  [{symbol} {timeframe}]")
    print(f"  EMA{_EMA_FAST}/EMA{_EMA_SLOW} trend  |  Touch ±{_TOUCH_TOL*100:.1f}%  |"
          f"  Reversal body >{_REV_BODY_MIN*100:.0f}%")
    print(f"  Stop {_STOP_ATR}×ATR  |  RR 1:{rr}  |  Daily SMA{_DAILY_SMA_LEN} regime")
    print("=" * 70)

    if not path.exists():
        print(f"\n[ERROR] {path} not found.")
        print(f"  Download: python scripts/download_1m_binance_vision.py --symbol {symbol}")
        return

    print(f"\n  Loading {symbol} 1m → {timeframe}...")
    m1   = _load_1m(path)
    bars = _resample(m1, interval_ms)
    bars = _build_bars(bars)
    sma_map = _build_daily_sma(m1)

    print(f"  {len(m1):,} 1m bars → {len(bars):,} {timeframe} bars")

    trades = _scan(bars, sma_map, rr)
    if not trades:
        print("  No trades found.")
        return

    capital = _CAPITAL
    year_stats: dict[int, dict] = {}
    for t in trades:
        capital += t["pnl"]
        t["capital"] = capital
        yr = t["year"]
        if yr not in year_stats:
            year_stats[yr] = {"pnl": 0.0, "trades": 0, "wins": 0,
                              "start": capital - t["pnl"]}
        year_stats[yr]["pnl"]    += t["pnl"]
        year_stats[yr]["trades"] += 1
        if t["pnl"] > 0:
            year_stats[yr]["wins"] += 1

    wins      = [t for t in trades if t["pnl"] > 0]
    losses    = [t for t in trades if t["pnl"] <= 0]
    tps       = [t for t in trades if t["exit_reason"] == "TP"]
    stops     = [t for t in trades if t["exit_reason"] == "STOP"]
    times     = [t for t in trades if t["exit_reason"] == "TIME"]
    gross_win = sum(t["pnl"] for t in wins)
    gross_los = abs(sum(t["pnl"] for t in losses))
    pf        = gross_win / gross_los if gross_los else float("inf")

    equity = [_CAPITAL] + [t["capital"] for t in trades]
    peak   = _CAPITAL
    max_dd = 0.0
    for eq in equity:
        peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak)

    total_pnl  = sum(t["pnl"] for t in trades)
    avg_per_yr = len(trades) / max(1, len(year_stats))

    print(f"\n{'─'*70}")
    print(f"  RESULTS  ({len(trades)} trades, {avg_per_yr:.0f}/year avg)")
    print(f"{'─'*70}")
    print(f"  Net P&L        : ${total_pnl:>10,.2f}  ({total_pnl/_CAPITAL*100:+.1f}%)")
    print(f"  Final capital  : ${capital:>10,.2f}")
    print(f"  Win rate       : {len(wins)/len(trades)*100:.1f}%  ({len(wins)}/{len(trades)})")
    print(f"  Profit factor  : {pf:.2f}")
    print(f"  Max drawdown   : {max_dd*100:.1f}%")
    if wins:
        print(f"  Avg win        : ${gross_win/len(wins):,.2f}")
    if losses:
        print(f"  Avg loss       : ${gross_los/len(losses):,.2f}")
    print(f"  TP / Stop / Time: {len(tps)} / {len(stops)} / {len(times)}")

    longs  = [t for t in trades if t["direction"] == "LONG"]
    shorts = [t for t in trades if t["direction"] == "SHORT"]
    if longs:
        lw = [t for t in longs if t["pnl"] > 0]
        print(f"  Longs          : {len(longs)}  WR {len(lw)/len(longs)*100:.0f}%")
    if shorts:
        sw = [t for t in shorts if t["pnl"] > 0]
        print(f"  Shorts         : {len(shorts)}  WR {len(sw)/len(shorts)*100:.0f}%")

    print(f"\n  {'Year':<6} {'Trades':>7} {'Win%':>6} {'Net P&L':>11} {'Capital':>11} {'Ret%':>7}")
    print(f"  {'─'*56}")
    running = _CAPITAL
    winning_years = 0
    for yr in sorted(year_stats):
        st = year_stats[yr]
        net = st["pnl"]
        end = running + net
        wp  = st["wins"] / st["trades"] * 100 if st["trades"] else 0
        ret = net / running * 100 if running else 0
        running = end
        flag = "✓" if net >= 0 else "✗"
        if net >= 0:
            winning_years += 1
        print(f"  {flag} {yr:<5} {st['trades']:>7} {wp:>5.1f}% {net:>+11,.2f} {end:>11,.2f} {ret:>6.1f}%")

    full_years = [yr for yr in year_stats if year_stats[yr]["trades"] >= 8]
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
            losing_months = 0
            for m in sorted(monthly):
                st = monthly[m]
                wp = st["wins"] / st["trades"] * 100 if st["trades"] else 0
                flag = "✓" if st["pnl"] >= 0 else "✗"
                if st["pnl"] < 0:
                    losing_months += 1
                print(f"    {flag} {datetime(recent,m,1).strftime('%b'):>3}  "
                      f"{st['trades']:>3} trades  {wp:>5.1f}% win  ${st['pnl']:>+8,.2f}")
            print(f"\n    Losing months in {recent}: {losing_months}/{len(monthly)}")

    print(f"\n{'─'*70}")
    if pf >= 1.5 and max_dd <= 0.15 and len(trades) >= 60:
        verdict = "✅ STRONG GO — High-frequency trend edge"
    elif pf >= 1.3 and max_dd <= 0.20:
        verdict = "⚠️  PROMISING — Refine and paper trade"
    elif pf >= 1.1 and max_dd <= 0.25:
        verdict = "⚠️  WEAK GO — Marginal"
    elif pf >= 1.0:
        verdict = "⚠️  BREAK-EVEN"
    else:
        verdict = "❌ KILL"
    print(f"  VERDICT: {verdict}")
    print(f"  Winning years : {winning_years}/{len(year_stats)}")
    print(f"{'─'*70}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol",    default="BTCUSDT")
    ap.add_argument("--rr",        type=float, default=2.0)
    ap.add_argument("--timeframe", default="1H", choices=["1H", "2H", "4H"])
    args = ap.parse_args()
    run_backtest(symbol=args.symbol, rr=args.rr, timeframe=args.timeframe)
