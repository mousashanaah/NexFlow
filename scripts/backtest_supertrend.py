#!/usr/bin/env python3
"""
backtest_supertrend.py — Supertrend Scalping (15m / 1H)

Supertrend is a dynamic trailing stop that flips direction when price
crosses it. Used by many professional traders on all markets. On 15m
BTC with 4H trend confirmation, it should generate 5-10 signals/week.

ENTRY RULES:
  1. 4H EMA8 > EMA21 = bull regime (only longs)
     4H EMA8 < EMA21 = bear regime (only shorts)
  2. On 15m: Supertrend flips to match the 4H direction
  3. Enter at next bar's open
  4. Stop: Supertrend line (adaptive)
  5. Exit: when Supertrend flips opposite, or 2R fixed TP (test both)

Usage:
  python scripts/backtest_supertrend.py
  python scripts/backtest_supertrend.py --timeframe 1H --rr 2.0
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

_FEE           = 0.0006
_CAPITAL       = 5_000.0
_RISK_PCT      = 0.01

# Supertrend params
_ST_ATR_PERIOD = 10
_ST_MULTIPLIER = 3.0      # classic: 3×ATR. Higher = fewer signals, better quality

# 4H trend
_HTF_FAST      = 8
_HTF_SLOW      = 21

# Daily regime
_DAILY_SMA_LEN = 50

# Target — fixed RR (alternative to Supertrend flip exit)
_RR            = 2.0
_MAX_HOLD_BARS = 48       # max hold in bars (15m: 48 = 12 hours)

_DATA_DIR = _REPO_ROOT / "data" / "candles"


def _load_1m(path: Path) -> list[dict]:
    t = pq.read_table(path, columns=["open_time","open","high","low","close","volume"])
    out = []
    for ot, o, h, l, c, v in zip(
        t["open_time"].to_pylist(), t["open"].to_pylist(),
        t["high"].to_pylist(), t["low"].to_pylist(),
        t["close"].to_pylist(), t["volume"].to_pylist()
    ):
        ts_raw = int(ot)
        ts_ms = ts_raw * 1000 if ts_raw < 2_000_000_000 else ts_raw
        out.append({"ts": ts_ms, "o": float(o), "h": float(h),
                    "l": float(l), "c": float(c), "v": float(v)})
    out.sort(key=lambda x: x["ts"])
    return out


def _resample(m1: list[dict], ms: int) -> list[dict]:
    bk: dict[int, dict] = {}
    for b in m1:
        t = (b["ts"] // ms) * ms
        if t not in bk:
            bk[t] = {"ts": t, "o": b["o"], "h": b["h"], "l": b["l"], "c": b["c"], "v": b["v"]}
        else:
            bk[t]["h"] = max(bk[t]["h"], b["h"])
            bk[t]["l"] = min(bk[t]["l"], b["l"])
            bk[t]["c"] = b["c"]
            bk[t]["v"] += b["v"]
    return sorted(bk.values(), key=lambda x: x["ts"])


def _build_daily_sma(m1: list[dict]) -> dict[int, float | None]:
    d1 = _resample(m1, 86_400_000)
    out: dict[int, float | None] = {}
    closes: list[float] = []
    for b in d1:
        closes.append(b["c"])
        out[b["ts"]] = sum(closes[-_DAILY_SMA_LEN:]) / _DAILY_SMA_LEN \
                       if len(closes) >= _DAILY_SMA_LEN else None
    return out


def _atr(bars: list[dict], i: int, n: int) -> float:
    trs = []
    for j in range(max(1, i - n + 1), i + 1):
        prev = bars[j-1]["c"]
        trs.append(max(bars[j]["h"] - bars[j]["l"],
                       abs(bars[j]["h"] - prev), abs(bars[j]["l"] - prev)))
    return sum(trs) / len(trs) if trs else bars[i]["h"] - bars[i]["l"]


def _build_supertrend(bars: list[dict]) -> list[dict]:
    n = len(bars)
    result = []
    st_val  = [0.0] * n
    st_dir  = [0]   * n   # 1 = bull (price above ST), -1 = bear

    for i in range(n):
        atr = _atr(bars, i, _ST_ATR_PERIOD)
        mid = (bars[i]["h"] + bars[i]["l"]) / 2
        upper_band = mid + _ST_MULTIPLIER * atr
        lower_band = mid - _ST_MULTIPLIER * atr

        if i == 0:
            st_val[i] = lower_band
            st_dir[i] = 1
        else:
            prev_st  = st_val[i-1]
            prev_dir = st_dir[i-1]
            prev_c   = bars[i-1]["c"]
            c        = bars[i]["c"]

            if prev_dir == 1:
                cur_st = max(lower_band, prev_st) if c > prev_st else upper_band
                cur_dir = 1 if c > cur_st else -1
            else:
                cur_st = min(upper_band, prev_st) if c < prev_st else lower_band
                cur_dir = -1 if c < cur_st else 1
            st_val[i] = cur_st
            st_dir[i] = cur_dir

        result.append({**bars[i], "st": st_val[i], "st_dir": st_dir[i]})
    return result


def _build_htf_trend(h4: list[dict]) -> dict[int, str]:
    af = 2 / (_HTF_FAST + 1)
    as_ = 2 / (_HTF_SLOW + 1)
    ef = es = None
    trend: dict[int, str] = {}
    for b in h4:
        c = b["c"]
        ef = c if ef is None else af * c + (1-af) * ef
        es = c if es is None else as_ * c + (1-as_) * es
        if ef and es:
            trend[b["ts"]] = "bull" if ef > es * 1.001 else "bear" if ef < es * 0.999 else "neutral"
    return trend


def _htf_at(ts_ms: int, trend: dict[int, str], interval_ms: int) -> str:
    bucket = (ts_ms // interval_ms) * interval_ms
    return trend.get(bucket, "neutral")


def _daily_at(ts_ms: int, sma_map: dict) -> float | None:
    return sma_map.get((ts_ms // 86_400_000) * 86_400_000)


def run_backtest(symbol: str = "BTCUSDT", rr: float = _RR,
                 timeframe: str = "15m") -> None:
    path = _DATA_DIR / f"{symbol}_1m.parquet"
    tf_ms = {"15m": 900_000, "30m": 1_800_000, "1H": 3_600_000}[timeframe]
    h4_ms = 14_400_000

    print("=" * 70)
    print(f"  NexFlow — Supertrend Scalping  [{symbol} {timeframe}]")
    print(f"  Supertrend({_ST_ATR_PERIOD}, {_ST_MULTIPLIER}×ATR)  |  4H EMA{_HTF_FAST}/{_HTF_SLOW} trend")
    print(f"  Fixed RR 1:{rr}  |  Daily SMA{_DAILY_SMA_LEN} regime")
    print("=" * 70)

    if not path.exists():
        print(f"[ERROR] {path} not found.")
        return

    print(f"\n  Loading and resampling {symbol}...")
    m1  = _load_1m(path)
    tf  = _resample(m1, tf_ms)
    h4  = _resample(m1, h4_ms)

    tf  = _build_supertrend(tf)
    h4_trend = _build_htf_trend(h4)
    sma_map  = _build_daily_sma(m1)

    print(f"  {len(m1):,} 1m → {len(tf):,} {timeframe} bars  |  {len(h4):,} 4H bars")

    trades: list[dict] = []
    capital = _CAPITAL
    year_stats: dict[int, dict] = {}
    last_exit_bar = -5

    n = len(tf)
    i = _ST_ATR_PERIOD + 1

    while i < n - 2:
        if i - last_exit_bar < 3:
            i += 1
            continue

        prev = tf[i-1]
        curr = tf[i]

        # Supertrend flip
        flipped_bull  = prev["st_dir"] == -1 and curr["st_dir"] == 1
        flipped_bear  = prev["st_dir"] == 1  and curr["st_dir"] == -1

        if not (flipped_bull or flipped_bear):
            i += 1
            continue

        direction = "LONG" if flipped_bull else "SHORT"

        # 4H trend alignment
        h4_t = _htf_at(curr["ts"], h4_trend, h4_ms)
        if h4_t == "neutral":
            i += 1
            continue
        if direction == "LONG"  and h4_t != "bull":
            i += 1
            continue
        if direction == "SHORT" and h4_t != "bear":
            i += 1
            continue

        # Daily SMA regime
        dsma = _daily_at(curr["ts"], sma_map)
        if dsma is not None:
            if direction == "LONG"  and curr["c"] < dsma:
                i += 1
                continue
            if direction == "SHORT" and curr["c"] > dsma:
                i += 1
                continue

        # Entry at next bar open
        if i + 1 >= n:
            break
        entry_bar   = tf[i + 1]
        entry_price = entry_bar["o"]
        stop_price  = curr["st"]  # Supertrend line is the stop

        risk_per_unit = abs(entry_price - stop_price)
        if risk_per_unit <= 0 or risk_per_unit / entry_price > 0.10:
            # Skip if stop is too far (>10% — probably a data anomaly)
            i += 1
            continue

        if direction == "LONG":
            target_price = entry_price + rr * risk_per_unit
        else:
            target_price = entry_price - rr * risk_per_unit

        qty  = (_CAPITAL * _RISK_PCT) / risk_per_unit
        pnl  = -(qty * entry_price * _FEE)

        exit_reason = "TIME"
        exit_price  = entry_price
        exit_bar_i  = min(i + 1 + _MAX_HOLD_BARS, n - 1)

        for j in range(i + 2, min(i + 2 + _MAX_HOLD_BARS, n)):
            b = tf[j]
            if direction == "LONG":
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
                # Supertrend flip exit
                if b["st_dir"] == -1:
                    pnl += qty * (b["c"] - entry_price)
                    pnl -= qty * b["c"] * _FEE
                    exit_reason = "FLIP"
                    exit_price  = b["c"]
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
                if b["st_dir"] == 1:
                    pnl += qty * (entry_price - b["c"])
                    pnl -= qty * b["c"] * _FEE
                    exit_reason = "FLIP"
                    exit_price  = b["c"]
                    exit_bar_i  = j
                    break
        else:
            ep = tf[exit_bar_i]["c"]
            diff = (ep - entry_price) if direction == "LONG" else (entry_price - ep)
            pnl += qty * diff
            pnl -= qty * ep * _FEE

        capital += pnl
        last_exit_bar = exit_bar_i
        dt = datetime.fromtimestamp(entry_bar["ts"] / 1000, tz=timezone.utc)
        tr = {
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "exit_reason": exit_reason,
            "year": dt.year,
            "month": dt.month,
            "capital": capital,
        }
        trades.append(tr)

        yr = dt.year
        if yr not in year_stats:
            year_stats[yr] = {"pnl": 0.0, "trades": 0, "wins": 0}
        year_stats[yr]["pnl"]    += pnl
        year_stats[yr]["trades"] += 1
        if pnl > 0:
            year_stats[yr]["wins"] += 1

        i = exit_bar_i + 1

    if not trades:
        print("  No trades found.")
        return

    wins      = [t for t in trades if t["pnl"] > 0]
    losses    = [t for t in trades if t["pnl"] <= 0]
    tps       = [t for t in trades if t["exit_reason"] == "TP"]
    stops     = [t for t in trades if t["exit_reason"] == "STOP"]
    flips     = [t for t in trades if t["exit_reason"] == "FLIP"]
    times     = [t for t in trades if t["exit_reason"] == "TIME"]
    gross_win = sum(t["pnl"] for t in wins)
    gross_los = abs(sum(t["pnl"] for t in losses))
    pf        = gross_win / gross_los if gross_los else float("inf")

    equity = [_CAPITAL] + [t["capital"] for t in trades]
    peak = _CAPITAL
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
    print(f"  TP/Stop/Flip/Time: {len(tps)}/{len(stops)}/{len(flips)}/{len(times)}")

    longs  = [t for t in trades if t["direction"] == "LONG"]
    shorts = [t for t in trades if t["direction"] == "SHORT"]
    if longs:
        lw = [t for t in longs if t["pnl"] > 0]
        print(f"  Longs  : {len(longs)}  WR {len(lw)/len(longs)*100:.0f}%")
    if shorts:
        sw = [t for t in shorts if t["pnl"] > 0]
        print(f"  Shorts : {len(shorts)}  WR {len(sw)/len(shorts)*100:.0f}%")

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

    print(f"\n{'─'*70}")
    if pf >= 1.5 and max_dd <= 0.15 and len(trades) >= 80:
        verdict = "✅ STRONG GO"
    elif pf >= 1.3 and max_dd <= 0.20:
        verdict = "⚠️  PROMISING"
    elif pf >= 1.1:
        verdict = "⚠️  WEAK GO"
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
    ap.add_argument("--timeframe", default="15m", choices=["15m", "30m", "1H"])
    args = ap.parse_args()
    run_backtest(symbol=args.symbol, rr=args.rr, timeframe=args.timeframe)
