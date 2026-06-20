#!/usr/bin/env python3
"""
backtest_bb_squeeze.py — Bollinger Band Squeeze (TTM Squeeze) Day Strategy

THE STRATEGY (John Carter, widely used on Nasdaq, gold, crypto):
  The "squeeze" is when Bollinger Bands contract inside Keltner Channels —
  volatility is compressing and a big move is loading. When BB expands back
  outside KC (squeeze "fires"), trade the direction of the expansion.

  Key insight: low volatility ALWAYS precedes high volatility. The squeeze
  catches the moment just before the explosive move — that's why it works
  across ALL liquid markets (BTC, ETH, Gold, Nasdaq).

ENTRY LOGIC:
  1. Squeeze ON: BB upper < KC upper AND BB lower > KC lower (both)
  2. Squeeze fires: BB breaks outside KC (squeeze OFF)
  3. Direction: determined by 1H EMA(8) vs EMA(21)
     — only take longs if 1H trend is bull, shorts if bear
  4. Entry: at the close of the first squeeze-release candle in trend direction
  5. Stop: ATR(14) × 1.5 below entry (long) or above (short)
  6. Target: 2R (test 1.5, 2.0, 3.0)
  7. Cool-down: don't enter another squeeze on same day as last trade

WHY THIS BEATS ORB ON CRYPTO:
  - ORB assumes a meaningful "open" — crypto has no overnight gap → 40% WR
  - Squeeze works on ANY timeframe, any market — volatility compression is universal
  - On 1H, average move after squeeze is 1-3% → fees (0.12%) are just 4-12% of profit
  - On 4H, moves are even larger and cleaner

FREQUENCY:
  - BTC: ~3-5 squeezes per week on 1H → ~150-200 signals/year
  - With trend filter: ~50-80 traded signals/year
  - Run on BTC + ETH + Gold = 150-240 signals/year → real day trading

Usage:
  python scripts/backtest_bb_squeeze.py
  python scripts/backtest_bb_squeeze.py --symbol ETHUSDT --rr 2.0
  python scripts/backtest_bb_squeeze.py --timeframe 4H --rr 3.0
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

# Bollinger Bands
_BB_PERIOD      = 20
_BB_STD         = 2.0

# Keltner Channel
_KC_PERIOD      = 20
_KC_ATR_PERIOD  = 14
_KC_MULT        = 1.5           # KC = EMA ± 1.5 × ATR

# Stop
_STOP_ATR_MULT  = 1.5           # stop = 1.5 × ATR from entry

# Trend alignment
_TREND_FAST     = 8
_TREND_SLOW     = 21

# Daily regime
_DAILY_SMA_LEN  = 50

# Cool-down: minimum bars between trades
_COOLDOWN_BARS  = 4             # skip setups within 4 bars of last trade

# Target
_RR             = 2.0
_TIMEFRAME      = "1H"          # "1H" or "4H"

_DATA_DIR = _REPO_ROOT / "data" / "candles"


# ─── Data loading & resampling ─────────────────────────────────────────────────

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
    """Resample 1m bars into any interval (3600000 = 1H, 14400000 = 4H)."""
    buckets: dict[int, dict] = {}
    for bar in m1:
        bucket_ts = (bar["ts"] // interval_ms) * interval_ms
        if bucket_ts not in buckets:
            buckets[bucket_ts] = {"ts": bucket_ts, "o": bar["o"], "h": bar["h"],
                                   "l": bar["l"], "c": bar["c"], "v": bar["v"]}
        else:
            b = buckets[bucket_ts]
            b["h"] = max(b["h"], bar["h"])
            b["l"] = min(b["l"], bar["l"])
            b["c"] = bar["c"]
            b["v"] += bar["v"]
    return sorted(buckets.values(), key=lambda x: x["ts"])


# ─── Indicators ───────────────────────────────────────────────────────────────

def _sma(values: list[float], i: int, n: int) -> float | None:
    if i < n - 1:
        return None
    return sum(values[i - n + 1: i + 1]) / n


def _stdev(values: list[float], i: int, n: int) -> float:
    if i < n - 1:
        return 0.0
    window = values[i - n + 1: i + 1]
    mean = sum(window) / n
    var  = sum((x - mean) ** 2 for x in window) / n
    return var ** 0.5


def _atr(candles: list[dict], i: int, n: int = _KC_ATR_PERIOD) -> float:
    if i == 0:
        return candles[0]["h"] - candles[0]["l"]
    trs = []
    for j in range(max(1, i - n + 1), i + 1):
        c_prev = candles[j - 1]["c"]
        trs.append(max(candles[j]["h"] - candles[j]["l"],
                       abs(candles[j]["h"] - c_prev),
                       abs(candles[j]["l"] - c_prev)))
    return sum(trs) / len(trs)


def _ema(values: list[float], i: int, n: int, cache: list) -> float:
    alpha = 2 / (n + 1)
    if i == 0 or cache[i - 1] is None:
        val = values[0] if i == 0 else values[i]
    else:
        val = alpha * values[i] + (1 - alpha) * cache[i - 1]
    cache[i] = val
    return val


# ─── Build indicators for all bars ───────────────────────────────────────────

def _build_indicators(candles: list[dict]) -> list[dict]:
    n = len(candles)
    closes = [c["c"] for c in candles]

    ema_fast = [None] * n
    ema_slow = [None] * n
    af = 2 / (_TREND_FAST + 1)
    as_ = 2 / (_TREND_SLOW + 1)

    for i in range(n):
        c = closes[i]
        ema_fast[i] = c if i == 0 or ema_fast[i-1] is None else af * c + (1-af) * ema_fast[i-1]
        ema_slow[i] = c if i == 0 or ema_slow[i-1] is None else as_ * c + (1-as_) * ema_slow[i-1]

    result = []
    for i in range(n):
        c = candles[i]

        # Bollinger Bands
        bb_mid = _sma(closes, i, _BB_PERIOD)
        bb_std = _stdev(closes, i, _BB_PERIOD)
        bb_upper = (bb_mid + _BB_STD * bb_std) if bb_mid is not None else None
        bb_lower = (bb_mid - _BB_STD * bb_std) if bb_mid is not None else None

        # Keltner Channel
        kc_mid_val = _sma(closes, i, _KC_PERIOD)
        atr_val = _atr(candles, i, _KC_ATR_PERIOD)
        kc_upper = (kc_mid_val + _KC_MULT * atr_val) if kc_mid_val is not None else None
        kc_lower = (kc_mid_val - _KC_MULT * atr_val) if kc_mid_val is not None else None

        # Squeeze state
        squeeze_on = False
        if all(x is not None for x in [bb_upper, bb_lower, kc_upper, kc_lower]):
            squeeze_on = bb_upper < kc_upper and bb_lower > kc_lower

        ef = ema_fast[i]
        es = ema_slow[i]
        trend = "bull" if (ef and es and ef > es * 1.001) else \
                "bear" if (ef and es and ef < es * 0.999) else "neutral"

        result.append({
            **c,
            "bb_upper":  bb_upper,
            "bb_lower":  bb_lower,
            "kc_upper":  kc_upper,
            "kc_lower":  kc_lower,
            "squeeze":   squeeze_on,
            "atr":       atr_val,
            "trend":     trend,
        })
    return result


def _build_daily_sma(m1: list[dict], sma_len: int = _DAILY_SMA_LEN) -> dict[int, float | None]:
    d1 = _resample(m1, 86_400_000)
    sma: dict[int, float | None] = {}
    closes = []
    for bar in d1:
        closes.append(bar["c"])
        sma[bar["ts"]] = sum(closes[-sma_len:]) / min(len(closes), sma_len) \
                         if len(closes) >= sma_len else None
    return sma


def _daily_sma_at(ts_ms: int, sma_map: dict[int, float | None]) -> float | None:
    day = (ts_ms // 86_400_000) * 86_400_000
    return sma_map.get(day)


# ─── Detect setups and simulate trades ───────────────────────────────────────

def _run(bars: list[dict], sma_map: dict[int, float | None],
         rr: float) -> list[dict]:
    trades = []
    n = len(bars)
    last_trade_bar = -_COOLDOWN_BARS - 1

    i = _BB_PERIOD + 1
    while i < n - 1:
        bar  = bars[i]
        prev = bars[i - 1]

        # Squeeze just fired: was ON, now OFF
        if not prev.get("squeeze") or bar.get("squeeze"):
            i += 1
            continue
        if bar["bb_upper"] is None:
            i += 1
            continue

        # Cool-down
        if i - last_trade_bar < _COOLDOWN_BARS:
            i += 1
            continue

        # Trend filter
        trend = bar["trend"]
        if trend == "neutral":
            i += 1
            continue

        # Daily regime
        dsma = _daily_sma_at(bar["ts"], sma_map)
        if dsma is not None:
            if trend == "bull" and bar["c"] < dsma:
                i += 1
                continue
            if trend == "bear" and bar["c"] > dsma:
                i += 1
                continue

        # Direction: first bar after squeeze fires, price direction
        direction = "LONG" if trend == "bull" else "SHORT"

        # Entry at open of next bar
        if i + 1 >= n:
            break
        entry_bar = bars[i + 1]
        entry_price = entry_bar["o"]
        atr_val = bar["atr"]

        if direction == "LONG":
            stop_price  = entry_price - _STOP_ATR_MULT * atr_val
            target_price = entry_price + rr * _STOP_ATR_MULT * atr_val
        else:
            stop_price  = entry_price + _STOP_ATR_MULT * atr_val
            target_price = entry_price - rr * _STOP_ATR_MULT * atr_val

        risk_per_unit = abs(entry_price - stop_price)
        if risk_per_unit <= 0:
            i += 1
            continue

        risk_dollars = _CAPITAL * _RISK_PCT  # simplified — would use running capital
        qty     = risk_dollars / risk_per_unit
        pnl     = -(qty * entry_price * _FEE)  # entry fee

        exit_reason = "TIME"
        exit_price  = entry_price
        exit_bar_i  = i + 1

        for j in range(i + 2, min(i + 50, n)):   # max 50-bar hold (~2 days on 1H)
            b = bars[j]
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
            last = bars[min(i + 49, n - 1)]
            ep = last["c"]
            diff = (ep - entry_price) if direction == "LONG" else (entry_price - ep)
            pnl += qty * diff
            pnl -= qty * ep * _FEE
            exit_bar_i = min(i + 49, n - 1)

        last_trade_bar = exit_bar_i
        dt = datetime.fromtimestamp(entry_bar["ts"] / 1000, tz=timezone.utc)
        trades.append({
            "direction":   direction,
            "entry_price": entry_price,
            "exit_price":  exit_price,
            "stop_price":  stop_price,
            "pnl":         pnl,
            "exit_reason": exit_reason,
            "year":        dt.year,
            "month":       dt.month,
            "entry_bar":   i + 1,
            "exit_bar":    exit_bar_i,
        })
        i = exit_bar_i + 1   # advance past the trade

    return trades


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_backtest(symbol: str = "BTCUSDT", rr: float = _RR,
                 timeframe: str = _TIMEFRAME) -> None:
    path = _DATA_DIR / f"{symbol}_1m.parquet"
    interval_ms = 3_600_000 if timeframe == "1H" else 14_400_000  # 4H
    tf_label = timeframe

    print("=" * 70)
    print(f"  NexFlow — Bollinger Band Squeeze  [{symbol} {tf_label}]")
    print(f"  BB({_BB_PERIOD}, {_BB_STD}) inside KC({_KC_PERIOD}, {_KC_MULT}×ATR)")
    print(f"  Trend: EMA{_TREND_FAST}/EMA{_TREND_SLOW}  |  Daily SMA{_DAILY_SMA_LEN} regime")
    print(f"  Stop: {_STOP_ATR_MULT}×ATR  |  RR 1:{rr}  |  Cooldown: {_COOLDOWN_BARS} bars")
    print("=" * 70)

    if not path.exists():
        print(f"\n[ERROR] {path} not found.")
        print(f"  Download: python scripts/download_1m_binance_vision.py --symbol {symbol}")
        return

    print(f"\n  Loading {symbol} 1m → resampling to {tf_label}...")
    m1   = _load_1m(path)
    bars = _resample(m1, interval_ms)
    bars = _build_indicators(bars)
    sma_map = _build_daily_sma(m1)

    print(f"  {len(m1):,} 1m bars → {len(bars):,} {tf_label} bars")

    trades  = _run(bars, sma_map, rr)
    capital = _CAPITAL

    if not trades:
        print("  No trades found. Check parameters.")
        return

    year_stats: dict[int, dict] = {}
    running_pnl = 0.0
    capitals = []
    for t in trades:
        running_pnl += t["pnl"]
        capitals.append(_CAPITAL + running_pnl)
        yr = t["year"]
        if yr not in year_stats:
            year_stats[yr] = {"pnl": 0.0, "trades": 0, "wins": 0,
                              "start": _CAPITAL + running_pnl - t["pnl"]}
        year_stats[yr]["pnl"]    += t["pnl"]
        year_stats[yr]["trades"] += 1
        if t["pnl"] > 0:
            year_stats[yr]["wins"] += 1
    capital = _CAPITAL + running_pnl

    wins      = [t for t in trades if t["pnl"] > 0]
    losses    = [t for t in trades if t["pnl"] <= 0]
    tps       = [t for t in trades if t["exit_reason"] == "TP"]
    stops     = [t for t in trades if t["exit_reason"] == "STOP"]
    times     = [t for t in trades if t["exit_reason"] == "TIME"]
    gross_win = sum(t["pnl"] for t in wins)
    gross_los = abs(sum(t["pnl"] for t in losses))
    pf        = gross_win / gross_los if gross_los else float("inf")

    equity = [_CAPITAL] + capitals
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

    full_years = [yr for yr in year_stats if year_stats[yr]["trades"] >= 10]
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
    if pf >= 1.5 and max_dd <= 0.15 and len(trades) >= 80:
        verdict = "✅ STRONG GO — Squeeze edge with controlled DD"
    elif pf >= 1.3 and max_dd <= 0.20:
        verdict = "⚠️  PROMISING — Refine entry, paper trade"
    elif pf >= 1.1 and max_dd <= 0.25:
        verdict = "⚠️  WEAK GO — Marginal, needs optimization"
    elif pf >= 1.0:
        verdict = "⚠️  BREAK-EVEN"
    else:
        verdict = "❌ KILL — No edge"
    print(f"  VERDICT: {verdict}")
    print(f"  Winning years : {winning_years}/{len(year_stats)}")
    print(f"{'─'*70}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Bollinger Band Squeeze backtest")
    ap.add_argument("--symbol",    default="BTCUSDT")
    ap.add_argument("--rr",        type=float, default=2.0)
    ap.add_argument("--timeframe", default="1H", choices=["1H", "4H"])
    args = ap.parse_args()
    run_backtest(symbol=args.symbol, rr=args.rr, timeframe=args.timeframe)
