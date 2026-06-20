#!/usr/bin/env python3
"""
backtest_ict_fvg.py — ICT/SMC Fair Value Gap Scalping Backtest

Mechanically implements Louis's ICT-style strategy:
  1. NY session only (13:30–20:00 UTC / 9:30–4:00 PM ET)
  2. Wait for a liquidity sweep (price takes out N-bar high/low then reverses)
  3. Strong displacement candle creates an FVG
  4. Price retraces into the FVG → entry
  5. Stop below/above the sweep extreme
  6. Target: fixed RR (tested at 1:3, 1:5, 1:8) + partial at 1:2

Requires 1-minute BTCUSDT data in data/candles/BTCUSDT_1m.parquet
Run the downloader first if needed:
  python scripts/download_candles.py --symbol BTCUSDT --start 2021-01-01

Usage:
  python scripts/backtest_ict_fvg.py
  python scripts/backtest_ict_fvg.py --rr 5 --sweep-bars 20
"""

from __future__ import annotations

import argparse
import os
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

# ─── Constants ────────────────────────────────────────────────────────────────
_FEE            = 0.0006        # 0.06% taker per side
_CAPITAL        = 5_000.0       # starting capital
_RISK_PCT       = 0.01          # risk 1% of capital per trade
_PARTIAL_R      = 2.0           # take half off at 2R, let runner go to full target
_PARTIAL_FRAC   = 0.5           # fraction closed at partial TP

# NY session in UTC (9:30 AM – 4:00 PM ET = 13:30 – 20:00 UTC)
_SESSION_START_H = 13
_SESSION_START_M = 30
_SESSION_END_H   = 20
_SESSION_END_M   = 0

# Displacement filter: the sweep candle + FVG candle body must be >= this
# multiple of the 20-bar average candle body (ensures it's a REAL move)
_DISP_MULT      = 2.0

# FVG retrace tolerance: price must enter the FVG zone within this many bars
_FVG_MAX_BARS   = 30

# Minimum FVG size as % of price (filters tiny gaps that disappear in spread)
_MIN_FVG_PCT    = 0.0005        # 0.05% of price minimum

# Max bars to hold a trade before time-stopping it
_MAX_HOLD_BARS  = 120           # 2 hours on 1-min chart

_DATA_PATH = _REPO_ROOT / "data" / "candles" / "BTCUSDT_1m.parquet"


# ─── Data loading ─────────────────────────────────────────────────────────────

def _load_candles() -> list[dict]:
    if not _DATA_PATH.exists():
        print(f"[ERROR] Missing 1-minute data: {_DATA_PATH}")
        print("Download it first:")
        print("  python scripts/download_candles.py --symbol BTCUSDT --start 2021-01-01")
        sys.exit(1)

    t = pq.read_table(_DATA_PATH, columns=["open_time", "open", "high", "low", "close", "volume"])
    candles = []
    for ot, o, h, l, c, v in zip(
        t["open_time"].to_pylist(), t["open"].to_pylist(),
        t["high"].to_pylist(), t["low"].to_pylist(),
        t["close"].to_pylist(), t["volume"].to_pylist()
    ):
        candles.append({"ts": int(ot), "o": float(o), "h": float(h),
                         "l": float(l), "c": float(c), "v": float(v)})
    candles.sort(key=lambda x: x["ts"])
    return candles


# ─── Session filter ───────────────────────────────────────────────────────────

def _in_ny_session(ts_ms: int) -> bool:
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    start = dt.replace(hour=_SESSION_START_H, minute=_SESSION_START_M, second=0, microsecond=0)
    end   = dt.replace(hour=_SESSION_END_H,   minute=_SESSION_END_M,   second=0, microsecond=0)
    return start <= dt < end


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _avg_body(candles: list[dict], i: int, n: int = 20) -> float:
    start = max(0, i - n)
    bodies = [abs(c["c"] - c["o"]) for c in candles[start:i]]
    return sum(bodies) / len(bodies) if bodies else 0.0


# ─── Core setup detection ─────────────────────────────────────────────────────

def _detect_setups(candles: list[dict], sweep_bars: int) -> list[dict]:
    """
    Scan the full candle series and return all valid ICT setups.

    A bullish setup:
      1. Price sweeps below the lowest LOW of the last `sweep_bars` bars
         (i.e., current low < min of prior sweep_bars lows) then CLOSES above
         that swept low → liquidity grab confirmed
      2. The NEXT candle is a strong bullish displacement (body >= _DISP_MULT × avg body)
      3. That displacement creates a bullish FVG:
         candle[i-2].high < candle[i].low  (gap in price)
      4. We record the FVG zone [fvg_low, fvg_high] and watch for retrace

    A bearish setup (mirror):
      1. Price sweeps above the highest HIGH of last sweep_bars → closes below
      2. Strong bearish displacement candle
      3. Bearish FVG: candle[i-2].low > candle[i].high
      4. Watch for retrace into gap
    """
    setups = []
    n = len(candles)

    for i in range(sweep_bars + 2, n - _FVG_MAX_BARS - 1):
        if not _in_ny_session(candles[i]["ts"]):
            continue

        avg_body = _avg_body(candles, i)
        if avg_body == 0:
            continue

        # ── Bullish setup ──────────────────────────────────────────────────
        prior_lows  = [candles[j]["l"] for j in range(i - sweep_bars, i)]
        swing_low   = min(prior_lows)
        swept_low   = candles[i]["l"] < swing_low     # wick below structure
        closed_above = candles[i]["c"] > swing_low    # but closed above = trap

        if swept_low and closed_above:
            # Check displacement: next candle must be a strong bull move
            d = candles[i + 1]
            disp_body = d["c"] - d["o"]
            if disp_body >= _DISP_MULT * avg_body:
                # Check for bullish FVG at candle i+1 (uses candle i-1 and i+1)
                # FVG = gap between high of candle before displacement and low of candle after
                fvg_low  = candles[i]["h"]     # top of the sweep candle
                fvg_high = candles[i + 2]["l"] if i + 2 < n else None  # bottom of next candle

                if fvg_high is not None and fvg_high > fvg_low:
                    fvg_size_pct = (fvg_high - fvg_low) / candles[i]["c"]
                    if fvg_size_pct >= _MIN_FVG_PCT:
                        setups.append({
                            "direction": "LONG",
                            "sweep_bar": i,
                            "sweep_extreme": candles[i]["l"],  # stop below this
                            "disp_bar": i + 1,
                            "fvg_low": fvg_low,
                            "fvg_high": fvg_high,
                            "fvg_mid": (fvg_low + fvg_high) / 2,
                            "valid_from": i + 2,
                            "avg_body": avg_body,
                        })

        # ── Bearish setup ──────────────────────────────────────────────────
        prior_highs  = [candles[j]["h"] for j in range(i - sweep_bars, i)]
        swing_high   = max(prior_highs)
        swept_high   = candles[i]["h"] > swing_high
        closed_below = candles[i]["c"] < swing_high

        if swept_high and closed_below:
            d = candles[i + 1]
            disp_body = d["o"] - d["c"]  # bearish body
            if disp_body >= _DISP_MULT * avg_body:
                fvg_high = candles[i]["l"]      # bottom of sweep candle
                fvg_low  = candles[i + 2]["h"] if i + 2 < n else None

                if fvg_low is not None and fvg_high > fvg_low:
                    fvg_size_pct = (fvg_high - fvg_low) / candles[i]["c"]
                    if fvg_size_pct >= _MIN_FVG_PCT:
                        setups.append({
                            "direction": "SHORT",
                            "sweep_bar": i,
                            "sweep_extreme": candles[i]["h"],  # stop above this
                            "disp_bar": i + 1,
                            "fvg_low": fvg_low,
                            "fvg_high": fvg_high,
                            "fvg_mid": (fvg_low + fvg_high) / 2,
                            "valid_from": i + 2,
                            "avg_body": avg_body,
                        })

    return setups


# ─── Trade simulation ─────────────────────────────────────────────────────────

def _simulate_trade(candles: list[dict], setup: dict, rr: float,
                    capital: float) -> dict | None:
    """
    Watch for price to retrace into the FVG, enter, manage the trade.
    Returns trade result dict or None if no entry occurred.
    """
    direction   = setup["direction"]
    fvg_low     = setup["fvg_low"]
    fvg_high    = setup["fvg_high"]
    fvg_mid     = setup["fvg_mid"]
    sweep_ext   = setup["sweep_extreme"]
    valid_from  = setup["valid_from"]
    n = len(candles)

    entry_price = stop_price = None
    entry_bar   = None

    # Wait for retrace into FVG zone
    for j in range(valid_from, min(valid_from + _FVG_MAX_BARS, n)):
        c = candles[j]
        if not _in_ny_session(c["ts"]):
            continue

        if direction == "LONG":
            # Price retraces down into the FVG
            if c["l"] <= fvg_high and c["h"] >= fvg_low:
                entry_price = fvg_mid       # enter at gap midpoint
                stop_price  = sweep_ext - (fvg_mid - sweep_ext) * 0.05  # just below sweep low
                entry_bar   = j
                break
        else:  # SHORT
            if c["h"] >= fvg_low and c["l"] <= fvg_high:
                entry_price = fvg_mid
                stop_price  = sweep_ext + (sweep_ext - fvg_mid) * 0.05
                entry_bar   = j
                break

    if entry_price is None:
        return None   # setup never triggered

    # Position sizing: risk _RISK_PCT of capital per trade
    if direction == "LONG":
        risk_per_unit = entry_price - stop_price
    else:
        risk_per_unit = stop_price - entry_price

    if risk_per_unit <= 0:
        return None

    risk_dollars  = capital * _RISK_PCT
    qty           = risk_dollars / risk_per_unit
    notional      = qty * entry_price
    entry_fee     = notional * _FEE

    # Targets
    if direction == "LONG":
        partial_tp = entry_price + _PARTIAL_R * risk_per_unit
        full_tp    = entry_price + rr * risk_per_unit
    else:
        partial_tp = entry_price - _PARTIAL_R * risk_per_unit
        full_tp    = entry_price - rr * risk_per_unit

    # Simulate bar by bar
    partial_hit  = False
    qty_remaining = qty
    pnl = -entry_fee   # pay entry fee upfront

    exit_reason = "TIME"
    exit_price  = entry_price

    for j in range(entry_bar + 1, min(entry_bar + _MAX_HOLD_BARS + 1, n)):
        c = candles[j]

        # Check partial TP
        if not partial_hit:
            if direction == "LONG" and c["h"] >= partial_tp:
                close_qty = qty * _PARTIAL_FRAC
                pnl += close_qty * (partial_tp - entry_price) - close_qty * partial_tp * _FEE
                qty_remaining -= close_qty
                partial_hit = True
            elif direction == "SHORT" and c["l"] <= partial_tp:
                close_qty = qty * _PARTIAL_FRAC
                pnl += close_qty * (entry_price - partial_tp) - close_qty * partial_tp * _FEE
                qty_remaining -= close_qty
                partial_hit = True

        # Check stop loss (worst case: gaps through on open)
        stop_hit_price = None
        if direction == "LONG" and c["l"] <= stop_price:
            stop_hit_price = min(c["o"], stop_price)   # gap-through pessimism
        elif direction == "SHORT" and c["h"] >= stop_price:
            stop_hit_price = max(c["o"], stop_price)

        if stop_hit_price is not None:
            pnl += qty_remaining * (stop_hit_price - entry_price) * (1 if direction == "LONG" else -1)
            pnl -= qty_remaining * stop_hit_price * _FEE
            exit_reason = "STOP"
            exit_price  = stop_hit_price
            break

        # Check full TP
        if direction == "LONG" and c["h"] >= full_tp:
            pnl += qty_remaining * (full_tp - entry_price) - qty_remaining * full_tp * _FEE
            exit_reason = "TP"
            exit_price  = full_tp
            break
        elif direction == "SHORT" and c["l"] <= full_tp:
            pnl += qty_remaining * (entry_price - full_tp) - qty_remaining * full_tp * _FEE
            exit_reason = "TP"
            exit_price  = full_tp
            break
    else:
        # Time stop: close at last bar's close
        last = candles[min(entry_bar + _MAX_HOLD_BARS, n - 1)]
        ep = last["c"]
        pnl += qty_remaining * (ep - entry_price) * (1 if direction == "LONG" else -1)
        pnl -= qty_remaining * ep * _FEE
        exit_reason = "TIME"
        exit_price  = ep

    dt_entry = datetime.fromtimestamp(candles[entry_bar]["ts"] / 1000, tz=timezone.utc)
    return {
        "direction":   direction,
        "entry_ts":    candles[entry_bar]["ts"],
        "entry_price": entry_price,
        "exit_price":  exit_price,
        "stop_price":  stop_price,
        "pnl":         pnl,
        "exit_reason": exit_reason,
        "year":        dt_entry.year,
        "month":       dt_entry.month,
    }


# ─── Main backtest ────────────────────────────────────────────────────────────

def run_backtest(rr: float = 5.0, sweep_bars: int = 20) -> None:
    print("=" * 68)
    print("  NexFlow — ICT/SMC Fair Value Gap Scalping Backtest")
    print(f"  Asset: BTCUSDT 1m | NY session | RR 1:{rr:.0f} | Sweep {sweep_bars}b")
    print(f"  Capital ${_CAPITAL:,.0f} | Risk {_RISK_PCT*100:.1f}%/trade | Partial at 1:{_PARTIAL_R:.0f}")
    print("=" * 68)

    candles = _load_candles()
    print(f"\n  Loaded {len(candles):,} 1-minute candles")

    ts_range = [candles[0]["ts"], candles[-1]["ts"]]
    print(f"  Range: {datetime.fromtimestamp(ts_range[0]/1000,tz=timezone.utc).strftime('%Y-%m-%d')} "
          f"→ {datetime.fromtimestamp(ts_range[1]/1000,tz=timezone.utc).strftime('%Y-%m-%d')}")

    print("\n  Scanning for setups...")
    setups = _detect_setups(candles, sweep_bars)
    print(f"  Found {len(setups):,} raw setups "
          f"({sum(1 for s in setups if s['direction']=='LONG')} long, "
          f"{sum(1 for s in setups if s['direction']=='SHORT')} short)")

    # Simulate trades — skip if previous trade still open (no overlapping)
    trades = []
    capital = _CAPITAL
    last_exit_bar = -1
    year_stats: dict[int, dict] = {}

    for setup in setups:
        if setup["valid_from"] <= last_exit_bar:
            continue   # don't overlap trades

        result = _simulate_trade(candles, setup, rr, capital)
        if result is None:
            continue

        capital += result["pnl"]
        result["capital"] = capital
        trades.append(result)

        yr = result["year"]
        if yr not in year_stats:
            year_stats[yr] = {"pnl": 0.0, "trades": 0, "wins": 0,
                              "stops": 0, "tps": 0, "times": 0, "start": capital - result["pnl"]}
        year_stats[yr]["pnl"]    += result["pnl"]
        year_stats[yr]["trades"] += 1
        if result["pnl"] > 0:
            year_stats[yr]["wins"] += 1
        year_stats[yr][result["exit_reason"].lower() + "s"] = \
            year_stats[yr].get(result["exit_reason"].lower() + "s", 0) + 1

        # Find the bar this trade exited on
        exit_ts = candles[setup["valid_from"]]["ts"]
        for k, c in enumerate(candles[setup["valid_from"]:], start=setup["valid_from"]):
            if c["ts"] >= exit_ts:
                last_exit_bar = k
                break

    if not trades:
        print("\n  No trades triggered — check data coverage and session filter.")
        return

    # ── Aggregate results ──────────────────────────────────────────────────────
    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    tps    = [t for t in trades if t["exit_reason"] == "TP"]
    stops  = [t for t in trades if t["exit_reason"] == "STOP"]
    times  = [t for t in trades if t["exit_reason"] == "TIME"]

    total_pnl = sum(t["pnl"] for t in trades)
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    pf = gross_win / gross_loss if gross_loss else float("inf")

    # Max drawdown
    equity = [_CAPITAL] + [t["capital"] for t in trades]
    peak = _CAPITAL
    max_dd = 0.0
    for eq in equity:
        peak = max(peak, eq)
        dd = (peak - eq) / peak
        max_dd = max(max_dd, dd)

    # ── Print ──────────────────────────────────────────────────────────────────
    print(f"\n{'─'*68}")
    print(f"  OVERALL RESULTS  ({len(trades)} trades)")
    print(f"{'─'*68}")
    print(f"  Net P&L       : ${total_pnl:>10,.2f}  ({total_pnl/_CAPITAL*100:+.1f}%)")
    print(f"  Final capital : ${capital:>10,.2f}")
    print(f"  Win rate      : {len(wins)/len(trades)*100:.1f}%  ({len(wins)}/{len(trades)})")
    print(f"  Profit factor : {pf:.2f}")
    print(f"  Max drawdown  : {max_dd*100:.1f}%")
    print(f"  Avg win       : ${gross_win/len(wins):,.2f}" if wins else "  Avg win       : n/a")
    print(f"  Avg loss      : ${gross_loss/len(losses):,.2f}" if losses else "  Avg loss       : n/a")
    print(f"  TP exits      : {len(tps)}  ({len(tps)/len(trades)*100:.1f}%)")
    print(f"  Stop exits    : {len(stops)}  ({len(stops)/len(trades)*100:.1f}%)")
    print(f"  Time exits    : {len(times)}  ({len(times)/len(trades)*100:.1f}%)")

    # Year-by-year
    print(f"\n  {'Year':<6} {'Trades':>7} {'Win%':>6} {'Net P&L':>11} {'Capital':>11} {'Ret%':>7}")
    print(f"  {'─'*54}")
    running = _CAPITAL
    for yr in sorted(year_stats):
        st = year_stats[yr]
        net = st["pnl"]
        end = running + net
        wp  = st["wins"] / st["trades"] * 100 if st["trades"] else 0
        ret = net / running * 100 if running else 0
        running = end
        print(f"  {yr:<6} {st['trades']:>7} {wp:>5.1f}% {net:>+11,.2f} {end:>11,.2f} {ret:>6.1f}%")

    # Monthly breakdown (2024 as example of recent performance)
    recent_year = max(year_stats.keys())
    monthly = {}
    for t in trades:
        if t["year"] == recent_year:
            m = t["month"]
            monthly.setdefault(m, {"pnl": 0.0, "trades": 0, "wins": 0})
            monthly[m]["pnl"]    += t["pnl"]
            monthly[m]["trades"] += 1
            if t["pnl"] > 0:
                monthly[m]["wins"] += 1

    if monthly:
        print(f"\n  Monthly breakdown ({recent_year}):")
        losing_months = 0
        for m in sorted(monthly):
            st = monthly[m]
            wp = st["wins"] / st["trades"] * 100 if st["trades"] else 0
            flag = "✓" if st["pnl"] >= 0 else "✗"
            if st["pnl"] < 0: losing_months += 1
            print(f"    {flag} {datetime(recent_year,m,1).strftime('%b'):>3}  "
                  f"{st['trades']:>3} trades  {wp:>5.1f}% win  ${st['pnl']:>+8,.2f}")
        print(f"\n    Losing months in {recent_year}: {losing_months}/{len(monthly)}")

    # Verdict
    print(f"\n{'─'*68}")
    if pf >= 1.3 and max_dd <= 0.25 and len(trades) >= 100:
        print("  VERDICT: ✅ GO — Edge survives mechanical rules + fees")
    elif pf >= 1.1:
        print("  VERDICT: ⚠️  WEAK — Marginal edge, needs live validation")
    else:
        print("  VERDICT: ❌ KILL — No edge after fees in mechanical form")
    print(f"{'─'*68}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rr", type=float, default=5.0, help="Risk:reward target (default 5)")
    ap.add_argument("--sweep-bars", type=int, default=20, help="Lookback for liquidity sweep")
    args = ap.parse_args()
    run_backtest(rr=args.rr, sweep_bars=args.sweep_bars)
