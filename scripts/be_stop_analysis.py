#!/usr/bin/env python3
"""Break-even stop analysis.

For every trade that hit TP1 then was stopped at break-even, simulate
four stop policies on the actual 1m candles that followed TP1:

    A  Current  — BE stop (what actually happened)
    B  Original — original stop maintained, no BE move
    C  Trail05  — 0.5× ATR trailing stop from the high watermark after TP1
    D  Trail1   — 1.0× ATR trailing stop from the high watermark after TP1

Candles are loaded from local parquet files if present, otherwise
fetched from Bitget public REST (no API key required).

Usage:
    python scripts/be_stop_analysis.py
    python scripts/be_stop_analysis.py --journal-dir logs/paper --candle-dir data/candles
    python scripts/be_stop_analysis.py --verbose
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    import pyarrow.parquet as pq
    _HAS_PARQUET = True
except ImportError:
    _HAS_PARQUET = False


# ---------------------------------------------------------------------------
# Tiny candle struct (don't import from nexflow to keep this standalone)
# ---------------------------------------------------------------------------

class Bar(NamedTuple):
    ts_ms: int       # close_time in ms
    open: float
    high: float
    low: float
    close: float


# ---------------------------------------------------------------------------
# Journal parsing
# ---------------------------------------------------------------------------

@dataclass
class BEStopTrade:
    symbol: str
    side: str
    entry_price: float
    original_stop: float
    tp1_price: float
    tp2_price: float
    tp3_price: float
    atr: float
    size: float
    entry_fee: float

    tp1_hit_time: float = 0.0        # epoch seconds
    tp1_fill_price: float = 0.0
    tp1_size: float = 0.0
    tp1_fee: float = 0.0
    tp1_latency_bars: int = 0

    stop_hit_time: float = 0.0       # epoch seconds
    stop_fill_price: float = 0.0
    stop_fee: float = 0.0
    stop_net_pnl: float = 0.0        # actual journal trade.pnl

    entry_time: float = 0.0


def _load_events(journal_dir: Path) -> list[dict]:
    events: list[dict] = []
    for path in sorted(journal_dir.glob("*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    events.sort(key=lambda e: e.get("ts_epoch", 0.0))
    return events


def extract_be_stop_trades(events: list[dict]) -> list[BEStopTrade]:
    """Reconstruct trades that hit TP1 then were stopped at break-even."""

    # State machines keyed by symbol
    open_fills: dict[str, dict] = {}          # symbol → FILL event
    open_signals: dict[str, dict] = {}        # symbol → most recent SIGNAL before fill
    open_tp_limits: dict[str, dict] = {}      # symbol → TP_LIMITS_PLACED event
    pending_tp1: dict[str, dict] = {}         # symbol → PARTIAL_TP (tp_idx=0) event
    pending_tp1_maker: dict[str, dict] = {}   # symbol → TP_MAKER_FILL (tp_idx=0) event

    last_signal: dict[str, dict] = {}         # symbol → most recent SIGNAL

    be_trades: list[BEStopTrade] = []

    for ev in events:
        evt = ev.get("event", "")
        sym = ev.get("symbol", "")

        if evt == "SIGNAL":
            last_signal[sym] = ev

        elif evt == "FILL":
            open_fills[sym] = ev
            open_signals[sym] = last_signal.get(sym, {})
            pending_tp1.pop(sym, None)
            pending_tp1_maker.pop(sym, None)

        elif evt == "TP_LIMITS_PLACED":
            open_tp_limits[sym] = ev

        elif evt == "PARTIAL_TP":
            if ev.get("tp_idx") == 0:
                pending_tp1[sym] = ev

        elif evt == "TP_MAKER_FILL":
            if ev.get("tp_idx") == 0:
                pending_tp1_maker[sym] = ev

        elif evt == "STOP_HIT":
            fill_ev = open_fills.pop(sym, None)
            sig_ev = open_signals.pop(sym, {})
            tp1_ev = pending_tp1.pop(sym, None)
            tp1_mk = pending_tp1_maker.pop(sym, None)
            open_tp_limits.pop(sym, None)

            # Only interested if TP1 was hit (BE stop condition)
            if fill_ev is None or tp1_ev is None:
                continue

            sig = sig_ev or {}
            tp_prices = sig.get("tp_prices", [])
            if len(tp_prices) < 3:
                continue

            atr = sig.get("atr", sig.get("features", {}).get("atr", 0.0))
            if atr <= 0:
                # fallback: derive from TP1 distance
                atr = abs(tp_prices[0] - fill_ev.get("fill_price", 0.0))

            trade = BEStopTrade(
                symbol=sym,
                side=fill_ev.get("direction", "long"),
                entry_price=fill_ev.get("fill_price", 0.0),
                original_stop=sig.get("stop_price", 0.0),
                tp1_price=tp_prices[0],
                tp2_price=tp_prices[1] if len(tp_prices) > 1 else 0.0,
                tp3_price=tp_prices[2] if len(tp_prices) > 2 else 0.0,
                atr=atr,
                size=fill_ev.get("size", 0.0),
                entry_fee=fill_ev.get("fee", 0.0),
                entry_time=fill_ev.get("ts_epoch", 0.0),

                tp1_hit_time=tp1_ev.get("ts_epoch", 0.0),
                tp1_fill_price=tp1_ev.get("fill_price", 0.0),
                tp1_size=tp1_ev.get("size", 0.0),
                tp1_fee=tp1_ev.get("fee", 0.0),
                tp1_latency_bars=(tp1_mk or {}).get("latency_bars", 0),

                stop_hit_time=ev.get("ts_epoch", 0.0),
                stop_fill_price=ev.get("fill_price", 0.0),
                stop_fee=ev.get("fee", 0.0),
                stop_net_pnl=ev.get("pnl", 0.0),
            )
            be_trades.append(trade)

    return be_trades


# ---------------------------------------------------------------------------
# Candle loading — parquet first, then Bitget REST fallback
# ---------------------------------------------------------------------------

def _load_parquet_bars(candle_dir: Path, symbol: str) -> list[Bar]:
    if not _HAS_PARQUET:
        return []
    path = candle_dir / f"{symbol}_1m.parquet"
    if not path.exists():
        return []
    rows = pq.read_table(path).to_pylist()
    bars = [
        Bar(ts_ms=r["close_time"], open=r["open"], high=r["high"],
            low=r["low"], close=r["close"])
        for r in rows if r.get("is_final")
    ]
    return sorted(bars, key=lambda b: b.ts_ms)


def _fetch_rest_bars(symbol: str, start_ms: int, end_ms: int) -> list[Bar]:
    """Fetch 1m klines from Bitget public REST — no auth required."""
    url = (
        f"https://api.bitget.com/api/v2/mix/market/candles"
        f"?symbol={symbol}&productType=USDT-FUTURES&granularity=1m"
        f"&startTime={start_ms}&endTime={end_ms}&limit=1000"
    )
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        if data.get("code") != "00000":
            return []
        bars = []
        for row in data.get("data", []):
            # Bitget v2: [ts, open, high, low, close, baseVol, quoteVol]
            bars.append(Bar(
                ts_ms=int(row[0]),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
            ))
        return sorted(bars, key=lambda b: b.ts_ms)
    except Exception as exc:
        print(f"  [WARN] REST fetch failed for {symbol}: {exc}")
        return []


def _get_bars_for_trade(
    t: BEStopTrade,
    parquet_bars: dict[str, list[Bar]],
    rest_cache: dict[str, list[Bar]],
    candle_dir: Path,
) -> list[Bar]:
    """Return 1m bars from TP1 hit time through stop hit time (plus buffer)."""
    sym = t.symbol

    # Load parquet once per symbol
    if sym not in parquet_bars:
        parquet_bars[sym] = _load_parquet_bars(candle_dir, sym)

    all_bars = parquet_bars[sym]

    # If parquet covers the window, use it
    start_ms = int(t.tp1_hit_time * 1000) - 60_000        # 1 bar before TP1
    end_ms = int(t.stop_hit_time * 1000) + 3 * 60_000     # 3 bars after stop

    window = [b for b in all_bars if start_ms <= b.ts_ms <= end_ms]
    if window:
        return window

    # Fallback: REST
    cache_key = f"{sym}_{start_ms}_{end_ms}"
    if cache_key not in rest_cache:
        print(f"  Fetching REST candles for {sym} {_fmt_ts(t.tp1_hit_time)} → {_fmt_ts(t.stop_hit_time)} ...")
        rest_cache[cache_key] = _fetch_rest_bars(sym, start_ms, end_ms)
        time.sleep(0.2)  # gentle rate limit
    return rest_cache[cache_key]


# ---------------------------------------------------------------------------
# Stop policy simulation
# ---------------------------------------------------------------------------

TAKER = 0.0006
MAKER = 0.0002


def _raw_pnl(entry: float, exit_price: float, size: float, side: str) -> float:
    sign = 1.0 if side == "long" else -1.0
    return (exit_price - entry) * size * sign


@dataclass
class PolicyResult:
    exit_reason: str        # "TP2", "TP3", "STOP", "EOD" (ran out of bars)
    exit_price: float
    exit_bar_idx: int
    mfe_after_tp1: float    # max favorable excursion after TP1 (price units)
    mfe_r: float            # MFE in R (relative to ATR)
    tp2_reached: bool
    tp3_reached: bool
    net_pnl: float          # full trade net including entry fee and TP1 partial


def _simulate_policy(
    t: BEStopTrade,
    bars: list[Bar],
    stop_policy: str,        # "be", "original", "trail05", "trail1"
) -> PolicyResult | None:
    """Simulate the post-TP1 portion of the trade under the given stop policy."""
    if not bars:
        return None

    is_long = t.side == "long"
    sign = 1.0 if is_long else -1.0

    # After TP1: remaining size is 50% of original (TP1 takes 50%)
    remaining = t.size * 0.50
    tp1_pnl_net = t.tp1_size * (t.tp1_fill_price - t.entry_price) * sign - t.tp1_fee

    # TP2 and TP3 sizes (25% each of original)
    tp2_size = t.size * 0.25
    tp3_size = t.size * 0.25

    # Initial stop level for this policy
    if stop_policy == "be":
        stop = t.entry_price          # BE stop (what actually happened)
    elif stop_policy == "original":
        stop = t.original_stop        # never moved
    elif stop_policy in ("trail05", "trail1"):
        mult = 0.5 if stop_policy == "trail05" else 1.0
        # Start trailing from TP1 fill level
        if is_long:
            stop = t.tp1_fill_price - mult * t.atr
        else:
            stop = t.tp1_fill_price + mult * t.atr
    else:
        return None

    high_watermark = t.tp1_fill_price   # tracks peak after TP1
    mfe_price = t.tp1_fill_price
    tp2_hit = False
    tp2_pnl = 0.0
    tp2_fee = 0.0

    for i, bar in enumerate(bars):
        # Update trailing stop from high watermark
        if stop_policy in ("trail05", "trail1"):
            mult = 0.5 if stop_policy == "trail05" else 1.0
            if is_long:
                new_hw = max(high_watermark, bar.high)
                high_watermark = new_hw
                stop = max(stop, new_hw - mult * t.atr)
            else:
                new_hw = min(high_watermark, bar.low)
                high_watermark = new_hw
                stop = min(stop, new_hw + mult * t.atr)

        # MFE tracking
        if is_long:
            mfe_price = max(mfe_price, bar.high)
        else:
            mfe_price = min(mfe_price, bar.low)

        # TP2 check (if not yet hit)
        if not tp2_hit and t.tp2_price > 0:
            if (is_long and bar.high >= t.tp2_price) or (not is_long and bar.low <= t.tp2_price):
                tp2_hit = True
                tp2_fee = tp2_size * t.tp2_price * MAKER
                tp2_pnl = _raw_pnl(t.entry_price, t.tp2_price, tp2_size, t.side) - tp2_fee
                remaining -= tp2_size

        # TP3 check (only if TP2 also hit)
        if tp2_hit and t.tp3_price > 0:
            if (is_long and bar.high >= t.tp3_price) or (not is_long and bar.low <= t.tp3_price):
                tp3_fee = tp3_size * t.tp3_price * MAKER
                tp3_pnl = _raw_pnl(t.entry_price, t.tp3_price, tp3_size, t.side) - tp3_fee
                total_pnl = -t.entry_fee + tp1_pnl_net + tp2_pnl + tp3_pnl
                mfe_r = (mfe_price - t.tp1_fill_price) * sign / t.atr if t.atr > 0 else 0.0
                return PolicyResult(
                    exit_reason="TP3", exit_price=t.tp3_price, exit_bar_idx=i,
                    mfe_after_tp1=abs(mfe_price - t.tp1_fill_price),
                    mfe_r=mfe_r, tp2_reached=True, tp3_reached=True,
                    net_pnl=round(total_pnl, 4),
                )

        # Stop check
        stop_triggered = (is_long and bar.low <= stop) or (not is_long and bar.high >= stop)
        if stop_triggered:
            stop_fill = stop  # assume fills at stop exactly (conservative)
            sf = remaining * stop_fill * TAKER
            stop_pnl = _raw_pnl(t.entry_price, stop_fill, remaining, t.side) - sf
            total_pnl = -t.entry_fee + tp1_pnl_net + tp2_pnl + stop_pnl
            mfe_r = (mfe_price - t.tp1_fill_price) * sign / t.atr if t.atr > 0 else 0.0
            return PolicyResult(
                exit_reason="STOP", exit_price=stop_fill, exit_bar_idx=i,
                mfe_after_tp1=abs(mfe_price - t.tp1_fill_price),
                mfe_r=mfe_r, tp2_reached=tp2_hit, tp3_reached=False,
                net_pnl=round(total_pnl, 4),
            )

    # Ran out of bars — use last close
    if bars:
        last_price = bars[-1].close
        sf = remaining * last_price * TAKER
        eod_pnl = _raw_pnl(t.entry_price, last_price, remaining, t.side) - sf
        total_pnl = -t.entry_fee + tp1_pnl_net + tp2_pnl + eod_pnl
        mfe_r = (mfe_price - t.tp1_fill_price) * sign / t.atr if t.atr > 0 else 0.0
        return PolicyResult(
            exit_reason="EOD", exit_price=last_price, exit_bar_idx=len(bars),
            mfe_after_tp1=abs(mfe_price - t.tp1_fill_price),
            mfe_r=mfe_r, tp2_reached=tp2_hit, tp3_reached=False,
            net_pnl=round(total_pnl, 4),
        )
    return None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_ts(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _pnl_col(v: float, w: int = 9) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}".rjust(w)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

POLICIES = ["be", "original", "trail05", "trail1"]
POLICY_LABELS = {
    "be":       "A  BE-stop  (current)",
    "original": "B  Original stop",
    "trail05":  "C  0.5 ATR trail",
    "trail1":   "D  1.0 ATR trail",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Break-even stop analysis")
    p.add_argument("--journal-dir", type=Path, default=Path("logs/paper"))
    p.add_argument("--candle-dir", type=Path, default=Path("data/candles"),
                   help="Parquet candle directory (1m files). Falls back to REST if not found.")
    p.add_argument("--verbose", action="store_true", help="Print per-trade detail")
    p.add_argument("--no-rest", action="store_true", help="Disable REST fallback")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.journal_dir.exists():
        print(f"Journal directory not found: {args.journal_dir}")
        sys.exit(1)

    print("Loading journal events ...")
    events = _load_events(args.journal_dir)
    print(f"  {len(events)} events loaded.")

    print("Extracting BE-stop trades ...")
    be_trades = extract_be_stop_trades(events)
    total_fills = sum(1 for e in events if e.get("event") == "FILL")
    total_stops = sum(1 for e in events if e.get("event") == "STOP_HIT")
    print(f"  Total fills  : {total_fills}")
    print(f"  Total stops  : {total_stops}")
    print(f"  BE-stop trades (TP1 hit, then stopped at BE): {len(be_trades)}")

    if not be_trades:
        print("\nNo BE-stop trades found. Nothing to analyse.")
        sys.exit(0)

    parquet_cache: dict[str, list[Bar]] = {}
    rest_cache: dict[str, list[Bar]] = {}

    # Per-policy accumulators
    policy_net: dict[str, float] = {p: 0.0 for p in POLICIES}
    policy_tp2: dict[str, int] = {p: 0 for p in POLICIES}
    policy_tp3: dict[str, int] = {p: 0 for p in POLICIES}
    policy_stop: dict[str, int] = {p: 0 for p in POLICIES}
    no_bar_count = 0

    results_table: list[dict] = []

    for t in be_trades:
        bars = _get_bars_for_trade(t, parquet_cache, {} if args.no_rest else rest_cache, args.candle_dir)
        if not bars:
            no_bar_count += 1

        row: dict = {
            "symbol": t.symbol, "side": t.side,
            "entry_time": _fmt_ts(t.entry_time),
            "tp1_time": _fmt_ts(t.tp1_hit_time),
            "stop_time": _fmt_ts(t.stop_hit_time),
            "entry_price": t.entry_price,
            "tp1_price": t.tp1_price,
            "tp2_price": t.tp2_price,
            "atr": round(t.atr, 4),
            "actual_net": round(t.stop_net_pnl, 2),
        }

        for pol in POLICIES:
            res = _simulate_policy(t, bars, pol)
            if res:
                policy_net[pol] += res.net_pnl
                if res.tp2_reached:
                    policy_tp2[pol] += 1
                if res.tp3_reached:
                    policy_tp3[pol] += 1
                if res.exit_reason == "STOP":
                    policy_stop[pol] += 1
                row[f"{pol}_net"] = round(res.net_pnl, 2)
                row[f"{pol}_reason"] = res.exit_reason
                row[f"{pol}_mfe_r"] = round(res.mfe_r, 2)
            else:
                row[f"{pol}_net"] = "N/A"
                row[f"{pol}_reason"] = "NO_DATA"
                row[f"{pol}_mfe_r"] = "N/A"

        results_table.append(row)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    n = len(be_trades)
    bar = "─" * 68
    print(f"\n{'═'*68}")
    print(f"  BREAK-EVEN STOP ANALYSIS  —  {n} trades")
    print(f"{'═'*68}")
    print(f"  Trades without candle data  : {no_bar_count} (REST fetch failed or no parquet)")
    print(bar)
    print(f"  {'Policy':<24}  {'Net PnL':>9}  {'TP2 hit':>7}  {'TP3 hit':>7}  {'Stopped':>7}")
    print(f"  {'─'*24}  {'─'*9}  {'─'*7}  {'─'*7}  {'─'*7}")
    for pol in POLICIES:
        net = policy_net[pol]
        tp2_pct = policy_tp2[pol] / n * 100
        tp3_pct = policy_tp3[pol] / n * 100
        stop_pct = policy_stop[pol] / n * 100
        label = POLICY_LABELS[pol]
        print(
            f"  {label:<24}  {_pnl_col(net):>9}  "
            f"{policy_tp2[pol]:>3} ({tp2_pct:4.0f}%)  "
            f"{policy_tp3[pol]:>3} ({tp3_pct:4.0f}%)  "
            f"{policy_stop[pol]:>3} ({stop_pct:4.0f}%)"
        )
    print(bar)

    # Answer the key question
    be_net = policy_net["be"]
    best_pol = max(POLICIES, key=lambda p: policy_net[p])
    best_net = policy_net[best_pol]
    delta = best_net - be_net
    print(f"\n  Key question: Is BE stop protecting capital or cutting winners?")
    print(f"  Best alternative : {POLICY_LABELS[best_pol]}")
    print(f"  Net PnL delta    : {_pnl_col(delta)} vs current BE stop")
    if delta > 0:
        print(f"  Verdict          : BE stop is SYSTEMATICALLY CUTTING WINNERS.")
        print(f"                     {POLICY_LABELS[best_pol]} adds {delta:+.2f} on these {n} trades.")
    else:
        print(f"  Verdict          : BE stop is PROTECTING CAPITAL.")
        print(f"                     No tested alternative improves net PnL.")
    print(f"{'═'*68}\n")

    # ------------------------------------------------------------------
    # MFE distribution after TP1
    # ------------------------------------------------------------------
    be_mfe_rs = [row.get("be_mfe_r", 0) for row in results_table
                 if isinstance(row.get("be_mfe_r"), (int, float))]
    if be_mfe_rs:
        sorted_mfe = sorted(be_mfe_rs)
        median = sorted_mfe[len(sorted_mfe) // 2]
        mean = sum(sorted_mfe) / len(sorted_mfe)
        pct75 = sorted_mfe[int(len(sorted_mfe) * 0.75)]
        pct90 = sorted_mfe[int(len(sorted_mfe) * 0.90)]
        print(f"  MFE after TP1 (in ATR units, across {len(be_mfe_rs)} trades):")
        print(f"  Mean={mean:.2f}R  Median={median:.2f}R  p75={pct75:.2f}R  p90={pct90:.2f}R")

        # Distribution buckets
        buckets = {"<0.25R": 0, "0.25-0.5R": 0, "0.5-1.0R": 0, "1.0-2.0R": 0, ">2.0R": 0}
        for v in be_mfe_rs:
            if v < 0.25:
                buckets["<0.25R"] += 1
            elif v < 0.5:
                buckets["0.25-0.5R"] += 1
            elif v < 1.0:
                buckets["0.5-1.0R"] += 1
            elif v < 2.0:
                buckets["1.0-2.0R"] += 1
            else:
                buckets[">2.0R"] += 1
        print(f"  Distribution: {buckets}")
        tp1_to_tp2 = abs(be_trades[0].tp2_price - be_trades[0].tp1_price) / be_trades[0].atr if be_trades[0].atr > 0 else 1.0
        print(f"  Note: TP1→TP2 distance ≈ {tp1_to_tp2:.2f}R (strategy config)")
        print()

    # ------------------------------------------------------------------
    # Per-trade verbose
    # ------------------------------------------------------------------
    if args.verbose:
        print(f"\n  {'#':>3}  {'Sym':8}  {'Side':5}  {'Entry':>9}  {'TP1':>9}  {'ATR':>6}  "
              f"{'A_net':>8}  {'B_net':>8}  {'C_net':>8}  {'D_net':>8}  {'A_rsn':>6}  {'B_rsn':>6}")
        print(f"  {'─'*3}  {'─'*8}  {'─'*5}  {'─'*9}  {'─'*9}  {'─'*6}  "
              f"{'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*6}")
        for i, row in enumerate(results_table, 1):
            def _v(k):
                v = row.get(k, "N/A")
                return f"{v:>8.2f}" if isinstance(v, float) else str(v).rjust(8)
            print(
                f"  {i:>3}  {row['symbol']:8}  {row['side']:5}  "
                f"{row['entry_price']:>9.2f}  {row['tp1_price']:>9.2f}  {row['atr']:>6.4f}  "
                f"{_v('be_net')}  {_v('original_net')}  {_v('trail05_net')}  {_v('trail1_net')}  "
                f"{str(row.get('be_reason','')):>6}  {str(row.get('original_reason','')):>6}"
            )
        print()


if __name__ == "__main__":
    main()
