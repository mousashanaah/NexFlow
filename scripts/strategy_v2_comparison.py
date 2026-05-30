#!/usr/bin/env python3
"""Strategy variant comparison: V1 (current) vs V2 (trail runner).

Uses the exact same entry set from paper trading journals and simulates
both exit strategies on the actual 1m candle data.

V1 — Current:
    TP1: close 50% at 1R (1×ATR)
    After TP1: move stop to break-even
    TP2: close 25% at 2R
    TP3: close 25% at 3R (or BE stop if hit first)

V2 — Trail Runner:
    TP1: close 50% at 1R (unchanged)
    After TP1: remaining 50% uses a 0.5×ATR trailing stop
    No TP2, no TP3 — let the trail decide

Entries, risk sizing, and TP1 price are IDENTICAL between variants.
Only post-TP1 management differs.

Usage:
    python scripts/strategy_v2_comparison.py
    python scripts/strategy_v2_comparison.py --journal-dir logs/paper --candle-dir data/candles
    python scripts/strategy_v2_comparison.py --verbose
"""

from __future__ import annotations

import argparse
import csv
import json
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

TAKER = 0.0006
MAKER = 0.0002


# ---------------------------------------------------------------------------
# Minimal bar struct
# ---------------------------------------------------------------------------

class Bar(NamedTuple):
    ts_ms: int
    open: float
    high: float
    low: float
    close: float


# ---------------------------------------------------------------------------
# Entry record — one per FILL event in the journal
# ---------------------------------------------------------------------------

@dataclass
class Entry:
    symbol: str
    side: str
    entry_time: float        # epoch seconds
    entry_price: float
    size: float
    entry_fee: float
    original_stop: float
    tp1_price: float
    tp2_price: float
    tp3_price: float
    atr: float


# ---------------------------------------------------------------------------
# Trade result from one simulation
# ---------------------------------------------------------------------------

@dataclass
class TradeResult:
    variant: str
    symbol: str
    side: str
    entry_time: str
    exit_time: str
    gross_pnl: float
    fees: float
    net_pnl: float
    exit_reason: str         # TP1_ONLY | TRAIL | TP2 | TP3 | STOP | EOD | NO_DATA
    tp1_hit: bool
    holding_bars: int


# ---------------------------------------------------------------------------
# Journal loading
# ---------------------------------------------------------------------------

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


def extract_entries(events: list[dict]) -> list[Entry]:
    """Extract one Entry per FILL, enriched with signal data."""
    last_signal: dict[str, dict] = {}
    entries: list[Entry] = []

    for ev in events:
        evt = ev.get("event", "")
        sym = ev.get("symbol", "")

        if evt == "SIGNAL":
            last_signal[sym] = ev

        elif evt == "FILL":
            sig = last_signal.get(sym, {})
            tp_prices = sig.get("tp_prices", [])
            atr = sig.get("atr", sig.get("features", {}).get("atr", 0.0))

            if not tp_prices or atr <= 0:
                continue  # can't simulate without TP targets and ATR

            entries.append(Entry(
                symbol=sym,
                side=ev.get("direction", "long"),
                entry_time=ev.get("ts_epoch", 0.0),
                entry_price=ev.get("fill_price", 0.0),
                size=ev.get("size", 0.0),
                entry_fee=ev.get("fee", 0.0),
                original_stop=sig.get("stop_price", 0.0),
                tp1_price=tp_prices[0],
                tp2_price=tp_prices[1] if len(tp_prices) > 1 else 0.0,
                tp3_price=tp_prices[2] if len(tp_prices) > 2 else 0.0,
                atr=atr,
            ))

    return entries


# ---------------------------------------------------------------------------
# Candle loading
# ---------------------------------------------------------------------------

def _load_parquet(candle_dir: Path, symbol: str) -> list[Bar]:
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


def _fetch_rest(symbol: str, start_ms: int, end_ms: int) -> list[Bar]:
    url = (
        f"https://api.bitget.com/api/v2/mix/market/candles"
        f"?symbol={symbol}&productType=USDT-FUTURES&granularity=1m"
        f"&startTime={start_ms}&endTime={end_ms}&limit=1000"
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read())
        if data.get("code") != "00000":
            return []
        bars = [
            Bar(ts_ms=int(r[0]), open=float(r[1]), high=float(r[2]),
                low=float(r[3]), close=float(r[4]))
            for r in data.get("data", [])
        ]
        return sorted(bars, key=lambda b: b.ts_ms)
    except Exception as exc:
        print(f"  [WARN] REST fetch failed for {symbol}: {exc}")
        return []


def _get_bars(
    entry: Entry,
    parquet_cache: dict[str, list[Bar]],
    rest_cache: dict[str, list[Bar]],
    candle_dir: Path,
    no_rest: bool,
    lookahead_bars: int = 240,
) -> list[Bar]:
    sym = entry.symbol
    if sym not in parquet_cache:
        parquet_cache[sym] = _load_parquet(candle_dir, sym)

    start_ms = int(entry.entry_time * 1000)
    end_ms = start_ms + lookahead_bars * 60_000

    # Try parquet window
    window = [b for b in parquet_cache[sym] if start_ms <= b.ts_ms <= end_ms]
    if window:
        return window

    if no_rest:
        return []

    cache_key = f"{sym}_{start_ms}_{end_ms}"
    if cache_key not in rest_cache:
        rest_cache[cache_key] = _fetch_rest(sym, start_ms, end_ms)
        time.sleep(0.2)
    return rest_cache[cache_key]


# ---------------------------------------------------------------------------
# Simulation primitives
# ---------------------------------------------------------------------------

def _sign(side: str) -> float:
    return 1.0 if side == "long" else -1.0


def _fee(price: float, size: float, is_maker: bool) -> float:
    return price * size * (MAKER if is_maker else TAKER)


def _gross(entry: float, exit_p: float, size: float, side: str) -> float:
    return (exit_p - entry) * size * _sign(side)


def _fmt_ts(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


# ---------------------------------------------------------------------------
# V1 simulation — TP1/TP2/TP3 with BE stop after TP1
# ---------------------------------------------------------------------------

def simulate_v1(entry: Entry, bars: list[Bar]) -> TradeResult:
    if not bars:
        return TradeResult(
            variant="V1", symbol=entry.symbol, side=entry.side,
            entry_time=_fmt_ts(entry.entry_time), exit_time="",
            gross_pnl=0, fees=entry.entry_fee, net_pnl=-entry.entry_fee,
            exit_reason="NO_DATA", tp1_hit=False, holding_bars=0,
        )

    is_long = entry.side == "long"
    total_fees = entry.entry_fee
    total_gross = 0.0

    remaining = entry.size
    tp1_size = entry.size * 0.50
    tp2_size = entry.size * 0.25
    tp3_size = entry.size * 0.25

    stop = entry.original_stop
    tp1_hit = False
    tp2_hit = False

    for i, bar in enumerate(bars):
        # TP1
        if not tp1_hit and (
            (is_long and bar.high >= entry.tp1_price) or
            (not is_long and bar.low <= entry.tp1_price)
        ):
            tp1_hit = True
            f = _fee(entry.tp1_price, tp1_size, is_maker=True)
            total_gross += _gross(entry.entry_price, entry.tp1_price, tp1_size, entry.side)
            total_fees += f
            remaining -= tp1_size
            stop = entry.entry_price  # move to BE

        # TP2 (only after TP1)
        if tp1_hit and not tp2_hit and entry.tp2_price > 0 and (
            (is_long and bar.high >= entry.tp2_price) or
            (not is_long and bar.low <= entry.tp2_price)
        ):
            tp2_hit = True
            f = _fee(entry.tp2_price, tp2_size, is_maker=True)
            total_gross += _gross(entry.entry_price, entry.tp2_price, tp2_size, entry.side)
            total_fees += f
            remaining -= tp2_size

        # TP3 (only after TP2)
        if tp2_hit and entry.tp3_price > 0 and (
            (is_long and bar.high >= entry.tp3_price) or
            (not is_long and bar.low <= entry.tp3_price)
        ):
            f = _fee(entry.tp3_price, tp3_size, is_maker=True)
            total_gross += _gross(entry.entry_price, entry.tp3_price, tp3_size, entry.side)
            total_fees += f
            remaining = 0.0
            net = total_gross - total_fees
            return TradeResult(
                variant="V1", symbol=entry.symbol, side=entry.side,
                entry_time=_fmt_ts(entry.entry_time),
                exit_time=_fmt_ts(bar.ts_ms / 1000),
                gross_pnl=round(total_gross, 4), fees=round(total_fees, 4),
                net_pnl=round(net, 4), exit_reason="TP3",
                tp1_hit=True, holding_bars=i + 1,
            )

        # Stop
        stop_hit = (is_long and bar.low <= stop) or (not is_long and bar.high >= stop)
        if stop_hit and remaining > 0:
            f = _fee(stop, remaining, is_maker=False)
            total_gross += _gross(entry.entry_price, stop, remaining, entry.side)
            total_fees += f
            net = total_gross - total_fees
            reason = "STOP" if not tp1_hit else ("TP2+STOP" if tp2_hit else "TP1+BESTOP")
            return TradeResult(
                variant="V1", symbol=entry.symbol, side=entry.side,
                entry_time=_fmt_ts(entry.entry_time),
                exit_time=_fmt_ts(bar.ts_ms / 1000),
                gross_pnl=round(total_gross, 4), fees=round(total_fees, 4),
                net_pnl=round(net, 4), exit_reason=reason,
                tp1_hit=tp1_hit, holding_bars=i + 1,
            )

    # End of bars: close at last price
    if remaining > 0 and bars:
        last = bars[-1].close
        f = _fee(last, remaining, is_maker=False)
        total_gross += _gross(entry.entry_price, last, remaining, entry.side)
        total_fees += f

    net = total_gross - total_fees
    return TradeResult(
        variant="V1", symbol=entry.symbol, side=entry.side,
        entry_time=_fmt_ts(entry.entry_time),
        exit_time=_fmt_ts(bars[-1].ts_ms / 1000) if bars else "",
        gross_pnl=round(total_gross, 4), fees=round(total_fees, 4),
        net_pnl=round(net, 4),
        exit_reason="EOD",
        tp1_hit=tp1_hit, holding_bars=len(bars),
    )


# ---------------------------------------------------------------------------
# V2 simulation — TP1 close 50%, then 0.5 ATR trailing stop on remainder
# ---------------------------------------------------------------------------

def simulate_v2(entry: Entry, bars: list[Bar]) -> TradeResult:
    if not bars:
        return TradeResult(
            variant="V2", symbol=entry.symbol, side=entry.side,
            entry_time=_fmt_ts(entry.entry_time), exit_time="",
            gross_pnl=0, fees=entry.entry_fee, net_pnl=-entry.entry_fee,
            exit_reason="NO_DATA", tp1_hit=False, holding_bars=0,
        )

    is_long = entry.side == "long"
    trail_mult = 0.5

    total_fees = entry.entry_fee
    total_gross = 0.0

    remaining = entry.size
    tp1_size = entry.size * 0.50

    stop = entry.original_stop       # pre-TP1: original stop
    trail_active = False
    tp1_hit = False

    for i, bar in enumerate(bars):
        # Update trailing stop once TP1 is hit
        if trail_active:
            if is_long:
                new_stop = bar.high - trail_mult * entry.atr
                stop = max(stop, new_stop)
            else:
                new_stop = bar.low + trail_mult * entry.atr
                stop = min(stop, new_stop)

        # TP1
        if not tp1_hit and (
            (is_long and bar.high >= entry.tp1_price) or
            (not is_long and bar.low <= entry.tp1_price)
        ):
            tp1_hit = True
            f = _fee(entry.tp1_price, tp1_size, is_maker=True)
            total_gross += _gross(entry.entry_price, entry.tp1_price, tp1_size, entry.side)
            total_fees += f
            remaining -= tp1_size

            # Initialise trail from TP1 level
            if is_long:
                stop = entry.tp1_price - trail_mult * entry.atr
            else:
                stop = entry.tp1_price + trail_mult * entry.atr
            trail_active = True

        # Stop check (original stop pre-TP1, trail post-TP1)
        stop_hit = (is_long and bar.low <= stop) or (not is_long and bar.high >= stop)
        if stop_hit and remaining > 0:
            f = _fee(stop, remaining, is_maker=False)
            total_gross += _gross(entry.entry_price, stop, remaining, entry.side)
            total_fees += f
            net = total_gross - total_fees
            reason = "TRAIL" if tp1_hit else "STOP"
            return TradeResult(
                variant="V2", symbol=entry.symbol, side=entry.side,
                entry_time=_fmt_ts(entry.entry_time),
                exit_time=_fmt_ts(bar.ts_ms / 1000),
                gross_pnl=round(total_gross, 4), fees=round(total_fees, 4),
                net_pnl=round(net, 4), exit_reason=reason,
                tp1_hit=tp1_hit, holding_bars=i + 1,
            )

    # End of bars
    if remaining > 0 and bars:
        last = bars[-1].close
        f = _fee(last, remaining, is_maker=False)
        total_gross += _gross(entry.entry_price, last, remaining, entry.side)
        total_fees += f

    net = total_gross - total_fees
    return TradeResult(
        variant="V2", symbol=entry.symbol, side=entry.side,
        entry_time=_fmt_ts(entry.entry_time),
        exit_time=_fmt_ts(bars[-1].ts_ms / 1000) if bars else "",
        gross_pnl=round(total_gross, 4), fees=round(total_fees, 4),
        net_pnl=round(net, 4),
        exit_reason="EOD",
        tp1_hit=tp1_hit, holding_bars=len(bars),
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics(results: list[TradeResult]) -> dict:
    if not results:
        return {}
    wins = [r for r in results if r.net_pnl > 0]
    losses = [r for r in results if r.net_pnl <= 0]
    gross_profit = sum(r.net_pnl for r in wins)
    gross_loss = sum(abs(r.net_pnl) for r in losses)
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    avg_win = gross_profit / len(wins) if wins else 0.0
    avg_loss = gross_loss / len(losses) if losses else 0.0
    expectancy = (
        (len(wins) / len(results) * avg_win - len(losses) / len(results) * avg_loss) / avg_loss
        if avg_loss > 0 else 0.0
    )
    exits = defaultdict(int)
    for r in results:
        exits[r.exit_reason] += 1
    return {
        "trades": len(results),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(results),
        "gross_pnl": sum(r.gross_pnl for r in results),
        "total_fees": sum(r.fees for r in results),
        "net_pnl": sum(r.net_pnl for r in results),
        "profit_factor": pf,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy_r": expectancy,
        "avg_hold_bars": sum(r.holding_bars for r in results) / len(results),
        "exits": dict(exits),
        "no_data": sum(1 for r in results if r.exit_reason == "NO_DATA"),
    }


def _print_comparison(m1: dict, m2: dict) -> None:
    if not m1 or not m2:
        print("Insufficient data for comparison.")
        return

    def _pct(v: float) -> str:
        return f"{v*100:.1f}%"

    def _col(v1, v2, fmt=".2f", higher_better=True):
        s1 = f"{v1:{fmt}}"
        s2 = f"{v2:{fmt}}"
        if isinstance(v1, float) and isinstance(v2, float):
            delta = v2 - v1
            arrow = "▲" if (delta > 0) == higher_better else "▼"
            if abs(delta) < 0.001:
                arrow = "─"
            return s1, s2, f"{arrow} {delta:+.2f}"
        return s1, s2, ""

    bar = "─" * 62
    print(f"\n{'═'*62}")
    print(f"  {'Metric':<22}  {'V1 (current)':>14}  {'V2 (trail)':>14}")
    print(f"  {'─'*22}  {'─'*14}  {'─'*14}")

    rows = [
        ("Trades",        m1["trades"],       m2["trades"],       "d",  True),
        ("No-data skips", m1["no_data"],       m2["no_data"],      "d",  False),
        ("Win rate",      m1["win_rate"],      m2["win_rate"],     "%",  True),
        ("Gross PnL",     m1["gross_pnl"],     m2["gross_pnl"],    "f",  True),
        ("Total fees",    m1["total_fees"],    m2["total_fees"],   "f",  False),
        ("Net PnL",       m1["net_pnl"],       m2["net_pnl"],      "f",  True),
        ("Profit factor", m1["profit_factor"], m2["profit_factor"],"f",  True),
        ("Avg winner",    m1["avg_win"],       m2["avg_win"],      "f",  True),
        ("Avg loser",     m1["avg_loss"],      m2["avg_loss"],     "f",  False),
        ("Expectancy R",  m1["expectancy_r"],  m2["expectancy_r"], "f",  True),
        ("Avg hold bars", m1["avg_hold_bars"], m2["avg_hold_bars"],"f",  None),
    ]

    for label, v1, v2, fmt, hb in rows:
        if fmt == "d":
            s1, s2 = str(v1), str(v2)
            delta = ""
        elif fmt == "%":
            s1, s2 = _pct(v1), _pct(v2)
            dv = (v2 - v1) * 100
            arrow = ("▲" if dv > 0 else "▼") if hb is not None else "─"
            if abs(dv) < 0.1:
                arrow = "─"
            delta = f"{arrow} {dv:+.1f}pp"
        else:
            if isinstance(v1, float) and v1 == float("inf"):
                s1 = "inf"
            else:
                s1 = f"{v1:+.2f}" if fmt == "f" and isinstance(v1, float) else str(v1)
            if isinstance(v2, float) and v2 == float("inf"):
                s2 = "inf"
            else:
                s2 = f"{v2:+.2f}" if fmt == "f" and isinstance(v2, float) else str(v2)
            if isinstance(v1, float) and isinstance(v2, float) and v1 != float("inf") and v2 != float("inf"):
                dv = v2 - v1
                if hb is None:
                    arrow = "─"
                else:
                    arrow = ("▲" if dv > 0 else "▼") if hb else ("▼" if dv > 0 else "▲")
                    if abs(dv) < 0.01:
                        arrow = "─"
                delta = f"{arrow} {dv:+.2f}"
            else:
                delta = ""
        print(f"  {label:<22}  {s1:>14}  {s2:>14}  {delta}")

    print(bar)
    print(f"  Exit breakdown V1: {m1['exits']}")
    print(f"  Exit breakdown V2: {m2['exits']}")
    print(f"{'═'*62}")


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def _write_csv(v1_results: list[TradeResult], v2_results: list[TradeResult], out: Path) -> None:
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "entry_time", "symbol", "side",
            "v1_gross", "v1_fees", "v1_net", "v1_exit", "v1_tp1", "v1_bars",
            "v2_gross", "v2_fees", "v2_net", "v2_exit", "v2_tp1", "v2_bars",
            "delta_net",
        ])
        for r1, r2 in zip(v1_results, v2_results):
            w.writerow([
                r1.entry_time, r1.symbol, r1.side,
                r1.gross_pnl, r1.fees, r1.net_pnl, r1.exit_reason, r1.tp1_hit, r1.holding_bars,
                r2.gross_pnl, r2.fees, r2.net_pnl, r2.exit_reason, r2.tp1_hit, r2.holding_bars,
                round(r2.net_pnl - r1.net_pnl, 4),
            ])
    print(f"CSV written: {out}  ({len(v1_results)} trades)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V1 vs V2 strategy comparison")
    p.add_argument("--journal-dir", type=Path, default=Path("logs/paper"))
    p.add_argument("--candle-dir", type=Path, default=Path("data/candles"))
    p.add_argument("--out", type=Path, default=Path("strategy_v2_comparison.csv"))
    p.add_argument("--lookahead", type=int, default=240,
                   help="Max bars to look ahead per trade (default 240 = 4 hours)")
    p.add_argument("--no-rest", action="store_true", help="Disable Bitget REST fallback")
    p.add_argument("--verbose", action="store_true", help="Print per-trade detail")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.journal_dir.exists():
        print(f"Journal directory not found: {args.journal_dir}")
        sys.exit(1)

    print("Loading journals ...")
    events = _load_events(args.journal_dir)
    entries = extract_entries(events)
    print(f"  {len(events)} events → {len(entries)} tradeable entries")

    if not entries:
        print("No entries found.")
        sys.exit(1)

    parquet_cache: dict[str, list[Bar]] = {}
    rest_cache: dict[str, list[Bar]] = {}

    v1_results: list[TradeResult] = []
    v2_results: list[TradeResult] = []
    no_data = 0

    print(f"Simulating {len(entries)} trades ...")
    for entry in entries:
        bars = _get_bars(entry, parquet_cache, rest_cache, args.candle_dir, args.no_rest, args.lookahead)
        if not bars:
            no_data += 1
        r1 = simulate_v1(entry, bars)
        r2 = simulate_v2(entry, bars)
        v1_results.append(r1)
        v2_results.append(r2)

    if no_data > 0:
        print(f"  {no_data} entries had no candle data (fetching from REST or parquet missing)")

    m1 = _metrics(v1_results)
    m2 = _metrics(v2_results)

    _print_comparison(m1, m2)

    if args.verbose:
        print(f"\n  {'#':>3}  {'Time':16}  {'Sym':8}  {'Side':5}  "
              f"{'V1 net':>9}  {'V1 exit':12}  {'V2 net':>9}  {'V2 exit':12}  {'Delta':>8}")
        print(f"  {'─'*3}  {'─'*16}  {'─'*8}  {'─'*5}  "
              f"{'─'*9}  {'─'*12}  {'─'*9}  {'─'*12}  {'─'*8}")
        for i, (r1, r2) in enumerate(zip(v1_results, v2_results), 1):
            delta = r2.net_pnl - r1.net_pnl
            marker = " ▲" if delta > 0 else (" ▼" if delta < 0 else "  ")
            print(
                f"  {i:>3}  {r1.entry_time:16}  {r1.symbol:8}  {r1.side:5}  "
                f"{r1.net_pnl:>+9.2f}  {r1.exit_reason:12}  "
                f"{r2.net_pnl:>+9.2f}  {r2.exit_reason:12}  {delta:>+8.2f}{marker}"
            )

    _write_csv(v1_results, v2_results, args.out)


if __name__ == "__main__":
    main()
