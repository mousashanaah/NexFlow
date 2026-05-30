#!/usr/bin/env python3
"""Daily summary report from paper trading journal files.

Reads all JSONL journals in a directory and prints a per-day table covering:
    trades, win rate, gross PnL, net PnL, fees, maker fill rate.

Usage:
    python scripts/daily_summary.py
    python scripts/daily_summary.py --journal-dir logs/paper
    python scripts/daily_summary.py --journal-dir logs/paper --days 7
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))


@dataclass
class _DayStats:
    fills: int = 0
    stops: int = 0
    tps: int = 0
    force_closes: int = 0
    gross_pnl: float = 0.0       # price-movement pnl before fees
    total_fees: float = 0.0      # all fees: entry + partial exits + stop/force
    maker_attempts: int = 0      # TP_LIMITS_PLACED n_limits sum
    maker_fills: int = 0         # TP_MAKER_FILL count
    latencies: list[int] = field(default_factory=list)
    wins: int = 0                # closed trades with net pnl > 0 (tracked via SESSION equity proxy)
    closed_pnls: list[float] = field(default_factory=list)  # per-trade pnl for win rate


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Daily paper trading summary")
    p.add_argument("--journal-dir", type=Path, default=Path("logs/paper"))
    p.add_argument("--days", type=int, default=0, help="Show last N days only (0 = all)")
    p.add_argument("--csv", action="store_true", help="Output CSV instead of table")
    return p.parse_args()


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
    return events


def _day_key(ts_epoch: float) -> str:
    return datetime.fromtimestamp(ts_epoch, tz=timezone.utc).strftime("%Y-%m-%d")


def _build_days(events: list[dict]) -> dict[str, _DayStats]:
    days: dict[str, _DayStats] = defaultdict(_DayStats)

    # Per-symbol fee accumulator: entry_fee + partial_fees, cleared on trade close.
    # Used to derive gross = net + total_fees for closed trades.
    open_fees: dict[str, float] = {}    # symbol → accumulated fees for open trade
    entry_prices: dict[str, float] = {}
    entry_sides: dict[str, str] = {}

    for ev in events:
        evt = ev.get("event", "")
        day = _day_key(ev.get("ts_epoch", 0.0))
        d = days[day]

        if evt == "FILL":
            sym = ev.get("symbol", "")
            fee = ev.get("fee", 0.0)
            d.fills += 1
            open_fees[sym] = fee
            entry_prices[sym] = ev.get("fill_price", 0.0)
            entry_sides[sym] = ev.get("direction", "long")

        elif evt == "PARTIAL_TP":
            sym = ev.get("symbol", "")
            # Accumulate exit-side fees; gross will be derived when trade closes.
            open_fees[sym] = open_fees.get(sym, 0.0) + ev.get("fee", 0.0)
            d.tps += 1

        elif evt == "STOP_HIT":
            # STOP_HIT.pnl = trade.pnl = authoritative total net pnl for the whole trade.
            sym = ev.get("symbol", "")
            stop_fee = ev.get("fee", 0.0)
            net = ev.get("pnl", 0.0)
            total_fees = open_fees.pop(sym, 0.0) + stop_fee
            gross = net + total_fees
            d.gross_pnl += gross
            d.total_fees += total_fees
            d.stops += 1
            d.closed_pnls.append(net)

        elif evt == "FORCE_CLOSE":
            sym = ev.get("symbol", "")
            net = ev.get("pnl", 0.0)
            total_fees = open_fees.pop(sym, 0.0)
            gross = net + total_fees
            d.gross_pnl += gross
            d.total_fees += total_fees
            d.force_closes += 1
            d.closed_pnls.append(net)

        elif evt == "TP_LIMITS_PLACED":
            d.maker_attempts += ev.get("n_limits", 0)

        elif evt == "TP_MAKER_FILL":
            d.maker_fills += 1
            lat = ev.get("latency_bars", 0)
            d.latencies.append(lat)

    # Compute wins from closed_pnls
    for d in days.values():
        d.wins = sum(1 for p in d.closed_pnls if p > 0)

    return days


def _print_table(days: dict[str, _DayStats], last_n: int, csv_mode: bool) -> None:
    keys = sorted(days.keys())
    if last_n > 0:
        keys = keys[-last_n:]

    if not keys:
        print("No journal data found.")
        return

    if csv_mode:
        print("date,trades,wins,win_rate,gross_pnl,fees,net_pnl,maker_attempts,maker_fills,maker_fill_rate,avg_latency_bars")
        for k in keys:
            d = days[k]
            trades = len(d.closed_pnls)
            wr = d.wins / trades * 100 if trades else 0.0
            net = d.gross_pnl - d.total_fees
            mfr = d.maker_fills / d.maker_attempts * 100 if d.maker_attempts else 0.0
            avg_lat = sum(d.latencies) / len(d.latencies) if d.latencies else 0.0
            print(f"{k},{trades},{d.wins},{wr:.1f},{d.gross_pnl:.2f},{d.total_fees:.2f},{net:.2f},{d.maker_attempts},{d.maker_fills},{mfr:.1f},{avg_lat:.1f}")
        return

    bar = "═" * 95
    print(f"\n╔{bar}╗")
    print(f"  {'NexFlow — Daily Summary':^93}")
    print(f"╠{bar}╣")
    print(
        f"  {'Date':10}  {'Trades':>6}  {'WR%':>5}  {'Gross PnL':>10}  "
        f"{'Fees':>9}  {'Net PnL':>10}  {'Mkr Att':>7}  {'Mkr Fill':>8}  {'Fill%':>5}  {'Lat(bars)':>9}"
    )
    print(f"  {'-'*10}  {'-'*6}  {'-'*5}  {'-'*10}  {'-'*9}  {'-'*10}  {'-'*7}  {'-'*8}  {'-'*5}  {'-'*9}")

    total_trades = total_wins = 0
    total_gross = total_fees_all = total_net = 0.0
    total_attempts = total_mk_fills = 0
    all_latencies: list[int] = []

    for k in keys:
        d = days[k]
        trades = len(d.closed_pnls)
        wr = d.wins / trades * 100 if trades else 0.0
        all_fees = d.total_fees
        net = d.gross_pnl - all_fees
        mfr = d.maker_fills / d.maker_attempts * 100 if d.maker_attempts else 0.0
        avg_lat = sum(d.latencies) / len(d.latencies) if d.latencies else 0.0

        gross_s = f"{d.gross_pnl:>+10.2f}"
        net_s = f"{net:>+10.2f}"

        print(
            f"  {k:10}  {trades:>6}  {wr:>4.1f}%  {gross_s}  "
            f"{all_fees:>9.2f}  {net_s}  {d.maker_attempts:>7}  {d.maker_fills:>8}  {mfr:>4.1f}%  {avg_lat:>9.1f}"
        )

        total_trades += trades
        total_wins += d.wins
        total_gross += d.gross_pnl
        total_fees_all += all_fees
        total_net += net
        total_attempts += d.maker_attempts
        total_mk_fills += d.maker_fills
        all_latencies.extend(d.latencies)

    total_wr = total_wins / total_trades * 100 if total_trades else 0.0
    total_mfr = total_mk_fills / total_attempts * 100 if total_attempts else 0.0
    total_avg_lat = sum(all_latencies) / len(all_latencies) if all_latencies else 0.0

    print(f"  {'─'*10}  {'─'*6}  {'─'*5}  {'─'*10}  {'─'*9}  {'─'*10}  {'─'*7}  {'─'*8}  {'─'*5}  {'─'*9}")
    print(
        f"  {'TOTAL':10}  {total_trades:>6}  {total_wr:>4.1f}%  {total_gross:>+10.2f}  "
        f"{total_fees_all:>9.2f}  {total_net:>+10.2f}  {total_attempts:>7}  {total_mk_fills:>8}  {total_mfr:>4.1f}%  {total_avg_lat:>9.1f}"
    )
    print(f"╚{bar}╝\n")


def main() -> None:
    args = _parse_args()
    if not args.journal_dir.exists():
        print(f"Journal directory not found: {args.journal_dir}")
        sys.exit(1)
    events = _load_events(args.journal_dir)
    if not events:
        print(f"No journal files found in {args.journal_dir}")
        sys.exit(1)
    days = _build_days(events)
    _print_table(days, last_n=args.days, csv_mode=args.csv)


if __name__ == "__main__":
    main()
