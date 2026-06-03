#!/usr/bin/env python3
"""Paper-trade cross-sectional momentum (mechanism #8) — live or replay mode.

REPLAY mode: replays cached daily candles, outputs per-rebalance log.
LIVE mode:   fetches current daily close at startup, schedules rebalances,
             places orders via Bitget Demo Trading (BITGET_PAPER adapter).

Usage:
    # Replay full history
    python scripts/run_cross_momentum_paper.py --mode replay

    # Replay from 2024 only
    python scripts/run_cross_momentum_paper.py --mode replay --from 2024-01-01

    # Live paper trading (requires BITGET_API_KEY, BITGET_API_SECRET, BITGET_PASSPHRASE)
    BITGET_PAPER=1 python scripts/run_cross_momentum_paper.py --mode live

Validated parameters: lookback=20 days, rebal=7 days, top_n=3, 12-coin universe.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    import pyarrow.parquet as pq
except ImportError:
    print("[ERROR] pyarrow required: pip install pyarrow")
    sys.exit(1)

from nexflow.services.strategy.cross_momentum_rebalancer import (
    CrossMomentumRebalancer, RebalanceOrder,
)

_DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT",
    "LINKUSDT", "LTCUSDT", "DOTUSDT", "TRXUSDT",
]
_CANDLE_DIR = _REPO_ROOT / "data" / "candles"
_DAY_MS = 86_400_000


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_daily_history(
    symbols: list[str],
    from_ts: int,
    to_ts: int,
) -> dict[str, list[tuple[int, float]]]:
    """Load cached daily parquet. Returns {symbol: [(ts_ms, close), ...]}."""
    result: dict[str, list[tuple[int, float]]] = {}
    for sym in symbols:
        path = _CANDLE_DIR / f"{sym}_1D.parquet"
        if not path.exists():
            print(f"  [WARN] No daily data for {sym}")
            result[sym] = []
            continue
        tbl = pq.read_table(path, columns=["open_time", "close"])
        rows = sorted(zip(
            tbl.column("open_time").to_pylist(),
            tbl.column("close").to_pylist(),
        ))
        result[sym] = [(ts, c) for ts, c in rows if from_ts <= ts <= to_ts]
    return result


def _fetch_live_daily_close(symbol: str) -> Optional[tuple[int, float]]:
    """Fetch the most recent daily close from Bitget REST API."""
    url = (
        f"https://api.bitget.com/api/v2/mix/market/history-candles"
        f"?symbol={symbol}&productType=USDT-FUTURES&granularity=1D&limit=2"
    )
    headers = {"User-Agent": "NexFlow/1.0", "Accept": "application/json"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        if data.get("code") != "00000":
            return None
        rows = data.get("data", [])
        # rows[0] = most recent completed daily bar (newest first)
        if not rows:
            return None
        ts_ms  = int(rows[0][0])
        close  = float(rows[0][4])
        return ts_ms, close
    except Exception as exc:
        print(f"  [WARN] Fetch failed for {symbol}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Replay mode
# ---------------------------------------------------------------------------
def run_replay(
    symbols: list[str],
    capital: float,
    from_ts: int,
    to_ts: int,
    lookback: int,
    rebal_days: int,
    top_n: int,
) -> None:
    print("Loading daily candles ...")
    history = _load_daily_history(symbols, from_ts, to_ts)
    n_bars = sum(len(v) for v in history.values())
    print(f"  {n_bars:,} bars across {len(symbols)} symbols")
    print()

    rebalancer = CrossMomentumRebalancer(
        symbols=symbols, capital=capital,
        lookback=lookback, rebal_days=rebal_days, top_n=top_n,
    )

    # Simulate: feed bars day by day in chronological order
    all_events: list[tuple[int, str, float]] = []
    for sym, bars in history.items():
        for ts, c in bars:
            all_events.append((ts, sym, c))
    all_events.sort()

    total_orders = 0
    for ts, sym, close in all_events:
        orders = rebalancer.on_daily_close(sym, close, ts)
        total_orders += len(orders)

    print()
    print(f"Replay complete: {total_orders} total orders")
    print(f"Final positions : {rebalancer.positions}")


# ---------------------------------------------------------------------------
# Live mode (paper trading via Bitget Demo)
# ---------------------------------------------------------------------------
def run_live(
    symbols: list[str],
    capital: float,
    lookback: int,
    rebal_days: int,
    top_n: int,
) -> None:
    from nexflow.exchange.bitget_client import BitgetClient
    from nexflow.execution.adapter import BitgetPaperAdapter

    client = BitgetClient.from_env()
    adapter = BitgetPaperAdapter(client)

    print("Cross-Sectional Momentum — LIVE PAPER MODE")
    print(f"Symbols  : {symbols}")
    print(f"Capital  : ${capital:,.0f}")
    print(f"Params   : lookback={lookback}d, rebal={rebal_days}d, top_n={top_n}")
    print()

    # Seed with historical data so rebalancer has a full lookback window
    today_ts = int(datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
    seed_from = today_ts - (lookback + 5) * _DAY_MS
    history = _load_daily_history(symbols, seed_from, today_ts)

    rebalancer = CrossMomentumRebalancer(
        symbols=symbols, capital=capital,
        lookback=lookback, rebal_days=rebal_days, top_n=top_n,
    )

    # Seed the rebalancer (silent — no orders on historical data)
    all_seed: list[tuple[int, str, float]] = []
    for sym, bars in history.items():
        for ts, c in bars:
            all_seed.append((ts, sym, c))
    all_seed.sort()
    print(f"Seeding {len(all_seed)} historical bars ...")
    for ts, sym, close in all_seed:
        rebalancer.on_daily_close(sym, close, ts)
    print("Seed complete. Monitoring for daily close ...")
    print()

    def _execute_orders(orders: list[RebalanceOrder]) -> None:
        for order in orders:
            try:
                if order.action == "OPEN_LONG":
                    adapter.on_entry(order.symbol, "long", order.notional / 100,
                                     0.0, 0.0, 0.0)  # notional-based sizing TODO
                elif order.action == "OPEN_SHORT":
                    adapter.on_entry(order.symbol, "short", order.notional / 100,
                                     0.0, 0.0, 0.0)
                elif order.action in ("CLOSE_LONG", "CLOSE_SHORT"):
                    adapter.on_close(order.symbol,
                                     "long" if order.action == "CLOSE_LONG" else "short",
                                     order.notional / 100, 0.0, "rebalance")
                print(f"  [{order.action}] {order.symbol} ${order.notional:,.0f}: {order.reason}")
            except Exception as exc:
                print(f"  [ERROR] {order.symbol} {order.action}: {exc}")

    # Main loop: fetch daily close once per hour, process at day boundary
    last_processed_ts: Optional[int] = None
    while True:
        now_ts = int(datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)

        if last_processed_ts is None or now_ts > last_processed_ts:
            print(f"\n[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] Fetching daily closes ...")
            for sym in symbols:
                result = _fetch_live_daily_close(sym)
                if result:
                    ts_ms, close = result
                    orders = rebalancer.on_daily_close(sym, close, ts_ms)
                    if orders:
                        _execute_orders(orders)
                time.sleep(0.2)  # rate limit
            last_processed_ts = now_ts

        # Sleep until next check (~1 hour)
        print(f"  Current positions: {rebalancer.positions}")
        time.sleep(3600)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["replay", "live"], default="replay")
    parser.add_argument("--symbols", nargs="+", default=_DEFAULT_SYMBOLS)
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--from", dest="from_date", default="2021-01-01")
    parser.add_argument("--to",   dest="to_date",   default=None)
    parser.add_argument("--lookback",  type=int, default=20)
    parser.add_argument("--rebal",     type=int, default=7)
    parser.add_argument("--top",       type=int, default=3)
    args = parser.parse_args()

    from_dt = datetime.strptime(args.from_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    to_dt   = (
        datetime.strptime(args.to_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.to_date else datetime.now(timezone.utc)
    )
    from_ts = int(from_dt.timestamp() * 1000)
    to_ts   = int(to_dt.timestamp() * 1000)

    if args.mode == "replay":
        run_replay(args.symbols, args.capital, from_ts, to_ts,
                   args.lookback, args.rebal, args.top)
    else:
        run_live(args.symbols, args.capital,
                 args.lookback, args.rebal, args.top)


if __name__ == "__main__":
    main()
