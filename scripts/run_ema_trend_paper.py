#!/usr/bin/env python3
"""Paper-trade EMA 8/21 Long-Only trend strategy (mechanism #12) — replay or live.

REPLAY: replays cached daily candles, logs every signal and equity curve.
LIVE  : seeds from historical data, polls Bitget daily close once per hour,
        executes OPEN_LONG / CLOSE_LONG via BitgetPaperAdapter.

Validated parameters: EMA fast=8, slow=21, long-only, 12-coin universe.
Results: CAGR 24%, DD 11%, PF 1.95 (2021-2026).

Usage:
    python scripts/run_ema_trend_paper.py --mode replay
    python scripts/run_ema_trend_paper.py --mode replay --from 2024-01-01
    BITGET_PAPER=1 python scripts/run_ema_trend_paper.py --mode live
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

from nexflow.services.strategy.ema_trend_strategy import EMATrendStrategy, EMASignal

_DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT",
    "LINKUSDT", "LTCUSDT", "DOTUSDT", "TRXUSDT",
]
_CANDLE_DIR = _REPO_ROOT / "data" / "candles"
_DAY_MS     = 86_400_000


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def _load_daily(symbols: list[str], from_ts: int, to_ts: int) -> dict[str, list[tuple[int, float]]]:
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
        result[sym] = [(ts, float(c)) for ts, c in rows if from_ts <= ts <= to_ts]
    return result


def _fetch_daily_close(symbol: str) -> Optional[tuple[int, float]]:
    """Fetch most recent completed daily bar from Bitget REST."""
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
        if not rows:
            return None
        return int(rows[0][0]), float(rows[0][4])
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
    fast: int,
    slow: int,
) -> None:
    print("Loading daily candles ...")
    history = _load_daily(symbols, from_ts, to_ts)
    n_bars = sum(len(v) for v in history.values())
    print(f"  {n_bars:,} bars across {len(symbols)} symbols")
    print()

    strategy = EMATrendStrategy(symbols=symbols, fast=fast, slow=slow)
    notional = capital / len(symbols)

    # Build chronological event list
    events: list[tuple[int, str, float]] = []
    for sym, bars in history.items():
        for ts, c in bars:
            events.append((ts, sym, c))
    events.sort()

    equity = capital
    positions: dict[str, tuple[float, float]] = {}  # {sym: (entry_price, notional)}
    total_signals = 0

    for ts, sym, close in events:
        sigs = strategy.on_daily_close(sym, close, ts)
        for sig in sigs:
            total_signals += 1
            dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            if sig.action == "OPEN_LONG":
                positions[sym] = (close, notional)
                fee = 0.0006 * notional
                equity -= fee
                print(f"  {dt}  BUY  {sym:<12}  @ {close:>12,.2f}  notional=${notional:,.0f}")
            elif sig.action == "CLOSE_LONG":
                if sym in positions:
                    entry, n = positions.pop(sym)
                    pnl = (close - entry) / entry * n
                    fee = 0.0006 * n
                    net = pnl - fee
                    equity += net
                    pct = (close - entry) / entry * 100
                    print(f"  {dt}  SELL {sym:<12}  @ {close:>12,.2f}  pnl={pct:+.1f}%  net=${net:+,.0f}  equity=${equity:,.0f}")

    # Mark open positions to market
    print()
    print("Open positions (mark-to-market):")
    last_prices: dict[str, float] = {}
    for sym, bars in history.items():
        if bars:
            last_prices[sym] = bars[-1][1]

    for sym, (entry, n) in positions.items():
        price = last_prices.get(sym, entry)
        pnl = (price - entry) / entry * n
        pct = (price - entry) / entry * 100
        print(f"  {sym:<12}  entry={entry:,.2f}  now={price:,.2f}  unrealised={pct:+.1f}%  ${pnl:+,.0f}")

    unrealised = sum((last_prices.get(s, e) - e) / e * n for s, (e, n) in positions.items())
    total_equity = equity + unrealised
    net = total_equity - capital
    cagr_years = (to_ts - from_ts) / (1000 * 86400 * 365.25)
    cagr = (total_equity / capital) ** (1 / cagr_years) - 1 if cagr_years > 0 else 0.0

    print()
    print(f"Replay complete:")
    print(f"  Signals       : {total_signals}")
    print(f"  Open positions: {len(positions)}")
    print(f"  Realised equity: ${equity:,.0f}")
    print(f"  Total equity   : ${total_equity:,.0f}  (net ${net:+,.0f}, {net/capital*100:.1f}%)")
    print(f"  CAGR (period)  : {cagr*100:.1f}%")
    print()
    print("Current EMA alignment:")
    for sym, state in strategy.current_signals().items():
        print(f"  {sym:<12} {state}")


# ---------------------------------------------------------------------------
# Live mode
# ---------------------------------------------------------------------------
def run_live(
    symbols: list[str],
    capital: float,
    fast: int,
    slow: int,
) -> None:
    from nexflow.exchange.bitget_client import BitgetClient
    from nexflow.execution.adapter import BitgetPaperAdapter

    client  = BitgetClient.from_env()
    adapter = BitgetPaperAdapter(client)

    print("EMA Trend Strategy — LIVE PAPER MODE")
    print(f"Symbols  : {symbols}")
    print(f"Capital  : ${capital:,.0f}")
    print(f"EMA      : fast={fast}, slow={slow}, long-only")
    print()

    # Seed with enough history for EMA warmup
    today_ts   = int(datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
    seed_from  = today_ts - (slow + 10) * _DAY_MS
    history    = _load_daily(symbols, seed_from, today_ts)

    strategy = EMATrendStrategy(symbols=symbols, fast=fast, slow=slow)

    all_seed: list[tuple[int, str, float]] = []
    for sym, bars in history.items():
        for ts, c in bars:
            all_seed.append((ts, sym, c))
    all_seed.sort()

    print(f"Seeding {len(all_seed)} historical bars ...")
    for ts, sym, close in all_seed:
        strategy.on_daily_close(sym, close, ts)
    print("Seed complete.")
    print()
    print("Current EMA alignment:")
    for sym, state in strategy.current_signals().items():
        print(f"  {sym:<12} {state}")
    print()

    notional = capital / len(symbols)

    def _execute(sig: EMASignal) -> None:
        try:
            if sig.action == "OPEN_LONG":
                adapter.on_entry(sig.symbol, "long", notional / sig.price, 0.0, 0.0, 0.0)
            elif sig.action == "CLOSE_LONG":
                adapter.on_close(sig.symbol, "long", notional / sig.price, 0.0, "ema_cross")
            print(f"  [{sig.action}] {sig.symbol}  @ {sig.price:,.2f}  ({sig.reason})")
        except Exception as exc:
            print(f"  [ERROR] {sig.symbol} {sig.action}: {exc}")

    last_processed_ts: Optional[int] = None
    while True:
        now_ts = int(datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)

        if last_processed_ts is None or now_ts > last_processed_ts:
            ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{ts_str}] Fetching daily closes ...")
            for sym in symbols:
                result = _fetch_daily_close(sym)
                if result:
                    ts_ms, close = result
                    sigs = strategy.on_daily_close(sym, close, ts_ms)
                    for sig in sigs:
                        _execute(sig)
                time.sleep(0.2)
            last_processed_ts = now_ts
            print(f"  EMA state: {strategy.current_signals()}")

        print(f"  Positions: {[s for s, v in strategy.positions.items() if v]}")
        time.sleep(3600)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",    choices=["replay", "live"], default="replay")
    parser.add_argument("--symbols", nargs="+", default=_DEFAULT_SYMBOLS)
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--fast",    type=int,   default=8)
    parser.add_argument("--slow",    type=int,   default=21)
    parser.add_argument("--from",    dest="from_date", default="2021-01-01")
    parser.add_argument("--to",      dest="to_date",   default=None)
    args = parser.parse_args()

    from_dt = datetime.strptime(args.from_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    to_dt   = (
        datetime.strptime(args.to_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.to_date else datetime.now(timezone.utc)
    )
    from_ts = int(from_dt.timestamp() * 1000)
    to_ts   = int(to_dt.timestamp() * 1000)

    if args.mode == "replay":
        run_replay(args.symbols, args.capital, from_ts, to_ts, args.fast, args.slow)
    else:
        run_live(args.symbols, args.capital, args.fast, args.slow)


if __name__ == "__main__":
    main()
