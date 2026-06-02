#!/usr/bin/env python3
"""Daily incremental data refresh for NexFlow.

Runs in sequence:
  1. Incremental daily candle update (last 5 bars, append new)
  2. Incremental 1H candle update (last 50 bars, append new)
  3. Current OI snapshot (append row)
  4. Write data/refresh_log.json with run metadata

Designed to run at 01:00 UTC daily via cron or Task Scheduler.
Typical runtime: under 60 seconds.

Usage:
  python scripts/daily_refresh.py
  python scripts/daily_refresh.py --symbols BTCUSDT ETHUSDT
  python scripts/daily_refresh.py --data-dir /path/to/data
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("[ERROR] pyarrow required: pip install pyarrow")
    sys.exit(1)

_HEADERS         = {"User-Agent": "NexFlow/1.0", "Accept": "application/json"}
_DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
_DELAY_S         = 0.2

_URL_CANDLES = "https://api.bitget.com/api/v2/mix/market/history-candles"
_URL_OI      = "https://api.bitget.com/api/v2/mix/market/open-interest"

_CANDLE_SCHEMA = pa.schema([
    pa.field("symbol",     pa.string()),
    pa.field("timeframe",  pa.string()),
    pa.field("open_time",  pa.int64()),
    pa.field("close_time", pa.int64()),
    pa.field("open",       pa.float64()),
    pa.field("high",       pa.float64()),
    pa.field("low",        pa.float64()),
    pa.field("close",      pa.float64()),
    pa.field("volume",     pa.float64()),
])

_OI_SCHEMA = pa.schema([
    pa.field("symbol",        pa.string()),
    pa.field("timestamp_ms",  pa.int64()),
    pa.field("open_interest", pa.float64()),
])


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read())
    if data.get("code") != "00000":
        raise RuntimeError(f"API error: {data.get('msg', data)}")
    return data


def _fetch_candles(symbol: str, granularity: str, end_ms: int, limit: int) -> list:
    url = (
        f"{_URL_CANDLES}?symbol={symbol}&productType=USDT-FUTURES"
        f"&granularity={granularity}&endTime={end_ms}&limit={limit}"
    )
    return _get(url).get("data", [])


def _update_candles(
    symbol: str,
    granularity: str,
    bar_ms: int,
    fetch_count: int,
    path: Path,
) -> tuple[int, int]:
    """Fetch recent bars and append any not already in parquet. Returns (existing, added)."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    existing_ts: set[int] = set()
    existing_table: pa.Table | None = None
    if path.exists():
        existing_table = pq.read_table(path)
        existing_ts = set(existing_table.column("open_time").to_pylist())

    rows = _fetch_candles(symbol, granularity, now_ms, fetch_count)
    new_bars = []
    for row in rows:
        ts = int(row[0])
        if ts in existing_ts:
            continue
        new_bars.append({
            "open_time":  ts,
            "close_time": ts + bar_ms - 1,
            "open":   float(row[1]),
            "high":   float(row[2]),
            "low":    float(row[3]),
            "close":  float(row[4]),
            "volume": float(row[5]) if len(row) > 5 else 0.0,
        })

    if not new_bars:
        return len(existing_ts), 0

    tf = "1D" if granularity == "1D" else "1H"
    new_table = pa.table({
        "symbol":     [symbol] * len(new_bars),
        "timeframe":  [tf]     * len(new_bars),
        "open_time":  [b["open_time"]  for b in new_bars],
        "close_time": [b["close_time"] for b in new_bars],
        "open":       [b["open"]       for b in new_bars],
        "high":       [b["high"]       for b in new_bars],
        "low":        [b["low"]        for b in new_bars],
        "close":      [b["close"]      for b in new_bars],
        "volume":     [b["volume"]     for b in new_bars],
    }, schema=_CANDLE_SCHEMA)

    if existing_table is not None:
        combined = pa.concat_tables([existing_table, new_table])
    else:
        combined = new_table

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(combined, path)
    return len(existing_ts), len(new_bars)


def _update_oi(symbol: str, path: Path) -> bool:
    url = f"{_URL_OI}?symbol={symbol}&productType=USDT-FUTURES"
    data = _get(url)["data"]

    oi_list = data.get("openInterestList", [])
    size_val = 0.0
    for entry in oi_list:
        if entry.get("currency") not in ("USDT",):
            size_val = float(entry.get("size", 0))
            break
    if not size_val and oi_list:
        size_val = float(oi_list[0].get("size", 0))

    ts_ms = int(data["ts"])

    new_row = pa.table({
        "symbol":        [symbol],
        "timestamp_ms":  [ts_ms],
        "open_interest": [size_val],
    }, schema=_OI_SCHEMA)

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pq.read_table(path)
        if ts_ms in set(existing.column("timestamp_ms").to_pylist()):
            return False
        pq.write_table(pa.concat_tables([existing, new_row]), path)
    else:
        pq.write_table(new_row, path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols",  nargs="+", default=_DEFAULT_SYMBOLS)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    data_dir = _REPO_ROOT / args.data_dir
    run_start = datetime.now(timezone.utc)
    errors: list[str] = []
    symbols_updated: list[str] = []

    print(f"[daily_refresh] {run_start.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Symbols: {args.symbols}")
    print()

    for symbol in args.symbols:
        print(f"[{symbol}] daily candles ...")
        try:
            existing, added = _update_candles(
                symbol, "1D", 86_400_000,
                5,
                data_dir / "candles" / f"{symbol}_1D.parquet",
            )
            print(f"  1D: {existing} existing, +{added} new")
        except Exception as exc:
            msg = f"{symbol} 1D candles: {exc}"
            print(f"  [ERROR] {msg}")
            errors.append(msg)
        time.sleep(_DELAY_S)

        print(f"[{symbol}] 1H candles ...")
        try:
            existing, added = _update_candles(
                symbol, "1H", 3_600_000,
                50,
                data_dir / "candles" / f"{symbol}_1H.parquet",
            )
            print(f"  1H: {existing} existing, +{added} new")
        except Exception as exc:
            msg = f"{symbol} 1H candles: {exc}"
            print(f"  [ERROR] {msg}")
            errors.append(msg)
        time.sleep(_DELAY_S)

        print(f"[{symbol}] open interest ...")
        try:
            appended = _update_oi(symbol, data_dir / "oi" / f"{symbol}_OI.parquet")
            print(f"  OI: {'appended new snapshot' if appended else 'already up to date'}")
        except Exception as exc:
            msg = f"{symbol} OI: {exc}"
            print(f"  [ERROR] {msg}")
            errors.append(msg)
        time.sleep(_DELAY_S)

        if not any(symbol in e for e in errors):
            symbols_updated.append(symbol)

        print()

    log_path = data_dir / "refresh_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = {
        "last_run":        run_start.isoformat(),
        "symbols_updated": symbols_updated,
        "errors":          errors,
    }
    log_path.write_text(json.dumps(log, indent=2))
    print(f"Refresh log → {log_path}")

    elapsed = (datetime.now(timezone.utc) - run_start).total_seconds()
    print(f"Done in {elapsed:.1f}s  |  {len(symbols_updated)}/{len(args.symbols)} symbols OK"
          + (f"  |  {len(errors)} error(s)" if errors else ""))

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
