#!/usr/bin/env python3
"""Download daily (1D) candles for BTCUSDT and ETHUSDT from Bitget.

Uses /api/v2/mix/market/history-candles with backward pagination
(endTime → startTime), same proven pattern as download_candles.py.
Saves to data/candles/{SYMBOL}_1D.parquet.

Usage:
  python scripts/download_daily_candles.py
  python scripts/download_daily_candles.py --start 2021-01-01
  python scripts/download_daily_candles.py --symbols BTCUSDT ETHUSDT
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
    print("[ERROR] pyarrow is required: pip install pyarrow")
    sys.exit(1)

_URL_HISTORY = "https://api.bitget.com/api/v2/mix/market/history-candles"
_LIMIT = 200
_DELAY_S = 0.15
_DAY_MS = 86_400_000
_DEFAULT_START = "2021-01-01"
_DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
_HEADERS = {"User-Agent": "NexFlow/1.0", "Accept": "application/json"}

_SCHEMA = pa.schema([
    pa.field("symbol",    pa.string()),
    pa.field("timeframe", pa.string()),
    pa.field("open_time", pa.int64()),
    pa.field("close_time", pa.int64()),
    pa.field("open",      pa.float64()),
    pa.field("high",      pa.float64()),
    pa.field("low",       pa.float64()),
    pa.field("close",     pa.float64()),
    pa.field("volume",    pa.float64()),
])


def _fetch_backward(symbol: str, end_ms: int) -> list:
    """Fetch up to 200 daily bars ending at end_ms (backward pagination)."""
    url = (
        f"{_URL_HISTORY}?symbol={symbol}&productType=USDT-FUTURES"
        f"&granularity=1D&endTime={end_ms}&limit={_LIMIT}"
    )
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read())
    if data.get("code") != "00000":
        raise RuntimeError(f"API error: {data.get('msg', data)}")
    return data.get("data", [])


def _download_symbol(symbol: str, start_ms: int, end_ms: int) -> list[dict]:
    """Paginate backward from end_ms to start_ms, collecting all daily bars."""
    bars: list[dict] = []
    seen: set[int] = set()
    cursor_end = end_ms
    errors = 0

    while cursor_end > start_ms:
        try:
            rows = _fetch_backward(symbol, cursor_end)
        except Exception as exc:
            errors += 1
            if errors >= 5:
                print(f"\n  [ERROR] Too many failures: {exc}")
                break
            print(f"\n  [WARN] Retrying after error: {exc}")
            time.sleep(2.0 * errors)
            continue
        errors = 0

        if not rows:
            break

        oldest_in_page = cursor_end
        new_count = 0
        for row in rows:
            ts = int(row[0])
            if ts in seen or ts > end_ms:
                continue
            seen.add(ts)
            new_count += 1
            oldest_in_page = min(oldest_in_page, ts)
            if ts < start_ms:
                continue  # collect but don't advance past start
            bars.append({
                "open_time":  ts,
                "close_time": ts + _DAY_MS - 1,
                "open":   float(row[1]),
                "high":   float(row[2]),
                "low":    float(row[3]),
                "close":  float(row[4]),
                "volume": float(row[5]),
            })

        print(f"  {symbol}: {len(bars)} bars collected ...", end="\r")

        if new_count == 0 or oldest_in_page <= start_ms:
            break

        cursor_end = oldest_in_page - 1
        time.sleep(_DELAY_S)

    print()
    # Filter to requested range and sort ascending
    bars = [b for b in bars if b["open_time"] >= start_ms]
    return sorted(bars, key=lambda b: b["open_time"])


def _save(symbol: str, bars: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{symbol}_1D.parquet"

    if path.exists():
        existing = pq.read_table(path).to_pydict()
        existing_ts: set[int] = set(existing["open_time"])
        cached = [
            {
                "open_time":  existing["open_time"][i],
                "close_time": existing["close_time"][i],
                "open":       existing["open"][i],
                "high":       existing["high"][i],
                "low":        existing["low"][i],
                "close":      existing["close"][i],
                "volume":     existing["volume"][i],
            }
            for i in range(len(existing["open_time"]))
        ]
        new_bars = [b for b in bars if b["open_time"] not in existing_ts]
        before = len(cached)
        bars = sorted(cached + new_bars, key=lambda b: b["open_time"])
        print(f"  Merged: {before:,} cached + {len(new_bars):,} new = {len(bars):,} total")

    table = pa.table({
        "symbol":    [symbol] * len(bars),
        "timeframe": ["1D"] * len(bars),
        "open_time": [b["open_time"] for b in bars],
        "close_time": [b["close_time"] for b in bars],
        "open":      [b["open"] for b in bars],
        "high":      [b["high"] for b in bars],
        "low":       [b["low"] for b in bars],
        "close":     [b["close"] for b in bars],
        "volume":    [b["volume"] for b in bars],
    }, schema=_SCHEMA)

    pq.write_table(table, path)
    return path


def _cached_start_ms(symbol: str, out_dir: Path, default_start_ms: int) -> int:
    """Return the timestamp just after the last cached bar, or default_start_ms."""
    path = out_dir / f"{symbol}_1D.parquet"
    if not path.exists():
        return default_start_ms
    tbl = pq.read_table(path, columns=["open_time"])
    if tbl.num_rows == 0:
        return default_start_ms
    last_ts = max(tbl.column("open_time").to_pylist())
    return last_ts + _DAY_MS


def main() -> None:
    parser = argparse.ArgumentParser(description="Download daily candles from Bitget")
    parser.add_argument("--symbols", nargs="+", default=_DEFAULT_SYMBOLS)
    parser.add_argument("--start", default=_DEFAULT_START, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--out", default="data/candles", help="Output directory")
    args = parser.parse_args()

    default_start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = (
        datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.end
        else datetime.now(timezone.utc)
    )
    default_start_ms = int(default_start_dt.timestamp() * 1000)
    end_ms   = int(end_dt.timestamp() * 1000)

    out_dir = _REPO_ROOT / args.out

    print(f"Downloading daily candles (incremental): up to {args.end or 'today'}")
    print(f"Symbols: {args.symbols}")
    print()

    for symbol in args.symbols:
        print(f"[{symbol}]")
        start_ms = _cached_start_ms(symbol, out_dir, default_start_ms)
        if start_ms >= end_ms:
            print(f"  Already up to date")
            continue
        start_str = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        print(f"  Fetching from {start_str} ...")
        bars = _download_symbol(symbol, start_ms, end_ms)
        if not bars:
            print(f"  No new data for {symbol}")
            continue
        path = _save(symbol, bars, out_dir)
        span_start = datetime.fromtimestamp(bars[0]["open_time"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        span_end   = datetime.fromtimestamp(bars[-1]["open_time"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        print(f"  Saved {len(bars)} bars ({span_start} → {span_end}) → {path}")


if __name__ == "__main__":
    main()
