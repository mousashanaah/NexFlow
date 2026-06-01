#!/usr/bin/env python3
"""Download daily (1D) candles for BTCUSDT and ETHUSDT from Bitget.

Uses /api/v2/mix/market/candles with granularity=1D, forward-paginating
from start_date to today. Saves to data/candles/{SYMBOL}_1D.parquet.

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

_URL = "https://api.bitget.com/api/v2/mix/market/candles"
_URL_HISTORY = "https://api.bitget.com/api/v2/mix/market/history-candles"
_LIMIT = 200
_DELAY_S = 0.15
_DEFAULT_START = "2021-01-01"
_DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]

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


def _fetch_page(symbol: str, start_ms: int, end_ms: int, use_history: bool = False) -> list:
    url_base = _URL_HISTORY if use_history else _URL
    if use_history:
        url = (
            f"{url_base}?symbol={symbol}&productType=USDT-FUTURES"
            f"&granularity=1D&endTime={end_ms}&limit={_LIMIT}"
        )
    else:
        url = (
            f"{url_base}?symbol={symbol}&productType=USDT-FUTURES"
            f"&granularity=1D&startTime={start_ms}&endTime={end_ms}&limit={_LIMIT}"
        )
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read())
        if data.get("code") != "00000":
            print(f"  [WARN] API error: {data.get('msg')}")
            return []
        return data.get("data", [])
    except Exception as e:
        print(f"  [WARN] Request failed: {e}")
        return []


def _download_symbol(symbol: str, start_ms: int, end_ms: int) -> list[dict]:
    bars = []
    seen = set()
    cursor_start = start_ms

    while cursor_start < end_ms:
        page = _fetch_page(symbol, cursor_start, end_ms, use_history=False)
        if not page:
            # Fall back to history endpoint
            page = _fetch_page(symbol, cursor_start, end_ms, use_history=True)
        if not page:
            break

        new_count = 0
        last_ts = cursor_start
        for row in page:
            ts = int(row[0])
            if ts in seen or ts < start_ms or ts > end_ms:
                continue
            seen.add(ts)
            new_count += 1
            last_ts = max(last_ts, ts)
            bars.append({
                "open_time":  ts,
                "close_time": ts + 86_400_000 - 1,  # daily bar: open_time + 24h - 1ms
                "open":   float(row[1]),
                "high":   float(row[2]),
                "low":    float(row[3]),
                "close":  float(row[4]),
                "volume": float(row[5]),
            })

        print(f"  {symbol}: fetched {len(bars)} bars total", end="\r")

        if new_count == 0:
            break
        cursor_start = last_ts + 86_400_000  # advance one day
        time.sleep(_DELAY_S)

    print()
    return sorted(bars, key=lambda b: b["open_time"])


def _save(symbol: str, bars: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{symbol}_1D.parquet"

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Download daily candles from Bitget")
    parser.add_argument("--symbols", nargs="+", default=_DEFAULT_SYMBOLS)
    parser.add_argument("--start", default=_DEFAULT_START, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--out", default="data/candles", help="Output directory")
    args = parser.parse_args()

    start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = (
        datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.end
        else datetime.now(timezone.utc)
    )
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms   = int(end_dt.timestamp() * 1000)

    out_dir = _REPO_ROOT / args.out

    print(f"Downloading daily candles: {args.start} → {args.end or 'today'}")
    print(f"Symbols: {args.symbols}")
    print()

    for symbol in args.symbols:
        print(f"[{symbol}]")
        bars = _download_symbol(symbol, start_ms, end_ms)
        if not bars:
            print(f"  No data returned for {symbol}")
            continue
        path = _save(symbol, bars, out_dir)
        span_start = datetime.fromtimestamp(bars[0]["open_time"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        span_end   = datetime.fromtimestamp(bars[-1]["open_time"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        print(f"  Saved {len(bars)} bars ({span_start} → {span_end}) → {path}")


if __name__ == "__main__":
    main()
