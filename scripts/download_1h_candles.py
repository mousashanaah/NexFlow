#!/usr/bin/env python3
"""Download 1H candles for BTCUSDT and ETHUSDT from Bitget.

Uses /api/v2/mix/market/history-candles with backward pagination,
same proven pattern as download_daily_candles.py.

Saves to data/candles/{SYMBOL}_1H.parquet.

Usage:
  python scripts/download_1h_candles.py
  python scripts/download_1h_candles.py --start 2021-01-01
  python scripts/download_1h_candles.py --symbols BTCUSDT ETHUSDT
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

_URL_HISTORY     = "https://api.bitget.com/api/v2/mix/market/history-candles"
_LIMIT           = 200
_DELAY_S         = 0.15
_HOUR_MS         = 3_600_000
_DEFAULT_START   = "2021-01-01"
_DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
_HEADERS         = {"User-Agent": "NexFlow/1.0", "Accept": "application/json"}

_SCHEMA = pa.schema([
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


def _fetch_page(symbol: str, end_ms: int) -> list:
    url = (
        f"{_URL_HISTORY}?symbol={symbol}&productType=USDT-FUTURES"
        f"&granularity=1H&endTime={end_ms}&limit={_LIMIT}"
    )
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read())
    if data.get("code") != "00000":
        raise RuntimeError(f"API error: {data.get('msg', data)}")
    return data.get("data", [])


def _download(symbol: str, start_ms: int, end_ms: int) -> list[dict]:
    bars: list[dict] = []
    seen: set[int]   = set()
    cursor_end = end_ms
    errors = 0

    while cursor_end > start_ms:
        try:
            rows = _fetch_page(symbol, cursor_end)
        except Exception as exc:
            errors += 1
            if errors >= 5:
                print(f"\n  [ERROR] Too many failures: {exc}")
                break
            print(f"\n  [WARN] Retrying: {exc}")
            time.sleep(2.0 * errors)
            continue
        errors = 0

        if not rows:
            break

        oldest = cursor_end
        new = 0
        for row in rows:
            ts = int(row[0])
            if ts in seen or ts > end_ms:
                continue
            seen.add(ts)
            new += 1
            oldest = min(oldest, ts)
            if ts >= start_ms:
                bars.append({
                    "open_time":  ts,
                    "close_time": ts + _HOUR_MS - 1,
                    "open":   float(row[1]),
                    "high":   float(row[2]),
                    "low":    float(row[3]),
                    "close":  float(row[4]),
                    "volume": float(row[5]) if len(row) > 5 else 0.0,
                })

        total = len(bars)
        pct   = (end_ms - oldest) / max(end_ms - start_ms, 1) * 100
        print(f"  {symbol}: {total:,} bars  ({pct:.0f}% complete) ...", end="\r")

        if new == 0 or oldest <= start_ms:
            break
        cursor_end = oldest - 1
        time.sleep(_DELAY_S)

    print()
    bars = [b for b in bars if b["open_time"] >= start_ms]
    return sorted(bars, key=lambda b: b["open_time"])


def _save(symbol: str, bars: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{symbol}_1H.parquet"

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
        "symbol":     [symbol] * len(bars),
        "timeframe":  ["1H"]   * len(bars),
        "open_time":  [b["open_time"]  for b in bars],
        "close_time": [b["close_time"] for b in bars],
        "open":       [b["open"]       for b in bars],
        "high":       [b["high"]       for b in bars],
        "low":        [b["low"]        for b in bars],
        "close":      [b["close"]      for b in bars],
        "volume":     [b["volume"]     for b in bars],
    }, schema=_SCHEMA)
    pq.write_table(table, path)
    return path


def _cached_start_ms(symbol: str, out_dir: Path, default_start_ms: int) -> int:
    """Return the timestamp just after the last cached bar, or default_start_ms."""
    path = out_dir / f"{symbol}_1H.parquet"
    if not path.exists():
        return default_start_ms
    tbl = pq.read_table(path, columns=["open_time"])
    if tbl.num_rows == 0:
        return default_start_ms
    last_ts = max(tbl.column("open_time").to_pylist())
    return last_ts + _HOUR_MS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=_DEFAULT_SYMBOLS)
    parser.add_argument("--start",   default=_DEFAULT_START)
    parser.add_argument("--end",     default=None)
    parser.add_argument("--out",     default="data/candles")
    args = parser.parse_args()

    default_start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt   = (
        datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.end else datetime.now(timezone.utc)
    )
    default_start_ms = int(default_start_dt.timestamp() * 1000)
    end_ms   = int(end_dt.timestamp() * 1000)
    out_dir  = _REPO_ROOT / args.out

    print(f"Downloading 1H candles (incremental): up to {args.end or 'today'}")
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
        bars = _download(symbol, start_ms, end_ms)
        if not bars:
            print(f"  No new data for {symbol}")
            continue
        path = _save(symbol, bars, out_dir)
        s = datetime.fromtimestamp(bars[0]["open_time"]  / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        e = datetime.fromtimestamp(bars[-1]["open_time"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        print(f"  Saved {len(bars):,} bars  ({s} → {e})  → {path}")


if __name__ == "__main__":
    main()
