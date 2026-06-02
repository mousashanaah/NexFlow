#!/usr/bin/env python3
"""Download 8H funding rate history for BTCUSDT and ETHUSDT from Bitget.

Uses /api/v2/mix/market/history-fund-rate with page-based forward pagination.
Saves to data/funding/{SYMBOL}_funding.parquet.

Usage:
  python scripts/download_funding_rates.py
  python scripts/download_funding_rates.py --start 2021-01-01
  python scripts/download_funding_rates.py --symbols BTCUSDT ETHUSDT
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

_URL_FUNDING     = "https://api.bitget.com/api/v2/mix/market/history-fund-rate"
_PAGE_SIZE       = 100
_DELAY_S         = 0.2
_DEFAULT_START   = "2021-01-01"
_DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
_HEADERS         = {"User-Agent": "NexFlow/1.0", "Accept": "application/json"}

_SCHEMA = pa.schema([
    pa.field("symbol",       pa.string()),
    pa.field("timestamp_ms", pa.int64()),
    pa.field("funding_rate", pa.float64()),
])


def _fetch_page(symbol: str, page_no: int) -> list:
    url = (
        f"{_URL_FUNDING}?symbol={symbol}&productType=USDT-FUTURES"
        f"&pageSize={_PAGE_SIZE}&pageNo={page_no}"
    )
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read())
    if data.get("code") != "00000":
        raise RuntimeError(f"API error: {data.get('msg', data)}")
    return data.get("data", [])


def _download(symbol: str, start_ms: int) -> list[dict]:
    records: list[dict] = []
    page_no = 1
    errors = 0

    while True:
        try:
            rows = _fetch_page(symbol, page_no)
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

        for row in rows:
            ts = int(row["fundingTime"])
            if ts < start_ms:
                continue
            records.append({
                "timestamp_ms": ts,
                "funding_rate": float(row["fundingRate"]),
            })

        print(f"  {symbol}: {len(records):,} records (page {page_no}) ...", end="\r")

        if len(rows) < _PAGE_SIZE:
            break

        page_no += 1
        time.sleep(_DELAY_S)

    print()
    return sorted(records, key=lambda r: r["timestamp_ms"])


def _save(symbol: str, records: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{symbol}_funding.parquet"
    table = pa.table({
        "symbol":       [symbol] * len(records),
        "timestamp_ms": [r["timestamp_ms"] for r in records],
        "funding_rate": [r["funding_rate"] for r in records],
    }, schema=_SCHEMA)
    pq.write_table(table, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=_DEFAULT_SYMBOLS)
    parser.add_argument("--start",   default=_DEFAULT_START)
    parser.add_argument("--out",     default="data/funding")
    args = parser.parse_args()

    start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    start_ms = int(start_dt.timestamp() * 1000)
    out_dir  = _REPO_ROOT / args.out

    print(f"Downloading funding rates from {args.start}")
    print(f"Symbols: {args.symbols}")
    print()

    for symbol in args.symbols:
        print(f"[{symbol}]")
        records = _download(symbol, start_ms)
        if not records:
            print(f"  No data for {symbol}")
            continue
        path = _save(symbol, records, out_dir)
        s = datetime.fromtimestamp(records[0]["timestamp_ms"]  / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        e = datetime.fromtimestamp(records[-1]["timestamp_ms"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        print(f"  Saved {len(records):,} records  ({s} → {e})  → {path}")


if __name__ == "__main__":
    main()
