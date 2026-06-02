#!/usr/bin/env python3
"""Download full funding-rate history from Binance USDT-Perpetuals.

Binance fapi provides complete history back to the instrument's launch date
(BTCUSDT since 2019-09-10, ETHUSDT since 2020-08-14). No API key required.

The resulting parquet is committed to the repo so CI/CD runners can load it
without any network calls. Re-run locally whenever you want to extend the cache.

Saves to data/funding/{SYMBOL}_funding.parquet

Usage:
  python scripts/download_funding_rates.py
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

_BASE_URL        = "https://fapi.binance.com/fapi/v1/fundingRate"
_PAGE_SIZE       = 1000       # Binance max per request
_DELAY_S         = 0.2        # polite delay between pages
_DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]

# Earliest funding records on Binance (approximate safe start)
_SYMBOL_START_MS: dict[str, int] = {
    "BTCUSDT": 1568592000000,   # 2019-09-16
    "ETHUSDT": 1597708800000,   # 2020-08-18
    "SOLUSDT": 1623024000000,   # 2021-06-07
    "BNBUSDT":  1597708800000,
}
_DEFAULT_START_MS = 1609459200000   # 2021-01-01 fallback

_HEADERS = {
    "User-Agent": "NexFlow/1.0",
    "Accept":     "application/json",
}

_SCHEMA = pa.schema([
    pa.field("symbol",       pa.string()),
    pa.field("timestamp_ms", pa.int64()),
    pa.field("funding_rate", pa.float64()),
    pa.field("exchange",     pa.string()),
])


def _fetch_page(symbol: str, start_ms: int) -> list[dict]:
    """Fetch one page of up to _PAGE_SIZE funding records starting at start_ms."""
    url = (
        f"{_BASE_URL}?symbol={symbol}"
        f"&startTime={start_ms}"
        f"&limit={_PAGE_SIZE}"
    )
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _download_all(symbol: str) -> list[dict]:
    """Paginate Binance fapi funding history for symbol, return all records."""
    start_ms = _SYMBOL_START_MS.get(symbol, _DEFAULT_START_MS)
    records: list[dict] = []

    while True:
        try:
            page = _fetch_page(symbol, start_ms)
        except urllib.error.HTTPError as exc:
            if exc.code == 451:
                print(
                    f"\n[ERROR] Binance returned HTTP 451 (geo-blocked).\n"
                    f"        You are likely running this on a US cloud server.\n"
                    f"        Run this script on your local machine and commit\n"
                    f"        the resulting data/funding/{symbol}_funding.parquet."
                )
                sys.exit(1)
            raise

        if not page:
            break

        for row in page:
            records.append({
                "timestamp_ms": int(row["fundingTime"]),
                "funding_rate": float(row["fundingRate"]),
            })

        last_ts = records[-1]["timestamp_ms"]
        pct_str = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        print(f"  {symbol}: {len(records):,} records ... {pct_str}", end="\r")

        if len(page) < _PAGE_SIZE:
            break

        # Next page starts just after the last record's timestamp
        start_ms = last_ts + 1
        time.sleep(_DELAY_S)

    print()
    return sorted(records, key=lambda r: r["timestamp_ms"])


def _save(symbol: str, records: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{symbol}_funding.parquet"

    # Merge with existing file so incremental updates work
    existing: list[dict] = []
    if path.exists():
        tbl = pq.read_table(path).to_pydict()
        existing_ts = set(tbl["timestamp_ms"])
        existing = [
            {"timestamp_ms": tbl["timestamp_ms"][i],
             "funding_rate": tbl["funding_rate"][i]}
            for i in range(len(tbl["timestamp_ms"]))
        ]
        before = len(existing)
        new_recs = [r for r in records if r["timestamp_ms"] not in existing_ts]
        records = sorted(existing + new_recs, key=lambda r: r["timestamp_ms"])
        print(f"  Merged: {before:,} existing + {len(new_recs):,} new = {len(records):,} total")

    table = pa.table({
        "symbol":       [symbol]        * len(records),
        "timestamp_ms": [r["timestamp_ms"] for r in records],
        "funding_rate": [r["funding_rate"]  for r in records],
        "exchange":     ["Binance"]     * len(records),
    }, schema=_SCHEMA)
    pq.write_table(table, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=_DEFAULT_SYMBOLS)
    parser.add_argument("--out", default="data/funding")
    args = parser.parse_args()

    out_dir = _REPO_ROOT / args.out
    print("Downloading full funding-rate history from Binance (no API key required)")
    print(f"Symbols : {args.symbols}")
    print(f"Output  : {out_dir}")
    print()

    for symbol in args.symbols:
        print(f"[{symbol}]")
        records = _download_all(symbol)
        if not records:
            print(f"  No data returned for {symbol}")
            continue
        path = _save(symbol, records, out_dir)
        s = datetime.fromtimestamp(records[0]["timestamp_ms"]  / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        e = datetime.fromtimestamp(records[-1]["timestamp_ms"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        print(f"  Saved {len(records):,} records  ({s} → {e})  → {path}")

    print()
    print("Done. Commit data/funding/ to the repo so CI runners can load it.")
    print("  git add data/funding/")
    print("  git commit -m 'add funding rate cache'")
    print("  git push")


if __name__ == "__main__":
    main()
