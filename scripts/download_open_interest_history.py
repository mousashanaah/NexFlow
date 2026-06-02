#!/usr/bin/env python3
"""Download historical 1H open-interest from Bybit USDT-Perpetuals (no API key).

Binance's open-interest history endpoint only retains the latest ~30 days, which
is useless for backtesting. Bybit's public /v5/market/open-interest endpoint
returns a much deeper history with no authentication, paginated backward via a
cursor. It is reachable from a normal home connection (it geo-blocks US cloud
IPs, same as Binance — so run this LOCALLY and commit the parquet).

OI is reported in CONTRACTS (base asset units) at each interval. We keep the raw
value; the backtest only ever uses its *percentage change*, so units cancel out.

Saves to data/oi/{SYMBOL}_OI_1H.parquet

Usage:
  python scripts/download_open_interest_history.py
  python scripts/download_open_interest_history.py --symbols BTCUSDT ETHUSDT
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
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

_BASE_URL        = "https://api.bybit.com/v5/market/open-interest"
_INTERVAL        = "1h"        # Bybit interval token for 1-hour buckets
_PAGE_LIMIT      = 200         # Bybit max per request
_DELAY_S         = 0.25
_DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]

_HEADERS = {
    "User-Agent": "NexFlow/1.0",
    "Accept":     "application/json",
}

_SCHEMA = pa.schema([
    pa.field("symbol",        pa.string()),
    pa.field("timestamp_ms",  pa.int64()),
    pa.field("open_interest", pa.float64()),
])


def _fetch_page(symbol: str, cursor: str | None) -> dict:
    """Fetch one page of OI history (newest-first). Returns the `result` dict."""
    params = {
        "category":     "linear",
        "symbol":       symbol,
        "intervalTime": _INTERVAL,
        "limit":        _PAGE_LIMIT,
    }
    if cursor:
        params["cursor"] = cursor
    url = f"{_BASE_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 451):
            print(
                f"\n[ERROR] Bybit returned HTTP {exc.code} (geo-blocked).\n"
                f"        You are likely on a US cloud server. Run this on your\n"
                f"        LOCAL machine and commit the resulting parquet."
            )
            sys.exit(1)
        raise
    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit API error: {data.get('retMsg', data)}")
    return data.get("result", {})


def _download_all(symbol: str) -> list[dict]:
    """Walk Bybit OI history backward via cursor until exhausted."""
    records: list[dict] = []
    seen: set[int] = set()
    cursor: str | None = None

    while True:
        result = _fetch_page(symbol, cursor)
        rows = result.get("list", []) or []
        if not rows:
            break

        for r in rows:
            ts = int(r["timestamp"])
            if ts in seen:
                continue
            seen.add(ts)
            records.append({
                "timestamp_ms": ts,
                "open_interest": float(r["openInterest"]),
            })

        oldest = min(r["timestamp_ms"] for r in records)
        oldest_str = datetime.fromtimestamp(oldest / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        print(f"  {symbol}: {len(records):,} rows ... back to {oldest_str}", end="\r")

        cursor = result.get("nextPageCursor")
        if not cursor:
            break
        time.sleep(_DELAY_S)

    print()
    return sorted(records, key=lambda r: r["timestamp_ms"])


def _save(symbol: str, records: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{symbol}_OI_1H.parquet"

    # Merge with existing for incremental top-ups
    if path.exists():
        tbl = pq.read_table(path).to_pydict()
        existing_ts = set(tbl["timestamp_ms"])
        existing = [
            {"timestamp_ms": tbl["timestamp_ms"][i],
             "open_interest": tbl["open_interest"][i]}
            for i in range(len(tbl["timestamp_ms"]))
        ]
        before = len(existing)
        new_recs = [r for r in records if r["timestamp_ms"] not in existing_ts]
        records = sorted(existing + new_recs, key=lambda r: r["timestamp_ms"])
        print(f"  Merged: {before:,} existing + {len(new_recs):,} new = {len(records):,} total")

    table = pa.table({
        "symbol":        [symbol] * len(records),
        "timestamp_ms":  [r["timestamp_ms"]  for r in records],
        "open_interest": [r["open_interest"] for r in records],
    }, schema=_SCHEMA)
    pq.write_table(table, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=_DEFAULT_SYMBOLS)
    parser.add_argument("--out", default="data/oi")
    args = parser.parse_args()

    out_dir = _REPO_ROOT / args.out
    print("Downloading 1H open-interest history from Bybit (no API key required)")
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
        print(f"  Saved {len(records):,} rows  ({s} → {e})  → {path}")

    print()
    print("Done. Commit data/oi/ so CI runners can load it:")
    print("  git add data/oi/")
    print("  git commit -m 'add open interest history cache'")
    print("  git push")


if __name__ == "__main__":
    main()
