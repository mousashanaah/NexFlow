#!/usr/bin/env python3
"""Snapshot current open interest for BTCUSDT and ETHUSDT from Bitget.

Uses /api/v2/mix/market/open-interest (current snapshot only).
Appends a new row to data/oi/{SYMBOL}_OI.parquet each run.

Usage:
  python scripts/download_open_interest.py
  python scripts/download_open_interest.py --symbols BTCUSDT ETHUSDT
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

_URL_OI          = "https://api.bitget.com/api/v2/mix/market/open-interest"
_DELAY_S         = 0.2
_DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
_HEADERS         = {"User-Agent": "NexFlow/1.0", "Accept": "application/json"}

_SCHEMA = pa.schema([
    pa.field("symbol",        pa.string()),
    pa.field("timestamp_ms",  pa.int64()),
    pa.field("open_interest", pa.float64()),
])


def _fetch_oi(symbol: str) -> dict:
    url = f"{_URL_OI}?symbol={symbol}&productType=USDT-FUTURES"
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read())
    if data.get("code") != "00000":
        raise RuntimeError(f"API error: {data.get('msg', data)}")
    return data["data"]


def _append(symbol: str, row: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{symbol}_OI.parquet"

    new_table = pa.table({
        "symbol":        [row["symbol"]],
        "timestamp_ms":  [row["timestamp_ms"]],
        "open_interest": [row["open_interest"]],
    }, schema=_SCHEMA)

    if path.exists():
        existing = pq.read_table(path)
        existing_ts = set(existing.column("timestamp_ms").to_pylist())
        # Skip if this exact timestamp already recorded (idempotent re-runs)
        if row["timestamp_ms"] in existing_ts:
            return path
        combined = pa.concat_tables([existing, new_table])
        pq.write_table(combined, path)
    else:
        pq.write_table(new_table, path)

    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=_DEFAULT_SYMBOLS)
    parser.add_argument("--out",     default="data/oi")
    args = parser.parse_args()

    out_dir = _REPO_ROOT / args.out

    print(f"Snapshotting open interest")
    print(f"Symbols: {args.symbols}")
    print()

    errors = 0
    for symbol in args.symbols:
        print(f"[{symbol}]")
        try:
            data = _fetch_oi(symbol)
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            errors += 1
            continue

        # openInterestList contains entries per currency; we want the contracts (size) entry
        oi_list = data.get("openInterestList", [])
        size_val = 0.0
        for entry in oi_list:
            if entry.get("currency") not in ("USDT",):
                # The non-USDT entry is contracts
                size_val = float(entry.get("size", 0))
                break
        if not size_val and oi_list:
            size_val = float(oi_list[0].get("size", 0))

        ts_ms = int(data["ts"])
        row = {"symbol": symbol, "timestamp_ms": ts_ms, "open_interest": size_val}
        path = _append(symbol, row, out_dir)
        ts_str = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  OI={size_val:,.0f} contracts  at {ts_str}  → {path}")

        time.sleep(_DELAY_S)

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
