#!/usr/bin/env python3
"""Download historical funding rate data from Coinglass (free, no API key required).

Bitget's own API only retains ~90 days of funding history. Coinglass provides
the full history back to 2020 for all major perpetual futures.

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

# Coinglass open API — no key required for funding rate history
_URL_COINGLASS   = "https://open-api.coinglass.com/public/v2/funding"
_DELAY_S         = 0.5
_DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]

# Coinglass uses coin symbol (BTC) not pair symbol (BTCUSDT)
_SYMBOL_MAP = {
    "BTCUSDT": "BTC",
    "ETHUSDT": "ETH",
    "SOLUSDT": "SOL",
    "BNBUSDT": "BNB",
}

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


def _fetch_coinglass(coin: str) -> list[dict]:
    """Fetch full funding rate history from Coinglass for a coin on Bitget."""
    url = f"{_URL_COINGLASS}?symbol={coin}&exchange=Bitget"
    req = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Coinglass HTTP {exc.code}: {exc.reason}")

    if not data.get("success", False):
        raise RuntimeError(f"Coinglass error: {data.get('msg', data)}")

    rows = data.get("data", {})
    # Response is a dict: {exchange -> [{t: ms, r: rate}, ...]}
    if isinstance(rows, dict):
        for exchange_key, entries in rows.items():
            if "bitget" in exchange_key.lower() or "Bitget" in exchange_key:
                return entries
        # fallback: return first exchange's data
        if rows:
            return next(iter(rows.values()))
    elif isinstance(rows, list):
        return rows
    return []


def _download_bitget_api(symbol: str) -> list[dict]:
    """Fallback: fetch recent funding rates directly from Bitget (last ~90 days)."""
    url = (
        f"https://api.bitget.com/api/v2/mix/market/history-fund-rate"
        f"?symbol={symbol}&productType=USDT-FUTURES&pageSize=100&pageNo=1"
    )
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read())
    if data.get("code") != "00000":
        raise RuntimeError(f"Bitget API error: {data.get('msg')}")
    records = []
    for row in data.get("data", []):
        records.append({
            "timestamp_ms": int(row["fundingTime"]),
            "funding_rate": float(row["fundingRate"]),
            "exchange":     "Bitget",
        })
    return sorted(records, key=lambda r: r["timestamp_ms"])


def _save(symbol: str, records: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{symbol}_funding.parquet"

    # Merge with existing if present
    existing: list[dict] = []
    if path.exists():
        tbl = pq.read_table(path).to_pydict()
        existing_ts = set(tbl["timestamp_ms"])
        existing = [
            {"timestamp_ms": tbl["timestamp_ms"][i],
             "funding_rate": tbl["funding_rate"][i],
             "exchange":     tbl["exchange"][i]}
            for i in range(len(tbl["timestamp_ms"]))
        ]
        before = len(existing)
        new = [r for r in records if r["timestamp_ms"] not in existing_ts]
        records = sorted(existing + new, key=lambda r: r["timestamp_ms"])
        print(f"  Merged: {before} existing + {len(new)} new = {len(records)} total")

    table = pa.table({
        "symbol":       [symbol] * len(records),
        "timestamp_ms": [r["timestamp_ms"] for r in records],
        "funding_rate": [r["funding_rate"]  for r in records],
        "exchange":     [r.get("exchange", "Bitget") for r in records],
    }, schema=_SCHEMA)
    pq.write_table(table, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=_DEFAULT_SYMBOLS)
    parser.add_argument("--out",     default="data/funding")
    parser.add_argument("--source",  default="auto",
                        choices=["auto", "coinglass", "bitget"],
                        help="Data source. auto tries Coinglass first, falls back to Bitget")
    args = parser.parse_args()

    out_dir = _REPO_ROOT / args.out
    print(f"Downloading funding rate history")
    print(f"Symbols: {args.symbols}  Source: {args.source}")
    print()

    for symbol in args.symbols:
        print(f"[{symbol}]")
        records: list[dict] = []

        if args.source in ("auto", "coinglass"):
            coin = _SYMBOL_MAP.get(symbol, symbol.replace("USDT", ""))
            try:
                raw = _fetch_coinglass(coin)
                for r in raw:
                    ts = r.get("t") or r.get("fundingTime") or r.get("timestamp")
                    rate = r.get("r") or r.get("fundingRate") or r.get("rate")
                    if ts and rate is not None:
                        records.append({
                            "timestamp_ms": int(ts),
                            "funding_rate": float(rate),
                            "exchange":     "Bitget",
                        })
                records = sorted(records, key=lambda r: r["timestamp_ms"])
                print(f"  Coinglass: {len(records)} records fetched")
            except Exception as exc:
                print(f"  [WARN] Coinglass failed: {exc}")
                if args.source == "coinglass":
                    print(f"  No data for {symbol}")
                    continue
                print(f"  Falling back to Bitget API (~90 days only) ...")
                args.source = "bitget"

        if args.source == "bitget" or (args.source == "auto" and not records):
            try:
                records = _download_bitget_api(symbol)
                print(f"  Bitget API: {len(records)} records (last ~90 days only)")
                print(f"  [WARN] Full history unavailable — Bitget API limited to ~90 days.")
                print(f"         For backtesting, Coinglass is required.")
            except Exception as exc:
                print(f"  [ERROR] Bitget API also failed: {exc}")
                continue

        time.sleep(_DELAY_S)

        if not records:
            print(f"  No data for {symbol}")
            continue

        path = _save(symbol, records, out_dir)
        s = datetime.fromtimestamp(records[0]["timestamp_ms"]  / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        e = datetime.fromtimestamp(records[-1]["timestamp_ms"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        print(f"  Saved {len(records):,} records  ({s} → {e})  → {path}")


if __name__ == "__main__":
    main()
