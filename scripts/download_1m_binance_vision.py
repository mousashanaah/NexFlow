#!/usr/bin/env python3
"""
download_1m_binance_vision.py — Download complete 1-minute BTC/ETH futures candles.

Source: https://data.binance.vision (free, no API key, authoritative)
  Monthly ZIP files: data/futures/um/monthly/klines/{SYMBOL}/1m/
  Each ZIP: one CSV with ~44,640 rows (31 days × 1440 min/day)

WHY THIS MATTERS:
  Bitget's API caps at 200 bars per request → only ~21% coverage on 1m.
  Binance Vision has every single minute bar back to 2020 with zero gaps.
  Full coverage = ~10× more ICT/FVG setups for statistical significance.

OUTPUT:
  data/candles/BTCUSDT_1m.parquet  (overwrites Bitget version)
  data/candles/ETHUSDT_1m.parquet  (optional)

Run on your LOCAL machine (Binance Vision is geo-blocked from cloud):
  python scripts/download_1m_binance_vision.py
  python scripts/download_1m_binance_vision.py --symbol ETHUSDT
  python scripts/download_1m_binance_vision.py --start 2022-01 --end 2026-06

Then push the parquet so the cloud session can run the backtest:
  git add data/candles/BTCUSDT_1m.parquet
  git commit -m "full 1m BTC data from Binance Vision (complete coverage)"
  git push origin claude/gallant-mendel-vJRA6
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("[ERROR] pip install pyarrow"); sys.exit(1)

# Binance Vision: USDT-M Futures monthly 1m klines
_BASE = "https://data.binance.vision/data/futures/um/monthly/klines"
_HEADERS = {"User-Agent": "NexFlow/1.0", "Accept": "*/*"}
_DELAY_S = 0.5   # polite delay between files

_SCHEMA = pa.schema([
    pa.field("symbol",   pa.string()),
    pa.field("timeframe",pa.string()),
    pa.field("open_time",pa.int64()),
    pa.field("close_time",pa.int64()),
    pa.field("open",     pa.float64()),
    pa.field("high",     pa.float64()),
    pa.field("low",      pa.float64()),
    pa.field("close",    pa.float64()),
    pa.field("volume",   pa.float64()),
])

# Binance Vision 1m CSV column layout
_COL_OPEN_TIME  = 0
_COL_OPEN       = 1
_COL_HIGH       = 2
_COL_LOW        = 3
_COL_CLOSE      = 4
_COL_VOLUME     = 5
_COL_CLOSE_TIME = 6


def _months_between(start: str, end: str) -> list[tuple[int, int]]:
    """Generate (year, month) pairs from 'YYYY-MM' to 'YYYY-MM' inclusive."""
    sy, sm = int(start[:4]), int(start[5:7])
    ey, em = int(end[:4]),   int(end[5:7])
    months = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1; y += 1
    return months


def _download_month(symbol: str, year: int, month: int) -> list[dict]:
    """Download one monthly ZIP and return list of candle dicts."""
    fname = f"{symbol}-1m-{year:04d}-{month:02d}.zip"
    url   = f"{_BASE}/{symbol}/1m/{fname}"
    req   = urllib.request.Request(url, headers=_HEADERS)

    backoff = 2
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return []   # month not available yet (future months)
            if e.code in (403, 451):
                print(f"\n  [GEO-BLOCK] HTTP {e.code} — Binance Vision is geo-blocked.")
                print("  Run this script on your LOCAL machine, not the cloud.")
                sys.exit(1)
            if attempt == 3:
                print(f"  [WARN] {fname}: HTTP {e.code} after 4 attempts, skipping")
                return []
            time.sleep(backoff); backoff *= 2
        except Exception as e:
            if attempt == 3:
                print(f"  [WARN] {fname}: {e}, skipping")
                return []
            time.sleep(backoff); backoff *= 2

    candles = []
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            reader = csv.reader(io.TextIOWrapper(f))
            for row in reader:
                if not row or not row[0].lstrip("-").isdigit():
                    continue  # skip header if present
                try:
                    candles.append({
                        "open_time":  int(row[_COL_OPEN_TIME]),
                        "close_time": int(row[_COL_CLOSE_TIME]),
                        "open":  float(row[_COL_OPEN]),
                        "high":  float(row[_COL_HIGH]),
                        "low":   float(row[_COL_LOW]),
                        "close": float(row[_COL_CLOSE]),
                        "volume":float(row[_COL_VOLUME]),
                    })
                except (ValueError, IndexError):
                    pass
    return candles


def _save(symbol: str, all_candles: list[dict], out_dir: Path) -> Path:
    # Deduplicate and sort
    seen = set()
    unique = []
    for c in all_candles:
        if c["open_time"] not in seen:
            seen.add(c["open_time"])
            unique.append(c)
    unique.sort(key=lambda x: x["open_time"])

    table = pa.table({
        "symbol":    [symbol] * len(unique),
        "timeframe": ["1m"]   * len(unique),
        "open_time":  [c["open_time"]  for c in unique],
        "close_time": [c["close_time"] for c in unique],
        "open":       [c["open"]       for c in unique],
        "high":       [c["high"]       for c in unique],
        "low":        [c["low"]        for c in unique],
        "close":      [c["close"]      for c in unique],
        "volume":     [c["volume"]     for c in unique],
    }, schema=_SCHEMA)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{symbol}_1m.parquet"
    pq.write_table(table, path)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Download complete 1m futures candles from Binance Vision")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--start", default="2021-01", help="Start month YYYY-MM")
    ap.add_argument("--end",   default=None,      help="End month YYYY-MM (default: current)")
    ap.add_argument("--out",   default="data/candles")
    args = ap.parse_args()

    if args.end is None:
        now = datetime.now(tz=timezone.utc)
        # Use previous month (current month may not be packaged yet)
        if now.month == 1:
            args.end = f"{now.year - 1}-12"
        else:
            args.end = f"{now.year}-{now.month - 1:02d}"

    months = _months_between(args.start, args.end)
    out_dir = _REPO_ROOT / args.out

    print("=" * 65)
    print(f"  Binance Vision — {args.symbol} 1m Futures Candles")
    print(f"  Range: {args.start} → {args.end}  ({len(months)} months)")
    print(f"  Expected bars: ~{len(months) * 44_640:,} (complete coverage)")
    print(f"  Output: {out_dir / (args.symbol + '_1m.parquet')}")
    print("=" * 65)
    print()

    all_candles: list[dict] = []

    for i, (year, month) in enumerate(months):
        label = f"{year:04d}-{month:02d}"
        candles = _download_month(args.symbol, year, month)
        all_candles.extend(candles)
        pct = (i + 1) / len(months) * 100
        bar_count = len(all_candles)
        print(f"  [{pct:5.1f}%] {label}: {len(candles):,} bars  (total {bar_count:,})", end="\r")
        if candles:
            time.sleep(_DELAY_S)

    print()
    print()

    if not all_candles:
        print("[ERROR] No candles downloaded. Check network and try again.")
        sys.exit(1)

    path = _save(args.symbol, all_candles, out_dir)

    ts = [c["open_time"] for c in all_candles]
    start_dt = datetime.fromtimestamp(min(ts) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    end_dt   = datetime.fromtimestamp(max(ts) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

    print(f"  Saved {len(all_candles):,} bars  ({start_dt} → {end_dt})")
    print(f"  File: {path}")
    print()
    print("  Now push and run the backtest:")
    print(f"    git add {path}")
    print('    git commit -m "full 1m BTC data from Binance Vision"')
    print("    git push origin claude/gallant-mendel-vJRA6")
    print()
    print("  Then the cloud session will run:")
    print("    python scripts/backtest_ict_fvg_v2.py")


if __name__ == "__main__":
    main()
