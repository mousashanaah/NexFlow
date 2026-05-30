#!/usr/bin/env python3
"""Download historical ETHUSDT (or any symbol) 1m candles from Bitget.

Uses the correct Bitget v2 endpoint for deep historical data:
  /api/v2/mix/market/history-candles
  — paginates BACKWARDS from a given endTime
  — returns up to 200 bars per request

Also fetches 5m and 15m bars for the same period (resampled from 1m).

Output: parquet files in data/candles/ compatible with BacktestRunner.
  ETHUSDT_1m.parquet
  ETHUSDT_5m.parquet
  ETHUSDT_15m.parquet

Usage:
  python scripts/download_candles.py
  python scripts/download_candles.py --symbol ETHUSDT --start 2023-01-01
  python scripts/download_candles.py --symbol ETHUSDT --start 2023-01-01 --end 2026-05-31
  python scripts/download_candles.py --resume           (extend existing parquet forward)
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
    _HAS_PARQUET = True
except ImportError:
    print("[ERROR] pyarrow is required: pip install pyarrow")
    sys.exit(1)

from nexflow.services.candles.candle_engine import Candle

_SCHEMA = pa.schema([
    pa.field("symbol",              pa.string()),
    pa.field("timeframe",           pa.string()),
    pa.field("open_time",           pa.int64()),
    pa.field("close_time",          pa.int64()),
    pa.field("open",                pa.float64()),
    pa.field("high",                pa.float64()),
    pa.field("low",                 pa.float64()),
    pa.field("close",               pa.float64()),
    pa.field("volume",              pa.float64()),
    pa.field("buy_volume",          pa.float64()),
    pa.field("sell_volume",         pa.float64()),
    pa.field("trade_count",         pa.int64()),
    pa.field("vwap",                pa.float64()),
    pa.field("spread_avg",          pa.float64()),
    pa.field("spread_max",          pa.float64()),
    pa.field("volatility_estimate", pa.float64()),
    pa.field("is_final",            pa.bool_()),
])

_BASE_URL = "https://api.bitget.com/api/v2/mix/market/history-candles"
_BARS_PER_REQUEST = 200
_REQUEST_DELAY_S  = 0.25    # 4 req/s — well within Bitget public rate limits


def _fetch_chunk(symbol: str, end_ms: int) -> list[dict]:
    """Fetch up to 200 bars ending at end_ms (exclusive). Returns newest-first list."""
    url = (
        f"{_BASE_URL}"
        f"?symbol={symbol}&productType=USDT-FUTURES&granularity=1m"
        f"&endTime={end_ms}&limit={_BARS_PER_REQUEST}"
    )
    req = urllib.request.Request(
        url, headers={"User-Agent": "NexFlow/1.0", "Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read())
    if data.get("code") != "00000":
        raise RuntimeError(f"API error: {data.get('msg', data)}")
    return data.get("data", [])


def _buy_vol_proxy(o: float, h: float, l: float, c: float, vol: float) -> float:
    if h > l:
        return vol * (c - l) / (h - l)
    return vol * 0.5


def _row_to_candle(r: list, symbol: str) -> Candle:
    ts_ms = int(r[0])
    o, h, l, c = float(r[1]), float(r[2]), float(r[3]), float(r[4])
    vol  = float(r[5]) if len(r) > 5 else 0.0
    bvol = _buy_vol_proxy(o, h, l, c, vol)
    open_s  = ts_ms // 1000
    close_s = open_s + 60
    return Candle(
        symbol=symbol, timeframe="1m",
        open_time=open_s, close_time=close_s,
        open=o, high=h, low=l, close=c,
        volume=vol, buy_volume=bvol, sell_volume=vol - bvol,
        trade_count=0,
        vwap=(o + h + l + c) / 4,
        spread_avg=0.0,   # not available from REST
        spread_max=0.0,
        volatility_estimate=(h - l) / o if o > 0 else 0.0,
        is_final=True,
    )


def download_1m(
    symbol: str, start_s: int, end_s: int,
    existing_close_times: set[int] | None = None,
) -> list[Candle]:
    """Fetch all 1m candles in [start_s, end_s].

    Paginates backwards from end_s using the history-candles endpoint.
    Skips bars already in existing_close_times (for --resume mode).
    """
    existing = existing_close_times or set()
    all_candles: list[Candle] = []
    cursor_ms = end_s * 1000
    start_ms  = start_s * 1000
    total_fetched = 0
    empty_retries = 0

    print(f"  Fetching {symbol} 1m  "
          f"{datetime.fromtimestamp(start_s, tz=timezone.utc).strftime('%Y-%m-%d')} → "
          f"{datetime.fromtimestamp(end_s,   tz=timezone.utc).strftime('%Y-%m-%d')}")

    while cursor_ms > start_ms:
        try:
            rows = _fetch_chunk(symbol, cursor_ms)
        except Exception as exc:
            print(f"\n  [WARN] fetch error at cursor={cursor_ms}: {exc}")
            time.sleep(2.0)
            empty_retries += 1
            if empty_retries >= 5:
                print("  [WARN] Too many errors — stopping early.")
                break
            continue

        if not rows:
            empty_retries += 1
            if empty_retries >= 3:
                break
            time.sleep(1.0)
            continue
        empty_retries = 0

        # rows are newest-first; filter to our time range and skip duplicates
        added = 0
        oldest_ms = cursor_ms
        for r in rows:
            ts_ms = int(r[0])
            if ts_ms < start_ms:
                continue
            close_s = ts_ms // 1000 + 60
            if close_s in existing:
                continue
            c = _row_to_candle(r, symbol)
            all_candles.append(c)
            existing.add(close_s)
            added += 1
            if ts_ms < oldest_ms:
                oldest_ms = ts_ms

        total_fetched += added
        # Move cursor to 1ms before the oldest bar in this chunk
        cursor_ms = oldest_ms - 1

        # Progress
        dt = datetime.fromtimestamp(oldest_ms / 1000, tz=timezone.utc)
        print(f"\r  {dt.strftime('%Y-%m-%d %H:%M')}  bars={total_fetched:>7,}", end="", flush=True)

        time.sleep(_REQUEST_DELAY_S)

    print(f"\r  Done. {total_fetched:,} bars fetched.                          ")
    all_candles.sort(key=lambda c: c.close_time)
    return all_candles


def _resample(candles_1m: list[Candle], symbol: str, tf: str, tf_s: int) -> list[Candle]:
    result: list[Candle] = []
    bucket: list[Candle] = []
    for c in candles_1m:
        bucket.append(c)
        if c.close_time % tf_s == 0 and bucket:
            o  = bucket[0].open
            h  = max(b.high for b in bucket)
            l  = min(b.low  for b in bucket)
            cl = bucket[-1].close
            vol  = sum(b.volume for b in bucket)
            bvol = sum(b.buy_volume for b in bucket)
            pv   = sum(b.vwap * b.volume for b in bucket)
            result.append(Candle(
                symbol=symbol, timeframe=tf,
                open_time=bucket[0].open_time,
                close_time=bucket[-1].close_time,
                open=o, high=h, low=l, close=cl,
                volume=vol, buy_volume=bvol, sell_volume=vol - bvol,
                trade_count=0,
                vwap=pv / vol if vol > 0 else (o + h + l + cl) / 4,
                spread_avg=0.0, spread_max=0.0,
                volatility_estimate=(h - l) / o if o > 0 else 0.0,
                is_final=True,
            ))
            bucket = []
    return result


def _candles_to_table(candles: list[Candle]) -> pa.Table:
    cols: dict[str, list] = {f.name: [] for f in _SCHEMA}
    for c in candles:
        cols["symbol"].append(c.symbol)
        cols["timeframe"].append(c.timeframe)
        cols["open_time"].append(c.open_time)
        cols["close_time"].append(c.close_time)
        cols["open"].append(c.open)
        cols["high"].append(c.high)
        cols["low"].append(c.low)
        cols["close"].append(c.close)
        cols["volume"].append(c.volume)
        cols["buy_volume"].append(c.buy_volume)
        cols["sell_volume"].append(c.sell_volume)
        cols["trade_count"].append(c.trade_count)
        cols["vwap"].append(c.vwap)
        cols["spread_avg"].append(c.spread_avg)
        cols["spread_max"].append(c.spread_max)
        cols["volatility_estimate"].append(c.volatility_estimate)
        cols["is_final"].append(c.is_final)
    arrays = [pa.array(cols[f.name], type=f.type) for f in _SCHEMA]
    return pa.table(arrays, schema=_SCHEMA)


def _load_existing(path: Path, symbol: str) -> tuple[list[Candle], set[int]]:
    """Load existing parquet, return (candles, set_of_close_times)."""
    if not path.exists():
        return [], set()
    rows = pq.read_table(path).to_pylist()
    candles = [
        Candle(
            symbol=r["symbol"], timeframe=r["timeframe"],
            open_time=r["open_time"], close_time=r["close_time"],
            open=r["open"], high=r["high"], low=r["low"], close=r["close"],
            volume=r["volume"], buy_volume=r["buy_volume"],
            sell_volume=r["sell_volume"], trade_count=r["trade_count"],
            vwap=r["vwap"], spread_avg=r["spread_avg"],
            spread_max=r["spread_max"],
            volatility_estimate=r["volatility_estimate"],
            is_final=True,
        )
        for r in rows if r.get("is_final")
    ]
    candles.sort(key=lambda c: c.close_time)
    return candles, {c.close_time for c in candles}


def _save(candles: list[Candle], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = _candles_to_table(candles)
    pq.write_table(table, path)
    first = candles[0]  if candles else None
    last  = candles[-1] if candles else None
    if first and last:
        span_days = (last.close_time - first.close_time) / 86400
        print(f"  Saved {len(candles):,} bars → {path.name}")
        print(f"  Range: {datetime.fromtimestamp(first.close_time, tz=timezone.utc).strftime('%Y-%m-%d')} "
              f"→ {datetime.fromtimestamp(last.close_time, tz=timezone.utc).strftime('%Y-%m-%d')} "
              f"({span_days:.0f} days)")


def main() -> None:
    p = argparse.ArgumentParser(description="Download ETHUSDT historical candles from Bitget")
    p.add_argument("--symbol",    default="ETHUSDT")
    p.add_argument("--start",     default="2023-01-01",
                   help="Start date YYYY-MM-DD (default: 2023-01-01)")
    p.add_argument("--end",       default=None,
                   help="End date YYYY-MM-DD (default: today)")
    p.add_argument("--out-dir",   default="data/candles",
                   help="Output directory for parquet files")
    p.add_argument("--resume",    action="store_true",
                   help="Load existing parquet and extend forward to --end")
    args = p.parse_args()

    out_dir = _REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        print("--start must be YYYY-MM-DD"); sys.exit(1)

    if args.end:
        try:
            end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            print("--end must be YYYY-MM-DD"); sys.exit(1)
    else:
        end_dt = datetime.now(tz=timezone.utc)

    path_1m = out_dir / f"{args.symbol}_1m.parquet"

    existing_candles: list[Candle] = []
    existing_ts: set[int] = set()

    if args.resume and path_1m.exists():
        print(f"Loading existing {path_1m.name} …")
        existing_candles, existing_ts = _load_existing(path_1m, args.symbol)
        if existing_candles:
            oldest = existing_candles[0].close_time
            newest = existing_candles[-1].close_time
            print(f"  {len(existing_candles):,} bars already present  "
                  f"({datetime.fromtimestamp(oldest, tz=timezone.utc).strftime('%Y-%m-%d')} "
                  f"→ {datetime.fromtimestamp(newest, tz=timezone.utc).strftime('%Y-%m-%d')})")
            # Only fetch the gap
            if newest >= int(end_dt.timestamp()):
                print("  Already up to date.")
                return
            # Extend forward from newest bar
            start_dt = datetime.fromtimestamp(newest, tz=timezone.utc)

    start_s = int(start_dt.timestamp())
    end_s   = int(end_dt.timestamp())
    span_days = (end_s - start_s) / 86400
    expected_bars = int(span_days * 1440)

    print(f"\nDownloading {args.symbol} 1m candles")
    print(f"  {start_dt.strftime('%Y-%m-%d')} → {end_dt.strftime('%Y-%m-%d')}"
          f"  (~{span_days:.0f} days, ~{expected_bars:,} bars expected)")
    print(f"  Endpoint: {_BASE_URL}")
    print(f"  Output:   {out_dir}")
    print()

    # Download
    new_candles = download_1m(args.symbol, start_s, end_s, existing_ts)
    all_candles_1m = sorted(existing_candles + new_candles, key=lambda c: c.close_time)

    if not all_candles_1m:
        print("[ERROR] No candles downloaded. Check network connectivity and symbol name.")
        sys.exit(1)

    # Verify coverage
    actual_days = (all_candles_1m[-1].close_time - all_candles_1m[0].close_time) / 86400
    coverage    = len(all_candles_1m) / max(actual_days * 1440, 1) * 100
    print(f"\n  Coverage: {coverage:.1f}% of expected bars")
    if coverage < 85:
        print("  [WARN] Coverage below 85%. Some periods may have gaps (exchange downtime,")
        print("         contract not yet listed, or API gaps for very old dates).")

    # Resample and save
    print("\nResampling to 5m and 15m …")
    candles_5m  = _resample(all_candles_1m, args.symbol, "5m",  300)
    candles_15m = _resample(all_candles_1m, args.symbol, "15m", 900)
    print(f"  5m bars:  {len(candles_5m):,}")
    print(f"  15m bars: {len(candles_15m):,}")

    print("\nWriting parquet files …")
    _save(all_candles_1m, out_dir / f"{args.symbol}_1m.parquet")
    _save(candles_5m,     out_dir / f"{args.symbol}_5m.parquet")
    _save(candles_15m,    out_dir / f"{args.symbol}_15m.parquet")

    # Month summary
    from collections import defaultdict
    month_counts: dict[str, int] = defaultdict(int)
    for c in all_candles_1m:
        dt = datetime.fromtimestamp(c.close_time, tz=timezone.utc)
        month_counts[f"{dt.year:04d}-{dt.month:02d}"] += 1
    print(f"\nBars per month ({len(month_counts)} months):")
    for month in sorted(month_counts):
        n = month_counts[month]
        expected = 43200  # 30 days × 1440 bars
        pct = n / expected * 100
        status = "✓" if pct >= 85 else "⚠"
        print(f"  {status} {month}  {n:6,} bars  ({pct:.0f}% of expected)")

    print("\nDone. Now run:")
    print(f"  python scripts/regime_robustness.py --symbol {args.symbol} --start 2023-01 --candle-dir {args.out_dir}")


if __name__ == "__main__":
    main()
