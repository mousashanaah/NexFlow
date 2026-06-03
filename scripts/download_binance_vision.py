#!/usr/bin/env python3
"""Download REAL daily candles for all 12 coins from Binance's official archive.

Source: https://data.binance.vision  (free, authoritative, no API key)
  Monthly daily-kline ZIPs:
    data/spot/monthly/klines/{SYMBOL}/1d/{SYMBOL}-1d-{YYYY}-{MM}.zip

Each ZIP contains one CSV with Binance kline columns:
  open_time, open, high, low, close, volume, close_time, quote_volume,
  num_trades, taker_buy_base, taker_buy_quote, ignore

Writes data/candles/{SYMBOL}_1D.parquet in the schema the backtest expects
(symbol, timeframe, open_time[ms], close_time[ms], open, high, low, close, volume).

Verifies BTC against known real all-time-high prices before trusting output.

Usage:
  python scripts/download_binance_vision.py
  python scripts/download_binance_vision.py --start 2021-01 --end 2026-06
"""

from __future__ import annotations

import argparse
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

_BASE = "https://data.binance.vision/data/spot/monthly/klines"
_DAY_MS = 86_400_000
_HEADERS = {"User-Agent": "NexFlow/1.0", "Accept": "*/*"}
_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT",
    "LINKUSDT", "LTCUSDT", "DOTUSDT", "TRXUSDT",
]

_SCHEMA = pa.schema([
    pa.field("symbol", pa.string()), pa.field("timeframe", pa.string()),
    pa.field("open_time", pa.int64()), pa.field("close_time", pa.int64()),
    pa.field("open", pa.float64()), pa.field("high", pa.float64()),
    pa.field("low", pa.float64()), pa.field("close", pa.float64()),
    pa.field("volume", pa.float64()),
])

# Known real BTC closes for sanity-check (approximate, USD)
_BTC_CHECKS = [
    ("2021-11-10", 64_000, 72_000),   # ATH region ~$69K
    ("2024-03-14", 68_000, 78_000),   # ATH region ~$73K
    ("2022-11-21", 14_000, 17_500),   # bear bottom region ~$15.7K
]


def _months(start: str, end: str):
    sy, sm = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    y, m = sy, sm
    while (y, m) <= (ey, em):
        yield y, m
        m += 1
        if m > 12:
            m = 1; y += 1


def _to_ms(ts: int) -> int:
    # Binance switched some 2025+ files to microseconds; normalize to ms.
    return ts // 1000 if ts > 10_000_000_000_000 else ts


def _fetch_month(symbol: str, y: int, m: int) -> list[dict]:
    url = f"{_BASE}/{symbol}/1d/{symbol}-1d-{y:04d}-{m:02d}.zip"
    req = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return []  # month not published yet / before listing
        raise
    rows = []
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        with zf.open(zf.namelist()[0]) as fh:
            for line in io.TextIOWrapper(fh, "utf-8"):
                c = line.strip().split(",")
                if not c or not c[0] or c[0][0].isalpha():
                    continue  # skip header row if present
                ot = _to_ms(int(c[0]))
                rows.append({
                    "open_time": ot, "close_time": ot + _DAY_MS - 1,
                    "open": float(c[1]), "high": float(c[2]), "low": float(c[3]),
                    "close": float(c[4]), "volume": float(c[5]),
                })
    return rows


def _download(symbol: str, start: str, end: str) -> list[dict]:
    bars: dict[int, dict] = {}
    for y, m in _months(start, end):
        try:
            for b in _fetch_month(symbol, y, m):
                bars[b["open_time"]] = b
        except Exception as exc:
            print(f"  [WARN] {symbol} {y}-{m:02d}: {exc}")
        print(f"  {symbol}: {len(bars)} bars ...", end="\r")
        time.sleep(0.05)
    print()
    return [bars[k] for k in sorted(bars)]


def _save(symbol: str, bars: list[dict]) -> Path:
    out = _REPO_ROOT / "data" / "candles" / f"{symbol}_1D.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    tbl = pa.table({
        "symbol": [symbol]*len(bars), "timeframe": ["1D"]*len(bars),
        "open_time": [b["open_time"] for b in bars],
        "close_time": [b["close_time"] for b in bars],
        "open": [b["open"] for b in bars], "high": [b["high"] for b in bars],
        "low": [b["low"] for b in bars], "close": [b["close"] for b in bars],
        "volume": [b["volume"] for b in bars],
    }, schema=_SCHEMA)
    pq.write_table(tbl, out)
    return out


def _verify_btc(bars: list[dict]) -> bool:
    by_date = {datetime.fromtimestamp(b["open_time"]/1000, tz=timezone.utc)
               .strftime("%Y-%m-%d"): b["close"] for b in bars}
    ok = True
    print("\n  BTC sanity check vs known real prices:")
    for date, lo, hi in _BTC_CHECKS:
        c = by_date.get(date)
        if c is None:
            print(f"    {date}: MISSING"); ok = False; continue
        good = lo <= c <= hi
        ok = ok and good
        print(f"    {date}: ${c:,.0f}  expected ${lo:,}-${hi:,}  {'OK' if good else 'WRONG'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01")
    ap.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m"))
    ap.add_argument("--symbols", nargs="+", default=_SYMBOLS)
    args = ap.parse_args()

    print(f"Binance archive download: {args.start} -> {args.end}")
    print(f"Symbols: {args.symbols}\n")

    # BTC first so we can verify data integrity before doing the rest
    syms = ["BTCUSDT"] + [s for s in args.symbols if s != "BTCUSDT"]
    for sym in syms:
        print(f"[{sym}]")
        bars = _download(sym, args.start, args.end)
        if not bars:
            print(f"  No data for {sym} — aborting."); sys.exit(1)
        if sym == "BTCUSDT" and not _verify_btc(bars):
            print("\n[ABORT] BTC data failed sanity check — not real prices. "
                  "Nothing was overwritten for the other coins.")
            sys.exit(1)
        path = _save(sym, bars)
        d0 = datetime.fromtimestamp(bars[0]["open_time"]/1000, tz=timezone.utc).date()
        d1 = datetime.fromtimestamp(bars[-1]["open_time"]/1000, tz=timezone.utc).date()
        print(f"  Saved {len(bars)} bars ({d0} -> {d1}) -> {path}\n")

    print("Done. All coins refreshed with real Binance data.")


if __name__ == "__main__":
    main()
