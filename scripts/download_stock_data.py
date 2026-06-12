#!/usr/bin/env python3
"""Download daily OHLC stock/ETF data from Stooq (keyless, free) into
data/stocks/<TICKER>.parquet — same layout idea as our crypto candles.

Run this from your LOCAL machine (cloud env blocks financial data hosts):

    python scripts/download_stock_data.py

Then commit the parquet files:

    git add data/stocks && git commit -m "Add stock daily data" && git push
"""

from __future__ import annotations

import io
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).parent.parent
_OUT_DIR   = _REPO_ROOT / "data" / "stocks"

# Universe: broad indices + mega-cap leaders + a few high-beta names.
# ETFs give the index exposure a futures account would trade (ES≈SPY, NQ≈QQQ).
_TICKERS = [
    # Index ETFs (proxies for index futures)
    "SPY", "QQQ", "IWM", "DIA",
    # Mega caps (the liquid single-name futures/CFD candidates)
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
    # High-beta / thematic
    "AMD", "NFLX", "COIN", "MSTR",
    # Defensive diversifier
    "GLD",
]

_FROM = "2018-01-01"   # extra history so SMA200 warms up before 2021


def _fetch_stooq(ticker: str) -> pd.DataFrame | None:
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  [WARN] {ticker}: {e}")
        return None
    if not raw or raw.startswith("<") or "Date" not in raw.splitlines()[0]:
        print(f"  [WARN] {ticker}: unexpected response ({raw[:60]!r})")
        return None
    df = pd.read_csv(io.StringIO(raw))
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"date": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= _FROM].reset_index(drop=True)
    # open_time in ms UTC midnight — matches crypto parquet convention
    df["open_time"] = (df["date"].astype("int64") // 10**6)
    return df[["open_time", "open", "high", "low", "close", "volume"]]


def main():
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok = 0
    for t in _TICKERS:
        df = _fetch_stooq(t)
        if df is None or len(df) < 500:
            print(f"  {t}: FAILED or too little data")
            continue
        path = _OUT_DIR / f"{t}_1D.parquet"
        df.to_parquet(path, index=False)
        d0 = pd.to_datetime(df['open_time'].iloc[0],  unit='ms').date()
        d1 = pd.to_datetime(df['open_time'].iloc[-1], unit='ms').date()
        print(f"  {t}: {len(df):,} bars  {d0} → {d1}  saved")
        ok += 1
        time.sleep(0.5)
    print(f"\nDone: {ok}/{len(_TICKERS)} tickers saved to {_OUT_DIR}")
    if ok:
        print("Now run:  git add data/stocks && "
              "git commit -m 'Add stock daily data' && git push")


if __name__ == "__main__":
    main()
