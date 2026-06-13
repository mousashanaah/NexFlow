#!/usr/bin/env python3
"""Download daily OHLC stock/ETF data via yfinance into
data/stocks/<TICKER>.parquet — same layout as our crypto candles.

Run this from your LOCAL machine:

    pip install yfinance
    python scripts/download_stock_data.py

Then commit the parquet files:

    git add data/stocks && git commit -m "Add stock daily data" && git push
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_OUT_DIR   = _REPO_ROOT / "data" / "stocks"

_TICKERS = [
    "SPY", "QQQ", "IWM", "DIA",
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
    "AMD", "NFLX", "COIN", "MSTR",
    "GLD",
]

_FROM = "2018-01-01"


def main():
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed. Run:  pip install yfinance")
        sys.exit(1)

    import pandas as pd

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok = 0
    for t in _TICKERS:
        try:
            df = yf.download(t, start=_FROM, auto_adjust=True, progress=False)
            if df is None or len(df) < 500:
                print(f"  {t}: FAILED or too little data ({len(df) if df is not None else 0} rows)")
                continue
            df = df.reset_index()
            df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                          for c in df.columns]
            df = df.rename(columns={"date": "date"})
            df["open_time"] = (pd.to_datetime(df["date"]).astype("int64") // 10**6)
            out = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
            path = _OUT_DIR / f"{t}_1D.parquet"
            out.to_parquet(path, index=False)
            d0 = df["date"].iloc[0].date()
            d1 = df["date"].iloc[-1].date()
            print(f"  {t}: {len(out):,} bars  {d0} → {d1}  saved")
            ok += 1
        except Exception as e:
            print(f"  {t}: ERROR — {e}")
        time.sleep(0.3)

    print(f"\nDone: {ok}/{len(_TICKERS)} tickers saved to {_OUT_DIR}")
    if ok:
        print("Now run:\n  git add data/stocks\n  git commit -m 'Add stock daily data'\n  git push")


if __name__ == "__main__":
    main()

