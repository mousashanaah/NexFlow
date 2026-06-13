#!/usr/bin/env python3
"""Download a de-biased stock universe for strategy testing.

Includes:
- Sector ETFs (non-cherry-picked anchors)
- Mega-caps (some winners, some losers)
- Mid/small volatile names (PLTR, HOOD, RBLX, etc.)
- High-beta / high-volatility names
- Beaten-down / failed names (INTC, BABA, NKLA, etc.)
- Crypto-adjacent stocks (COIN, MSTR, RIOT, MARA, HUT)
- Commodity / defensive plays (GLD, SLV, USO, XOM, CVX)

Goal: find the BEST combo for Bitget TradFi stock perps,
not cherry-pick hindsight winners.
"""

from __future__ import annotations
import sys, time
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_OUT_DIR   = _REPO_ROOT / "data" / "stocks"
_FROM      = "2018-01-01"

# De-biased universe: diverse sectors, volatility profiles, including failures
_TICKERS = [
    # -- Broad ETFs (regime anchors, low bias) --
    "SPY", "QQQ", "IWM", "DIA",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLU", "XLRE", "XLP",

    # -- Mega-cap tech (winners AND one loser) --
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META",
    "INTC",       # loser - deliberate inclusion to avoid bias

    # -- High-beta growth / meme / volatile --
    "TSLA", "PLTR", "RBLX", "HOOD", "UPST", "SOFI",
    "LCID", "RIVN", "AFRM",

    # -- Crypto-adjacent (high vol, crypto correlation) --
    "COIN", "MSTR", "RIOT", "MARA", "HUT", "CLSK",

    # -- China / emerging (BABA = deliberate loser inclusion) --
    "BABA", "JD", "BIDU",

    # -- Semiconductors (beyond NVDA) --
    "AMD", "AVGO", "QCOM", "MU", "AMAT", "LRCX",

    # -- Biotech / healthcare volatile --
    "MRNA", "BNTX", "CRSP", "EDIT",

    # -- Consumer / entertainment --
    "NFLX", "SPOT", "SNAP", "PINS", "UBER", "LYFT",

    # -- Financials / fintech --
    "SQ", "PYPL", "V", "MA",

    # -- Energy / commodity --
    "XOM", "CVX", "USO",

    # -- Precious metals --
    "GLD", "SLV",

    # -- Retail --
    "WMT", "TGT", "COST",

    # -- Failed / beaten names (to de-bias and test robustness) --
    "NKLA",       # near-zero, fraud-adjacent
    "GME",        # meme, chaotic
    "AMC",        # meme, chaotic
]


def main():
    try:
        import yfinance as yf
        import pandas as pd
    except ImportError:
        print("Run:  pip install yfinance pandas pyarrow")
        sys.exit(1)

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok, fail = 0, []

    for t in _TICKERS:
        try:
            df = yf.download(t, start=_FROM, auto_adjust=True, progress=False)
            if df is None or len(df) < 200:
                n = len(df) if df is not None else 0
                print(f"  {t}: SKIP — only {n} bars")
                fail.append(t)
                continue
            df = df.reset_index()
            df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                          for c in df.columns]
            ts = pd.to_datetime(df["date"]).astype("datetime64[ns]")
            df["open_time"] = ts.astype("int64") // 10**6
            out = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
            path = _OUT_DIR / f"{t}_1D.parquet"
            out.to_parquet(path, index=False)
            d0 = df["date"].iloc[0].date()
            d1 = df["date"].iloc[-1].date()
            print(f"  {t}: {len(out):,} bars  {d0} → {d1}")
            ok += 1
        except Exception as e:
            print(f"  {t}: ERROR — {e}")
            fail.append(t)
        time.sleep(0.35)

    print(f"\nDone: {ok}/{len(_TICKERS)} saved  failed={fail}")


if __name__ == "__main__":
    main()
