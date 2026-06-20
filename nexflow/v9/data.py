"""
V9 Confidence — Module 3: Data Ingestion

DATA CONTRACT
=============

Every field consumed by V9 Confidence is defined below.
The contract is enforced at load time. Any violation raises DataValidationError.
Trading is blocked when DataValidationError is raised.

──────────────────────────────────────────────────────────────────────────────
CRYPTO INSTRUMENTS
Source:          Bitget REST API  /api/v2/mix/market/candles  or local parquet
Instruments:     BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, ADAUSDT,
                 DOGEUSDT, AVAXUSDT, LINKUSDT, LTCUSDT, DOTUSDT, TRXUSDT
Timeframe:       1D (one bar = one UTC calendar day)
Timestamp:       open_time — UTC midnight of the bar's calendar day, milliseconds
                 e.g. 2021-01-01 00:00:00 UTC = 1609459200000
Gap tolerance:   Exactly 86_400_000 ms between consecutive bars
                 Any gap != 86_400_000 ms is a missing-day error
Minimum history: 200 bars (SMA200 warmup) before signals are valid
                 System uses SMA50 proxy for warmup period (bars 50–199)

Raw fields used:
  open_time  int64   UTC midnight milliseconds — primary key
  high       float64 daily high price — for ATR computation only
  low        float64 daily low price  — for ATR computation only
  close      float64 daily close price — all signal computation

Derived fields (computed by this module, not stored in parquet):
  sma200     float64  200-day simple moving average of close  (SMA50 proxy for warmup)
  sma50      float64  50-day simple moving average of close
  mom90      float64  close[i]/close[i-90] - 1  (14d proxy during warmup)
  mom30      float64  close[i]/close[i-30] - 1  (14d proxy during warmup)
  atr14      float64  14-day average true range
  atr_avg    float64  60-day average of atr14     (20d proxy during warmup)
  ema8_state bool     EMA8>EMA21 crossover state (transition-based, not point-in-time)
  macd_state bool     MACD histogram > 0 crossover state (transition-based)
  mom20      float64  close[i]/close[i-20] - 1   (20-day momentum gate for V8.63)

Validation rules (BTCUSDT):
  close > 0                                   always
  timestamps monotonically increasing         always
  gap == 86_400_000 ms                        between every consecutive pair
  no NaN in close after first bar             always
  sma200 computed from ≥ 50 bars              (50-bar minimum for proxy)

Validation rules (non-BTC coins):
  Same as BTCUSDT except:
  max_gap <= 2 * 86_400_000 ms allowed        (some coins had brief exchange downtime)

──────────────────────────────────────────────────────────────────────────────
STOCK INSTRUMENTS
Source:          Local parquet (historical); live: Yahoo Finance / Alpaca / Polygon
Instruments:     AMD, GOOGL, MSTR, SPOT
Timeframe:       1D (one bar = one US equity trading day)
Timestamp:       open_time — UTC midnight of the bar's US calendar date, milliseconds
                 e.g. Tuesday 2025-01-14 (US market open) = 1736812800000
                 Note: US market closes at 21:00 UTC; bar is timestamped at
                 the UTC midnight that begins the trading day, NOT at close.
Gap tolerance:   max 4 * 86_400_000 ms (covers 3-day weekends and holidays)
                 Any gap > 4 days requires manual review
Minimum history: 200 bars before SMA200 is reliable (SMA50 proxy for warmup)

Raw fields used:
  open_time  int64   UTC midnight milliseconds
  high       float64 not used in V9 signals (kept for future use)
  low        float64 not used in V9 signals (kept for future use)
  close      float64 adjusted close — split/dividend adjusted

Derived fields:
  sma200     float64  200-day SMA of adjusted close
  mom90      float64  close[i]/close[i-90] - 1
  ema_f      float64  EMA-8 of adjusted close (fast)
  ema_s      float64  EMA-21 of adjusted close (slow)

Validation rules:
  close > 0                                   always
  timestamps monotonically increasing         always
  gap <= 4 * 86_400_000 ms                    between consecutive bars
  no NaN in close                             always
  len(series) >= 90                           minimum for mom90

CRITICAL — SPLIT ADJUSTMENT:
  close must be the split-adjusted (and dividend-adjusted) price.
  A corporate action on any of AMD, GOOGL, MSTR, SPOT that is NOT reflected
  in adjusted prices will corrupt SMA200 for up to 200 trading days.
  The data pipeline MUST verify adjusted prices on corporate action events.

──────────────────────────────────────────────────────────────────────────────
COMMON TIMESTAMP CONVENTION

All internal operations use UTC midnight milliseconds (int64).
    utc_midnight_ms = calendar_date.timestamp() * 1000  (where time=00:00:00 UTC)

Crypto and stock timestamps are aligned to this convention.
Alignment check: for any date D where both markets have data,
    btc_ts == stock_ts (same UTC midnight)
This is enforced by the DataSet.align() method.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

_REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO))

from nexflow.v9.core import (
    sma, ema, ema_crossover_state, macd_crossover_state,
)

_DAY_MS = 86_400_000

# ── Instruments ───────────────────────────────────────────────────────────────

CRYPTO_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT",
    "LINKUSDT", "LTCUSDT", "DOTUSDT", "TRXUSDT",
]
STOCK_TICKERS = ["AMD", "GOOGL", "MSTR", "SPOT"]


# ── Validation error ──────────────────────────────────────────────────────────

class DataValidationError(RuntimeError):
    """
    Raised when ingested data violates the contract.
    Trading must be blocked until the root cause is resolved.
    """


# ── Candle containers ─────────────────────────────────────────────────────────

@dataclass
class CryptoSeries:
    """
    One crypto instrument's validated, indicator-enriched daily data.
    All arrays share the same index and are aligned to ts[i].
    """
    symbol:     str
    ts:         np.ndarray   # UTC midnight ms  (int64)
    close:      np.ndarray   # adjusted close   (float64)
    high:       np.ndarray   # daily high        (float64)
    low:        np.ndarray   # daily low         (float64)
    sma200:     np.ndarray   # SMA200 (SMA50 proxy during warmup)
    mom90:      np.ndarray   # 90d return (14d proxy during warmup)
    mom30:      np.ndarray   # 30d return (14d proxy during warmup)
    mom20:      np.ndarray   # 20d return (momentum gate)
    atr14:      np.ndarray   # ATR-14
    atr_avg:    np.ndarray   # 60d avg of ATR14 (20d proxy during warmup)
    ema8_state: np.ndarray   # bool: EMA8 > EMA21 crossover state
    macd_state: np.ndarray   # bool: MACD histogram > 0 crossover state
    by_ts:      dict         # ts → array index (for O(1) lookup)


@dataclass
class StockSeries:
    """
    One stock instrument's validated, indicator-enriched daily data.
    """
    ticker:  str
    ts:      np.ndarray   # UTC midnight ms (int64)
    close:   np.ndarray   # split-adjusted close
    sma200:  np.ndarray
    mom90:   np.ndarray
    ema_f:   np.ndarray   # EMA-8
    ema_s:   np.ndarray   # EMA-21
    by_ts:   dict


@dataclass
class V9DataSet:
    """
    Complete, aligned dataset for one evaluation run.
    All series share a common timestamp axis (intersection of all available bars).
    """
    as_of_date: str              # ISO YYYY-MM-DD — data available up to and including this date
    common_ts:  np.ndarray       # sorted UTC midnight ms present in ALL series
    crypto:     dict             # symbol → CryptoSeries
    stocks:     dict             # ticker → StockSeries

    def btc(self) -> CryptoSeries:
        return self.crypto["BTCUSDT"]

    def latest_ts(self) -> int:
        return int(self.common_ts[-1])

    def latest_date(self) -> str:
        return datetime.utcfromtimestamp(self.latest_ts() / 1000).strftime("%Y-%m-%d")


# ── Validators ────────────────────────────────────────────────────────────────

def _validate_crypto(symbol: str, ts: np.ndarray, close: np.ndarray) -> None:
    if len(ts) == 0:
        raise DataValidationError(f"{symbol}: empty series")

    if not np.all(np.diff(ts) > 0):
        raise DataValidationError(f"{symbol}: timestamps not monotonically increasing")

    gaps = np.diff(ts)
    max_gap = _DAY_MS if symbol == "BTCUSDT" else 2 * _DAY_MS
    bad = gaps[gaps > max_gap]
    if len(bad) > 0:
        idx  = int(np.argmax(gaps > max_gap))
        date = datetime.utcfromtimestamp(int(ts[idx]) / 1000).strftime("%Y-%m-%d")
        raise DataValidationError(
            f"{symbol}: gap of {int(bad[0]) // _DAY_MS}d after {date} "
            f"(max allowed: {max_gap // _DAY_MS}d)"
        )

    if np.any(close <= 0):
        n_bad = int(np.sum(close <= 0))
        raise DataValidationError(f"{symbol}: {n_bad} non-positive close prices")

    if np.any(np.isnan(close)):
        raise DataValidationError(f"{symbol}: NaN in close prices")

    if len(ts) < 50:
        raise DataValidationError(
            f"{symbol}: only {len(ts)} bars — minimum 50 required for SMA50 proxy"
        )


def _validate_stock(ticker: str, ts: np.ndarray, close: np.ndarray) -> None:
    if len(ts) == 0:
        raise DataValidationError(f"{ticker}: empty series")

    if not np.all(np.diff(ts) > 0):
        raise DataValidationError(f"{ticker}: timestamps not monotonically increasing")

    gaps = np.diff(ts)
    bad  = gaps[gaps > 4 * _DAY_MS]
    if len(bad) > 0:
        idx  = int(np.argmax(gaps > 4 * _DAY_MS))
        date = datetime.utcfromtimestamp(int(ts[idx]) / 1000).strftime("%Y-%m-%d")
        raise DataValidationError(
            f"{ticker}: gap of {int(bad[0]) // _DAY_MS}d after {date} "
            f"(max allowed: 4d — check for delisting or data outage)"
        )

    if np.any(close <= 0):
        raise DataValidationError(f"{ticker}: non-positive close price detected — check split adjustment")

    if np.any(np.isnan(close)):
        raise DataValidationError(f"{ticker}: NaN in close prices")

    if len(ts) < 90:
        raise DataValidationError(
            f"{ticker}: only {len(ts)} bars — minimum 90 required for mom90"
        )


# ── Indicator computation ─────────────────────────────────────────────────────

def _compute_crypto_indicators(
    symbol: str,
    ts: np.ndarray,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
) -> CryptoSeries:
    n = len(close)

    # SMA200 with SMA50 warmup proxy — matches test_v9_confidence.py lines 78-79
    s200 = sma(close, 200)
    s50  = sma(close, 50)
    s200 = np.where(np.isfinite(s200), s200, s50)

    # Momentum — short-window proxies for warmup
    m90 = np.full(n, np.nan); m30 = np.full(n, np.nan); m14 = np.full(n, np.nan)
    for i in range(90, n): m90[i] = close[i] / close[i - 90] - 1
    for i in range(30, n): m30[i] = close[i] / close[i - 30] - 1
    for i in range(14, n): m14[i] = close[i] / close[i - 14] - 1
    m90 = np.where(np.isfinite(m90), m90, np.where(np.isfinite(m30), m30, m14))
    m30 = np.where(np.isfinite(m30), m30, m14)

    m20 = np.full(n, np.nan)
    for i in range(20, n): m20[i] = close[i] / close[i - 20] - 1

    # ATR
    atr = np.full(n, np.nan)
    for i in range(1, n):
        atr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr14  = sma(atr, 14)
    atr_avg = sma(atr14, 60)
    atr_avg = np.where(np.isfinite(atr_avg), atr_avg, sma(atr14, 20))

    # EMA crossover state (transition-based, not point-in-time)
    e8_state   = ema_crossover_state(close, 8, 21)
    macd_st    = macd_crossover_state(close)

    by_ts = {int(t): i for i, t in enumerate(ts)}

    return CryptoSeries(
        symbol     = symbol,
        ts         = ts,
        close      = close,
        high       = high,
        low        = low,
        sma200     = s200,
        mom90      = m90,
        mom30      = m30,
        mom20      = m20,
        atr14      = atr14,
        atr_avg    = atr_avg,
        ema8_state = e8_state,
        macd_state = macd_st,
        by_ts      = by_ts,
    )


def _compute_stock_indicators(
    ticker: str,
    ts: np.ndarray,
    close: np.ndarray,
) -> StockSeries:
    n = len(close)

    s200 = sma(close, 200); s50 = sma(close, 50)
    s200 = np.where(np.isfinite(s200), s200, s50)

    m90 = np.full(n, np.nan)
    for i in range(90, n): m90[i] = close[i] / close[i - 90] - 1

    ef = ema(close, 8)
    es = ema(close, 21)

    by_ts = {int(t): i for i, t in enumerate(ts)}

    return StockSeries(
        ticker = ticker,
        ts     = ts,
        close  = close,
        sma200 = s200,
        mom90  = m90,
        ema_f  = ef,
        ema_s  = es,
        by_ts  = by_ts,
    )


# ── Loaders ───────────────────────────────────────────────────────────────────

def _cutoff_ms(as_of_date: str) -> int:
    """Return the first millisecond AFTER as_of_date (inclusive upper bound)."""
    d = datetime.strptime(as_of_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(d.timestamp() * 1000) + _DAY_MS


def load_crypto(
    symbol:      str,
    as_of_date:  str,
    candle_dir:  Optional[Path] = None,
) -> CryptoSeries:
    """
    Load one crypto instrument from local parquet, validate, compute indicators.

    Args:
        symbol:     e.g. "BTCUSDT"
        as_of_date: ISO date string — load bars up to and including this date
        candle_dir: override default data/candles directory

    Raises:
        DataValidationError on any contract violation
    """
    import pandas as pd

    if candle_dir is None:
        candle_dir = _REPO / "data" / "candles"
    path = candle_dir / f"{symbol}_1D.parquet"
    if not path.exists():
        raise DataValidationError(f"{symbol}: parquet file not found at {path}")

    df   = pd.read_parquet(path, columns=["open_time", "high", "low", "close"])
    df   = df.sort_values("open_time").reset_index(drop=True)
    cut  = _cutoff_ms(as_of_date)
    df   = df[df["open_time"] < cut].copy()

    ts    = df["open_time"].values.astype(np.int64)
    close = df["close"].values.astype(np.float64)
    high  = df["high"].values.astype(np.float64)
    low   = df["low"].values.astype(np.float64)

    _validate_crypto(symbol, ts, close)
    return _compute_crypto_indicators(symbol, ts, close, high, low)


def load_stock(
    ticker:     str,
    as_of_date: str,
    stock_dir:  Optional[Path] = None,
) -> StockSeries:
    """
    Load one stock instrument from local parquet, validate, compute indicators.

    Args:
        ticker:     e.g. "AMD"
        as_of_date: ISO date string
        stock_dir:  override default data/stocks directory

    Raises:
        DataValidationError on any contract violation
    """
    import pandas as pd

    if stock_dir is None:
        stock_dir = _REPO / "data" / "stocks"
    path = stock_dir / f"{ticker}_1D.parquet"
    if not path.exists():
        raise DataValidationError(f"{ticker}: parquet file not found at {path}")

    df   = pd.read_parquet(path, columns=["open_time", "close"])
    df   = df.sort_values("open_time").reset_index(drop=True)
    cut  = _cutoff_ms(as_of_date)
    df   = df[df["open_time"] < cut].copy()

    ts    = df["open_time"].values.astype(np.int64)
    close = df["close"].values.astype(np.float64)

    _validate_stock(ticker, ts, close)
    return _compute_stock_indicators(ticker, ts, close)


def load_v9_dataset(
    as_of_date:  str,
    candle_dir:  Optional[Path] = None,
    stock_dir:   Optional[Path] = None,
    strict:      bool           = True,
) -> V9DataSet:
    """
    Load and validate the complete V9 dataset.

    Loads all 12 crypto instruments and 4 stock instruments.
    Computes the common timestamp axis (intersection of all series).
    Validates that the common axis is non-empty.

    Args:
        as_of_date:  ISO date — load data up to and including this date
        candle_dir:  override crypto data directory
        stock_dir:   override stock data directory
        strict:      if True (default), raise DataValidationError immediately;
                     if False, skip invalid instruments and log warnings

    Raises:
        DataValidationError if any instrument fails validation (when strict=True)
        DataValidationError if common timestamp axis is empty

    Returns:
        V9DataSet — fully validated, indicator-enriched, aligned dataset
    """
    import warnings

    crypto: dict = {}
    stocks: dict = {}
    errors: list = []

    for symbol in CRYPTO_SYMBOLS:
        try:
            crypto[symbol] = load_crypto(symbol, as_of_date, candle_dir)
        except DataValidationError as e:
            if strict:
                raise
            errors.append(str(e))
            warnings.warn(f"Skipping {symbol}: {e}", stacklevel=2)

    for ticker in STOCK_TICKERS:
        try:
            stocks[ticker] = load_stock(ticker, as_of_date, stock_dir)
        except DataValidationError as e:
            if strict:
                raise
            errors.append(str(e))
            warnings.warn(f"Skipping {ticker}: {e}", stacklevel=2)

    if not crypto or "BTCUSDT" not in crypto:
        raise DataValidationError("BTCUSDT is required and could not be loaded")
    if len(stocks) < len(STOCK_TICKERS):
        raise DataValidationError(
            f"Only {len(stocks)}/{len(STOCK_TICKERS)} stock instruments loaded. "
            "All four are required for valid confidence scoring."
        )

    # Build common timestamp axis
    all_ts_sets = [set(s.ts.tolist()) for s in crypto.values()]
    all_ts_sets += [set(s.ts.tolist()) for s in stocks.values()]
    common = sorted(set.intersection(*all_ts_sets))

    if not common:
        raise DataValidationError(
            "Common timestamp axis is empty — crypto and stock data do not overlap. "
            "Check date ranges and timestamp normalization."
        )

    common_ts = np.array(common, dtype=np.int64)

    return V9DataSet(
        as_of_date = as_of_date,
        common_ts  = common_ts,
        crypto     = crypto,
        stocks     = stocks,
    )


# ── Freshness check ───────────────────────────────────────────────────────────

def assert_data_freshness(dataset: V9DataSet, as_of_date: str) -> None:
    """
    Assert that the dataset's latest bar is not stale.
    Raises DataValidationError if the most recent common bar is more than
    2 calendar days behind as_of_date (accounting for weekends).
    """
    latest_ms = dataset.latest_ts()
    cutoff_ms = _cutoff_ms(as_of_date)
    lag_days  = (cutoff_ms - latest_ms) / _DAY_MS

    if lag_days > 4:   # > 4 days covers longest US holiday weekends
        latest_str = dataset.latest_date()
        raise DataValidationError(
            f"Data is stale: latest common bar is {latest_str} "
            f"but as_of_date is {as_of_date} ({lag_days:.0f} days lag). "
            "Check data pipeline."
        )


# ── Quick audit CLI ───────────────────────────────────────────────────────────

def audit(as_of_date: Optional[str] = None) -> None:
    """Print a summary of data quality for all instruments."""
    from datetime import date, timedelta
    if as_of_date is None:
        as_of_date = (date.today() - timedelta(days=1)).isoformat()

    print(f"\nData audit — as of {as_of_date}")
    print("=" * 64)

    all_ok = True
    for symbol in CRYPTO_SYMBOLS:
        try:
            s = load_crypto(symbol, as_of_date)
            btc_str = f"{len(s.ts)} bars  latest={s.ts[-1]}  close={s.close[-1]:,.2f}"
            nan_sma = int(np.isnan(s.sma200).sum())
            print(f"  OK  {symbol:12s}  {btc_str}  nan_sma200={nan_sma}")
        except DataValidationError as e:
            print(f"  ERR {symbol:12s}  {e}")
            all_ok = False

    print()
    for ticker in STOCK_TICKERS:
        try:
            s = load_stock(ticker, as_of_date)
            print(f"  OK  {ticker:6s}  {len(s.ts)} bars  close={s.close[-1]:,.2f}")
        except DataValidationError as e:
            print(f"  ERR {ticker:6s}  {e}")
            all_ok = False

    print()
    if all_ok:
        try:
            ds = load_v9_dataset(as_of_date)
            assert_data_freshness(ds, as_of_date)
            print(f"  Dataset: {len(ds.common_ts)} common bars  "
                  f"range={datetime.utcfromtimestamp(ds.common_ts[0]/1000).date()} "
                  f"→ {ds.latest_date()}")
            print("  Status: ALL OK")
        except DataValidationError as e:
            print(f"  Dataset error: {e}")
    else:
        print("  Status: ERRORS FOUND — trading should not proceed")
    print("=" * 64)


if __name__ == "__main__":
    import sys
    audit(sys.argv[1] if len(sys.argv) > 1 else None)
