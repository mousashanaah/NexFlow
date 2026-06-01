#!/usr/bin/env python3
"""Generate synthetic daily OHLCV data for backtest engine validation.

Produces data/candles/{SYMBOL}_1D.parquet with realistic crypto-like
trending + choppy regimes. Used only when live API is unavailable.

NOT for production use — run download_daily_candles.py for real data.
"""

from __future__ import annotations

import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("[ERROR] pyarrow required")
    sys.exit(1)

_SCHEMA = pa.schema([
    pa.field("symbol",    pa.string()),
    pa.field("timeframe", pa.string()),
    pa.field("open_time", pa.int64()),
    pa.field("close_time", pa.int64()),
    pa.field("open",      pa.float64()),
    pa.field("high",      pa.float64()),
    pa.field("low",       pa.float64()),
    pa.field("close",     pa.float64()),
    pa.field("volume",    pa.float64()),
])

_DAY_MS = 86_400_000


def _generate(
    symbol: str,
    start_price: float,
    n_days: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)

    # Regime sequence: trending up, chop, trending down, chop, trending up
    # Approximates BTC 2021-2026 loosely
    regimes: list[tuple[int, float, float]] = [
        # (days, daily_drift, daily_vol)
        (180,  0.003,  0.040),   # 2021 bull
        (90,  -0.002,  0.050),   # 2021 correction
        (120,  0.004,  0.045),   # late 2021 run
        (200, -0.005,  0.055),   # 2022 bear
        (90,   0.001,  0.040),   # 2022 relief
        (120, -0.003,  0.060),   # late 2022 crash
        (180,  0.002,  0.035),   # 2023 recovery
        (150,  0.005,  0.045),   # 2024 bull
        (60,  -0.001,  0.040),   # 2024/25 consolidation
        (120,  0.003,  0.038),   # 2025 continuation
    ]

    bars = []
    open_price = start_price
    day = 0
    # Start: 2021-01-01
    base_ms = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

    for reg_days, drift, vol in regimes:
        for _ in range(min(reg_days, n_days - day)):
            if day >= n_days:
                break
            log_ret = drift + vol * rng.gauss(0, 1)
            close = open_price * math.exp(log_ret)
            close = max(close, 1.0)

            # Realistic OHLC within the bar
            intra_vol = vol * 0.6
            high  = close * math.exp(abs(rng.gauss(0, intra_vol)))
            low   = close * math.exp(-abs(rng.gauss(0, intra_vol)))
            high  = max(high, open_price, close)
            low   = min(low,  open_price, close)

            open_ms  = base_ms + day * _DAY_MS
            close_ms = open_ms + _DAY_MS - 1
            volume   = start_price * 1000 * (0.5 + rng.random())

            bars.append({
                "open_time":  open_ms,
                "close_time": close_ms,
                "open":       round(open_price, 4),
                "high":       round(high, 4),
                "low":        round(low, 4),
                "close":      round(close, 4),
                "volume":     round(volume, 2),
            })

            open_price = close
            day += 1

    return bars


def main() -> None:
    out_dir = _REPO_ROOT / "data" / "candles"
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("BTCUSDT", 29_000.0, 1200, 42),
        ("ETHUSDT",  1_000.0, 1200, 99),
    ]

    for symbol, start_price, n_days, seed in configs:
        bars = _generate(symbol, start_price, n_days, seed)
        path = out_dir / f"{symbol}_1D.parquet"
        table = pa.table({
            "symbol":    [symbol] * len(bars),
            "timeframe": ["1D"]   * len(bars),
            "open_time": [b["open_time"]  for b in bars],
            "close_time":[b["close_time"] for b in bars],
            "open":      [b["open"]   for b in bars],
            "high":      [b["high"]   for b in bars],
            "low":       [b["low"]    for b in bars],
            "close":     [b["close"]  for b in bars],
            "volume":    [b["volume"] for b in bars],
        }, schema=_SCHEMA)
        pq.write_table(table, path)
        print(f"  {symbol}: {len(bars)} synthetic bars → {path}")


if __name__ == "__main__":
    print("Generating synthetic daily data (for engine validation only) ...")
    main()
    print("Done. Run download_daily_candles.py for real data.")
