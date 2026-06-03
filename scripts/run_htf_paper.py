#!/usr/bin/env python3
"""Paper-trade the 4H HTF trend strategy (mechanism #7d) in replay mode.

Replays cached 4H candles through HTFTrendStrategy → LiveSignalRouter.
Produces the same trade log, equity curve, and performance metrics as the
paper trader, without connecting to any exchange.

Usage:
    python scripts/run_htf_paper.py
    python scripts/run_htf_paper.py --symbols SOLUSDT TRXUSDT
    python scripts/run_htf_paper.py --capital 50000 --from 2024-01-01

Validated universe: BTCUSDT ETHUSDT SOLUSDT TRXUSDT  (regime_ma=200)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    import pyarrow.parquet as pq
except ImportError:
    print("[ERROR] pyarrow required: pip install pyarrow")
    sys.exit(1)

from nexflow.services.candles.candle_engine import Candle
from nexflow.services.paper_trading.paper_trader import PaperTrader, PaperTraderConfig
from nexflow.services.strategy.htf_trend_strategy import HTFTrendStrategy
from nexflow.services.strategy.risk_engine import RiskConfig

_DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "TRXUSDT"]
_CANDLE_DIR = _REPO_ROOT / "data" / "candles"


def _load_4h_candles(symbols: list[str], from_ts: int, to_ts: int) -> list[Candle]:
    """Load 1H parquet and aggregate to 4H candles, sorted by close_time."""
    candles: list[Candle] = []
    _HOUR_MS = 3_600_000

    for symbol in symbols:
        path = _CANDLE_DIR / f"{symbol}_1H.parquet"
        if not path.exists():
            print(f"[WARN] Missing 1H data for {symbol}: {path}")
            continue
        tbl = pq.read_table(path, columns=["open_time", "open", "high", "low", "close", "volume"])
        rows = sorted(zip(
            tbl.column("open_time").to_pylist(),
            tbl.column("open").to_pylist(),
            tbl.column("high").to_pylist(),
            tbl.column("low").to_pylist(),
            tbl.column("close").to_pylist(),
            tbl.column("volume").to_pylist(),
        ))

        # Aggregate into 4H buckets
        bucket: list = []
        for ot, o, h, l, c, v in rows:
            if ot < from_ts or ot > to_ts:
                continue
            # 4H bucket: align to 4-hour boundary
            bucket_start = (ot // (4 * _HOUR_MS)) * (4 * _HOUR_MS)
            if bucket and bucket[0] != bucket_start:
                # Emit previous bucket
                candles.append(_emit_4h(symbol, bucket[0], bucket[1:]))
                bucket = []
            if not bucket:
                bucket = [bucket_start, o, h, l, c, v]
            else:
                bucket[2] = max(bucket[2], h)   # high
                bucket[3] = min(bucket[3], l)   # low
                bucket[4] = c                    # close
                bucket[5] += v                   # volume
        if len(bucket) > 1:
            candles.append(_emit_4h(symbol, bucket[0], bucket[1:]))

    return sorted(candles, key=lambda c: c.close_time)


def _emit_4h(symbol: str, bucket_start: int, ohlcv: list) -> Candle:
    _HOUR_MS = 3_600_000
    o, h, l, c, v = ohlcv
    return Candle(
        symbol=symbol,
        timeframe="4H",
        open_time=bucket_start,
        close_time=bucket_start + 4 * _HOUR_MS - 1,
        open=o, high=h, low=l, close=c, volume=v,
        is_final=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=_DEFAULT_SYMBOLS)
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--from",    dest="from_date", default="2021-01-01")
    parser.add_argument("--to",      dest="to_date",   default=None)
    parser.add_argument("--risk-pct", type=float, default=0.01,
                        help="Fraction of equity to risk per trade (default 0.01 = 1%%)")
    args = parser.parse_args()

    from_dt = datetime.strptime(args.from_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    to_dt   = (
        datetime.strptime(args.to_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.to_date else datetime.now(timezone.utc)
    )
    from_ts = int(from_dt.timestamp() * 1000)
    to_ts   = int(to_dt.timestamp() * 1000)

    print(f"HTF Trend Paper Trader (replay mode)")
    print(f"Symbols  : {args.symbols}")
    print(f"Capital  : ${args.capital:,.0f}")
    print(f"Period   : {args.from_date} → {args.to_date or 'today'}")
    print(f"Risk/trade: {args.risk_pct*100:.1f}%")
    print()

    print("Loading 4H candles ...")
    candles = _load_4h_candles(args.symbols, from_ts, to_ts)
    print(f"  {len(candles):,} 4H bars across {len(args.symbols)} symbols")
    print()

    # run_replay expects {symbol: {timeframe: [Candle]}}
    from collections import defaultdict
    candle_map: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for c in candles:
        candle_map[c.symbol][c.timeframe].append(c)

    strategy = HTFTrendStrategy()
    cfg = PaperTraderConfig(
        initial_equity=args.capital,
        risk=RiskConfig(risk_per_trade=args.risk_pct),
        enable_dashboard=False,
    )
    trader = PaperTrader(cfg=cfg, symbols=args.symbols, strategy=strategy)
    state = trader.run_replay(dict(candle_map))

    print()
    print(f"Replay complete:")
    print(f"  Equity  : ${state.equity:,.0f}")
    print(f"  Trades  : {state.total_trades}")
    print(f"  Win rate: {state.win_rate*100:.1f}%")
    print(f"  DD      : {state.drawdown*100:.1f}%")


if __name__ == "__main__":
    main()
