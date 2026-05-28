#!/usr/bin/env python3
"""Live candle engine — connects to Bitget, prints finalized candles, persists to parquet.

Usage:
    python scripts/run_candle_engine.py
    python scripts/run_candle_engine.py --symbols BTCUSDT,ETHUSDT
    python scripts/run_candle_engine.py --timeframes 1m,5m
    python scripts/run_candle_engine.py --data-dir /tmp/candles --log-level DEBUG
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from nexflow.config import get_config, NexFlowConfig
from nexflow.logging import configure_logging, get_logger
from nexflow.services.candles.candle_engine import Candle, CandleEngine, CandleStore, TIMEFRAMES
from nexflow.services.market_data.bitget_ws import BitgetWSClient


def _fmt_candle(c: Candle) -> str:
    return (
        f"[{c.timeframe}] {c.symbol} "
        f"O={c.open:.4f} H={c.high:.4f} L={c.low:.4f} C={c.close:.4f} "
        f"V={c.volume:.4f} VWAP={c.vwap:.4f} "
        f"buys={c.buy_volume:.4f} sells={c.sell_volume:.4f} "
        f"trades={c.trade_count} vol%={c.volatility_estimate*100:.3f}% "
        f"spread_avg={c.spread_avg:.4f} spread_max={c.spread_max:.4f}"
    )


async def run(cfg: NexFlowConfig, data_dir: Path, timeframe_filter: set[str]) -> None:
    log = get_logger(__name__)

    store = CandleStore(data_dir)
    engine = CandleEngine(cfg, store=store)

    async def on_candle_close(symbol: str, timeframe: str, candle: Candle) -> None:
        if timeframe_filter and timeframe not in timeframe_filter:
            return
        print(_fmt_candle(candle), flush=True)

    engine.on_candle_close(on_candle_close)

    client = BitgetWSClient(cfg)
    engine.attach(client)

    log.info(
        "candle_engine.starting",
        symbols=cfg.market_data.symbols,
        timeframes=sorted(TIMEFRAMES.keys()),
        data_dir=str(data_dir),
    )

    try:
        await client.start()
    except KeyboardInterrupt:
        pass
    finally:
        log.info("candle_engine.stopping")
        await engine.stop()
        await client.stop()
        log.info("candle_engine.stopped")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NexFlow candle engine")
    p.add_argument("--symbols", type=str, default=None,
                   help="Comma-separated symbols, e.g. BTCUSDT,ETHUSDT")
    p.add_argument("--timeframes", type=str, default=None,
                   help="Comma-separated timeframes to print, e.g. 1m,5m (default: all)")
    p.add_argument("--data-dir", type=Path, default=Path("data/candles"),
                   help="Directory for parquet output (default: data/candles)")
    p.add_argument("--log-level", type=str, default=None,
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    import json, os

    if args.symbols:
        os.environ["NEXFLOW_MARKET_DATA__SYMBOLS"] = json.dumps(
            [s.strip() for s in args.symbols.split(",") if s.strip()]
        )

    cfg = get_config()
    configure_logging(args.log_level or cfg.app.log_level)

    tf_filter: set[str] = set()
    if args.timeframes:
        tf_filter = {tf.strip() for tf in args.timeframes.split(",") if tf.strip()}
        unknown = tf_filter - set(TIMEFRAMES)
        if unknown:
            print(f"Unknown timeframes: {unknown}. Valid: {set(TIMEFRAMES)}", file=sys.stderr)
            sys.exit(1)

    try:
        asyncio.run(run(cfg, args.data_dir, tf_filter))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
