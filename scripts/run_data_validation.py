#!/usr/bin/env python3
"""Live market data transport validation.

Connects to Bitget, streams real-time futures data, and continuously validates
transport integrity. Prints heartbeat diagnostics every 30 seconds and writes
a persistent telemetry log to logs/market_telemetry.csv.

Usage:
    python scripts/run_data_validation.py
    python scripts/run_data_validation.py --symbols BTCUSDT,ETHUSDT
    python scripts/run_data_validation.py --heartbeat 60
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure the repo root is importable regardless of how the script is invoked
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from nexflow.config import get_config, NexFlowConfig
from nexflow.logging import configure_logging, get_logger
from nexflow.services.market_data.bitget_ws import BitgetWSClient
from nexflow.services.market_data.telemetry_monitor import TelemetryMonitor


async def run(cfg: NexFlowConfig, heartbeat_s: float) -> None:
    log = get_logger(__name__)

    client = BitgetWSClient(cfg)
    monitor = TelemetryMonitor(cfg, heartbeat_interval_s=heartbeat_s)
    monitor.attach(client)

    log.info(
        "validation.starting",
        symbols=cfg.market_data.symbols,
        product_type=cfg.market_data.product_type,
        ws_url=cfg.exchange.ws_url,
        heartbeat_interval_s=heartbeat_s,
    )

    await monitor.start()

    try:
        await client.start()
    except KeyboardInterrupt:
        pass
    finally:
        await client.stop()
        await monitor.stop()
        log.info("validation.done")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NexFlow market data transport validator")
    p.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated symbols, e.g. BTCUSDT,ETHUSDT (overrides config)",
    )
    p.add_argument(
        "--heartbeat",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="Heartbeat print interval in seconds (default: 30)",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity (default: from config)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Config is loaded once; apply CLI overrides before locking it in
    import os

    if args.symbols:
        import json
        os.environ["NEXFLOW_MARKET_DATA__SYMBOLS"] = json.dumps(
            [s.strip() for s in args.symbols.split(",") if s.strip()]
        )

    cfg = get_config()
    log_level = args.log_level or cfg.app.log_level
    configure_logging(log_level)

    try:
        asyncio.run(run(cfg, heartbeat_s=args.heartbeat))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
