"""Process entrypoint — wire up logging, config, and market data service."""

from __future__ import annotations

import asyncio

from nexflow.config import get_config
from nexflow.logging import configure_logging, get_logger
from nexflow.models.market_state import MarketState
from nexflow.services.market_data import BitgetWSClient


async def _on_market_update(state: MarketState) -> None:
    log = get_logger(__name__)
    log.debug("market_update", symbol=state.symbol, mid=state.mid_price)


async def main() -> None:
    cfg = get_config()
    configure_logging(cfg.app.log_level)
    log = get_logger(__name__)

    log.info("nexflow.starting", env=cfg.app.env, symbols=cfg.market_data.symbols)

    client = BitgetWSClient(cfg)
    client.on_update(_on_market_update)

    try:
        await client.start()
    except KeyboardInterrupt:
        pass
    finally:
        await client.stop()
        log.info("nexflow.stopped")


if __name__ == "__main__":
    asyncio.run(main())
