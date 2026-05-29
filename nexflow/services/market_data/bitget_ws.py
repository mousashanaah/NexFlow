"""Bitget V2 WebSocket market data client.

Subscribes to orderbook, trade, and ticker channels for configured symbols
and keeps MarketState objects up to date.

Uses aiohttp for the WebSocket transport — aiohttp is already a dependency
and its WS client is stable across Python 3.11–3.14, unlike the websockets
library whose legacy asyncio.Protocol backend is broken on Python 3.14.

Usage:
    client = BitgetWSClient(config)
    await client.start()          # runs until cancelled
    state = client.get_state("BTCUSDT")
"""

from __future__ import annotations

import asyncio
import json
import time

from collections.abc import Callable, Coroutine
from typing import Any

import aiohttp

from nexflow.config import NexFlowConfig
from nexflow.logging import get_logger
from nexflow.models.market_state import MarketState, Trade, TradeSide


_log = get_logger(__name__)

# Bitget V2 channel names for futures
_CHANNEL_BOOKS = "books"
_CHANNEL_TRADE = "trade"
_CHANNEL_TICKER = "ticker"

StateCallback = Callable[[MarketState], Coroutine[Any, Any, None]]


class BitgetWSClient:
    """Async WebSocket client that maintains live MarketState for each symbol."""

    def __init__(self, config: NexFlowConfig, symbols: list[str] | None = None) -> None:
        self._cfg = config
        self._ex = config.exchange
        self._md = config.market_data
        self._symbols = symbols if symbols is not None else list(self._md.symbols)

        self._states: dict[str, MarketState] = {
            sym: MarketState(symbol=sym, product_type=self._md.product_type)
            for sym in self._symbols
        }

        self._callbacks: list[StateCallback] = []
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._running = False
        self._reconnect_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def reconnect_count(self) -> int:
        return self._reconnect_count

    def get_state(self, symbol: str) -> MarketState | None:
        return self._states.get(symbol)

    def on_update(self, callback: StateCallback) -> None:
        """Register an async callback invoked with the updated MarketState."""
        self._callbacks.append(callback)

    async def start(self) -> None:
        """Connect and run until cancelled or max reconnects exceeded."""
        self._running = True
        while self._running:
            try:
                await self._connect_and_run()
            except asyncio.CancelledError:
                _log.info("market_data.cancelled")
                break
            except Exception as exc:
                self._reconnect_count += 1
                if self._reconnect_count > self._ex.ws_max_reconnect_attempts:
                    _log.error(
                        "market_data.max_reconnects_exceeded",
                        attempts=self._reconnect_count,
                    )
                    raise
                delay = self._ex.ws_reconnect_delay
                _log.warning(
                    "market_data.reconnecting",
                    error=str(exc),
                    attempt=self._reconnect_count,
                    delay_s=delay,
                )
                await asyncio.sleep(delay)

    async def stop(self) -> None:
        self._running = False
        if self._ws is not None:
            await self._ws.close()

    # ------------------------------------------------------------------
    # Internal connection lifecycle
    # ------------------------------------------------------------------

    async def _connect_and_run(self) -> None:
        url = self._ex.ws_url
        _log.info("market_data.connecting", url=url)

        timeout = aiohttp.ClientTimeout(
            connect=self._ex.ws_connect_timeout,
            total=None,  # no total timeout — connection stays open indefinitely
        )

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.ws_connect(
                url,
                heartbeat=None,      # disable aiohttp's own ping; we send Bitget's text ping
                receive_timeout=None,
                autoclose=True,
                autoping=False,      # we handle pong ourselves
            ) as ws:
                self._ws = ws
                self._reconnect_count = 0
                _log.info("market_data.connected", url=url)

                await self._subscribe(ws)
                await asyncio.gather(
                    self._receive_loop(ws),
                    self._heartbeat_loop(ws),
                )

    async def _subscribe(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        args: list[dict[str, str]] = []
        inst_type = _product_type_to_inst_type(self._md.product_type)

        for symbol in self._symbols:
            for channel in (_CHANNEL_BOOKS, _CHANNEL_TRADE, _CHANNEL_TICKER):
                args.append({"instType": inst_type, "channel": channel, "instId": symbol})

        payload = {"op": "subscribe", "args": args}
        await ws.send_str(json.dumps(payload))
        _log.info(
            "market_data.subscribed",
            symbols=self._symbols,
            channels=list({a["channel"] for a in args}),
        )

    async def _receive_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        async for msg in ws:
            if not self._running:
                break
            if msg.type == aiohttp.WSMsgType.TEXT:
                await self._handle_message(ws, msg.data)
            elif msg.type == aiohttp.WSMsgType.BINARY:
                await self._handle_message(ws, msg.data)
            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                raise aiohttp.ClientConnectionError(
                    f"WebSocket closed: {msg.type} {msg.data}"
                )

    async def _heartbeat_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Send Bitget application-level 'ping' every 20 s to keep the connection alive."""
        while self._running:
            await asyncio.sleep(self._ex.ws_ping_interval)
            try:
                await ws.send_str("ping")
            except Exception:
                break

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    async def _handle_message(
        self, ws: aiohttp.ClientWebSocketResponse, raw: str | bytes
    ) -> None:
        # Bitget heartbeat: server sends "ping" → we reply "pong".
        # Our heartbeat_loop sends "ping" → server replies "pong" → discard.
        if raw in ("ping", b"ping"):
            await ws.send_str("pong")
            return
        if raw in ("pong", b"pong"):
            return

        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            _log.warning("market_data.bad_json", raw=str(raw)[:200])
            return

        # Subscription ack / error responses
        if "event" in msg:
            _log.debug("market_data.event", ws_event=msg.get("event"), msg=msg)
            return

        action = msg.get("action")
        arg = msg.get("arg", {})
        data = msg.get("data", [])
        channel = arg.get("channel", "")
        symbol = arg.get("instId", "")
        exchange_ts_ms: int = int(msg.get("ts", 0) or 0)

        state = self._states.get(symbol)
        if state is None:
            return

        if exchange_ts_ms:
            state.exchange_ts_ms = exchange_ts_ms

        try:
            if channel == _CHANNEL_BOOKS:
                self._handle_books(state, action, data)
            elif channel == _CHANNEL_TRADE:
                self._handle_trades(state, data)
            elif channel == _CHANNEL_TICKER:
                self._handle_ticker(state, data)
        except Exception as exc:
            _log.error("market_data.parse_error", channel=channel, symbol=symbol, error=str(exc))
            return

        await self._fire_callbacks(state)

    def _handle_books(self, state: MarketState, action: str | None, data: list[Any]) -> None:
        for entry in data:
            bids = [(float(p), float(s)) for p, s, *_ in entry.get("bids", [])]
            asks = [(float(p), float(s)) for p, s, *_ in entry.get("asks", [])]

            if action == "snapshot":
                bids = bids[: self._md.orderbook_depth]
                asks = asks[: self._md.orderbook_depth]
                state.apply_orderbook_snapshot(bids, asks)
            else:
                state.apply_orderbook_delta(bids, asks)
                state.bids = state.bids[: self._md.orderbook_depth]
                state.asks = state.asks[: self._md.orderbook_depth]

    def _handle_trades(self, state: MarketState, data: list[Any]) -> None:
        for entry in data:
            trade = Trade(
                trade_id=str(entry.get("tradeId", "")),
                price=float(entry.get("price", 0)),
                size=float(entry.get("size", 0)),
                side=TradeSide.BUY if entry.get("side", "").lower() == "buy" else TradeSide.SELL,
                timestamp_ms=int(entry.get("ts", int(time.time() * 1000))),
            )
            state.add_trade(trade, max_history=self._md.max_trade_history)

    def _handle_ticker(self, state: MarketState, data: list[Any]) -> None:
        for entry in data:
            state.apply_ticker(
                {
                    "open24h": float(entry.get("open24h", 0) or 0),
                    "high24h": float(entry.get("high24h", 0) or 0),
                    "low24h": float(entry.get("low24h", 0) or 0),
                    "close24h": float(entry.get("lastPr", 0) or 0),
                    "baseVolume": float(entry.get("baseVolume", 0) or 0),
                    "quoteVolume": float(entry.get("quoteVolume", 0) or 0),
                }
            )

    async def _fire_callbacks(self, state: MarketState) -> None:
        for cb in self._callbacks:
            try:
                await cb(state)
            except Exception as exc:
                _log.error("market_data.callback_error", error=str(exc))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _product_type_to_inst_type(product_type: str) -> str:
    mapping = {
        "USDT-FUTURES": "USDT-FUTURES",
        "COIN-FUTURES": "COIN-FUTURES",
        "USDC-FUTURES": "USDC-FUTURES",
    }
    return mapping.get(product_type.upper(), "USDT-FUTURES")
