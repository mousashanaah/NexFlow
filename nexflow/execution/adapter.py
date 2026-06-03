"""Execution adapter abstraction for NexFlow engines.

Three execution modes:
  BACKTEST      — no I/O; caller owns all state updates (used by backtest scripts)
  LOCAL_PAPER   — local simulation; fills/stops recorded in local JSON (no exchange)
  BITGET_PAPER  — real orders to Bitget Demo Trading via authenticated REST API

The strategy layer is identical across all three modes. Only the execution
adapter changes. Each engine daemon selects a mode at startup via --mode
or the NEXFLOW_EXEC_MODE environment variable.

Adapter contract (ExecutionAdapter ABC):
  on_entry(symbol, direction, size, entry_price, stop_price, atr) → EntryResult
  on_stop_update(symbol, direction, new_stop_price)               → None
  on_close(symbol, direction, size, exit_price, reason)           → None
  sync_position(symbol)                                           → ExchangePosition | None

BACKTEST mode is intentionally not an ExecutionAdapter subclass — it is
handled inline by the backtest script without any adapter overhead.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ExecMode(str, Enum):
    BACKTEST     = "BACKTEST"
    LOCAL_PAPER  = "LOCAL_PAPER"
    BITGET_PAPER = "BITGET_PAPER"

    @classmethod
    def from_env(cls) -> "ExecMode":
        raw = os.environ.get("NEXFLOW_EXEC_MODE", "LOCAL_PAPER").upper().strip()
        try:
            return cls(raw)
        except ValueError:
            raise ValueError(
                f"Unknown NEXFLOW_EXEC_MODE={raw!r}. "
                f"Valid values: {[m.value for m in cls]}"
            )


@dataclass
class EntryResult:
    """Outcome of an entry attempt."""
    accepted: bool
    fill_price: float
    fill_size:  float
    stop_order_id: Optional[str] = None  # set by BITGET_PAPER adapter
    note: str = ""


@dataclass
class ExchangePosition:
    """Position state as reported by the exchange (BITGET_PAPER only)."""
    symbol:      str
    direction:   str       # "long" | "short"
    size:        float
    entry_price: float
    unrealized_pnl: float = 0.0
    stop_order_id:  Optional[str] = None


class ExecutionAdapter(ABC):
    """Interface that all execution adapters must implement."""

    @abstractmethod
    def on_entry(
        self,
        symbol:      str,
        direction:   str,    # "long" | "short"
        size:        float,
        entry_price: float,
        stop_price:  float,
        atr:         float,
    ) -> EntryResult:
        """Execute an entry order and attach an initial stop.

        Returns an EntryResult. On rejection, accepted=False and the
        daemon skips adding the position to local state.
        """

    @abstractmethod
    def on_stop_update(
        self,
        symbol:         str,
        direction:      str,
        new_stop_price: float,
        stop_order_id:  Optional[str] = None,
    ) -> Optional[str]:
        """Update the trailing stop to a new price level.

        Returns the new stop_order_id (BITGET_PAPER: cancel + replace).
        LOCAL_PAPER: no-op, returns None.
        """

    @abstractmethod
    def on_close(
        self,
        symbol:     str,
        direction:  str,
        size:       float,
        exit_price: float,
        reason:     str,   # "stop" | "flip" | "manual"
    ) -> None:
        """Record or execute a position close."""

    @abstractmethod
    def sync_position(self, symbol: str) -> Optional[ExchangePosition]:
        """Query the exchange for the current position state.

        Returns None if flat or if the adapter does not support live queries
        (LOCAL_PAPER returns None always).
        """


# ── LOCAL_PAPER adapter ───────────────────────────────────────────────────────

class LocalPaperAdapter(ExecutionAdapter):
    """Simulates execution locally. No exchange calls. Fills at requested price."""

    def __init__(self, taker_fee: float = 0.0006) -> None:
        self._taker_fee = taker_fee

    def on_entry(self, symbol, direction, size, entry_price, stop_price, atr) -> EntryResult:
        return EntryResult(
            accepted=True,
            fill_price=entry_price,
            fill_size=size,
            note="local paper fill",
        )

    def on_stop_update(self, symbol, direction, new_stop_price, stop_order_id=None):
        return None  # stop is tracked only in local state JSON

    def on_close(self, symbol, direction, size, exit_price, reason):
        pass  # caller updates local state

    def sync_position(self, symbol):
        return None  # no exchange to query


# ── BITGET_PAPER adapter ──────────────────────────────────────────────────────

class BitgetPaperAdapter(ExecutionAdapter):
    """Sends real orders to Bitget Demo Trading (paptrading: 1 header)."""

    def __init__(self, client: "BitgetClient") -> None:  # type: ignore[name-defined]
        self._client = client

    def on_entry(self, symbol, direction, size, entry_price, stop_price, atr) -> EntryResult:
        from nexflow.exchange.bitget_order import (
            place_market_entry, place_stop, get_position
        )
        from nexflow.services.strategy.signal_models import Direction as Dir

        dir_enum = Dir.LONG if direction == "long" else Dir.SHORT
        try:
            resp = place_market_entry(self._client, symbol, dir_enum, size)
            fill_price = float(resp.get("price") or entry_price)
            fill_size  = float(resp.get("size")  or size)
        except Exception as exc:
            return EntryResult(accepted=False, fill_price=0.0, fill_size=0.0,
                               note=f"entry rejected: {exc}")

        stop_id: Optional[str] = None
        try:
            stop_resp = place_stop(self._client, symbol, dir_enum, stop_price)
            stop_id   = stop_resp.get("orderId") if stop_resp else None
        except Exception as exc:
            return EntryResult(accepted=True, fill_price=fill_price, fill_size=fill_size,
                               stop_order_id=None, note=f"stop placement failed: {exc}")

        return EntryResult(
            accepted=True,
            fill_price=fill_price,
            fill_size=fill_size,
            stop_order_id=stop_id,
            note="bitget paper fill",
        )

    def on_stop_update(self, symbol, direction, new_stop_price, stop_order_id=None):
        from nexflow.exchange.bitget_order import cancel_stop, place_stop
        from nexflow.services.strategy.signal_models import Direction as Dir

        dir_enum = Dir.LONG if direction == "long" else Dir.SHORT

        if stop_order_id:
            cancel_stop(self._client, symbol, stop_order_id)

        try:
            resp   = place_stop(self._client, symbol, dir_enum, new_stop_price)
            new_id = resp.get("orderId") if resp else None
            return new_id
        except Exception:
            return None

    def on_close(self, symbol, direction, size, exit_price, reason):
        # Send a real market close for any signal-driven exit.
        # Only skip if reason == "stop" (exchange already closed it via stop order).
        if reason == "stop":
            return
        from nexflow.exchange.bitget_order import close_market_position
        from nexflow.services.strategy.signal_models import Direction as Dir
        dir_enum = Dir.LONG if direction == "long" else Dir.SHORT
        try:
            close_market_position(self._client, symbol, dir_enum, size)
        except Exception as exc:
            print(f"  [ERROR] on_close {symbol} {direction}: {exc}")

    def sync_position(self, symbol):
        from nexflow.exchange.bitget_order import get_position, get_open_stops
        raw = get_position(self._client, symbol)
        if raw is None:
            return None

        hold_side = raw.get("holdSide", "")
        direction = "long" if hold_side == "long" else "short"
        size      = float(raw.get("total", 0.0))
        if size == 0:
            return None

        stop_id: Optional[str] = None
        try:
            from nexflow.services.strategy.signal_models import Direction as Dir
            dir_enum = Dir.LONG if direction == "long" else Dir.SHORT
            stops = get_open_stops(self._client, symbol)
            if stops:
                stop_id = stops[0].get("orderId")
        except Exception:
            pass

        return ExchangePosition(
            symbol=symbol,
            direction=direction,
            size=size,
            entry_price=float(raw.get("openPriceAvg", 0.0)),
            unrealized_pnl=float(raw.get("unrealizedPL", 0.0)),
            stop_order_id=stop_id,
        )


def build_adapter(mode: ExecMode) -> ExecutionAdapter:
    """Factory: return the correct adapter for the given mode."""
    if mode is ExecMode.LOCAL_PAPER:
        return LocalPaperAdapter()
    if mode is ExecMode.BITGET_PAPER:
        from nexflow.exchange.bitget_client import BitgetClient
        client = BitgetClient.from_env()
        if not client._paper:
            raise RuntimeError(
                "BITGET_PAPER mode requires BITGET_PAPER=1 in environment. "
                "Set it to prevent accidentally trading a live account."
            )
        return BitgetPaperAdapter(client)
    raise ValueError(f"build_adapter called with BACKTEST mode — handle inline")
