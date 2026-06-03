"""Thin order-management wrappers for Bitget USDT-Perpetual Futures.

Covers the four operations Engine #1 needs:
  place_market_entry  — open a new position (market order)
  place_stop          — attach a position-level stop-loss
  cancel_stop         — cancel the current stop before placing an updated one
  get_position        — query current position state from the exchange
  get_account_balance — query available USDT balance

All functions accept a BitgetClient and return plain Python dicts or None.
They never mutate local state — the caller (execution adapter) owns state.

One-way position mode is assumed throughout:
  - Entries use tradeSide="open"
  - Stops use reduce_only and are placed as TPSL orders
  - Position flips require a separate close + open sequence
"""

from __future__ import annotations

from typing import Any

from nexflow.exchange.bitget_client import BitgetClient
from nexflow.exchange.bitget_constraints import get_constraints, round_price, round_size
from nexflow.services.strategy.signal_models import Direction

_PRODUCT_TYPE = "USDT-FUTURES"
_MARGIN_COIN  = "USDT"

# ── Endpoints ────────────────────────────────────────────────────────────────

_EP_PLACE_ORDER     = "/api/v2/mix/order/place-order"
_EP_CLOSE_POSITIONS = "/api/v2/mix/order/close-positions"
_EP_PLACE_TPSL      = "/api/v2/mix/order/place-tpsl-order"
_EP_CANCEL_TPSL     = "/api/v2/mix/order/cancel-plan-order"
_EP_TPSL_LIST       = "/api/v2/mix/order/tpsl-order-list"
_EP_POSITION        = "/api/v2/mix/position/all-position"
_EP_ACCOUNT         = "/api/v2/mix/account/account"
_EP_SET_LEVERAGE    = "/api/v2/mix/account/set-leverage"

_TARGET_LEVERAGE    = "1"  # always trade at 1x — no leverage


# ── Leverage ──────────────────────────────────────────────────────────────────

def set_leverage(
    client: BitgetClient,
    symbol: str,
    hold_side: str = "long",  # "long" or "short"
) -> None:
    """Set leverage to 1x for this symbol before placing any order.

    Silently ignores errors (e.g. already set) so it never blocks an entry.
    """
    try:
        client.post(_EP_SET_LEVERAGE, {
            "symbol":      symbol,
            "productType": _PRODUCT_TYPE,
            "marginCoin":  _MARGIN_COIN,
            "leverage":    _TARGET_LEVERAGE,
            "holdSide":    hold_side,
        })
    except Exception:
        pass  # non-fatal — proceed with entry regardless


# ── Entry ─────────────────────────────────────────────────────────────────────

def place_market_entry(
    client: BitgetClient,
    symbol: str,
    direction: Direction,
    size: float,
) -> dict[str, Any]:
    """Place a market entry order. Returns the exchange order response dict.

    Size is floored to the symbol's lot size before submission.
    Raises BitgetAPIError on rejection.
    """
    constraints = get_constraints(symbol)
    size = round_size(size, constraints)

    hold_side  = "long" if direction is Direction.LONG else "short"
    set_leverage(client, symbol, hold_side)

    side      = "buy"  if direction is Direction.LONG  else "sell"
    trade_side = "open"

    body = {
        "symbol":      symbol,
        "productType": _PRODUCT_TYPE,
        "marginMode":  "crossed",
        "marginCoin":  _MARGIN_COIN,
        "size":        str(size),
        "side":        side,
        "tradeSide":   trade_side,
        "orderType":   "market",
    }
    return client.post(_EP_PLACE_ORDER, body)


def close_market_position(
    client: BitgetClient,
    symbol: str,
    direction: Direction,
    size: float,
) -> dict[str, Any]:
    """Flash-close the entire position for symbol using Bitget's close-positions endpoint.

    This endpoint closes the full position without needing to specify size,
    avoiding precision and parameter issues with place-order on closes.
    """
    hold_side = "long" if direction is Direction.LONG else "short"
    body = {
        "symbol":      symbol,
        "productType": _PRODUCT_TYPE,
        "holdSide":    hold_side,
    }
    return client.post(_EP_CLOSE_POSITIONS, body)


# ── Stop-loss ────────────────────────────────────────────────────────────────

def place_stop(
    client: BitgetClient,
    symbol: str,
    direction: Direction,
    stop_price: float,
) -> dict[str, Any]:
    """Attach a position-level stop-loss (TPSL order) for the entire position.

    Uses planType="pos_loss" which targets the whole position and is
    reduce_only by design. Returns the exchange response dict.
    """
    constraints = get_constraints(symbol)
    stop_price  = round_price(stop_price, constraints)

    hold_side = "long" if direction is Direction.LONG else "short"

    body = {
        "symbol":              symbol,
        "productType":         _PRODUCT_TYPE,
        "marginCoin":          _MARGIN_COIN,
        "planType":            "pos_loss",
        "triggerPrice":        str(stop_price),
        "triggerType":         "mark_price",
        "holdSide":            hold_side,
        "executePrice":        "0",   # 0 = market execution when triggered
    }
    return client.post(_EP_PLACE_TPSL, body)


def cancel_stop(
    client: BitgetClient,
    symbol: str,
    order_id: str,
) -> dict[str, Any] | None:
    """Cancel a TPSL stop-loss order by its orderId.

    Returns None if the order no longer exists (already triggered or cancelled).
    Does NOT raise on "order not found" — idempotent.
    """
    from nexflow.exchange.bitget_client import BitgetAPIError
    body = {
        "symbol":      symbol,
        "productType": _PRODUCT_TYPE,
        "orderId":     order_id,
    }
    try:
        return client.post(_EP_CANCEL_TPSL, body)
    except BitgetAPIError as exc:
        # 40768 = order not found / already cancelled
        if exc.code in ("40768", "40014"):
            return None
        raise


def get_open_stops(
    client: BitgetClient,
    symbol: str,
) -> list[dict]:
    """Return all open TPSL stop orders for this symbol."""
    data = client.get(_EP_TPSL_LIST, {
        "symbol":      symbol,
        "productType": _PRODUCT_TYPE,
        "planType":    "pos_loss",
    })
    if not data:
        return []
    return data if isinstance(data, list) else data.get("entrustedList", [])


# ── Position / account state ─────────────────────────────────────────────────

def get_position(
    client: BitgetClient,
    symbol: str,
) -> dict | None:
    """Return the current position dict for symbol, or None if flat.

    Exchange response fields include: holdSide, openPriceAvg, total, unrealizedPL.
    """
    data = client.get(_EP_POSITION, {
        "symbol":      symbol,
        "productType": _PRODUCT_TYPE,
        "marginCoin":  _MARGIN_COIN,
    })
    if not data:
        return None
    positions = data if isinstance(data, list) else [data]
    for pos in positions:
        if pos.get("symbol") == symbol and float(pos.get("total", 0)) > 0:
            return pos
    return None


def get_account_balance(client: BitgetClient) -> float:
    """Return available USDT balance from the futures account."""
    data = client.get(_EP_ACCOUNT, {
        "symbol":      "BTCUSDT",
        "productType": _PRODUCT_TYPE,
        "marginCoin":  _MARGIN_COIN,
    })
    if not data:
        return 0.0
    return float(data.get("available", 0.0))
