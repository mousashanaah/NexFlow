"""Bitget USDT-Perpetual Futures exchange constraints and execution helpers.

Architecture notes (NexFlow is Bitget USDT-Perp only):

One-way position mode:
    All positions are single-direction per symbol. The exchange rejects orders
    that would flip the position; all exits must use reduce_only=True.

Reduce-only semantics:
    Stop-loss and take-profit orders are reduce_only. The execution layer must
    tag these orders correctly; submitting a non-reduce-only exit when flat
    would open a new position in the opposite direction.

Tick / lot size:
    Prices must be rounded to price_precision decimal places.
    Quantities must be rounded DOWN (floor) to size_precision decimal places
    to avoid exceeding the desired risk budget.

Minimum notional:
    Bitget rejects orders whose notional value (price × size) is below
    min_notional. Position sizing must validate this after rounding.

Funding:
    USDT-Perp contracts pay/receive funding every 8 hours. Long positions pay
    when the rate is positive (spot > perp). This is a holding-time cost that
    the portfolio layer must track and deduct from PnL for honest accounting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from nexflow.services.strategy.signal_models import Direction


# ---------------------------------------------------------------------------
# Symbol constraints
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SymbolConstraints:
    symbol: str
    price_precision: int         # decimal places for limit/stop prices
    size_precision: int          # decimal places for order quantity
    min_order_qty: float         # minimum order size in base units
    min_notional: float          # minimum order value in USDT
    contract_multiplier: float = 1.0   # USDT-perp: 1 contract = 1 base unit
    max_leverage: int = 125
    taker_fee: float = 0.0006
    maker_fee: float = 0.0002


# Real Bitget USDT-Futures specs (as of 2024 — verify against API before live use)
SYMBOL_REGISTRY: dict[str, SymbolConstraints] = {
    "BTCUSDT": SymbolConstraints(
        "BTCUSDT", price_precision=1, size_precision=3,
        min_order_qty=0.001, min_notional=5.0, max_leverage=125,
    ),
    "ETHUSDT": SymbolConstraints(
        "ETHUSDT", price_precision=2, size_precision=2,
        min_order_qty=0.01, min_notional=5.0, max_leverage=100,
    ),
    "SOLUSDT": SymbolConstraints(
        "SOLUSDT", price_precision=3, size_precision=1,
        min_order_qty=0.1, min_notional=5.0, max_leverage=75,
    ),
    "BNBUSDT": SymbolConstraints(
        "BNBUSDT", price_precision=3, size_precision=2,
        min_order_qty=0.01, min_notional=5.0, max_leverage=50,
    ),
    "XRPUSDT": SymbolConstraints(
        "XRPUSDT", price_precision=5, size_precision=0,
        min_order_qty=1.0, min_notional=5.0, max_leverage=75,
    ),
    "ADAUSDT": SymbolConstraints(
        "ADAUSDT", price_precision=5, size_precision=0,
        min_order_qty=1.0, min_notional=5.0, max_leverage=75,
    ),
    "DOGEUSDT": SymbolConstraints(
        "DOGEUSDT", price_precision=6, size_precision=0,
        min_order_qty=1.0, min_notional=5.0, max_leverage=75,
    ),
    "AVAXUSDT": SymbolConstraints(
        "AVAXUSDT", price_precision=3, size_precision=1,
        min_order_qty=0.1, min_notional=5.0, max_leverage=75,
    ),
    "LINKUSDT": SymbolConstraints(
        "LINKUSDT", price_precision=4, size_precision=1,
        min_order_qty=0.1, min_notional=5.0, max_leverage=75,
    ),
    "LTCUSDT": SymbolConstraints(
        "LTCUSDT", price_precision=3, size_precision=2,
        min_order_qty=0.01, min_notional=5.0, max_leverage=75,
    ),
    "DOTUSDT": SymbolConstraints(
        "DOTUSDT", price_precision=4, size_precision=1,
        min_order_qty=0.1, min_notional=5.0, max_leverage=75,
    ),
    "TRXUSDT": SymbolConstraints(
        "TRXUSDT", price_precision=6, size_precision=0,
        min_order_qty=1.0, min_notional=5.0, max_leverage=75,
    ),
}


def get_constraints(symbol: str) -> SymbolConstraints:
    """Return constraints for a known symbol, or a safe default for unknown ones."""
    if symbol not in SYMBOL_REGISTRY:
        # Safe fallback: 3 decimal price, 1 decimal size, $5 min notional
        return SymbolConstraints(symbol, price_precision=3, size_precision=1,
                                 min_order_qty=0.1, min_notional=5.0)
    return SYMBOL_REGISTRY[symbol]


# ---------------------------------------------------------------------------
# Price and size rounding
# ---------------------------------------------------------------------------

def round_price(price: float, constraints: SymbolConstraints) -> float:
    """Round price to the symbol's tick precision (standard rounding)."""
    factor = 10 ** constraints.price_precision
    return round(price * factor) / factor


def round_size(size: float, constraints: SymbolConstraints) -> float:
    """Floor quantity to the symbol's lot precision (never exceed intended risk budget)."""
    factor = 10 ** constraints.size_precision
    return math.floor(size * factor) / factor


def enforce_min_notional(
    price: float, size: float, constraints: SymbolConstraints
) -> tuple[float, bool]:
    """Return (adjusted_size, is_valid).

    If size × price < min_notional, size is raised to meet the minimum.
    Returns is_valid=False if even the minimum order would be rejected by
    the exchange (e.g., insufficient equity).
    """
    if size <= 0 or price <= 0:
        return 0.0, False

    notional = price * size
    if notional >= constraints.min_notional:
        return size, True

    # Raise size to meet minimum — caller must check equity sufficiency
    min_size = constraints.min_notional / price
    adjusted = round_size(min_size + 10 ** -constraints.size_precision, constraints)
    return adjusted, True


def is_valid_order(price: float, size: float, constraints: SymbolConstraints) -> tuple[bool, str]:
    """Validate an order against all exchange constraints.

    Returns (valid, reason). reason is empty on success.
    """
    if size < constraints.min_order_qty:
        return False, f"size {size} below min_order_qty {constraints.min_order_qty}"
    notional = price * size
    if notional < constraints.min_notional:
        return False, f"notional {notional:.2f} below min_notional {constraints.min_notional}"
    return True, ""


# ---------------------------------------------------------------------------
# One-way position mode / reduce-only semantics
# ---------------------------------------------------------------------------

def is_reduce_only_safe(existing_direction: Direction, order_direction: Direction) -> bool:
    """Return True if an order in order_direction would reduce (not flip) the position.

    In one-way mode:
    - A LONG exit (SHORT order) reduces a LONG position ✓
    - A SHORT exit (LONG order) reduces a SHORT position ✓
    - A LONG order when SHORT → would flip the position ✗
    """
    if existing_direction is Direction.LONG:
        return order_direction is Direction.SHORT
    if existing_direction is Direction.SHORT:
        return order_direction is Direction.LONG
    return False  # FLAT has nothing to reduce


def required_reduce_only(existing_direction: Direction, order_direction: Direction) -> bool:
    """Return True if reduce_only=True must be set on this order.

    All closing orders in one-way mode should be reduce_only to prevent
    accidental position flips if the fill arrives after the position closes.
    """
    return is_reduce_only_safe(existing_direction, order_direction)


# ---------------------------------------------------------------------------
# Funding awareness (data model for future portfolio integration)
# ---------------------------------------------------------------------------

@dataclass
class FundingPayment:
    """A single funding settlement event.

    Funding occurs every 8 hours on Bitget USDT-Perp.
    Positive rate: longs pay shorts. Negative rate: shorts pay longs.

    payment = position_value × rate   (negative payment = received)
    """
    symbol: str
    timestamp: int          # epoch seconds of settlement
    rate: float             # e.g., 0.0001 = 0.01%
    position_value: float   # abs(size × mark_price) at settlement
    payment: float          # positive = cost for the holder (long in positive rate)


def estimate_funding_drag(
    position_value: float,
    hold_seconds: float,
    annual_funding_rate: float = 0.1095,   # ~0.01% × 3 per day × 365 = 10.95%
) -> float:
    """Estimate cumulative funding cost for a given hold time.

    Uses a simplified continuous funding model. For accurate accounting,
    use actual settlement records from the exchange.

    Returns the estimated total funding payment (positive = cost to holder).
    """
    hold_fraction = hold_seconds / (365.25 * 86_400)
    return position_value * annual_funding_rate * hold_fraction
