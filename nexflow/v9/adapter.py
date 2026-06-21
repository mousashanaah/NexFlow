"""
V9 Confidence — Module 7: Exchange Adapter

This module is intentionally minimal.

Responsibilities — exactly four, no more:
  1. Read balances
  2. Submit orders
  3. Query fills
  4. Verify execution

No allocation logic.
No signal logic.
No scoring.
No state transitions.
No business decisions.

The adapter is the only module that talks to the exchange.
Everything above it stays unchanged when the adapter is swapped.

Components:
  ExchangeAdapter  — abstract base class defining the interface contract
  NullAdapter      — silent no-op for dry-run operation (no network)
  BitgetAdapter    — Bitget REST API implementation
  ExecutionBridge  — translates IntendedOrders to OrderRequests + fills

ExecutionBridge is the only place where IntendedOrder.delta_weight
becomes an exchange order size.  It contains arithmetic only — no
strategy decisions.

Environment variables:
  BITGET_API_KEY        — live credentials
  BITGET_API_SECRET
  BITGET_PASSPHRASE
  BITGET_PAPER=1        — routes to Bitget demo account (default: safe)
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from nexflow.v9.paper import IntendedOrder, CRYPTO_BOOK

# ── Minimum order size guard ──────────────────────────────────────────────────

MIN_ORDER_USDT = 5.0       # below this, skip — avoid exchange rejections
ORDER_FILL_TIMEOUT_S = 30  # seconds to wait for a market fill confirmation


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Balance:
    symbol:    str
    available: float   # quantity available for trading
    total:     float   # total quantity (including locked in open orders)
    usdt_value: float  # approximate USDT equivalent (if known)


@dataclass
class TickerSnapshot:
    symbol:     str
    last_price: float
    timestamp_ms: int


@dataclass
class OrderRequest:
    symbol:      str    # e.g. "BTCUSDT", "AMD"
    side:        str    # "buy" | "sell"
    size:        float  # quantity in base currency (BTC, shares, etc.)
    order_type:  str    = "market"
    client_order_id: Optional[str] = None


@dataclass
class FillResult:
    order_id:    str
    symbol:      str
    side:        str
    filled_size: float
    avg_price:   float
    filled_usdt: float
    status:      str   # "full" | "partial" | "cancelled" | "null_fill" | "pending"
    timestamp_ms: int
    note:        str   = ""


# ── Abstract interface ────────────────────────────────────────────────────────

class ExchangeAdapter(ABC):
    """
    Defines the complete exchange interface.

    All implementations must satisfy this contract exactly.
    No method may contain allocation logic, signal logic, or state decisions.
    """

    @abstractmethod
    def get_balances(self) -> dict[str, Balance]:
        """
        Return current account balances.

        Returns:
            dict mapping symbol → Balance.  Always includes "USDT" key
            if USDT is held.  Never raises on empty account.
        """

    @abstractmethod
    def get_price(self, symbol: str) -> float:
        """
        Return the current last trade price for a symbol.

        Raises:
            AdapterError if the symbol is unknown or the request fails.
        """

    @abstractmethod
    def submit_order(self, request: OrderRequest) -> str:
        """
        Submit a market or limit order.

        Returns:
            order_id (str) — use with query_fill() to confirm execution.

        Raises:
            AdapterError if the order is rejected.
        """

    @abstractmethod
    def query_fill(self, order_id: str) -> FillResult:
        """
        Query the fill status of a previously submitted order.

        Returns:
            FillResult with current fill state.

        Raises:
            AdapterError if order_id is unknown.
        """

    @abstractmethod
    def verify_execution(
        self,
        order_id:      str,
        expected_size: float,
        tolerance:     float = 0.01,
    ) -> bool:
        """
        Confirm that an order was fully filled within tolerance.

        Args:
            order_id:      as returned by submit_order()
            expected_size: target quantity
            tolerance:     fractional tolerance (0.01 = 1%)

        Returns:
            True if |filled_size - expected_size| / expected_size <= tolerance
        """


# ── Exceptions ────────────────────────────────────────────────────────────────

class AdapterError(RuntimeError):
    """Raised when the exchange rejects a request or returns an error."""


# ── NullAdapter — dry-run / testing ──────────────────────────────────────────

class NullAdapter(ExchangeAdapter):
    """
    Silent no-op adapter for dry-run and testing.

    All calls are logged and return safe, valid defaults.
    No network access.  No capital at risk.

    Use this adapter during the 30-day shadow deployment gate.
    Replace with BitgetAdapter only after assert_module7_gate() passes.
    """

    def __init__(self, portfolio_value: float = 5_000.0) -> None:
        self._portfolio_value = portfolio_value
        self._call_log: list[dict] = []

    @property
    def call_log(self) -> list[dict]:
        return list(self._call_log)

    def _log(self, method: str, **kwargs) -> None:
        self._call_log.append({
            "method":    method,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs,
        })

    def get_balances(self) -> dict[str, Balance]:
        self._log("get_balances")
        return {
            "USDT": Balance(
                symbol     = "USDT",
                available  = self._portfolio_value,
                total      = self._portfolio_value,
                usdt_value = self._portfolio_value,
            )
        }

    def get_price(self, symbol: str) -> float:
        self._log("get_price", symbol=symbol)
        # Return a nominal price — null adapter never uses this for real fills
        _nominal = {
            "BTCUSDT": 50_000.0,
            "AMD":     150.0,
            "GOOGL":   170.0,
            "MSTR":    300.0,
            "SPOT":     20.0,
        }
        return _nominal.get(symbol, 1.0)

    def submit_order(self, request: OrderRequest) -> str:
        order_id = f"NULL-{int(time.time() * 1000)}-{request.symbol}"
        self._log("submit_order", request=vars(request), order_id=order_id)
        return order_id

    def query_fill(self, order_id: str) -> FillResult:
        self._log("query_fill", order_id=order_id)
        return FillResult(
            order_id     = order_id,
            symbol       = "NULL",
            side         = "buy",
            filled_size  = 0.0,
            avg_price    = 0.0,
            filled_usdt  = 0.0,
            status       = "null_fill",
            timestamp_ms = int(time.time() * 1000),
            note         = "NullAdapter: no real fill",
        )

    def verify_execution(
        self,
        order_id:      str,
        expected_size: float,
        tolerance:     float = 0.01,
    ) -> bool:
        self._log("verify_execution", order_id=order_id, expected_size=expected_size)
        return True   # null adapter always "succeeds"


# ── BitgetAdapter — live exchange ─────────────────────────────────────────────

class BitgetAdapter(ExchangeAdapter):
    """
    Bitget REST API adapter.

    Reads BITGET_PAPER from environment.  If set to "1", routes to
    Bitget's simulated trading environment.

    This class contains only HTTP mechanics and Bitget-specific encoding.
    It has no knowledge of allocation, scoring, or portfolio weights.
    """

    _BASE_URL = "https://api.bitget.com"

    def __init__(
        self,
        api_key:    Optional[str] = None,
        api_secret: Optional[str] = None,
        passphrase: Optional[str] = None,
    ) -> None:
        self._api_key    = api_key    or os.environ.get("BITGET_API_KEY",    "")
        self._api_secret = api_secret or os.environ.get("BITGET_API_SECRET", "")
        self._passphrase = passphrase or os.environ.get("BITGET_PASSPHRASE", "")
        self._is_paper   = os.environ.get("BITGET_PAPER", "1") == "1"

        if not self._api_key:
            raise AdapterError(
                "BITGET_API_KEY not set.  "
                "Set env var or pass api_key= to BitgetAdapter()."
            )

    @property
    def is_paper(self) -> bool:
        return self._is_paper

    # ── Authentication ────────────────────────────────────────────────────────

    def _sign(
        self,
        timestamp: str,
        method:    str,
        path:      str,
        body:      str = "",
    ) -> str:
        """HMAC-SHA256 signature per Bitget API spec."""
        message = timestamp + method.upper() + path + body
        raw     = hmac.new(
            self._api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return base64.b64encode(raw).decode("utf-8")

    def _headers(self, method: str, path: str, body: str = "") -> dict:
        ts = str(int(time.time() * 1000))
        return {
            "ACCESS-KEY":       self._api_key,
            "ACCESS-SIGN":      self._sign(ts, method, path, body),
            "ACCESS-TIMESTAMP": ts,
            "ACCESS-PASSPHRASE": self._passphrase,
            "Content-Type":    "application/json",
            "locale":          "en-US",
            **({"paperId": "1"} if self._is_paper else {}),
        }

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        if params:
            path = path + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(
            self._BASE_URL + path,
            headers = self._headers("GET", path),
            method  = "GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
        except Exception as exc:
            raise AdapterError(f"GET {path} failed: {exc}") from exc
        if data.get("code") != "00000":
            raise AdapterError(
                f"Bitget error {data.get('code')}: {data.get('msg')}"
            )
        return data.get("data", {})

    def _post(self, path: str, payload: dict) -> dict:
        body = json.dumps(payload)
        req  = urllib.request.Request(
            self._BASE_URL + path,
            data    = body.encode("utf-8"),
            headers = self._headers("POST", path, body),
            method  = "POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
        except Exception as exc:
            raise AdapterError(f"POST {path} failed: {exc}") from exc
        if data.get("code") != "00000":
            raise AdapterError(
                f"Bitget error {data.get('code')}: {data.get('msg')}"
            )
        return data.get("data", {})

    # ── Interface implementation ───────────────────────────────────────────────

    def get_balances(self) -> dict[str, Balance]:
        """GET /api/v2/mix/account/accounts — USDT-M perpetuals."""
        data = self._get("/api/v2/mix/account/accounts", {"productType": "USDT-FUTURES"})
        balances: dict[str, Balance] = {}
        for item in data if isinstance(data, list) else []:
            sym = item.get("marginCoin", "")
            balances[sym] = Balance(
                symbol     = sym,
                available  = float(item.get("available", 0)),
                total      = float(item.get("equity", 0)),
                usdt_value = float(item.get("usdtEquity", 0)),
            )
        return balances

    def get_price(self, symbol: str) -> float:
        """GET /api/v2/mix/market/ticker — last price for a symbol."""
        data = self._get("/api/v2/mix/market/ticker", {
            "symbol":      symbol,
            "productType": "USDT-FUTURES",
        })
        try:
            return float(data["lastPr"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AdapterError(f"Could not parse price for {symbol}: {exc}") from exc

    def submit_order(self, request: OrderRequest) -> str:
        """POST /api/v2/mix/order/place-order."""
        payload = {
            "symbol":      request.symbol,
            "productType": "USDT-FUTURES",
            "marginMode":  "crossed",
            "marginCoin":  "USDT",
            "size":        str(request.size),
            "side":        request.side.lower(),
            "tradeSide":   "open",
            "orderType":   request.order_type,
        }
        if request.client_order_id:
            payload["clientOid"] = request.client_order_id
        data = self._post("/api/v2/mix/order/place-order", payload)
        order_id = data.get("orderId", "")
        if not order_id:
            raise AdapterError(f"No orderId in response: {data}")
        return str(order_id)

    def query_fill(self, order_id: str) -> FillResult:
        """GET /api/v2/mix/order/detail."""
        data = self._get("/api/v2/mix/order/detail", {
            "orderId":     order_id,
            "productType": "USDT-FUTURES",
        })
        status_map = {
            "filled":    "full",
            "partially_filled": "partial",
            "cancelled": "cancelled",
            "new":       "pending",
            "live":      "pending",
        }
        raw_status = data.get("state", "").lower()
        return FillResult(
            order_id     = order_id,
            symbol       = data.get("symbol", ""),
            side         = data.get("side", ""),
            filled_size  = float(data.get("baseVolume", 0)),
            avg_price    = float(data.get("priceAvg",  0)),
            filled_usdt  = float(data.get("quoteVolume", 0)),
            status       = status_map.get(raw_status, raw_status),
            timestamp_ms = int(data.get("cTime", 0)),
        )

    def verify_execution(
        self,
        order_id:      str,
        expected_size: float,
        tolerance:     float = 0.01,
    ) -> bool:
        fill = self.query_fill(order_id)
        if fill.status not in ("full", "partial"):
            return False
        if expected_size == 0.0:
            return True
        diff = abs(fill.filled_size - expected_size) / expected_size
        return diff <= tolerance


# ── ExecutionBridge — arithmetic translation only ─────────────────────────────

@dataclass
class BridgeResult:
    date:         str
    orders_sent:  int
    fills:        list[FillResult]
    skipped:      list[str]   # instruments skipped (below min size, HOLD, etc.)
    errors:       list[str]

    @property
    def all_succeeded(self) -> bool:
        return not self.errors and all(
            f.status in ("full", "null_fill") for f in self.fills
        )


class ExecutionBridge:
    """
    Translates IntendedOrder objects into exchange orders.

    This is the ONLY place where delta_weight becomes a size in base
    currency.  The arithmetic is:
      delta_usdt = delta_weight * portfolio_value
      size       = abs(delta_usdt) / price
      side       = "buy" if delta_weight > 0 else "sell"

    No allocation decisions.  No signal logic.  Arithmetic only.
    """

    def __init__(
        self,
        adapter:         ExchangeAdapter,
        portfolio_value: float,
    ) -> None:
        self._adapter         = adapter
        self._portfolio_value = portfolio_value

    def execute_orders(
        self,
        orders: list[IntendedOrder],
        date:   str,
    ) -> BridgeResult:
        """
        Translate IntendedOrders into exchange orders and collect fills.

        HOLD orders are skipped.
        Orders below MIN_ORDER_USDT are skipped.
        CRYPTO_BOOK is treated as BTCUSDT for price lookup.
        """
        fills:   list[FillResult] = []
        skipped: list[str]        = []
        errors:  list[str]        = []
        sent                       = 0

        for order in orders:
            if order.side == "HOLD":
                skipped.append(f"{order.instrument}:HOLD")
                continue

            delta_usdt = abs(order.delta_weight) * self._portfolio_value
            if delta_usdt < MIN_ORDER_USDT:
                skipped.append(
                    f"{order.instrument}:below_min({delta_usdt:.2f} USDT)"
                )
                continue

            # Map instrument to exchange symbol
            exchange_sym = (
                "BTCUSDT" if order.instrument == CRYPTO_BOOK
                else order.instrument
            )

            try:
                price = self._adapter.get_price(exchange_sym)
                size  = round(delta_usdt / price, 6)
                side  = "buy" if order.delta_weight > 0 else "sell"

                order_id = self._adapter.submit_order(OrderRequest(
                    symbol     = exchange_sym,
                    side       = side,
                    size       = size,
                    order_type = "market",
                ))
                sent += 1

                fill = self._adapter.query_fill(order_id)
                fills.append(fill)

            except AdapterError as exc:
                errors.append(f"{order.instrument}: {exc}")

        return BridgeResult(
            date        = date,
            orders_sent = sent,
            fills       = fills,
            skipped     = skipped,
            errors      = errors,
        )
