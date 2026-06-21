"""
V9 Exchange Adapter — Module 7 tests.

Test classes:
  TestBalance, TestOrderRequest, TestFillResult  — dataclass construction
  TestNullAdapter            — full NullAdapter contract coverage
  TestBitgetAdapterStructure — init, sign, header structure (no live calls)
  TestExecutionBridge        — order translation, size arithmetic, skips
  TestAdapterIsolation       — hard rule: no strategy logic in adapter.py
"""
from __future__ import annotations

import hashlib
import hmac
import base64
import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO))

from nexflow.v9.adapter import (
    AdapterError,
    Balance,
    BitgetAdapter,
    BridgeResult,
    ExecutionBridge,
    FillResult,
    MIN_ORDER_USDT,
    NullAdapter,
    OrderRequest,
    TickerSnapshot,
)
from nexflow.v9.paper import CRYPTO_BOOK, IntendedOrder


# ── Data class construction ───────────────────────────────────────────────────

class TestBalance:
    def test_construction(self):
        b = Balance(symbol="USDT", available=1000.0, total=1000.0, usdt_value=1000.0)
        assert b.symbol == "USDT"
        assert b.available == 1000.0

    def test_zero_balance(self):
        b = Balance(symbol="BTC", available=0.0, total=0.0, usdt_value=0.0)
        assert b.total == 0.0


class TestOrderRequest:
    def test_default_order_type_is_market(self):
        r = OrderRequest(symbol="BTCUSDT", side="buy", size=0.001)
        assert r.order_type == "market"

    def test_optional_client_order_id(self):
        r = OrderRequest(symbol="BTCUSDT", side="sell", size=0.001)
        assert r.client_order_id is None

    def test_limit_order_type(self):
        r = OrderRequest(symbol="BTCUSDT", side="buy", size=0.001, order_type="limit")
        assert r.order_type == "limit"


class TestFillResult:
    def test_construction(self):
        f = FillResult(
            order_id="123", symbol="BTCUSDT", side="buy",
            filled_size=0.001, avg_price=50000.0, filled_usdt=50.0,
            status="full", timestamp_ms=1700000000000,
        )
        assert f.status == "full"
        assert f.filled_usdt == 50.0

    def test_default_note_empty(self):
        f = FillResult(
            order_id="x", symbol="X", side="buy",
            filled_size=0, avg_price=0, filled_usdt=0,
            status="pending", timestamp_ms=0,
        )
        assert f.note == ""


# ── NullAdapter ───────────────────────────────────────────────────────────────

class TestNullAdapter:
    @pytest.fixture
    def adapter(self):
        return NullAdapter(portfolio_value=5_000.0)

    def test_get_balances_returns_usdt(self, adapter):
        balances = adapter.get_balances()
        assert "USDT" in balances
        assert balances["USDT"].available == 5_000.0

    def test_get_balances_usdt_value_matches_portfolio(self, adapter):
        b = adapter.get_balances()["USDT"]
        assert b.usdt_value == 5_000.0

    def test_get_price_btc(self, adapter):
        price = adapter.get_price("BTCUSDT")
        assert price > 0

    def test_get_price_stock(self, adapter):
        price = adapter.get_price("AMD")
        assert price > 0

    def test_get_price_unknown_returns_one(self, adapter):
        price = adapter.get_price("UNKNOWN")
        assert price == 1.0

    def test_submit_order_returns_order_id(self, adapter):
        r  = OrderRequest(symbol="BTCUSDT", side="buy", size=0.001)
        oid = adapter.submit_order(r)
        assert isinstance(oid, str)
        assert "NULL" in oid

    def test_submit_order_id_contains_symbol(self, adapter):
        r   = OrderRequest(symbol="BTCUSDT", side="buy", size=0.001)
        oid = adapter.submit_order(r)
        assert "BTCUSDT" in oid

    def test_query_fill_returns_null_fill(self, adapter):
        fill = adapter.query_fill("any-order-id")
        assert fill.status == "null_fill"

    def test_query_fill_order_id_preserved(self, adapter):
        fill = adapter.query_fill("order-abc")
        assert fill.order_id == "order-abc"

    def test_verify_execution_always_true(self, adapter):
        result = adapter.verify_execution("any", expected_size=1.0)
        assert result is True

    def test_verify_execution_zero_size(self, adapter):
        result = adapter.verify_execution("any", expected_size=0.0)
        assert result is True

    def test_call_log_records_all_calls(self, adapter):
        adapter.get_balances()
        adapter.get_price("BTCUSDT")
        adapter.submit_order(OrderRequest("BTCUSDT", "buy", 0.001))
        log = adapter.call_log
        methods = [e["method"] for e in log]
        assert "get_balances" in methods
        assert "get_price"    in methods
        assert "submit_order" in methods

    def test_call_log_immutable(self, adapter):
        adapter.get_balances()
        log = adapter.call_log
        log.append({"injected": True})
        assert len(adapter.call_log) == 1  # internal log unaffected

    def test_call_log_has_timestamp(self, adapter):
        adapter.get_balances()
        assert "timestamp" in adapter.call_log[0]

    def test_null_adapter_no_network(self, adapter):
        """NullAdapter must never make HTTP calls."""
        import socket
        original_connect = socket.socket.connect

        def fail_connect(*args, **kwargs):
            raise AssertionError("NullAdapter made a network call")

        socket.socket.connect = fail_connect
        try:
            adapter.get_balances()
            adapter.get_price("BTCUSDT")
            adapter.submit_order(OrderRequest("BTCUSDT", "buy", 0.001))
            adapter.query_fill("test-id")
            adapter.verify_execution("test-id", 0.001)
        finally:
            socket.socket.connect = original_connect

    def test_satisfies_adapter_interface(self, adapter):
        """NullAdapter must implement all abstract methods."""
        from nexflow.v9.adapter import ExchangeAdapter
        assert isinstance(adapter, ExchangeAdapter)


# ── BitgetAdapter — structure and auth (no live calls) ────────────────────────

class TestBitgetAdapterStructure:
    @pytest.fixture
    def adapter(self):
        return BitgetAdapter(
            api_key    = "test-key",
            api_secret = "test-secret",
            passphrase = "test-pass",
        )

    def test_init_reads_credentials(self, adapter):
        assert adapter._api_key    == "test-key"
        assert adapter._api_secret == "test-secret"
        assert adapter._passphrase == "test-pass"

    def test_init_defaults_to_paper_mode(self, monkeypatch):
        monkeypatch.delenv("BITGET_PAPER", raising=False)
        # Default is paper (BITGET_PAPER not set → use "1" default)
        a = BitgetAdapter(api_key="k", api_secret="s", passphrase="p")
        assert a.is_paper is True

    def test_paper_mode_from_env(self, monkeypatch):
        monkeypatch.setenv("BITGET_PAPER", "1")
        a = BitgetAdapter(api_key="k", api_secret="s", passphrase="p")
        assert a.is_paper is True

    def test_live_mode_when_paper_unset(self, monkeypatch):
        monkeypatch.setenv("BITGET_PAPER", "0")
        a = BitgetAdapter(api_key="k", api_secret="s", passphrase="p")
        assert a.is_paper is False

    def test_init_raises_without_credentials(self, monkeypatch):
        monkeypatch.delenv("BITGET_API_KEY", raising=False)
        with pytest.raises(AdapterError, match="BITGET_API_KEY"):
            BitgetAdapter()

    def test_sign_is_hmac_sha256_base64(self, adapter):
        """Verify the signing algorithm matches Bitget spec."""
        ts       = "1700000000000"
        method   = "GET"
        path     = "/api/v2/mix/account/accounts"
        body     = ""
        message  = ts + method + path + body
        expected = base64.b64encode(
            hmac.new(
                b"test-secret",
                message.encode("utf-8"),
                hashlib.sha256,
            ).digest()
        ).decode("utf-8")
        assert adapter._sign(ts, method, path, body) == expected

    def test_sign_includes_body(self, adapter):
        ts  = "1700000000000"
        s1  = adapter._sign(ts, "POST", "/path", "")
        s2  = adapter._sign(ts, "POST", "/path", '{"key":"val"}')
        assert s1 != s2

    def test_headers_contain_required_fields(self, adapter):
        h = adapter._headers("GET", "/test")
        assert "ACCESS-KEY"       in h
        assert "ACCESS-SIGN"      in h
        assert "ACCESS-TIMESTAMP" in h
        assert "ACCESS-PASSPHRASE" in h
        assert h["Content-Type"] == "application/json"

    def test_paper_mode_adds_paper_header(self, adapter):
        # adapter defaults to paper mode
        h = adapter._headers("GET", "/test")
        assert "paperId" in h

    def test_live_mode_no_paper_header(self, monkeypatch):
        monkeypatch.setenv("BITGET_PAPER", "0")
        a = BitgetAdapter(api_key="k", api_secret="s", passphrase="p")
        h = a._headers("GET", "/test")
        assert "paperId" not in h

    def test_satisfies_adapter_interface(self, adapter):
        from nexflow.v9.adapter import ExchangeAdapter
        assert isinstance(adapter, ExchangeAdapter)

    def test_get_price_raises_adapter_error_on_network_failure(self, adapter):
        """Any HTTP failure must raise AdapterError, not urllib error."""
        with patch.object(adapter, "_get", side_effect=AdapterError("timeout")):
            with pytest.raises(AdapterError):
                adapter.get_price("BTCUSDT")

    def test_submit_order_raises_on_rejection(self, adapter):
        with patch.object(adapter, "_post", side_effect=AdapterError("insufficient funds")):
            with pytest.raises(AdapterError):
                adapter.submit_order(OrderRequest("BTCUSDT", "buy", 0.001))


# ── ExecutionBridge ───────────────────────────────────────────────────────────

def _order(instrument: str, side: str, delta: float, portfolio: float = 5_000.0) -> IntendedOrder:
    return IntendedOrder(
        instrument     = instrument,
        side           = side,
        target_weight  = 0.65 if side == "BUY" else 0.0,
        current_weight = 0.0  if side == "BUY" else 0.65,
        delta_weight   = delta,
        notional_delta = round(delta * portfolio, 2),
    )


class TestExecutionBridge:
    @pytest.fixture
    def null_adapter(self):
        return NullAdapter(portfolio_value=5_000.0)

    @pytest.fixture
    def bridge(self, null_adapter):
        return ExecutionBridge(null_adapter, portfolio_value=5_000.0)

    def test_hold_orders_skipped(self, bridge):
        orders = [_order("AMD", "HOLD", 0.0)]
        orders[0] = IntendedOrder("AMD", "HOLD", 0.10, 0.10, 0.0, 0.0)
        result = bridge.execute_orders(orders, "2024-03-01")
        assert result.orders_sent == 0
        assert any("HOLD" in s for s in result.skipped)

    def test_below_min_usdt_skipped(self, bridge):
        # delta_weight=0.001 → delta_usdt = 0.001 * 5000 = 5.0 → right at boundary
        tiny = IntendedOrder("AMD", "BUY", 0.001, 0.0, 0.0001, 0.50)
        result = bridge.execute_orders([tiny], "2024-03-01")
        assert result.orders_sent == 0
        assert len(result.skipped) == 1

    def test_valid_order_sent_to_adapter(self, bridge, null_adapter):
        order = _order(CRYPTO_BOOK, "BUY", 0.65)
        result = bridge.execute_orders([order], "2024-03-01")
        assert result.orders_sent == 1
        assert len(result.fills) == 1

    def test_crypto_book_mapped_to_btcusdt(self, null_adapter):
        bridge = ExecutionBridge(null_adapter, portfolio_value=5_000.0)
        order  = _order(CRYPTO_BOOK, "BUY", 0.65)
        result = bridge.execute_orders([order], "2024-03-01")
        # Verify get_price was called with BTCUSDT (logged in NullAdapter)
        price_calls = [
            e for e in null_adapter.call_log if e["method"] == "get_price"
        ]
        assert any(e.get("symbol") == "BTCUSDT" for e in price_calls)

    def test_size_arithmetic_correct(self):
        """size = abs(delta_weight * portfolio) / price."""
        price     = 50_000.0
        delta_w   = 0.65
        portfolio = 10_000.0
        expected_size = round((delta_w * portfolio) / price, 6)

        adapter = NullAdapter(portfolio_value=portfolio)
        with patch.object(adapter, "get_price", return_value=price) as mock_price:
            bridge = ExecutionBridge(adapter, portfolio_value=portfolio)
            order  = _order(CRYPTO_BOOK, "BUY", delta_w, portfolio)
            bridge.execute_orders([order], "2024-03-01")

        # Check submitted order size
        order_calls = [e for e in adapter.call_log if e["method"] == "submit_order"]
        assert len(order_calls) == 1
        req = order_calls[0]["request"]
        assert abs(req["size"] - expected_size) < 1e-6

    def test_sell_order_side_is_sell(self):
        adapter = NullAdapter(portfolio_value=5_000.0)
        bridge  = ExecutionBridge(adapter, portfolio_value=5_000.0)
        order   = _order("AMD", "SELL", -0.10)
        bridge.execute_orders([order], "2024-03-01")
        order_calls = [e for e in adapter.call_log if e["method"] == "submit_order"]
        assert order_calls[0]["request"]["side"] == "sell"

    def test_adapter_error_captured_not_raised(self):
        bad_adapter = NullAdapter()
        with patch.object(bad_adapter, "submit_order", side_effect=AdapterError("rejected")):
            bridge = ExecutionBridge(bad_adapter, portfolio_value=5_000.0)
            order  = _order(CRYPTO_BOOK, "BUY", 0.65)
            result = bridge.execute_orders([order], "2024-03-01")
        assert len(result.errors) == 1
        assert "rejected" in result.errors[0]
        assert result.orders_sent == 0

    def test_all_succeeded_true_for_null_fills(self, bridge):
        order  = _order(CRYPTO_BOOK, "BUY", 0.65)
        result = bridge.execute_orders([order], "2024-03-01")
        assert result.all_succeeded is True

    def test_all_succeeded_false_with_errors(self):
        adapter = NullAdapter(portfolio_value=5_000.0)
        with patch.object(adapter, "submit_order", side_effect=AdapterError("fail")):
            bridge = ExecutionBridge(adapter, portfolio_value=5_000.0)
            order  = _order(CRYPTO_BOOK, "BUY", 0.65)
            result = bridge.execute_orders([order], "2024-03-01")
        assert result.all_succeeded is False

    def test_multiple_orders_processed(self, bridge):
        orders = [
            _order(CRYPTO_BOOK, "BUY",  0.65),
            _order("AMD",        "BUY",  0.0875),
            _order("GOOGL",      "BUY",  0.0875),
        ]
        result = bridge.execute_orders(orders, "2024-03-01")
        assert result.orders_sent == 3

    def test_result_date_preserved(self, bridge):
        order  = _order(CRYPTO_BOOK, "BUY", 0.65)
        result = bridge.execute_orders([order], "2024-03-15")
        assert result.date == "2024-03-15"

    def test_min_order_usdt_constant_is_positive(self):
        assert MIN_ORDER_USDT > 0


# ── Isolation: no strategy logic in adapter.py ───────────────────────────────

class TestAdapterIsolation:
    def test_no_allocation_thresholds_in_adapter(self):
        import inspect
        import nexflow.v9.adapter as mod
        src = inspect.getsource(mod)
        assert "BOTH_HOT_THRESHOLD" not in src, "Allocation threshold in adapter.py"
        assert "CRYPTO_SCORE_MAX"   not in src, "Score constant in adapter.py"
        assert "REBALANCE_DAYS"     not in src, "Rebalance constant in adapter.py"

    def test_no_regime_machine_in_adapter(self):
        import inspect
        import nexflow.v9.adapter as mod
        src = inspect.getsource(mod)
        assert "RegimeMachine" not in src
        assert "in_bear"       not in src

    def test_no_score_computation_in_adapter(self):
        import inspect
        import nexflow.v9.adapter as mod
        src = inspect.getsource(mod)
        assert "crypto_score"  not in src
        assert "stock_score"   not in src
        assert "allocate("     not in src

    def test_abstract_methods_are_four(self):
        """The interface has exactly four responsibilities."""
        import inspect
        from nexflow.v9.adapter import ExchangeAdapter
        abstract_methods = {
            name for name, method in inspect.getmembers(ExchangeAdapter)
            if getattr(method, "__isabstractmethod__", False)
        }
        assert abstract_methods == {
            "get_balances",
            "get_price",
            "submit_order",
            "query_fill",
            "verify_execution",
        }

    def test_null_adapter_is_default_safe(self):
        """NullAdapter must be instantiable with no arguments."""
        a = NullAdapter()
        assert a is not None
