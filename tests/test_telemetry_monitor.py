"""Unit tests for TelemetryMonitor — no network, no real WS connection."""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path

import pytest

from nexflow.config import get_config
from nexflow.models.market_state import MarketState, OrderBookLevel
from nexflow.services.market_data.telemetry_monitor import (
    TelemetryMonitor,
    _FROZEN_OB_THRESHOLD_S,
    _LATENCY_SPIKE_MS,
    _MAX_SPREAD_PCT,
    _SILENCE_THRESHOLD_S,
    _SymbolMetrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(
    symbol: str = "BTCUSDT",
    bid: float = 30000.0,
    ask: float = 30001.0,
    exchange_ts_ms: int = 0,
) -> MarketState:
    s = MarketState(symbol=symbol, product_type="USDT-FUTURES")
    s.bids = [OrderBookLevel(bid, 1.0)]
    s.asks = [OrderBookLevel(ask, 1.0)]
    s.exchange_ts_ms = exchange_ts_ms or int(time.time() * 1000)
    return s


def _monitor_with_tmp_csv() -> tuple[TelemetryMonitor, Path]:
    cfg = get_config()
    tmp = tempfile.mkdtemp()
    csv_path = Path(tmp) / "test_telemetry.csv"
    monitor = TelemetryMonitor(cfg, heartbeat_interval_s=9999, csv_path=csv_path)
    return monitor, csv_path


# ---------------------------------------------------------------------------
# _SymbolMetrics unit tests
# ---------------------------------------------------------------------------

def test_msgs_per_sec_empty() -> None:
    m = _SymbolMetrics(symbol="X")
    assert m.msgs_per_sec() == 0.0


def test_percentile_empty() -> None:
    m = _SymbolMetrics(symbol="X")
    assert m.percentile(50) == 0.0


def test_percentile_single_value() -> None:
    m = _SymbolMetrics(symbol="X")
    m.latencies_ms.append(42.0)
    assert m.percentile(50) == 42.0
    assert m.percentile(99) == 42.0


def test_percentile_ordered() -> None:
    m = _SymbolMetrics(symbol="X")
    for v in [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]:
        m.latencies_ms.append(v)
    assert m.percentile(50) == 60.0   # idx = 10 * 50/100 = 5 → 60.0
    assert m.percentile(95) == 100.0


# ---------------------------------------------------------------------------
# TelemetryMonitor — on_update logic
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_packet_sequence_increments() -> None:
    monitor, csv_path = _monitor_with_tmp_csv()
    monitor._open_csv()

    state = _make_state()
    await monitor._on_update(state)
    await monitor._on_update(state)

    assert monitor._metrics["BTCUSDT"].packet_sequence == 2
    monitor._close_csv()


@pytest.mark.asyncio
async def test_latency_recorded() -> None:
    monitor, csv_path = _monitor_with_tmp_csv()
    monitor._open_csv()

    # Exchange timestamp 100ms in the past
    exchange_ts_ms = int((time.time() - 0.1) * 1000)
    state = _make_state(exchange_ts_ms=exchange_ts_ms)
    await monitor._on_update(state)

    m = monitor._metrics["BTCUSDT"]
    assert len(m.latencies_ms) == 1
    assert 50 < m.latencies_ms[0] < 2000  # somewhere around 100ms ± jitter
    monitor._close_csv()


@pytest.mark.asyncio
async def test_no_latency_when_exchange_ts_zero() -> None:
    monitor, csv_path = _monitor_with_tmp_csv()
    monitor._open_csv()

    state = _make_state()
    state.exchange_ts_ms = 0
    await monitor._on_update(state)

    m = monitor._metrics["BTCUSDT"]
    assert len(m.latencies_ms) == 0
    monitor._close_csv()


@pytest.mark.asyncio
async def test_orderbook_fingerprint_updates() -> None:
    monitor, csv_path = _monitor_with_tmp_csv()
    monitor._open_csv()

    s1 = _make_state(bid=100.0, ask=101.0)
    await monitor._on_update(s1)
    fp1 = monitor._metrics["BTCUSDT"].last_ob_fingerprint

    s2 = _make_state(bid=100.5, ask=101.5)
    await monitor._on_update(s2)
    fp2 = monitor._metrics["BTCUSDT"].last_ob_fingerprint

    assert fp1 == (100.0, 101.0)
    assert fp2 == (100.5, 101.5)
    monitor._close_csv()


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def test_detect_latency_spike() -> None:
    monitor, _ = _monitor_with_tmp_csv()
    monitor._open_csv()

    m = _SymbolMetrics(symbol="X")
    state = _make_state()
    flags = monitor._detect_anomalies(state, m, latency_ms=_LATENCY_SPIKE_MS + 1, now=time.time())

    assert "latency_spike" in flags
    assert m.anomalies["latency_spike"] == 1
    monitor._close_csv()


def test_no_latency_spike_below_threshold() -> None:
    monitor, _ = _monitor_with_tmp_csv()
    monitor._open_csv()

    m = _SymbolMetrics(symbol="X")
    state = _make_state()
    flags = monitor._detect_anomalies(state, m, latency_ms=_LATENCY_SPIKE_MS - 1, now=time.time())

    assert "latency_spike" not in flags
    monitor._close_csv()


def test_detect_invalid_spread_crossed() -> None:
    monitor, _ = _monitor_with_tmp_csv()
    monitor._open_csv()

    m = _SymbolMetrics(symbol="X")
    # bid > ask → negative spread
    state = _make_state(bid=101.0, ask=100.0)
    flags = monitor._detect_anomalies(state, m, latency_ms=0.0, now=time.time())

    assert "invalid_spread" in flags
    monitor._close_csv()


def test_detect_abnormal_spread_pct() -> None:
    monitor, _ = _monitor_with_tmp_csv()
    monitor._open_csv()

    m = _SymbolMetrics(symbol="X")
    # spread = 200, mid = 10100 → ~1.98% > 1% threshold
    state = _make_state(bid=10000.0, ask=10200.0)
    flags = monitor._detect_anomalies(state, m, latency_ms=0.0, now=time.time())

    assert "invalid_spread" in flags
    monitor._close_csv()


def test_detect_out_of_order_timestamp() -> None:
    monitor, _ = _monitor_with_tmp_csv()
    monitor._open_csv()

    m = _SymbolMetrics(symbol="X")
    m.last_exchange_ts_ms = 1_000_000
    state = _make_state()
    state.exchange_ts_ms = 999_899   # 101ms earlier than last seen (above jitter threshold)

    flags = monitor._detect_anomalies(state, m, latency_ms=0.0, now=time.time())
    assert "out_of_order_ts" in flags
    monitor._close_csv()


def test_detect_message_gap() -> None:
    monitor, _ = _monitor_with_tmp_csv()
    monitor._open_csv()

    m = _SymbolMetrics(symbol="X")
    m.prev_message_at = time.time() - 10.0   # 10s ago, threshold is 5s
    state = _make_state()

    flags = monitor._detect_anomalies(state, m, latency_ms=0.0, now=time.time())
    assert "message_gap" in flags
    monitor._close_csv()


def test_no_message_gap_below_threshold() -> None:
    monitor, _ = _monitor_with_tmp_csv()
    monitor._open_csv()

    m = _SymbolMetrics(symbol="X")
    m.prev_message_at = time.time() - 1.0   # 1s ago, well below threshold
    state = _make_state()

    flags = monitor._detect_anomalies(state, m, latency_ms=0.0, now=time.time())
    assert "message_gap" not in flags
    monitor._close_csv()


def test_ws_silence_detected() -> None:
    monitor, _ = _monitor_with_tmp_csv()
    monitor._open_csv()

    monitor._last_any_message_at = time.time() - (_SILENCE_THRESHOLD_S + 1)
    is_silent = monitor._check_ws_silence()

    assert is_silent is True
    monitor._close_csv()


def test_ws_no_silence_recent() -> None:
    monitor, _ = _monitor_with_tmp_csv()
    monitor._open_csv()

    monitor._last_any_message_at = time.time() - 1.0
    assert monitor._check_ws_silence() is False
    monitor._close_csv()


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_csv_written_on_update() -> None:
    monitor, csv_path = _monitor_with_tmp_csv()
    monitor._open_csv()

    state = _make_state()
    await monitor._on_update(state)
    monitor._close_csv()

    lines = csv_path.read_text().strip().splitlines()
    assert len(lines) == 2              # header + 1 data row
    assert "local_ts" in lines[0]       # header present
    assert "BTCUSDT" in lines[1]


@pytest.mark.asyncio
async def test_csv_appends_on_reopen() -> None:
    monitor, csv_path = _monitor_with_tmp_csv()

    monitor._open_csv()
    await monitor._on_update(_make_state())
    monitor._close_csv()

    # Re-open (simulates process restart)
    monitor2, _ = _monitor_with_tmp_csv()
    monitor2._csv_path = csv_path
    monitor2._open_csv()
    await monitor2._on_update(_make_state())
    monitor2._close_csv()

    lines = csv_path.read_text().strip().splitlines()
    assert len(lines) == 3   # header + 2 data rows
