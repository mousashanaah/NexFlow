"""Market data telemetry monitor.

Tracks transport health, data quality, and anomalies for every symbol.
Runs a background heartbeat task and exports rows to a CSV file.

Usage:
    monitor = TelemetryMonitor(config)
    monitor.attach(client)      # registers callbacks on BitgetWSClient
    await monitor.start()       # launch background heartbeat (runs until cancelled)
    await monitor.stop()
"""

from __future__ import annotations

import asyncio
import csv
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from nexflow.logging import get_logger
from nexflow.models.market_state import MarketState

if TYPE_CHECKING:
    from nexflow.config import NexFlowConfig
    from nexflow.services.market_data.bitget_ws import BitgetWSClient


_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Anomaly thresholds (reasonable production defaults)
# ---------------------------------------------------------------------------
_SILENCE_THRESHOLD_S: float = 10.0       # no message received for this long
_FROZEN_OB_THRESHOLD_S: float = 30.0     # orderbook top unchanged for this long
_LATENCY_SPIKE_MS: float = 2_000.0       # single-message latency above this
_MAX_SPREAD_PCT: float = 1.0             # spread > 1% of mid → invalid
_MSG_GAP_SPIKE_S: float = 5.0            # gap between consecutive messages

# Number of latency samples kept for percentile computation
_LATENCY_WINDOW = 1_000
# Window for computing messages/sec (keep timestamps for last N seconds)
_FREQ_WINDOW_S: float = 60.0

CSV_HEADER = [
    "local_ts",
    "exchange_ts",
    "latency_ms",
    "spread",
    "mid_price",
    "reconnect_id",
    "packet_sequence",
    "symbol",
    "anomaly_flags",
]


# ---------------------------------------------------------------------------
# Per-symbol rolling state
# ---------------------------------------------------------------------------

@dataclass
class _SymbolMetrics:
    symbol: str
    first_seen_at: float = field(default_factory=time.time)
    packet_sequence: int = 0
    last_message_at: float = 0.0
    prev_message_at: float = 0.0
    last_exchange_ts_ms: int = 0       # for out-of-order detection
    last_ob_fingerprint: tuple[float, float] | None = None  # (bid_px, ask_px)
    last_ob_changed_at: float = field(default_factory=time.time)

    # Latency samples (ms), capped at _LATENCY_WINDOW
    latencies_ms: deque[float] = field(default_factory=lambda: deque(maxlen=_LATENCY_WINDOW))

    # Arrival times for msg/sec (epoch float), capped at last _FREQ_WINDOW_S worth
    message_times: deque[float] = field(default_factory=lambda: deque(maxlen=10_000))

    # Cumulative anomaly counters
    anomalies: dict[str, int] = field(default_factory=lambda: {
        "latency_spike": 0,
        "frozen_orderbook": 0,
        "invalid_spread": 0,
        "out_of_order_ts": 0,
        "message_gap": 0,
        "ws_silence": 0,
    })

    def msgs_per_sec(self) -> float:
        now = time.time()
        cutoff = now - _FREQ_WINDOW_S
        # Trim stale entries from the left
        while self.message_times and self.message_times[0] < cutoff:
            self.message_times.popleft()
        elapsed = now - (self.message_times[0] if self.message_times else now)
        count = len(self.message_times)
        return count / elapsed if elapsed > 0 else 0.0

    def percentile(self, pct: float) -> float:
        """Return the p{pct} latency. pct in [0, 100]."""
        if not self.latencies_ms:
            return 0.0
        sorted_lats = sorted(self.latencies_ms)
        idx = int(len(sorted_lats) * pct / 100)
        idx = min(idx, len(sorted_lats) - 1)
        return sorted_lats[idx]

    def uptime_s(self) -> float:
        return time.time() - self.first_seen_at


# ---------------------------------------------------------------------------
# Main monitor
# ---------------------------------------------------------------------------

class TelemetryMonitor:
    """Validates transport integrity and emits heartbeat diagnostics."""

    def __init__(
        self,
        config: NexFlowConfig,
        heartbeat_interval_s: float = 30.0,
        csv_path: str | Path | None = None,
    ) -> None:
        self._cfg = config
        self._heartbeat_interval = heartbeat_interval_s
        self._csv_path = Path(csv_path) if csv_path else Path("logs/market_telemetry.csv")

        self._metrics: dict[str, _SymbolMetrics] = {}
        self._client: BitgetWSClient | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._csv_writer: csv.DictWriter | None = None
        self._csv_file: object | None = None

        # Global ws-level silence tracking (across all symbols)
        self._last_any_message_at: float = time.time()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def attach(self, client: BitgetWSClient) -> None:
        """Register callbacks on a BitgetWSClient. Call before client.start()."""
        self._client = client
        client.on_update(self._on_update)
        _log.info("telemetry.attached", symbols=client._md.symbols)

    async def start(self) -> None:
        self._open_csv()
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(), name="telemetry-heartbeat")
        _log.info("telemetry.started", csv=str(self._csv_path), interval_s=self._heartbeat_interval)

    async def stop(self) -> None:
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        self._close_csv()
        _log.info("telemetry.stopped")

    # ------------------------------------------------------------------
    # Callback — called by BitgetWSClient after each state update
    # ------------------------------------------------------------------

    async def _on_update(self, state: MarketState) -> None:
        now = time.time()
        self._last_any_message_at = now

        m = self._metrics.setdefault(state.symbol, _SymbolMetrics(symbol=state.symbol, first_seen_at=now))

        # --- basic counters ---
        m.packet_sequence += 1
        m.prev_message_at = m.last_message_at
        m.last_message_at = now
        m.message_times.append(now)

        reconnect_id = self._client.reconnect_count if self._client else 0

        # --- latency ---
        latency_ms = 0.0
        if state.exchange_ts_ms:
            latency_ms = now * 1000.0 - state.exchange_ts_ms
            if latency_ms > 0:
                m.latencies_ms.append(latency_ms)

        # --- anomaly detection ---
        anomaly_flags = self._detect_anomalies(state, m, latency_ms, now)

        # --- orderbook fingerprint update ---
        if state.best_bid and state.best_ask:
            fp = (state.best_bid.price, state.best_ask.price)
            if fp != m.last_ob_fingerprint:
                m.last_ob_fingerprint = fp
                m.last_ob_changed_at = now

        # --- OOO timestamp tracking ---
        if state.exchange_ts_ms:
            m.last_exchange_ts_ms = state.exchange_ts_ms

        # --- CSV export ---
        self._write_csv_row(
            local_ts=now,
            exchange_ts=state.exchange_ts_ms / 1000.0 if state.exchange_ts_ms else 0.0,
            latency_ms=round(latency_ms, 3),
            spread=round(state.spread, 6) if state.spread is not None else "",
            mid_price=round(state.mid_price, 6) if state.mid_price is not None else "",
            reconnect_id=reconnect_id,
            packet_sequence=m.packet_sequence,
            symbol=state.symbol,
            anomaly_flags="|".join(anomaly_flags) if anomaly_flags else "",
        )

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------

    def _detect_anomalies(
        self,
        state: MarketState,
        m: _SymbolMetrics,
        latency_ms: float,
        now: float,
    ) -> list[str]:
        flags: list[str] = []

        # Latency spike
        if latency_ms > _LATENCY_SPIKE_MS:
            m.anomalies["latency_spike"] += 1
            flags.append("latency_spike")
            _log.warning(
                "telemetry.anomaly.latency_spike",
                symbol=state.symbol,
                latency_ms=round(latency_ms, 1),
                threshold_ms=_LATENCY_SPIKE_MS,
            )

        # Frozen orderbook
        if m.last_ob_fingerprint is not None and (now - m.last_ob_changed_at) > _FROZEN_OB_THRESHOLD_S:
            m.anomalies["frozen_orderbook"] += 1
            flags.append("frozen_orderbook")
            _log.warning(
                "telemetry.anomaly.frozen_orderbook",
                symbol=state.symbol,
                frozen_s=round(now - m.last_ob_changed_at, 1),
            )

        # Invalid / crossed spread
        if state.spread is not None and state.mid_price:
            if state.spread <= 0:
                m.anomalies["invalid_spread"] += 1
                flags.append("invalid_spread")
                _log.warning(
                    "telemetry.anomaly.invalid_spread",
                    symbol=state.symbol,
                    spread=state.spread,
                    mid=state.mid_price,
                )
            elif state.mid_price > 0:
                spread_pct = (state.spread / state.mid_price) * 100
                if spread_pct > _MAX_SPREAD_PCT:
                    m.anomalies["invalid_spread"] += 1
                    flags.append("invalid_spread")
                    _log.warning(
                        "telemetry.anomaly.abnormal_spread",
                        symbol=state.symbol,
                        spread_pct=round(spread_pct, 4),
                        threshold_pct=_MAX_SPREAD_PCT,
                    )

        # Out-of-order exchange timestamp
        if state.exchange_ts_ms and m.last_exchange_ts_ms:
            if state.exchange_ts_ms < m.last_exchange_ts_ms:
                m.anomalies["out_of_order_ts"] += 1
                flags.append("out_of_order_ts")
                _log.warning(
                    "telemetry.anomaly.out_of_order_ts",
                    symbol=state.symbol,
                    current_ts=state.exchange_ts_ms,
                    prev_ts=m.last_exchange_ts_ms,
                    delta_ms=m.last_exchange_ts_ms - state.exchange_ts_ms,
                )

        # Message gap between consecutive messages for this symbol
        if m.prev_message_at > 0:
            gap_s = now - m.prev_message_at
            if gap_s > _MSG_GAP_SPIKE_S:
                m.anomalies["message_gap"] += 1
                flags.append("message_gap")
                _log.warning(
                    "telemetry.anomaly.message_gap",
                    symbol=state.symbol,
                    gap_s=round(gap_s, 2),
                    threshold_s=_MSG_GAP_SPIKE_S,
                )

        return flags

    def _check_ws_silence(self) -> bool:
        """Return True if no message has been received across all symbols."""
        silence_s = time.time() - self._last_any_message_at
        if silence_s > _SILENCE_THRESHOLD_S:
            for m in self._metrics.values():
                m.anomalies["ws_silence"] += 1
            _log.error(
                "telemetry.anomaly.ws_silence",
                silence_s=round(silence_s, 1),
                threshold_s=_SILENCE_THRESHOLD_S,
            )
            return True
        return False

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(self._heartbeat_interval)
            self._emit_heartbeat()

    def _emit_heartbeat(self) -> None:
        ws_silent = self._check_ws_silence()
        reconnect_id = self._client.reconnect_count if self._client else 0

        for symbol, m in self._metrics.items():
            p50 = round(m.percentile(50), 2)
            p95 = round(m.percentile(95), 2)
            mps = round(m.msgs_per_sec(), 2)
            uptime = round(m.uptime_s(), 1)
            total_anomalies = sum(m.anomalies.values())
            stale = ws_silent or (
                m.last_message_at > 0 and (time.time() - m.last_message_at) > _SILENCE_THRESHOLD_S
            )

            _log.info(
                "[HEARTBEAT]",
                symbol=symbol,
                uptime_s=uptime,
                latency_p50_ms=p50,
                latency_p95_ms=p95,
                reconnects=reconnect_id,
                messages_per_sec=mps,
                packet_sequence=m.packet_sequence,
                stale_state=stale,
                anomalies_total=total_anomalies,
                anomaly_breakdown=dict(m.anomalies),
            )

    # ------------------------------------------------------------------
    # CSV persistence
    # ------------------------------------------------------------------

    def _open_csv(self) -> None:
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self._csv_path.exists()
        # Open in append mode so restarts accumulate history
        self._csv_file = open(self._csv_path, "a", newline="", buffering=1)  # line-buffered
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=CSV_HEADER)  # type: ignore[arg-type]
        if is_new:
            self._csv_writer.writeheader()

    def _close_csv(self) -> None:
        if self._csv_file:
            self._csv_file.close()  # type: ignore[union-attr]
            self._csv_file = None
            self._csv_writer = None

    def _write_csv_row(self, **kwargs: object) -> None:
        if self._csv_writer is None:
            return
        try:
            self._csv_writer.writerow(kwargs)
        except Exception as exc:
            _log.error("telemetry.csv_write_error", error=str(exc))
