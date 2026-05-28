"""Persistent JSONL journal for all paper-trading lifecycle events.

Every observable event in the paper engine is appended to a single JSONL
file (one JSON object per line). The journal is the audit trail — all other
reporting layers derive from it.

Event schema (shared fields):
    ts          : ISO-8601 UTC timestamp of the local system clock
    ts_epoch    : epoch seconds (float) for programmatic parsing
    event       : EventType string
    session_id  : UUID4 assigned at engine startup (groups one run)

Additional fields depend on event type — see EventType docs below.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class EventType(str, Enum):
    # Signal lifecycle
    SIGNAL          = "SIGNAL"           # strategy emitted a signal
    REJECTED        = "REJECTED"         # risk engine or kill-switch blocked entry
    FILL            = "FILL"             # entry order simulated
    PARTIAL_TP      = "PARTIAL_TP"       # one TP level hit
    STOP_HIT        = "STOP_HIT"         # stop loss triggered
    FORCE_CLOSE     = "FORCE_CLOSE"      # engine shutdown forced close

    # Risk events
    KILL_SWITCH     = "KILL_SWITCH"      # hard kill engaged
    KILL_CLEARED    = "KILL_CLEARED"     # kill condition cleared (new session)
    RISK_WARNING    = "RISK_WARNING"     # soft warning, no action taken

    # Operational snapshots
    EQUITY_SNAPSHOT = "EQUITY_SNAPSHOT"  # periodic equity / drawdown record
    SESSION_START   = "SESSION_START"    # engine started
    SESSION_END     = "SESSION_END"      # engine stopped cleanly

    # Feed health
    FEED_STALE      = "FEED_STALE"       # candle gap detected
    LATENCY_SPIKE   = "LATENCY_SPIKE"    # exchange→local latency exceeded threshold
    SPREAD_ANOMALY  = "SPREAD_ANOMALY"   # abnormal spread detected


@dataclass
class ExecutionJournal:
    """Append-only JSONL event log.

    Args:
        log_dir: directory for journal files (created if absent)
        session_id: unique run identifier; auto-generated if None
    """

    log_dir: Path
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    _file: Any = field(init=False, default=None)
    _path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.log_dir = Path(self.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._path = self.log_dir / f"journal_{ts}_{self.session_id[:8]}.jsonl"
        self._file = open(self._path, "a", buffering=1)  # line-buffered
        self._write(EventType.SESSION_START, {"session_id": self.session_id})

    @property
    def path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------
    # Signal lifecycle
    # ------------------------------------------------------------------

    def log_signal(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_price: float,
        tp_prices: list[float],
        atr: float,
        features: dict[str, float],
    ) -> None:
        self._write(EventType.SIGNAL, {
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "tp_prices": tp_prices,
            "atr": atr,
            "features": features,
        })

    def log_rejected(self, symbol: str, direction: str, reason: str) -> None:
        self._write(EventType.REJECTED, {
            "symbol": symbol,
            "direction": direction,
            "reason": reason,
        })

    def log_fill(
        self,
        symbol: str,
        direction: str,
        fill_price: float,
        size: float,
        fee: float,
        slippage: float,
        equity_after: float,
    ) -> None:
        self._write(EventType.FILL, {
            "symbol": symbol,
            "direction": direction,
            "fill_price": fill_price,
            "size": size,
            "fee": fee,
            "slippage": slippage,
            "equity_after": equity_after,
        })

    def log_partial_tp(
        self,
        symbol: str,
        tp_idx: int,
        fill_price: float,
        size: float,
        pnl: float,
        fee: float,
        equity_after: float,
    ) -> None:
        self._write(EventType.PARTIAL_TP, {
            "symbol": symbol,
            "tp_idx": tp_idx,
            "fill_price": fill_price,
            "size": size,
            "pnl": pnl,
            "fee": fee,
            "equity_after": equity_after,
        })

    def log_stop_hit(
        self,
        symbol: str,
        fill_price: float,
        size: float,
        pnl: float,
        fee: float,
        equity_after: float,
    ) -> None:
        self._write(EventType.STOP_HIT, {
            "symbol": symbol,
            "fill_price": fill_price,
            "size": size,
            "pnl": pnl,
            "fee": fee,
            "equity_after": equity_after,
        })

    def log_force_close(self, symbol: str, price: float, pnl: float, reason: str) -> None:
        self._write(EventType.FORCE_CLOSE, {
            "symbol": symbol,
            "price": price,
            "pnl": pnl,
            "reason": reason,
        })

    # ------------------------------------------------------------------
    # Risk events
    # ------------------------------------------------------------------

    def log_kill_switch(self, reason: str, detail: str = "") -> None:
        self._write(EventType.KILL_SWITCH, {"reason": reason, "detail": detail})

    def log_kill_cleared(self) -> None:
        self._write(EventType.KILL_CLEARED, {})

    def log_risk_warning(self, warning: str, detail: str = "") -> None:
        self._write(EventType.RISK_WARNING, {"warning": warning, "detail": detail})

    # ------------------------------------------------------------------
    # Operational snapshots
    # ------------------------------------------------------------------

    def log_equity_snapshot(
        self,
        equity: float,
        realized_pnl: float,
        unrealized_pnl: float,
        drawdown: float,
        open_positions: int,
    ) -> None:
        self._write(EventType.EQUITY_SNAPSHOT, {
            "equity": equity,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "drawdown": drawdown,
            "open_positions": open_positions,
        })

    def log_session_end(self, final_equity: float, total_trades: int) -> None:
        self._write(EventType.SESSION_END, {
            "final_equity": final_equity,
            "total_trades": total_trades,
        })
        if self._file:
            self._file.flush()

    # ------------------------------------------------------------------
    # Feed health
    # ------------------------------------------------------------------

    def log_feed_stale(self, symbol: str, gap_seconds: float) -> None:
        self._write(EventType.FEED_STALE, {"symbol": symbol, "gap_seconds": gap_seconds})

    def log_latency_spike(self, symbol: str, latency_ms: float) -> None:
        self._write(EventType.LATENCY_SPIKE, {"symbol": symbol, "latency_ms": latency_ms})

    def log_spread_anomaly(self, symbol: str, spread: float, atr: float) -> None:
        self._write(EventType.SPREAD_ANOMALY, {
            "symbol": symbol,
            "spread": spread,
            "atr": atr,
            "spread_atr_ratio": spread / atr if atr > 0 else 0.0,
        })

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write(self, event_type: EventType, payload: dict[str, Any]) -> None:
        now = time.time()
        record = {
            "ts": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
            "ts_epoch": now,
            "event": event_type.value,
            "session_id": self.session_id,
            **payload,
        }
        self._file.write(json.dumps(record, default=str) + "\n")

    def close(self) -> None:
        if self._file:
            self._file.flush()
            self._file.close()
            self._file = None

    def __del__(self) -> None:
        self.close()
