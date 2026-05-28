"""Live risk monitor: real-time checks beyond the static RiskEngine.

The RiskEngine handles per-trade risk budget (position size, cooldown, daily DD).
This layer handles operational conditions that can only be evaluated at runtime:

    - WebSocket health (reconnect count, last message age)
    - Stale candle detection (gap between last finalized candle and wall clock)
    - Spread anomaly (spread_avg / ATR ratio exceeds threshold)
    - Latency spike (exchange→local round-trip exceeds threshold)
    - Slippage anomaly (observed fill slippage >> expected)
    - Consecutive loss streak independent of cooldown bars

Kill conditions halt ALL new entries until the engine is restarted (or the
condition explicitly clears). Each condition is logged separately.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class KillReason(str, Enum):
    WS_UNHEALTHY       = "ws_unhealthy"        # reconnects > threshold
    STALE_FEED         = "stale_feed"           # no candle for > threshold seconds
    SPREAD_ANOMALY     = "spread_anomaly"       # spread / ATR > threshold
    DRAWDOWN_EXCEEDED  = "drawdown_exceeded"    # portfolio DD > limit
    CONSECUTIVE_LOSSES = "consecutive_losses"   # streak > limit
    LATENCY_SPIKE      = "latency_spike"        # persistent high latency
    SLIPPAGE_ANOMALY   = "slippage_anomaly"     # observed slip >> model


@dataclass
class LiveRiskConfig:
    # WebSocket health
    max_reconnects_per_hour: int = 5
    stale_candle_threshold_s: float = 120.0        # seconds without a 1m candle

    # Spread
    max_spread_atr_ratio: float = 0.30             # spread > 30% ATR → anomaly

    # Latency
    latency_spike_ms: float = 1_000.0             # warn above this
    latency_kill_ms: float = 3_000.0              # kill above this (sustained)
    latency_kill_consecutive: int = 3             # N consecutive spikes → kill

    # Slippage
    slippage_kill_fraction: float = 0.005         # 0.5% of price → anomaly

    # Streak / drawdown (mirrors RiskEngine but independent)
    max_consecutive_losses: int = 6
    max_drawdown_kill: float = 0.05               # 5% of initial equity → kill


@dataclass
class LiveRiskMonitor:
    """Stateful monitor that tracks live operational conditions.

    Call update_* methods as events arrive. Query is_killed / kill_reasons
    before routing any new signal.
    """

    cfg: LiveRiskConfig = field(default_factory=LiveRiskConfig)
    initial_equity: float = 100_000.0

    # Internal state
    _kill_reasons: set[KillReason] = field(init=False, default_factory=set)
    _last_candle_ts: dict[str, float] = field(init=False, default_factory=dict)  # symbol → wall_clock
    _consecutive_losses: int = field(init=False, default=0)
    _consecutive_latency_spikes: int = field(init=False, default=0)
    _reconnect_times: list[float] = field(init=False, default_factory=list)
    _last_spreads: dict[str, float] = field(init=False, default_factory=dict)
    _last_atrs: dict[str, float] = field(init=False, default_factory=dict)

    @property
    def is_killed(self) -> bool:
        return bool(self._kill_reasons)

    @property
    def kill_reasons(self) -> set[KillReason]:
        return frozenset(self._kill_reasons)

    # ------------------------------------------------------------------
    # Update methods — call these from the paper trader event loop
    # ------------------------------------------------------------------

    def on_candle_received(self, symbol: str, atr: float = 0.0, spread: float = 0.0) -> None:
        """Record that a finalized candle arrived for this symbol."""
        self._last_candle_ts[symbol] = time.time()
        if atr > 0:
            self._last_atrs[symbol] = atr
        if spread > 0:
            self._last_spreads[symbol] = spread
            self._check_spread(symbol, spread, atr)

    def on_trade_result(self, pnl: float) -> None:
        """Record the PnL of a closed trade for streak tracking."""
        if pnl < 0:
            self._consecutive_losses += 1
            if self._consecutive_losses >= self.cfg.max_consecutive_losses:
                self._kill_reasons.add(KillReason.CONSECUTIVE_LOSSES)
        else:
            self._consecutive_losses = 0
            self._kill_reasons.discard(KillReason.CONSECUTIVE_LOSSES)

    def on_drawdown(self, current_equity: float) -> None:
        """Check if portfolio drawdown has breached the kill threshold."""
        dd = (self.initial_equity - current_equity) / self.initial_equity
        if dd >= self.cfg.max_drawdown_kill:
            self._kill_reasons.add(KillReason.DRAWDOWN_EXCEEDED)
        else:
            self._kill_reasons.discard(KillReason.DRAWDOWN_EXCEEDED)

    def on_latency(self, latency_ms: float) -> bool:
        """Record a latency measurement. Returns True if a spike was detected."""
        if latency_ms >= self.cfg.latency_kill_ms:
            self._consecutive_latency_spikes += 1
            if self._consecutive_latency_spikes >= self.cfg.latency_kill_consecutive:
                self._kill_reasons.add(KillReason.LATENCY_SPIKE)
            return True
        elif latency_ms >= self.cfg.latency_spike_ms:
            self._consecutive_latency_spikes = 0
            return True
        else:
            self._consecutive_latency_spikes = 0
            self._kill_reasons.discard(KillReason.LATENCY_SPIKE)
            return False

    def on_ws_reconnect(self) -> None:
        """Record a WebSocket reconnection event."""
        now = time.time()
        self._reconnect_times = [t for t in self._reconnect_times if now - t < 3600]
        self._reconnect_times.append(now)
        if len(self._reconnect_times) >= self.cfg.max_reconnects_per_hour:
            self._kill_reasons.add(KillReason.WS_UNHEALTHY)
        else:
            self._kill_reasons.discard(KillReason.WS_UNHEALTHY)

    def on_fill_slippage(self, expected_price: float, fill_price: float) -> bool:
        """Check if observed slippage exceeds the anomaly threshold. Returns True if anomalous."""
        if expected_price <= 0:
            return False
        slip_fraction = abs(fill_price - expected_price) / expected_price
        if slip_fraction >= self.cfg.slippage_kill_fraction:
            self._kill_reasons.add(KillReason.SLIPPAGE_ANOMALY)
            return True
        self._kill_reasons.discard(KillReason.SLIPPAGE_ANOMALY)
        return False

    def check_stale_feeds(self, symbols: list[str]) -> list[str]:
        """Return list of symbols with stale feeds (no candle for > threshold).

        Call periodically (e.g., every 30s) to detect dead feeds.
        """
        now = time.time()
        stale: list[str] = []
        for sym in symbols:
            last = self._last_candle_ts.get(sym, 0.0)
            if last > 0 and (now - last) > self.cfg.stale_candle_threshold_s:
                stale.append(sym)
        if stale:
            self._kill_reasons.add(KillReason.STALE_FEED)
        else:
            self._kill_reasons.discard(KillReason.STALE_FEED)
        return stale

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_spread(self, symbol: str, spread: float, atr: float) -> None:
        if atr <= 0:
            return
        ratio = spread / atr
        if ratio > self.cfg.max_spread_atr_ratio:
            self._kill_reasons.add(KillReason.SPREAD_ANOMALY)
        else:
            self._kill_reasons.discard(KillReason.SPREAD_ANOMALY)

    def status_summary(self) -> dict:
        """Return a snapshot dict for dashboard / logging."""
        return {
            "is_killed": self.is_killed,
            "kill_reasons": [r.value for r in self._kill_reasons],
            "consecutive_losses": self._consecutive_losses,
            "reconnects_last_hour": len(self._reconnect_times),
            "consecutive_latency_spikes": self._consecutive_latency_spikes,
        }
