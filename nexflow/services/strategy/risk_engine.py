"""Risk engine — pre-trade checks and position sizing.

All checks are stateless relative to the signal itself; state lives in Portfolio.
The engine's own mutable state tracks cooldown periods.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nexflow.services.strategy.portfolio import Portfolio
from nexflow.services.strategy.signal_models import Signal


@dataclass
class RiskConfig:
    max_risk_per_trade: float = 0.005       # 0.5% of equity per trade
    max_concurrent_positions: int = 3
    cooldown_after_loss_bars: int = 3       # 1m bars to wait after a losing trade
    daily_drawdown_kill: float = 0.02       # 2% daily loss → block new entries
    min_stop_distance_atr: float = 0.5      # stop must be at least 0.5× ATR from entry
    max_position_equity_fraction: float = 0.20  # single position ≤ 20% of equity


class RiskEngine:
    """Validates entries and sizes positions.

    Stateful only for cooldown tracking. All other checks derive from Portfolio.
    """

    def __init__(self, cfg: RiskConfig | None = None) -> None:
        self._cfg = cfg or RiskConfig()
        # cooldown_remaining: bars remaining before the next entry is allowed
        self._cooldown_remaining: int = 0
        # track how many bars have elapsed since last cooldown was set
        self._bars_since_last_trade: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Advance one 1m bar. Call once per bar regardless of trades."""
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

    def on_loss(self) -> None:
        """Trigger a cooldown period after a losing trade."""
        self._cooldown_remaining = self._cfg.cooldown_after_loss_bars

    def cooldown_active(self) -> bool:
        return self._cooldown_remaining > 0

    def check_entry(self, signal: Signal, portfolio: Portfolio) -> tuple[bool, str]:
        """Return (allowed, reason). reason is empty string when allowed."""
        cfg = self._cfg

        # 1. Daily drawdown kill-switch
        if portfolio.daily_drawdown() >= cfg.daily_drawdown_kill:
            return False, "daily_drawdown_limit"

        # 2. Cooldown after loss
        if self._cooldown_remaining > 0:
            return False, f"cooldown_active:{self._cooldown_remaining}"

        # 3. Max concurrent positions
        if portfolio.open_position_count() >= cfg.max_concurrent_positions:
            return False, "max_positions_reached"

        # 4. Already have a position in this symbol
        if portfolio.has_position(signal.symbol):
            return False, "duplicate_symbol_position"

        # 5. Stop distance sanity
        stop_dist = abs(signal.entry_price - signal.stop_price)
        if stop_dist < cfg.min_stop_distance_atr * signal.atr:
            return False, "stop_too_close"

        # 6. Minimum equity guard
        if portfolio.current_equity <= 0:
            return False, "insufficient_equity"

        return True, ""

    def compute_position_size(self, signal: Signal, portfolio: Portfolio) -> float:
        """Risk-based position sizing: equity × risk_pct / stop_distance.

        Capped at max_position_equity_fraction × equity / entry_price.
        Returns 0.0 if the position cannot be sized (e.g., stop distance = 0).
        """
        cfg = self._cfg
        equity = portfolio.current_equity
        stop_dist = abs(signal.entry_price - signal.stop_price)

        if stop_dist <= 0 or signal.entry_price <= 0:
            return 0.0

        # Risk-based size (in base units)
        risk_amount = equity * cfg.max_risk_per_trade
        size = risk_amount / stop_dist

        # Cap by max fraction of equity
        max_notional = equity * cfg.max_position_equity_fraction
        max_size = max_notional / signal.entry_price
        size = min(size, max_size)

        return max(0.0, size)

    def reset(self) -> None:
        self._cooldown_remaining = 0
