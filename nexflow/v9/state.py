"""
V9 Confidence — Module 1: State Persistence Layer

Addresses Risk 01 (Bear Regime State Loss) — the highest-severity risk.

Responsibilities:
  - Persist bear regime state and consecutive-above counter after every daily bar
  - Persist allocation weights and last rebalance date
  - On startup: load stored state, re-derive regime from historical data,
    assert they match before any trading is permitted
  - Provide a hard gate: trading is BLOCKED until reconciliation passes

State file: JSON, written atomically via temp-file rename.

Startup sequence (mandatory, enforced by SystemState.startup_gate()):
  1. Load state.json
  2. Load last N days of BTC historical candles
  3. Re-derive regime state independently from those candles
  4. Assert stored state == derived state (within tolerance)
  5. If mismatch: raise ReconciliationError — system halts, alert fires
  6. If match: set gate_open = True — trading permitted
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np

from nexflow.v9.core import RegimeMachine, sma

# ── Paths ─────────────────────────────────────────────────────────────────────

DEFAULT_STATE_PATH = Path(
    os.environ.get("NEXFLOW_STATE_PATH", "/var/nexflow/state.json")
)
STATE_VERSION = "1.0"

# How many historical bars to use for startup reconciliation
RECONCILE_WINDOW = 30   # days — covers BEAR_CONFIRM_DAYS=10 with ample margin


# ── Exceptions ────────────────────────────────────────────────────────────────

class ReconciliationError(RuntimeError):
    """
    Raised when stored regime state does not match state derived from
    historical candles.  Trading must halt until this is resolved manually.
    """


class StateMissingError(RuntimeError):
    """Raised when state.json does not exist and no initialization was done."""


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class RegimeSnapshot:
    in_bear:           bool
    consecutive_above: int
    last_bar_date:     str   # ISO YYYY-MM-DD of the last bar the machine processed

    def to_machine(self) -> RegimeMachine:
        return RegimeMachine(
            in_bear           = self.in_bear,
            consecutive_above = self.consecutive_above,
        )


@dataclass
class AllocationSnapshot:
    wc:                   float   # crypto weight currently in effect
    ws:                   float   # stock weight currently in effect
    last_rebalance_date:  str     # ISO YYYY-MM-DD
    trading_days_since:   int     # trading days elapsed since last rebalance


@dataclass
class SystemState:
    regime:     RegimeSnapshot
    allocation: AllocationSnapshot
    version:    str = STATE_VERSION
    gate_open:  bool = field(default=False, init=False)

    # ── Serialisation ────────────────────────────────────────────────────────

    def save(self, path: Path = DEFAULT_STATE_PATH) -> None:
        """Atomic write: write to temp file, then rename."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version":    self.version,
            "regime":     asdict(self.regime),
            "allocation": asdict(self.allocation),
            "saved_at":   datetime.utcnow().isoformat() + "Z",
        }
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path: Path = DEFAULT_STATE_PATH) -> "SystemState":
        if not path.exists():
            raise StateMissingError(
                f"State file not found: {path}\n"
                "Run SystemState.initialize() before starting the system."
            )
        with open(path) as f:
            d = json.load(f)
        if d.get("version") != STATE_VERSION:
            raise ReconciliationError(
                f"State file version {d.get('version')!r} != expected {STATE_VERSION!r}. "
                "Manual migration required."
            )
        return cls(
            regime     = RegimeSnapshot(**d["regime"]),
            allocation = AllocationSnapshot(**d["allocation"]),
            version    = d["version"],
        )

    # ── Startup gate ─────────────────────────────────────────────────────────

    def startup_gate(
        self,
        btc_closes:  np.ndarray,
        btc_sma200:  np.ndarray,
        btc_mom30:   np.ndarray,
        bar_dates:   list[str],
    ) -> None:
        """
        Hard gate — must pass before trading is permitted.

        Re-derives regime state from the last RECONCILE_WINDOW bars of BTC data
        and asserts it matches the stored state.

        Args:
            btc_closes:  array of BTC daily close prices, oldest first
            btc_sma200:  corresponding SMA200 values
            btc_mom30:   corresponding 30d momentum values
            bar_dates:   corresponding dates as ISO strings (YYYY-MM-DD)

        Raises:
            ReconciliationError if stored state != derived state.
        """
        if len(btc_closes) < RECONCILE_WINDOW:
            raise ReconciliationError(
                f"Need at least {RECONCILE_WINDOW} BTC bars for reconciliation, "
                f"got {len(btc_closes)}."
            )

        # Use the full provided window to derive state
        machine_fresh = RegimeMachine()
        for i in range(len(btc_closes)):
            machine_fresh.step(
                float(btc_closes[i]),
                float(btc_sma200[i]),
                float(btc_mom30[i]),
            )

        stored = self.regime
        derived_in_bear = machine_fresh.in_bear
        derived_consec  = machine_fresh.consecutive_above

        mismatch_bear   = stored.in_bear != derived_in_bear
        # consecutive_above only matters inside bear; in bull mode it is always 0
        mismatch_consec = (
            stored.in_bear  # only check inside bear
            and abs(stored.consecutive_above - derived_consec) > 1
        )

        if mismatch_bear or mismatch_consec:
            raise ReconciliationError(
                f"Regime state mismatch.\n"
                f"  Stored:  in_bear={stored.in_bear}, "
                f"consecutive_above={stored.consecutive_above}\n"
                f"  Derived: in_bear={derived_in_bear}, "
                f"consecutive_above={derived_consec}\n"
                f"  Last bar: {bar_dates[-1]}\n"
                f"Trading is BLOCKED. Investigate state.json and historical data."
            )

        self.gate_open = True

    def assert_gate(self) -> None:
        """Call before any trading action. Raises if startup_gate() was not run."""
        if not self.gate_open:
            raise ReconciliationError(
                "startup_gate() has not been called or failed. "
                "Trading is BLOCKED."
            )

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def initialize(
        cls,
        path:          Path  = DEFAULT_STATE_PATH,
        wc:            float = 0.50,
        ws:            float = 0.50,
        rebalance_date: str  = "",
    ) -> "SystemState":
        """
        Create a fresh state file.  Only run once on first deployment.
        If a state file already exists, this will raise to prevent accidental reset.
        """
        if path.exists():
            raise FileExistsError(
                f"State file already exists: {path}\n"
                "Delete it manually if you intend to reset."
            )
        today = date.today().isoformat()
        state = cls(
            regime=RegimeSnapshot(
                in_bear           = False,
                consecutive_above = 0,
                last_bar_date     = rebalance_date or today,
            ),
            allocation=AllocationSnapshot(
                wc                  = wc,
                ws                  = ws,
                last_rebalance_date = rebalance_date or today,
                trading_days_since  = 0,
            ),
        )
        state.save(path)
        return state

    # ── Updaters ─────────────────────────────────────────────────────────────

    def update_regime(
        self,
        in_bear:           bool,
        consecutive_above: int,
        bar_date:          str,
        path:              Path = DEFAULT_STATE_PATH,
    ) -> None:
        """Update regime state and persist immediately."""
        self.assert_gate()
        self.regime = RegimeSnapshot(
            in_bear           = in_bear,
            consecutive_above = consecutive_above,
            last_bar_date     = bar_date,
        )
        self.save(path)

    def update_allocation(
        self,
        wc:                  float,
        ws:                  float,
        rebalance_date:      str,
        trading_days_since:  int,
        path:                Path = DEFAULT_STATE_PATH,
    ) -> None:
        """Update allocation weights and persist immediately."""
        self.assert_gate()
        self.allocation = AllocationSnapshot(
            wc                  = wc,
            ws                  = ws,
            last_rebalance_date = rebalance_date,
            trading_days_since  = trading_days_since,
        )
        self.save(path)

    def increment_trading_days(
        self,
        path: Path = DEFAULT_STATE_PATH,
    ) -> None:
        """Increment the trading-days counter and persist."""
        self.assert_gate()
        self.allocation = AllocationSnapshot(
            wc                  = self.allocation.wc,
            ws                  = self.allocation.ws,
            last_rebalance_date = self.allocation.last_rebalance_date,
            trading_days_since  = self.allocation.trading_days_since + 1,
        )
        self.save(path)
