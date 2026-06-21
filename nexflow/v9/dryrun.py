"""
V9 Confidence — Dry Run Tracker

Enforces the 30-consecutive-day shadow deployment gate before Module 7
(live exchange integration) is authorised.

Gate definition:
  30 consecutive calendar days of successful paper operation, where each
  day must pass ALL of:
    - Daily parity check (production == research engine)
    - State reconciliation pass (startup_gate)
    - No audit inconsistencies
    - No lifecycle failures (AllocationRunnerError)
    - Runner snapshot validates cleanly

Any single failure resets the consecutive count to zero.

The gate is enforced by assert_module7_gate() — call this before any
live exchange code is touched.

State file: JSON at DRYRUN_STATE_PATH (default /var/nexflow/dryrun.json)
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

DEFAULT_DRYRUN_PATH = Path(
    os.environ.get("NEXFLOW_DRYRUN_PATH", "/var/nexflow/dryrun.json")
)

SHADOW_GATE_DAYS = 30


# ── Exceptions ────────────────────────────────────────────────────────────────

class ShadowGateNotMet(RuntimeError):
    """
    Raised when Module 7 (live trading) is attempted before the 30-day
    dry-run gate has been satisfied.
    """


# ── Day result ────────────────────────────────────────────────────────────────

@dataclass
class DryRunDayResult:
    date:             str        # YYYY-MM-DD
    parity_passed:    bool
    reconcile_passed: bool
    audit_clean:      bool
    lifecycle_clean:  bool
    snapshot_valid:   bool
    note:             str = ""

    @property
    def passed(self) -> bool:
        return (
            self.parity_passed    and
            self.reconcile_passed and
            self.audit_clean      and
            self.lifecycle_clean  and
            self.snapshot_valid
        )

    def failure_reasons(self) -> list[str]:
        reasons = []
        if not self.parity_passed:    reasons.append("parity_failed")
        if not self.reconcile_passed: reasons.append("reconcile_failed")
        if not self.audit_clean:      reasons.append("audit_dirty")
        if not self.lifecycle_clean:  reasons.append("lifecycle_error")
        if not self.snapshot_valid:   reasons.append("snapshot_invalid")
        return reasons

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DryRunDayResult":
        return cls(**d)


# ── Dry run state ─────────────────────────────────────────────────────────────

@dataclass
class DryRunState:
    consecutive_successes: int                = 0
    gate_met:              bool               = False
    gate_met_date:         Optional[str]      = None
    start_date:            Optional[str]      = None
    last_run_date:         Optional[str]      = None
    total_days_run:        int                = 0
    total_failures:        int                = 0
    history:               list               = field(default_factory=list)
    # list[dict] — DryRunDayResult.to_dict()

    # ── Serialisation ────────────────────────────────────────────────────────

    def save(self, path: Path = DEFAULT_DRYRUN_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
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
    def load(cls, path: Path = DEFAULT_DRYRUN_PATH) -> "DryRunState":
        if not path.exists():
            return cls()
        with open(path) as f:
            d = json.load(f)
        state = cls(
            consecutive_successes = d.get("consecutive_successes", 0),
            gate_met              = d.get("gate_met", False),
            gate_met_date         = d.get("gate_met_date"),
            start_date            = d.get("start_date"),
            last_run_date         = d.get("last_run_date"),
            total_days_run        = d.get("total_days_run", 0),
            total_failures        = d.get("total_failures", 0),
            history               = d.get("history", []),
        )
        return state

    # ── Record a day ─────────────────────────────────────────────────────────

    def record_day(
        self,
        result: DryRunDayResult,
        path:   Path = DEFAULT_DRYRUN_PATH,
    ) -> None:
        """
        Record one day's result, update counters, persist.

        If result.passed: increment consecutive_successes.
        If not:           reset consecutive_successes to 0.
        Gate is considered met once consecutive_successes reaches 30.
        """
        today = result.date

        if self.start_date is None:
            self.start_date = today

        self.last_run_date  = today
        self.total_days_run += 1
        self.history.append(result.to_dict())

        if result.passed:
            self.consecutive_successes += 1
            if (not self.gate_met and
                    self.consecutive_successes >= SHADOW_GATE_DAYS):
                self.gate_met      = True
                self.gate_met_date = today
        else:
            self.consecutive_successes = 0
            self.total_failures       += 1

        self.save(path)

    # ── Gate enforcement ──────────────────────────────────────────────────────

    def assert_module7_gate(self) -> None:
        """
        Raise ShadowGateNotMet if the 30-day gate has not been met.

        Call this before ANY live exchange code is touched.
        """
        if not self.gate_met:
            remaining = SHADOW_GATE_DAYS - self.consecutive_successes
            raise ShadowGateNotMet(
                f"Shadow deployment gate not met.\n"
                f"  Consecutive successes : {self.consecutive_successes}/{SHADOW_GATE_DAYS}\n"
                f"  Remaining             : {remaining} days\n"
                f"  Total days run        : {self.total_days_run}\n"
                f"  Total failures        : {self.total_failures}\n"
                f"\nNo live capital may be deployed until {SHADOW_GATE_DAYS} "
                f"consecutive dry-run days pass without failure."
            )

    def status_report(self) -> str:
        """Human-readable status for daily reporting."""
        lines = [
            "DRY RUN STATUS",
            f"  Gate met              : {'YES — ' + str(self.gate_met_date) if self.gate_met else 'NO'}",
            f"  Consecutive successes : {self.consecutive_successes}/{SHADOW_GATE_DAYS}",
            f"  Total days run        : {self.total_days_run}",
            f"  Total failures        : {self.total_failures}",
            f"  Start date            : {self.start_date}",
            f"  Last run date         : {self.last_run_date}",
        ]
        if not self.gate_met and self.consecutive_successes > 0:
            lines.append(
                f"  Days until gate       : "
                f"{SHADOW_GATE_DAYS - self.consecutive_successes}"
            )
        if self.total_failures > 0 and self.history:
            failures = [
                DryRunDayResult.from_dict(d)
                for d in self.history
                if not DryRunDayResult.from_dict(d).passed
            ]
            lines.append(f"  Failure dates         : "
                         + ", ".join(f.date for f in failures[-5:]))
        return "\n".join(lines)

    def recent_history(self, n: int = 10) -> list[DryRunDayResult]:
        """Return the N most recent day results."""
        return [DryRunDayResult.from_dict(d) for d in self.history[-n:]]
