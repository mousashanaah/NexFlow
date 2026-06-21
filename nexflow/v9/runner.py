"""
V9 Confidence — Module 5: Allocation Runner

The runner is a state machine, not a calculation.

The calculation (allocate, should_rebalance) lives in core.py.
This module handles state transitions only.

State machine:

    IDLE ──(should_rebalance=True)──► REBALANCE_PENDING
                                              │
                          ┌───────────────────┼───────────────────┐
                          ▼                   ▼                   ▼
                       IDLE           REBALANCE_FAILED     RECOVERY_REQUIRED
                   (confirm_execution)  (mark_failed)         (request_recovery)
                                              │                   │
                                              ▼                   ▼
                                      RECOVERY_REQUIRED         IDLE
                                       (request_recovery)   (execute_recovery)

On startup, if snapshot shows REBALANCE_PENDING (crash during execution),
startup_reconcile() automatically transitions to RECOVERY_REQUIRED.

Hard rules enforced here:
  - No allocation formulas. All decisions come from core.py only.
  - No implicit transitions. Every state change is an explicit method call.
  - Every allocation change produces an AllocationChangeRecord.
  - State file is written atomically before and after every transition.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from nexflow.v9 import core
from nexflow.v9.signals import DailySignalRecord
from nexflow.v9.state import SystemState


# ── States ────────────────────────────────────────────────────────────────────

class RunnerState(str, Enum):
    IDLE               = "IDLE"
    REBALANCE_PENDING  = "REBALANCE_PENDING"
    REBALANCE_FAILED   = "REBALANCE_FAILED"
    RECOVERY_REQUIRED  = "RECOVERY_REQUIRED"


# ── Exceptions ────────────────────────────────────────────────────────────────

class AllocationRunnerError(RuntimeError):
    """Raised on illegal state transitions or precondition violations."""


# ── Audit record ──────────────────────────────────────────────────────────────

@dataclass
class AllocationChangeRecord:
    date:                 str
    prev_wc:              float
    prev_ws:              float
    new_wc:               float
    new_ws:               float
    reason:               str
    trading_days_elapsed: int
    # Forensic context — why the allocation changed
    crypto_score:         float = 0.0
    stock_score:          float = 0.0
    in_bear:              bool  = False

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "AllocationChangeRecord":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_json(cls, s: str) -> "AllocationChangeRecord":
        return cls.from_dict(json.loads(s))


# ── Runner decision (pure, returned by evaluate) ──────────────────────────────

@dataclass
class RunnerDecision:
    should_rebalance:   bool
    target_wc:          float
    target_ws:          float
    reason:             str
    trading_days_since: int


# ── Runner snapshot (persistent state) ───────────────────────────────────────

_SNAPSHOT_VERSION = "1.0"


@dataclass
class RunnerSnapshot:
    runner_state:   RunnerState
    pending_wc:     Optional[float] = None
    pending_ws:     Optional[float] = None
    pending_reason: Optional[str]   = None
    pending_date:   Optional[str]   = None
    version:        str             = _SNAPSHOT_VERSION

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version":       self.version,
            "runner_state":  self.runner_state,
            "pending_wc":    self.pending_wc,
            "pending_ws":    self.pending_ws,
            "pending_reason": self.pending_reason,
            "pending_date":  self.pending_date,
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

    def validate(self) -> None:
        """Enforce state machine invariants. Raises AllocationRunnerError on violation."""
        if self.runner_state == RunnerState.REBALANCE_PENDING:
            if self.pending_wc is None or self.pending_ws is None:
                raise AllocationRunnerError(
                    "Corrupt snapshot: REBALANCE_PENDING requires pending_wc and pending_ws."
                )
            if self.pending_reason is None or self.pending_date is None:
                raise AllocationRunnerError(
                    "Corrupt snapshot: REBALANCE_PENDING requires pending_reason and pending_date."
                )
        if self.runner_state == RunnerState.IDLE:
            if self.pending_wc is not None or self.pending_ws is not None:
                raise AllocationRunnerError(
                    "Corrupt snapshot: IDLE state must not carry pending weights. "
                    "State file may have been manually edited."
                )

    @classmethod
    def load(cls, path: Path) -> "RunnerSnapshot":
        with open(path) as f:
            d = json.load(f)
        snap = cls(
            runner_state   = RunnerState(d["runner_state"]),
            pending_wc     = d.get("pending_wc"),
            pending_ws     = d.get("pending_ws"),
            pending_reason = d.get("pending_reason"),
            pending_date   = d.get("pending_date"),
            version        = d.get("version", _SNAPSHOT_VERSION),
        )
        snap.validate()
        return snap


# ── Audit log helpers ─────────────────────────────────────────────────────────

def append_audit_record(record: AllocationChangeRecord, audit_path: Path) -> None:
    """Append one record to the audit log (JSON Lines). File grows only."""
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_path, "a") as f:
        f.write(record.to_json() + "\n")


def load_audit_log(audit_path: Path) -> list[AllocationChangeRecord]:
    """Return all audit records in chronological order."""
    if not audit_path.exists():
        return []
    records = []
    with open(audit_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(AllocationChangeRecord.from_json(line))
    return records


# ── Allocation runner ─────────────────────────────────────────────────────────

class AllocationRunner:
    """
    State machine wrapper around the allocation decision.

    Never computes allocations directly — always delegates to core.py.
    """

    def __init__(self, snapshot: RunnerSnapshot) -> None:
        self._snap = snapshot

    @property
    def state(self) -> RunnerState:
        return self._snap.runner_state

    @property
    def pending_target(self) -> Optional[tuple[float, float]]:
        """Returns (wc, ws) if a rebalance is pending, else None."""
        if self._snap.pending_wc is None:
            return None
        return (self._snap.pending_wc, self._snap.pending_ws)

    # ── Pure evaluation (no state change) ────────────────────────────────────

    def evaluate(
        self,
        signal:             DailySignalRecord,
        trading_days_since: int,
    ) -> RunnerDecision:
        """
        Determine whether a rebalance is needed and what the target should be.

        Pure — reads signal and calls core.py only.  No side effects.
        """
        wc, ws = core.allocate(signal.crypto_score, signal.stock_score)
        regime = core.allocation_regime_name(signal.crypto_score, signal.stock_score)
        needs  = core.should_rebalance(trading_days_since)

        if needs and trading_days_since > core.REBALANCE_DAYS:
            lateness = trading_days_since - core.REBALANCE_DAYS
            reason   = f"21-day rebalance ({lateness}d late) — {regime}"
        elif needs:
            reason   = f"21-day rebalance — {regime}"
        else:
            reason   = f"no rebalance ({trading_days_since}d since last) — {regime}"

        return RunnerDecision(
            should_rebalance   = needs,
            target_wc          = wc,
            target_ws          = ws,
            reason             = reason,
            trading_days_since = trading_days_since,
        )

    # ── State transitions ─────────────────────────────────────────────────────

    def step(
        self,
        signal:             DailySignalRecord,
        trading_days_since: int,
        snapshot_path:      Path,
    ) -> Optional[RunnerDecision]:
        """
        Evaluate the signal and, if a rebalance is due, transition IDLE →
        REBALANCE_PENDING and return the decision.

        Returns None if no rebalance is needed.
        Raises AllocationRunnerError if not in IDLE state.
        """
        self._assert_state(RunnerState.IDLE, "step")

        decision = self.evaluate(signal, trading_days_since)
        if not decision.should_rebalance:
            return None

        self._snap.runner_state  = RunnerState.REBALANCE_PENDING
        self._snap.pending_wc    = decision.target_wc
        self._snap.pending_ws    = decision.target_ws
        self._snap.pending_reason = decision.reason
        self._snap.pending_date  = signal.date
        self._snap.save(snapshot_path)
        return decision

    def confirm_execution(
        self,
        executed_wc:          float,
        executed_ws:          float,
        date:                 str,
        system_state:         SystemState,
        state_path:           Path,
        snapshot_path:        Path,
        trading_days_elapsed: int,
        audit_path:           Optional[Path] = None,
        crypto_score:         float = 0.0,
        stock_score:          float = 0.0,
        in_bear:              bool  = False,
    ) -> AllocationChangeRecord:
        """
        REBALANCE_PENDING → IDLE.

        Records the previous and new allocation, updates SystemState,
        resets the runner, and appends an audit record.
        """
        self._assert_state(RunnerState.REBALANCE_PENDING, "confirm_execution")

        record = AllocationChangeRecord(
            date                 = date,
            prev_wc              = system_state.allocation.wc,
            prev_ws              = system_state.allocation.ws,
            new_wc               = executed_wc,
            new_ws               = executed_ws,
            reason               = self._snap.pending_reason or "rebalance",
            trading_days_elapsed = trading_days_elapsed,
            crypto_score         = crypto_score,
            stock_score          = stock_score,
            in_bear              = in_bear,
        )

        # Update system state (persists allocation weights)
        system_state.update_allocation(
            wc                 = executed_wc,
            ws                 = executed_ws,
            rebalance_date     = date,
            trading_days_since = 0,
            path               = state_path,
        )

        # Transition runner to IDLE
        self._clear_pending(RunnerState.IDLE, snapshot_path)

        if audit_path is not None:
            append_audit_record(record, audit_path)

        return record

    def mark_failed(self, snapshot_path: Path) -> None:
        """
        REBALANCE_PENDING → REBALANCE_FAILED.

        Call this when execution was attempted but the exchange rejected it,
        or a network error left completion uncertain.
        """
        self._assert_state(RunnerState.REBALANCE_PENDING, "mark_failed")
        self._snap.runner_state = RunnerState.REBALANCE_FAILED
        self._snap.save(snapshot_path)

    def request_recovery(self, snapshot_path: Path) -> None:
        """
        Transition to RECOVERY_REQUIRED from PENDING or FAILED.

        Use when: manual intervention is needed, or startup_reconcile detected
        the system crashed mid-execution.
        """
        _recoverable = {
            RunnerState.REBALANCE_PENDING,
            RunnerState.REBALANCE_FAILED,
            RunnerState.RECOVERY_REQUIRED,
        }
        if self._snap.runner_state not in _recoverable:
            raise AllocationRunnerError(
                f"request_recovery called in state {self._snap.runner_state}. "
                f"Only valid from: {', '.join(s.value for s in _recoverable)}."
            )
        self._snap.runner_state = RunnerState.RECOVERY_REQUIRED
        self._snap.save(snapshot_path)

    def execute_recovery(
        self,
        executed_wc:          float,
        executed_ws:          float,
        date:                 str,
        reason:               str,
        system_state:         SystemState,
        state_path:           Path,
        snapshot_path:        Path,
        trading_days_elapsed: int,
        audit_path:           Optional[Path] = None,
        crypto_score:         float = 0.0,
        stock_score:          float = 0.0,
        in_bear:              bool  = False,
    ) -> AllocationChangeRecord:
        """
        RECOVERY_REQUIRED → IDLE.

        Accepts the actually-executed weights (which may differ from pending
        if recovery used the prior allocation or a manual target).
        """
        self._assert_state(RunnerState.RECOVERY_REQUIRED, "execute_recovery")

        record = AllocationChangeRecord(
            date                 = date,
            prev_wc              = system_state.allocation.wc,
            prev_ws              = system_state.allocation.ws,
            new_wc               = executed_wc,
            new_ws               = executed_ws,
            reason               = f"recovery — {reason}",
            trading_days_elapsed = trading_days_elapsed,
            crypto_score         = crypto_score,
            stock_score          = stock_score,
            in_bear              = in_bear,
        )

        system_state.update_allocation(
            wc                 = executed_wc,
            ws                 = executed_ws,
            rebalance_date     = date,
            trading_days_since = 0,
            path               = state_path,
        )

        self._clear_pending(RunnerState.IDLE, snapshot_path)

        if audit_path is not None:
            append_audit_record(record, audit_path)

        return record

    # ── Factory / startup ─────────────────────────────────────────────────────

    @classmethod
    def initialize(cls, snapshot_path: Path) -> "AllocationRunner":
        """Create a fresh runner in IDLE state."""
        snap = RunnerSnapshot(runner_state=RunnerState.IDLE)
        snap.save(snapshot_path)
        return cls(snap)

    @classmethod
    def startup_reconcile(cls, snapshot_path: Path) -> "AllocationRunner":
        """
        Load runner from disk on system startup.

        If the snapshot shows REBALANCE_PENDING (system crashed between
        step() and confirm_execution()), automatically transitions to
        RECOVERY_REQUIRED.  The operator must then call execute_recovery()
        before normal operation resumes.
        """
        snap   = RunnerSnapshot.load(snapshot_path)
        runner = cls(snap)
        if snap.runner_state == RunnerState.REBALANCE_PENDING:
            runner.request_recovery(snapshot_path)
        return runner

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _assert_state(self, expected: RunnerState, method: str) -> None:
        if self._snap.runner_state != expected:
            raise AllocationRunnerError(
                f"{method}() requires state {expected.value}, "
                f"but current state is {self._snap.runner_state.value}."
            )

    def _clear_pending(self, target_state: RunnerState, snapshot_path: Path) -> None:
        self._snap.runner_state  = target_state
        self._snap.pending_wc    = None
        self._snap.pending_ws    = None
        self._snap.pending_reason = None
        self._snap.pending_date  = None
        self._snap.save(snapshot_path)


# ── Forensic reconstruction ───────────────────────────────────────────────────

from dataclasses import dataclass as _dc


@_dc
class AllocationHistory:
    """Complete reconstructable record of all allocation decisions."""
    current_wc:          float
    current_ws:          float
    current_cash:        float
    previous_wc:         float
    previous_ws:         float
    last_rebalance_date: str
    total_rebalances:    int
    records:             list  # list[AllocationChangeRecord]

    def last_record(self) -> Optional[AllocationChangeRecord]:
        return self.records[-1] if self.records else None


def reconstruct_history(state_path: Path, audit_path: Path) -> AllocationHistory:
    """
    Reconstruct the full allocation history from state.json + audit log.

    Provides:
      - current_wc / current_ws: live allocation from state file
      - previous_wc / previous_ws: allocation before the last change
      - last_rebalance_date: from state file
      - total_rebalances: count of audit records
      - records: ordered list of every AllocationChangeRecord ever written

    This is the forensic entry point.  Given only these two files the
    complete history of every portfolio change is reconstructable.
    """
    state   = SystemState.load(state_path)
    records = load_audit_log(audit_path)

    if records:
        last     = records[-1]
        prev_wc  = last.prev_wc
        prev_ws  = last.prev_ws
    else:
        prev_wc  = state.allocation.wc
        prev_ws  = state.allocation.ws

    wc   = state.allocation.wc
    ws   = state.allocation.ws
    cash = round(1.0 - wc - ws, 10)

    return AllocationHistory(
        current_wc          = wc,
        current_ws          = ws,
        current_cash        = cash,
        previous_wc         = prev_wc,
        previous_ws         = prev_ws,
        last_rebalance_date = state.allocation.last_rebalance_date,
        total_rebalances    = len(records),
        records             = records,
    )
