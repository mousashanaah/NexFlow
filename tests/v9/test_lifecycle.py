"""
V9 Engineering Tests 04, 05, 06 — lifecycle validation.

TEST 04 — Full multi-cycle lifecycle simulation
  Day 0 → IDLE → Day 21 → PENDING → confirm → IDLE
  → Day 42 → PENDING → crash → RECOVERY_REQUIRED → recover → IDLE

TEST 05 — Audit reconstruction
  Given only state.json + audit.jsonl:
  - current allocation recoverable
  - previous allocation recoverable
  - full rebalance history recoverable
  - last successful execution date recoverable
  - forensic context (scores, bear flag) recoverable

TEST 06 — Impossible states enforced by design
  - PENDING with no target weights → load raises
  - IDLE with pending weights set → load raises
  - Double confirm → raises
  - step() from non-IDLE → raises
  - Corrupted snapshot → load raises
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import pytest

_REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO))

from nexflow.v9 import core
from nexflow.v9.runner import (
    AllocationChangeRecord,
    AllocationHistory,
    AllocationRunner,
    AllocationRunnerError,
    RunnerSnapshot,
    RunnerState,
    append_audit_record,
    load_audit_log,
    reconstruct_history,
)
from nexflow.v9.signals import CryptoSignal, DailySignalRecord
from nexflow.v9.state import AllocationSnapshot, RegimeSnapshot, SystemState


# ── helpers ───────────────────────────────────────────────────────────────────

def _ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _signal(date: str = "2024-01-15", c_sc: float = 3.5, s_sc: float = 2.25) -> DailySignalRecord:
    wc, ws = core.allocate(c_sc, s_sc)
    btc = CryptoSignal(
        close=50000.0, sma200=40000.0, mom90=0.30, mom30=0.05,
        atr14=800.0, atr_avg=1200.0,
        pts_sma200=2.0, pts_mom90=1.0, pts_mom30=0.5, pts_vol=0.5,
        pts_bonus=0.0, raw=c_sc, score=c_sc,
    )
    return DailySignalRecord(
        date=date, timestamp_ms=_ms(date), in_bear=False,
        btc=btc, crypto_score=c_sc, stocks=[], stock_score=s_sc,
        allocation_regime=core.allocation_regime_name(c_sc, s_sc),
        wc=wc, ws=ws, cash=round(1.0 - wc - ws, 10),
    )


def _sys_state(wc: float = 0.65, ws: float = 0.35, days: int = 0) -> SystemState:
    ss = SystemState(
        regime=RegimeSnapshot(in_bear=False, consecutive_above=0, last_bar_date="2024-01-01"),
        allocation=AllocationSnapshot(
            wc=wc, ws=ws, last_rebalance_date="2024-01-01", trading_days_since=days,
        ),
    )
    ss.gate_open = True
    return ss


@pytest.fixture
def dirs(tmp_path):
    return {
        "snap":  tmp_path / "runner.json",
        "state": tmp_path / "state.json",
        "audit": tmp_path / "audit.jsonl",
    }


# ── TEST 04: Full multi-cycle lifecycle simulation ─────────────────────────────

class TestLifecycleFull:
    """
    Simulates the complete two-cycle lifecycle:
      Cycle 1: Day 0→21 normal rebalance
      Cycle 2: Day 22→42 rebalance + crash + recovery
    """

    def _run_cycle_1(self, dirs) -> tuple[AllocationRunner, SystemState]:
        """Day 0→21: normal rebalance, confirm, return to IDLE."""
        runner = AllocationRunner.initialize(dirs["snap"])
        ss     = _sys_state(wc=0.50, ws=0.50, days=0)

        # Days 1-20: no rebalance
        for day in range(1, 21):
            sig    = _signal()
            result = runner.step(sig, trading_days_since=day, snapshot_path=dirs["snap"])
            assert result is None, f"Unexpected rebalance on day {day}"

        # Day 21: rebalance due
        sig = _signal(date="2024-02-01", c_sc=3.5, s_sc=2.25)
        dec = runner.step(sig, trading_days_since=21, snapshot_path=dirs["snap"])
        assert dec is not None
        assert runner.state == RunnerState.REBALANCE_PENDING

        # Confirm execution
        runner.confirm_execution(
            executed_wc=dec.target_wc, executed_ws=dec.target_ws,
            date="2024-02-01",
            system_state=ss, state_path=dirs["state"],
            snapshot_path=dirs["snap"],
            trading_days_elapsed=21,
            audit_path=dirs["audit"],
            crypto_score=3.5, stock_score=2.25, in_bear=False,
        )
        assert runner.state == RunnerState.IDLE
        return runner, ss

    def test_cycle_1_ends_in_idle(self, dirs):
        runner, _ = self._run_cycle_1(dirs)
        assert runner.state == RunnerState.IDLE

    def test_cycle_1_audit_has_one_record(self, dirs):
        self._run_cycle_1(dirs)
        log = load_audit_log(dirs["audit"])
        assert len(log) == 1

    def test_cycle_1_audit_record_correct(self, dirs):
        self._run_cycle_1(dirs)
        rec = load_audit_log(dirs["audit"])[0]
        assert rec.date == "2024-02-01"
        assert rec.prev_wc == 0.50
        assert rec.trading_days_elapsed == 21
        assert rec.crypto_score == 3.5

    def test_cycle_2_crash_and_recovery(self, dirs):
        """
        Full cycle 2: Day 42 rebalance → crash before confirm → restart →
        RECOVERY_REQUIRED → execute_recovery → IDLE.
        """
        runner, ss = self._run_cycle_1(dirs)
        # ss now has wc set from cycle 1; reset trading days
        ss.allocation = AllocationSnapshot(
            wc=runner._snap.pending_wc or ss.allocation.wc,
            ws=runner._snap.pending_ws or ss.allocation.ws,
            last_rebalance_date="2024-02-01",
            trading_days_since=0,
        )

        # Days 22-41: no rebalance (0 days since last)
        for day in range(1, 21):
            sig    = _signal()
            result = runner.step(sig, trading_days_since=day, snapshot_path=dirs["snap"])
            assert result is None

        # Day 42 (21st day since last rebalance)
        sig2 = _signal(date="2024-03-01", c_sc=4.0, s_sc=3.0)
        dec2 = runner.step(sig2, trading_days_since=21, snapshot_path=dirs["snap"])
        assert dec2 is not None
        assert runner.state == RunnerState.REBALANCE_PENDING

        # ── CRASH ── simulate by reloading from disk
        restarted = AllocationRunner.startup_reconcile(dirs["snap"])
        assert restarted.state == RunnerState.RECOVERY_REQUIRED

        # Execute recovery
        rec = restarted.execute_recovery(
            executed_wc=dec2.target_wc, executed_ws=dec2.target_ws,
            date="2024-03-01", reason="post-crash recovery",
            system_state=ss, state_path=dirs["state"],
            snapshot_path=dirs["snap"],
            trading_days_elapsed=21,
            audit_path=dirs["audit"],
            crypto_score=4.0, stock_score=3.0, in_bear=False,
        )
        assert restarted.state == RunnerState.IDLE
        assert rec.reason.startswith("recovery — ")

    def test_cycle_2_audit_has_two_records(self, dirs):
        """After both cycles complete, audit must have exactly two records."""
        runner, ss = self._run_cycle_1(dirs)
        ss.allocation = AllocationSnapshot(
            wc=0.65, ws=0.35, last_rebalance_date="2024-02-01", trading_days_since=0,
        )
        sig2 = _signal(date="2024-03-01", c_sc=4.0, s_sc=3.0)
        runner.step(sig2, trading_days_since=21, snapshot_path=dirs["snap"])
        restarted = AllocationRunner.startup_reconcile(dirs["snap"])
        restarted.execute_recovery(
            executed_wc=0.65, executed_ws=0.35,
            date="2024-03-01", reason="crash recovery",
            system_state=ss, state_path=dirs["state"],
            snapshot_path=dirs["snap"],
            trading_days_elapsed=21,
            audit_path=dirs["audit"],
        )
        log = load_audit_log(dirs["audit"])
        assert len(log) == 2

    def test_full_state_machine_sequence_of_states(self, dirs):
        """Assert exact state at each lifecycle step."""
        runner = AllocationRunner.initialize(dirs["snap"])
        ss     = _sys_state()

        assert runner.state == RunnerState.IDLE

        sig = _signal(date="2024-02-01")
        dec = runner.step(sig, trading_days_since=21, snapshot_path=dirs["snap"])
        assert runner.state == RunnerState.REBALANCE_PENDING

        runner.confirm_execution(
            executed_wc=dec.target_wc, executed_ws=dec.target_ws,
            date="2024-02-01",
            system_state=ss, state_path=dirs["state"],
            snapshot_path=dirs["snap"],
            trading_days_elapsed=21,
        )
        assert runner.state == RunnerState.IDLE

        sig2 = _signal(date="2024-03-01")
        runner.step(sig2, trading_days_since=21, snapshot_path=dirs["snap"])
        assert runner.state == RunnerState.REBALANCE_PENDING

        # Crash
        r2 = AllocationRunner.startup_reconcile(dirs["snap"])
        assert r2.state == RunnerState.RECOVERY_REQUIRED

        r2.execute_recovery(
            executed_wc=0.65, executed_ws=0.35,
            date="2024-03-01", reason="post-crash",
            system_state=ss, state_path=dirs["state"],
            snapshot_path=dirs["snap"],
            trading_days_elapsed=21,
        )
        assert r2.state == RunnerState.IDLE

    def test_cycle_1_no_rebalance_before_day_21(self, dirs):
        runner = AllocationRunner.initialize(dirs["snap"])
        for day in range(0, 21):
            sig    = _signal()
            result = runner.step(sig, trading_days_since=day, snapshot_path=dirs["snap"])
            assert result is None, f"Should not rebalance on day {day}"

    def test_mark_failed_then_recovery_cycle(self, dirs):
        """Explicit FAILED path: PENDING → FAILED → RECOVERY_REQUIRED → IDLE."""
        runner = AllocationRunner.initialize(dirs["snap"])
        ss     = _sys_state()

        sig = _signal()
        runner.step(sig, trading_days_since=21, snapshot_path=dirs["snap"])
        assert runner.state == RunnerState.REBALANCE_PENDING

        runner.mark_failed(dirs["snap"])
        assert runner.state == RunnerState.REBALANCE_FAILED

        runner.request_recovery(dirs["snap"])
        assert runner.state == RunnerState.RECOVERY_REQUIRED

        runner.execute_recovery(
            executed_wc=0.65, executed_ws=0.35,
            date="2024-02-01", reason="manual after failure",
            system_state=ss, state_path=dirs["state"],
            snapshot_path=dirs["snap"],
            trading_days_elapsed=21,
        )
        assert runner.state == RunnerState.IDLE


# ── TEST 05: Audit reconstruction ─────────────────────────────────────────────

class TestAuditReconstruction:
    """
    Given only state.json + audit.jsonl, every allocation decision
    must be reconstructable.
    """

    def _build_two_cycle_audit(self, dirs) -> None:
        """Write two audit records with different forensic context."""
        rec1 = AllocationChangeRecord(
            date="2024-02-01", prev_wc=0.50, prev_ws=0.50,
            new_wc=0.65, new_ws=0.35,
            reason="21-day rebalance — BOTH_HOT",
            trading_days_elapsed=21,
            crypto_score=3.5, stock_score=2.25, in_bear=False,
        )
        rec2 = AllocationChangeRecord(
            date="2024-03-01", prev_wc=0.65, prev_ws=0.35,
            new_wc=0.40, new_ws=0.40,
            reason="21-day rebalance — DEFENSIVE",
            trading_days_elapsed=21,
            crypto_score=0.5, stock_score=0.4, in_bear=False,
        )
        append_audit_record(rec1, dirs["audit"])
        append_audit_record(rec2, dirs["audit"])

        # Write a state.json reflecting the current (post-cycle-2) allocation
        ss = _sys_state(wc=0.40, ws=0.40, days=5)
        ss.save(dirs["state"])

    def test_current_allocation_recoverable(self, dirs):
        self._build_two_cycle_audit(dirs)
        history = reconstruct_history(dirs["state"], dirs["audit"])
        assert history.current_wc == 0.40
        assert history.current_ws == 0.40

    def test_current_cash_recoverable(self, dirs):
        self._build_two_cycle_audit(dirs)
        history = reconstruct_history(dirs["state"], dirs["audit"])
        assert abs(history.current_cash - 0.20) < 1e-9

    def test_previous_allocation_recoverable(self, dirs):
        self._build_two_cycle_audit(dirs)
        history = reconstruct_history(dirs["state"], dirs["audit"])
        assert history.previous_wc == 0.65
        assert history.previous_ws == 0.35

    def test_last_rebalance_date_recoverable(self, dirs):
        self._build_two_cycle_audit(dirs)
        history = reconstruct_history(dirs["state"], dirs["audit"])
        assert history.last_rebalance_date == "2024-01-01"  # from state.json

    def test_rebalance_count_recoverable(self, dirs):
        self._build_two_cycle_audit(dirs)
        history = reconstruct_history(dirs["state"], dirs["audit"])
        assert history.total_rebalances == 2

    def test_full_history_recoverable(self, dirs):
        self._build_two_cycle_audit(dirs)
        history = reconstruct_history(dirs["state"], dirs["audit"])
        assert len(history.records) == 2
        assert history.records[0].date == "2024-02-01"
        assert history.records[1].date == "2024-03-01"

    def test_last_successful_execution_recoverable(self, dirs):
        self._build_two_cycle_audit(dirs)
        history = reconstruct_history(dirs["state"], dirs["audit"])
        last = history.last_record()
        assert last is not None
        assert last.date == "2024-03-01"

    def test_forensic_context_recoverable(self, dirs):
        """Scores and bear flag must survive round-trip through audit log."""
        self._build_two_cycle_audit(dirs)
        history = reconstruct_history(dirs["state"], dirs["audit"])
        rec1 = history.records[0]
        assert rec1.crypto_score == 3.5
        assert rec1.stock_score  == 2.25
        assert rec1.in_bear      is False

    def test_why_allocation_changed_recoverable(self, dirs):
        """reason field must tell WHY the allocation changed."""
        self._build_two_cycle_audit(dirs)
        history = reconstruct_history(dirs["state"], dirs["audit"])
        assert "BOTH_HOT" in history.records[0].reason
        assert "DEFENSIVE" in history.records[1].reason

    def test_transition_from_to_visible_in_records(self, dirs):
        """Each record shows the full before/after transition."""
        self._build_two_cycle_audit(dirs)
        history = reconstruct_history(dirs["state"], dirs["audit"])
        r = history.records[1]
        assert r.prev_wc == 0.65  # was BOTH_HOT
        assert r.new_wc  == 0.40  # now DEFENSIVE

    def test_empty_audit_log_handled(self, dirs):
        """No audit records yet — current allocation is initial allocation."""
        ss = _sys_state(wc=0.50, ws=0.50, days=0)
        ss.save(dirs["state"])
        history = reconstruct_history(dirs["state"], dirs["audit"])
        assert history.total_rebalances == 0
        assert history.current_wc == 0.50
        # previous == current when no records
        assert history.previous_wc == 0.50

    def test_record_chronological_order_preserved(self, dirs):
        for i in range(5):
            rec = AllocationChangeRecord(
                date=f"2024-0{i+1}-01", prev_wc=0.50, prev_ws=0.50,
                new_wc=0.65, new_ws=0.35, reason=f"rebalance {i+1}",
                trading_days_elapsed=21,
            )
            append_audit_record(rec, dirs["audit"])
        ss = _sys_state()
        ss.save(dirs["state"])
        history = reconstruct_history(dirs["state"], dirs["audit"])
        dates = [r.date for r in history.records]
        assert dates == sorted(dates)

    def test_recovery_record_identifiable(self, dirs):
        """Recovery records must be distinguishable from normal rebalances."""
        rec = AllocationChangeRecord(
            date="2024-03-01", prev_wc=0.65, prev_ws=0.35,
            new_wc=0.65, new_ws=0.35,
            reason="recovery — post-crash recovery",
            trading_days_elapsed=21,
        )
        append_audit_record(rec, dirs["audit"])
        ss = _sys_state()
        ss.save(dirs["state"])
        history = reconstruct_history(dirs["state"], dirs["audit"])
        assert history.records[0].reason.startswith("recovery — ")


# ── TEST 06: Impossible states enforced by design ─────────────────────────────

class TestImpossibleStates:
    """
    Verify that the state machine enforces its invariants.
    Every impossible state below must be unreachable through the public API
    or must raise when attempted via direct snapshot manipulation.
    """

    def test_pending_with_no_target_wc_raises_on_load(self, dirs):
        """REBALANCE_PENDING without pending_wc is corrupt. load() must raise."""
        payload = {
            "version": "1.0",
            "runner_state": "REBALANCE_PENDING",
            "pending_wc": None,       # ← corrupt: missing target
            "pending_ws": None,
            "pending_reason": None,
            "pending_date": None,
        }
        dirs["snap"].parent.mkdir(parents=True, exist_ok=True)
        dirs["snap"].write_text(json.dumps(payload))
        with pytest.raises(AllocationRunnerError, match="pending_wc"):
            RunnerSnapshot.load(dirs["snap"])

    def test_pending_with_partial_target_raises(self, dirs):
        """PENDING with wc but no ws is also corrupt."""
        payload = {
            "version": "1.0",
            "runner_state": "REBALANCE_PENDING",
            "pending_wc": 0.65,
            "pending_ws": None,      # ← missing
            "pending_reason": "test",
            "pending_date": "2024-01-01",
        }
        dirs["snap"].parent.mkdir(parents=True, exist_ok=True)
        dirs["snap"].write_text(json.dumps(payload))
        with pytest.raises(AllocationRunnerError, match="pending_ws"):
            RunnerSnapshot.load(dirs["snap"])

    def test_pending_without_reason_raises(self, dirs):
        """PENDING without reason/date is corrupt."""
        payload = {
            "version": "1.0",
            "runner_state": "REBALANCE_PENDING",
            "pending_wc": 0.65,
            "pending_ws": 0.35,
            "pending_reason": None,  # ← missing
            "pending_date": None,
        }
        dirs["snap"].parent.mkdir(parents=True, exist_ok=True)
        dirs["snap"].write_text(json.dumps(payload))
        with pytest.raises(AllocationRunnerError, match="pending_reason"):
            RunnerSnapshot.load(dirs["snap"])

    def test_idle_with_stale_pending_weights_raises(self, dirs):
        """IDLE must not carry pending weights — indicates half-cleared state."""
        payload = {
            "version": "1.0",
            "runner_state": "IDLE",
            "pending_wc": 0.65,      # ← stale, should have been cleared
            "pending_ws": 0.35,
            "pending_reason": None,
            "pending_date": None,
        }
        dirs["snap"].parent.mkdir(parents=True, exist_ok=True)
        dirs["snap"].write_text(json.dumps(payload))
        with pytest.raises(AllocationRunnerError, match="IDLE"):
            RunnerSnapshot.load(dirs["snap"])

    def test_step_from_pending_raises(self, dirs):
        """IDLE → PENDING via step(), then step() again must raise."""
        runner = AllocationRunner.initialize(dirs["snap"])
        sig    = _signal()
        runner.step(sig, trading_days_since=21, snapshot_path=dirs["snap"])
        with pytest.raises(AllocationRunnerError, match="REBALANCE_PENDING"):
            runner.step(sig, trading_days_since=21, snapshot_path=dirs["snap"])

    def test_step_from_recovery_required_raises(self, dirs):
        """Cannot step while in RECOVERY_REQUIRED."""
        runner = AllocationRunner.initialize(dirs["snap"])
        sig    = _signal()
        runner.step(sig, trading_days_since=21, snapshot_path=dirs["snap"])
        runner.request_recovery(dirs["snap"])
        with pytest.raises(AllocationRunnerError, match="RECOVERY_REQUIRED"):
            runner.step(sig, trading_days_since=21, snapshot_path=dirs["snap"])

    def test_step_from_failed_raises(self, dirs):
        """Cannot step while in REBALANCE_FAILED."""
        runner = AllocationRunner.initialize(dirs["snap"])
        sig    = _signal()
        runner.step(sig, trading_days_since=21, snapshot_path=dirs["snap"])
        runner.mark_failed(dirs["snap"])
        with pytest.raises(AllocationRunnerError, match="REBALANCE_FAILED"):
            runner.step(sig, trading_days_since=21, snapshot_path=dirs["snap"])

    def test_double_confirm_raises(self, dirs):
        """confirm_execution() twice must raise on the second call."""
        runner = AllocationRunner.initialize(dirs["snap"])
        ss     = _sys_state()
        sig    = _signal(date="2024-02-01")
        runner.step(sig, trading_days_since=21, snapshot_path=dirs["snap"])
        runner.confirm_execution(
            executed_wc=0.65, executed_ws=0.35,
            date="2024-02-01",
            system_state=ss, state_path=dirs["state"],
            snapshot_path=dirs["snap"],
            trading_days_elapsed=21,
        )
        with pytest.raises(AllocationRunnerError):
            runner.confirm_execution(
                executed_wc=0.65, executed_ws=0.35,
                date="2024-02-01",
                system_state=ss, state_path=dirs["state"],
                snapshot_path=dirs["snap"],
                trading_days_elapsed=21,
            )

    def test_confirm_on_idle_raises(self, dirs):
        """confirm_execution on IDLE (no pending rebalance) must raise."""
        runner = AllocationRunner.initialize(dirs["snap"])
        ss     = _sys_state()
        with pytest.raises(AllocationRunnerError, match="IDLE"):
            runner.confirm_execution(
                executed_wc=0.65, executed_ws=0.35,
                date="2024-02-01",
                system_state=ss, state_path=dirs["state"],
                snapshot_path=dirs["snap"],
                trading_days_elapsed=21,
            )

    def test_recover_from_idle_raises(self, dirs):
        """request_recovery() on IDLE is illegal."""
        runner = AllocationRunner.initialize(dirs["snap"])
        with pytest.raises(AllocationRunnerError):
            runner.request_recovery(dirs["snap"])

    def test_execute_recovery_on_idle_raises(self, dirs):
        """execute_recovery() on IDLE is illegal."""
        runner = AllocationRunner.initialize(dirs["snap"])
        ss     = _sys_state()
        with pytest.raises(AllocationRunnerError, match="IDLE"):
            runner.execute_recovery(
                executed_wc=0.65, executed_ws=0.35,
                date="2024-02-01", reason="test",
                system_state=ss, state_path=dirs["state"],
                snapshot_path=dirs["snap"],
                trading_days_elapsed=21,
            )

    def test_mark_failed_on_idle_raises(self, dirs):
        runner = AllocationRunner.initialize(dirs["snap"])
        with pytest.raises(AllocationRunnerError):
            runner.mark_failed(dirs["snap"])

    def test_mark_failed_on_recovery_required_raises(self, dirs):
        runner = AllocationRunner.initialize(dirs["snap"])
        sig    = _signal()
        runner.step(sig, trading_days_since=21, snapshot_path=dirs["snap"])
        runner.request_recovery(dirs["snap"])
        with pytest.raises(AllocationRunnerError, match="RECOVERY_REQUIRED"):
            runner.mark_failed(dirs["snap"])

    def test_pending_weights_cleared_after_confirm(self, dirs):
        """After confirm, pending_wc/ws must be None — no stale data."""
        runner = AllocationRunner.initialize(dirs["snap"])
        ss     = _sys_state()
        sig    = _signal(date="2024-02-01")
        runner.step(sig, trading_days_since=21, snapshot_path=dirs["snap"])
        runner.confirm_execution(
            executed_wc=0.65, executed_ws=0.35,
            date="2024-02-01",
            system_state=ss, state_path=dirs["state"],
            snapshot_path=dirs["snap"],
            trading_days_elapsed=21,
        )
        snap = RunnerSnapshot.load(dirs["snap"])
        assert snap.pending_wc is None
        assert snap.pending_ws is None
        assert snap.pending_reason is None

    def test_valid_pending_snapshot_loads_without_error(self, dirs):
        """Verify a valid PENDING snapshot passes validation."""
        payload = {
            "version": "1.0",
            "runner_state": "REBALANCE_PENDING",
            "pending_wc": 0.65,
            "pending_ws": 0.35,
            "pending_reason": "21-day rebalance — BOTH_HOT",
            "pending_date": "2024-01-01",
        }
        dirs["snap"].parent.mkdir(parents=True, exist_ok=True)
        dirs["snap"].write_text(json.dumps(payload))
        snap = RunnerSnapshot.load(dirs["snap"])  # must not raise
        assert snap.runner_state == RunnerState.REBALANCE_PENDING
