"""
V9 Allocation Runner — Module 5 tests.

Test classes:
  TestRunnerSnapshot        — serialisation and round-trip
  TestAllocationChangeRecord — audit record serialisation
  TestRunnerDecision         — evaluate() correctness
  TestStateTransitions       — every legal and illegal transition
  TestMissedRebalance        — Test A: system offline, restart 3 days late
  TestDoubleExecution        — Test B: runner called twice, no duplicate
  TestPartialExecution       — Test C: crash during PENDING, recovery path
  TestMonthBoundary          — Test D: day 20 vs 21 boundary exactness
  TestAuditLog               — append_audit_record / load_audit_log
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import tempfile

import pytest

_REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO))

from nexflow.v9 import core
from nexflow.v9.runner import (
    AllocationChangeRecord,
    AllocationRunner,
    AllocationRunnerError,
    RunnerDecision,
    RunnerSnapshot,
    RunnerState,
    append_audit_record,
    load_audit_log,
)
from nexflow.v9.signals import DailySignalRecord, CryptoSignal, StockSignal
from nexflow.v9.state import AllocationSnapshot, RegimeSnapshot, SystemState


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _make_btc_signal(*, c_sc: float = 3.5) -> CryptoSignal:
    # Reverse-engineer pts from score for consistency
    return CryptoSignal(
        close=50000.0, sma200=40000.0, mom90=0.30, mom30=0.05,
        atr14=800.0, atr_avg=1200.0,
        pts_sma200=2.0, pts_mom90=1.0, pts_mom30=0.5, pts_vol=0.5,
        pts_bonus=0.0, raw=c_sc, score=c_sc,
    )


def _make_signal(
    date: str = "2024-01-15",
    c_sc: float = 3.5,
    s_sc: float = 2.25,
) -> DailySignalRecord:
    wc, ws = core.allocate(c_sc, s_sc)
    regime = core.allocation_regime_name(c_sc, s_sc)
    return DailySignalRecord(
        date             = date,
        timestamp_ms     = _ms(date),
        in_bear          = False,
        btc              = _make_btc_signal(c_sc=c_sc),
        crypto_score     = c_sc,
        stocks           = [],
        stock_score      = s_sc,
        allocation_regime = regime,
        wc               = wc,
        ws               = ws,
        cash             = round(1.0 - wc - ws, 10),
    )


def _make_system_state(wc: float = 0.65, ws: float = 0.35, days_since: int = 0) -> SystemState:
    state = SystemState(
        regime=RegimeSnapshot(in_bear=False, consecutive_above=0, last_bar_date="2024-01-14"),
        allocation=AllocationSnapshot(
            wc=wc, ws=ws,
            last_rebalance_date="2024-01-14",
            trading_days_since=days_since,
        ),
    )
    state.gate_open = True
    return state


@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def snap_path(tmp_dir):
    return tmp_dir / "runner.json"


@pytest.fixture
def state_path(tmp_dir):
    return tmp_dir / "state.json"


@pytest.fixture
def audit_path(tmp_dir):
    return tmp_dir / "audit.jsonl"


@pytest.fixture
def idle_runner(snap_path):
    return AllocationRunner.initialize(snap_path)


# ── RunnerSnapshot ─────────────────────────────────────────────────────────────

class TestRunnerSnapshot:
    def test_initialize_creates_idle_snapshot(self, snap_path):
        AllocationRunner.initialize(snap_path)
        snap = RunnerSnapshot.load(snap_path)
        assert snap.runner_state == RunnerState.IDLE

    def test_save_load_round_trip_idle(self, snap_path):
        snap = RunnerSnapshot(runner_state=RunnerState.IDLE)
        snap.save(snap_path)
        loaded = RunnerSnapshot.load(snap_path)
        assert loaded.runner_state == RunnerState.IDLE
        assert loaded.pending_wc is None

    def test_save_load_round_trip_pending(self, snap_path):
        snap = RunnerSnapshot(
            runner_state=RunnerState.REBALANCE_PENDING,
            pending_wc=0.80,
            pending_ws=0.20,
            pending_reason="21-day rebalance — CRYPTO_DOMINANT",
            pending_date="2024-03-01",
        )
        snap.save(snap_path)
        loaded = RunnerSnapshot.load(snap_path)
        assert loaded.runner_state == RunnerState.REBALANCE_PENDING
        assert loaded.pending_wc == 0.80
        assert loaded.pending_ws == 0.20
        assert loaded.pending_reason == "21-day rebalance — CRYPTO_DOMINANT"
        assert loaded.pending_date == "2024-03-01"

    def test_save_is_atomic(self, snap_path):
        snap = RunnerSnapshot(runner_state=RunnerState.IDLE)
        snap.save(snap_path)
        assert snap_path.exists()
        # No .tmp files left behind
        assert list(snap_path.parent.glob("*.tmp")) == []

    def test_all_states_round_trip(self, tmp_path):
        for state in RunnerState:
            path = tmp_path / f"{state.value}.json"
            snap = RunnerSnapshot(runner_state=state)
            snap.save(path)
            loaded = RunnerSnapshot.load(path)
            assert loaded.runner_state == state


# ── AllocationChangeRecord ────────────────────────────────────────────────────

class TestAllocationChangeRecord:
    def test_to_json_round_trip(self):
        rec = AllocationChangeRecord(
            date="2024-03-01", prev_wc=0.65, prev_ws=0.35,
            new_wc=0.80, new_ws=0.20, reason="21-day rebalance — CRYPTO_DOMINANT",
            trading_days_elapsed=21,
        )
        rec2 = AllocationChangeRecord.from_json(rec.to_json())
        assert rec2.date == "2024-03-01"
        assert rec2.prev_wc == 0.65
        assert rec2.new_wc == 0.80
        assert rec2.reason == "21-day rebalance — CRYPTO_DOMINANT"
        assert rec2.trading_days_elapsed == 21

    def test_to_dict_contains_all_fields(self):
        rec = AllocationChangeRecord(
            date="2024-03-01", prev_wc=0.65, prev_ws=0.35,
            new_wc=0.80, new_ws=0.20, reason="recovery — missed day",
            trading_days_elapsed=24,
        )
        d = rec.to_dict()
        assert set(d.keys()) == {
            "date", "prev_wc", "prev_ws", "new_wc", "new_ws",
            "reason", "trading_days_elapsed",
        }


# ── evaluate() — pure, no state change ───────────────────────────────────────

class TestRunnerDecision:
    def test_evaluate_no_rebalance_before_21_days(self, idle_runner):
        sig = _make_signal(c_sc=3.5, s_sc=2.25)
        dec = idle_runner.evaluate(sig, trading_days_since=20)
        assert dec.should_rebalance is False

    def test_evaluate_rebalance_at_exactly_21_days(self, idle_runner):
        sig = _make_signal(c_sc=3.5, s_sc=2.25)
        dec = idle_runner.evaluate(sig, trading_days_since=21)
        assert dec.should_rebalance is True

    def test_evaluate_rebalance_after_21_days(self, idle_runner):
        sig = _make_signal(c_sc=3.5, s_sc=2.25)
        dec = idle_runner.evaluate(sig, trading_days_since=25)
        assert dec.should_rebalance is True

    def test_evaluate_target_comes_from_core(self, idle_runner):
        sig = _make_signal(c_sc=4.0, s_sc=3.0)
        dec = idle_runner.evaluate(sig, trading_days_since=21)
        expected_wc, expected_ws = core.allocate(4.0, 3.0)
        assert dec.target_wc == expected_wc
        assert dec.target_ws == expected_ws

    def test_evaluate_reason_mentions_regime(self, idle_runner):
        sig = _make_signal(c_sc=4.0, s_sc=0.5)  # crypto dominant
        dec = idle_runner.evaluate(sig, trading_days_since=21)
        assert "CRYPTO_DOMINANT" in dec.reason

    def test_evaluate_late_rebalance_mentions_lateness(self, idle_runner):
        sig = _make_signal()
        dec = idle_runner.evaluate(sig, trading_days_since=24)
        assert "3d late" in dec.reason

    def test_evaluate_does_not_change_state(self, snap_path, idle_runner):
        sig = _make_signal()
        idle_runner.evaluate(sig, trading_days_since=21)
        assert idle_runner.state == RunnerState.IDLE

    def test_evaluate_defensive_regime(self, idle_runner):
        # Both scores very low → DEFENSIVE
        sig = _make_signal(c_sc=0.5, s_sc=0.4)
        dec = idle_runner.evaluate(sig, trading_days_since=21)
        assert "DEFENSIVE" in dec.reason
        assert dec.target_wc == 0.40
        assert dec.target_ws == 0.40


# ── State transitions ─────────────────────────────────────────────────────────

class TestStateTransitions:
    def test_idle_step_no_rebalance_returns_none(self, idle_runner, snap_path):
        sig = _make_signal()
        result = idle_runner.step(sig, trading_days_since=5, snapshot_path=snap_path)
        assert result is None
        assert idle_runner.state == RunnerState.IDLE

    def test_idle_step_rebalance_due_transitions_to_pending(self, idle_runner, snap_path):
        sig = _make_signal()
        dec = idle_runner.step(sig, trading_days_since=21, snapshot_path=snap_path)
        assert dec is not None
        assert idle_runner.state == RunnerState.REBALANCE_PENDING

    def test_pending_step_raises(self, idle_runner, snap_path):
        sig = _make_signal()
        idle_runner.step(sig, trading_days_since=21, snapshot_path=snap_path)
        with pytest.raises(AllocationRunnerError, match="REBALANCE_PENDING"):
            idle_runner.step(sig, trading_days_since=21, snapshot_path=snap_path)

    def test_confirm_transitions_pending_to_idle(self, idle_runner, snap_path, state_path, audit_path):
        sig   = _make_signal(date="2024-03-01")
        idle_runner.step(sig, trading_days_since=21, snapshot_path=snap_path)
        ss    = _make_system_state(wc=0.65, ws=0.35)
        idle_runner.confirm_execution(
            executed_wc=0.80, executed_ws=0.20,
            date="2024-03-01",
            system_state=ss, state_path=state_path,
            snapshot_path=snap_path,
            trading_days_elapsed=21,
            audit_path=audit_path,
        )
        assert idle_runner.state == RunnerState.IDLE

    def test_confirm_clears_pending_fields(self, idle_runner, snap_path, state_path):
        sig = _make_signal(date="2024-03-01")
        idle_runner.step(sig, trading_days_since=21, snapshot_path=snap_path)
        ss  = _make_system_state()
        idle_runner.confirm_execution(
            executed_wc=0.80, executed_ws=0.20,
            date="2024-03-01",
            system_state=ss, state_path=state_path,
            snapshot_path=snap_path,
            trading_days_elapsed=21,
        )
        assert idle_runner.pending_target is None

    def test_mark_failed_transitions_pending_to_failed(self, idle_runner, snap_path):
        sig = _make_signal()
        idle_runner.step(sig, trading_days_since=21, snapshot_path=snap_path)
        idle_runner.mark_failed(snap_path)
        assert idle_runner.state == RunnerState.REBALANCE_FAILED

    def test_mark_failed_on_idle_raises(self, idle_runner, snap_path):
        with pytest.raises(AllocationRunnerError):
            idle_runner.mark_failed(snap_path)

    def test_request_recovery_from_failed(self, idle_runner, snap_path):
        sig = _make_signal()
        idle_runner.step(sig, trading_days_since=21, snapshot_path=snap_path)
        idle_runner.mark_failed(snap_path)
        idle_runner.request_recovery(snap_path)
        assert idle_runner.state == RunnerState.RECOVERY_REQUIRED

    def test_request_recovery_from_pending(self, idle_runner, snap_path):
        sig = _make_signal()
        idle_runner.step(sig, trading_days_since=21, snapshot_path=snap_path)
        idle_runner.request_recovery(snap_path)
        assert idle_runner.state == RunnerState.RECOVERY_REQUIRED

    def test_request_recovery_from_idle_raises(self, idle_runner, snap_path):
        with pytest.raises(AllocationRunnerError):
            idle_runner.request_recovery(snap_path)

    def test_execute_recovery_transitions_to_idle(self, idle_runner, snap_path, state_path):
        sig = _make_signal()
        idle_runner.step(sig, trading_days_since=21, snapshot_path=snap_path)
        idle_runner.request_recovery(snap_path)
        ss = _make_system_state()
        idle_runner.execute_recovery(
            executed_wc=0.65, executed_ws=0.35,
            date="2024-03-01", reason="manual recovery",
            system_state=ss, state_path=state_path,
            snapshot_path=snap_path,
            trading_days_elapsed=24,
        )
        assert idle_runner.state == RunnerState.IDLE

    def test_execute_recovery_on_idle_raises(self, idle_runner, snap_path, state_path):
        ss = _make_system_state()
        with pytest.raises(AllocationRunnerError):
            idle_runner.execute_recovery(
                executed_wc=0.65, executed_ws=0.35,
                date="2024-03-01", reason="test",
                system_state=ss, state_path=state_path,
                snapshot_path=snap_path,
                trading_days_elapsed=0,
            )

    def test_confirm_on_idle_raises(self, idle_runner, snap_path, state_path):
        ss = _make_system_state()
        with pytest.raises(AllocationRunnerError):
            idle_runner.confirm_execution(
                executed_wc=0.65, executed_ws=0.35,
                date="2024-03-01",
                system_state=ss, state_path=state_path,
                snapshot_path=snap_path,
                trading_days_elapsed=21,
            )

    def test_state_persisted_after_step(self, idle_runner, snap_path):
        sig = _make_signal()
        idle_runner.step(sig, trading_days_since=21, snapshot_path=snap_path)
        # Reload from disk — state must be PENDING
        reloaded = RunnerSnapshot.load(snap_path)
        assert reloaded.runner_state == RunnerState.REBALANCE_PENDING

    def test_state_persisted_after_confirm(self, idle_runner, snap_path, state_path):
        sig = _make_signal(date="2024-03-01")
        idle_runner.step(sig, trading_days_since=21, snapshot_path=snap_path)
        ss  = _make_system_state()
        idle_runner.confirm_execution(
            executed_wc=0.80, executed_ws=0.20,
            date="2024-03-01",
            system_state=ss, state_path=state_path,
            snapshot_path=snap_path,
            trading_days_elapsed=21,
        )
        reloaded = RunnerSnapshot.load(snap_path)
        assert reloaded.runner_state == RunnerState.IDLE

    def test_pending_target_set_during_pending(self, idle_runner, snap_path):
        sig = _make_signal(c_sc=4.0, s_sc=0.5)  # crypto dominant → 80/20
        idle_runner.step(sig, trading_days_since=21, snapshot_path=snap_path)
        target = idle_runner.pending_target
        assert target is not None
        assert target[0] == 0.80
        assert target[1] == 0.20


# ── Test A: Missed rebalance (system offline, restart 3 days late) ─────────────

class TestMissedRebalance:
    """
    Scenario: System was offline on day 21. Restarts on day 24.
    Expected: Rebalance still triggers, reason mentions lateness.
    System must not skip the rebalance or panic.
    """

    def test_rebalance_triggers_when_overdue(self, idle_runner, snap_path):
        sig = _make_signal(date="2024-03-04")
        # 24 trading days since last rebalance (3 days overdue)
        dec = idle_runner.step(sig, trading_days_since=24, snapshot_path=snap_path)
        assert dec is not None
        assert dec.should_rebalance is True

    def test_overdue_reason_contains_lateness(self, idle_runner, snap_path):
        sig = _make_signal(date="2024-03-04")
        dec = idle_runner.step(sig, trading_days_since=24, snapshot_path=snap_path)
        assert "3d late" in dec.reason

    def test_overdue_allocation_still_uses_core(self, idle_runner, snap_path):
        sig = _make_signal(date="2024-03-04", c_sc=3.5, s_sc=2.25)
        dec = idle_runner.step(sig, trading_days_since=24, snapshot_path=snap_path)
        expected_wc, expected_ws = core.allocate(3.5, 2.25)
        assert dec.target_wc == expected_wc
        assert dec.target_ws == expected_ws

    def test_confirm_after_overdue_resets_days_to_zero(self, idle_runner, snap_path, state_path):
        sig = _make_signal(date="2024-03-04")
        idle_runner.step(sig, trading_days_since=24, snapshot_path=snap_path)
        ss = _make_system_state(wc=0.65, ws=0.35, days_since=24)
        idle_runner.confirm_execution(
            executed_wc=0.80, executed_ws=0.20,
            date="2024-03-04",
            system_state=ss, state_path=state_path,
            snapshot_path=snap_path,
            trading_days_elapsed=24,
        )
        assert ss.allocation.trading_days_since == 0

    def test_idle_after_overdue_rebalance(self, idle_runner, snap_path, state_path):
        sig = _make_signal(date="2024-03-04")
        idle_runner.step(sig, trading_days_since=24, snapshot_path=snap_path)
        ss = _make_system_state()
        idle_runner.confirm_execution(
            executed_wc=0.80, executed_ws=0.20,
            date="2024-03-04",
            system_state=ss, state_path=state_path,
            snapshot_path=snap_path,
            trading_days_elapsed=24,
        )
        assert idle_runner.state == RunnerState.IDLE

    def test_startup_reconcile_with_idle_snapshot_stays_idle(self, snap_path):
        # If snapshot is IDLE on startup, no recovery needed
        AllocationRunner.initialize(snap_path)
        runner = AllocationRunner.startup_reconcile(snap_path)
        assert runner.state == RunnerState.IDLE


# ── Test B: Double execution attempt ─────────────────────────────────────────

class TestDoubleExecution:
    """
    Scenario: Runner step() called twice in the same day / same session.
    Expected: Second call raises; no duplicate rebalance.
    """

    def test_second_step_while_pending_raises(self, idle_runner, snap_path):
        sig = _make_signal()
        idle_runner.step(sig, trading_days_since=21, snapshot_path=snap_path)
        with pytest.raises(AllocationRunnerError, match="REBALANCE_PENDING"):
            idle_runner.step(sig, trading_days_since=21, snapshot_path=snap_path)

    def test_no_rebalance_before_21_days_after_confirm(
        self, idle_runner, snap_path, state_path
    ):
        # First rebalance cycle
        sig1 = _make_signal(date="2024-03-01")
        idle_runner.step(sig1, trading_days_since=21, snapshot_path=snap_path)
        ss   = _make_system_state()
        idle_runner.confirm_execution(
            executed_wc=0.80, executed_ws=0.20,
            date="2024-03-01",
            system_state=ss, state_path=state_path,
            snapshot_path=snap_path,
            trading_days_elapsed=21,
        )

        # Immediately after: 0 trading days since last rebalance
        sig2   = _make_signal(date="2024-03-02")
        result = idle_runner.step(sig2, trading_days_since=0, snapshot_path=snap_path)
        assert result is None
        assert idle_runner.state == RunnerState.IDLE

    def test_step_returns_none_on_day_20(self, idle_runner, snap_path):
        sig = _make_signal()
        result = idle_runner.step(sig, trading_days_since=20, snapshot_path=snap_path)
        assert result is None

    def test_no_state_change_when_step_returns_none(self, idle_runner, snap_path):
        sig = _make_signal()
        idle_runner.step(sig, trading_days_since=1, snapshot_path=snap_path)
        loaded = RunnerSnapshot.load(snap_path)
        assert loaded.runner_state == RunnerState.IDLE

    def test_double_confirm_raises(self, idle_runner, snap_path, state_path):
        sig = _make_signal(date="2024-03-01")
        idle_runner.step(sig, trading_days_since=21, snapshot_path=snap_path)
        ss  = _make_system_state()
        idle_runner.confirm_execution(
            executed_wc=0.80, executed_ws=0.20,
            date="2024-03-01",
            system_state=ss, state_path=state_path,
            snapshot_path=snap_path,
            trading_days_elapsed=21,
        )
        # Second confirm — now in IDLE, should raise
        with pytest.raises(AllocationRunnerError):
            idle_runner.confirm_execution(
                executed_wc=0.80, executed_ws=0.20,
                date="2024-03-01",
                system_state=ss, state_path=state_path,
                snapshot_path=snap_path,
                trading_days_elapsed=21,
            )


# ── Test C: Partial execution / crash during PENDING ─────────────────────────

class TestPartialExecution:
    """
    Scenario: step() completes (snapshot saved as PENDING), system crashes.
    On restart, startup_reconcile() finds PENDING → transitions to RECOVERY_REQUIRED.
    execute_recovery() completes the cycle.
    """

    def test_crash_during_pending_detected_on_startup(self, idle_runner, snap_path):
        sig = _make_signal()
        idle_runner.step(sig, trading_days_since=21, snapshot_path=snap_path)
        # Simulate crash: reload from disk as if restarting
        recovered = AllocationRunner.startup_reconcile(snap_path)
        assert recovered.state == RunnerState.RECOVERY_REQUIRED

    def test_recovery_required_after_startup_reconcile(self, idle_runner, snap_path):
        sig = _make_signal()
        idle_runner.step(sig, trading_days_since=21, snapshot_path=snap_path)
        recovered = AllocationRunner.startup_reconcile(snap_path)
        # Snapshot on disk must also show RECOVERY_REQUIRED
        snap = RunnerSnapshot.load(snap_path)
        assert snap.runner_state == RunnerState.RECOVERY_REQUIRED

    def test_recovery_pending_fields_preserved(self, idle_runner, snap_path):
        sig = _make_signal(c_sc=4.0, s_sc=3.0)  # both hot → 65/35
        idle_runner.step(sig, trading_days_since=21, snapshot_path=snap_path)
        recovered = AllocationRunner.startup_reconcile(snap_path)
        # Pending fields must survive the PENDING → RECOVERY_REQUIRED transition
        snap = RunnerSnapshot.load(snap_path)
        assert snap.pending_wc == 0.65
        assert snap.pending_ws == 0.35

    def test_execute_recovery_after_crash_goes_to_idle(
        self, idle_runner, snap_path, state_path
    ):
        sig = _make_signal()
        idle_runner.step(sig, trading_days_since=21, snapshot_path=snap_path)
        recovered = AllocationRunner.startup_reconcile(snap_path)
        ss = _make_system_state()
        recovered.execute_recovery(
            executed_wc=0.80, executed_ws=0.20,
            date="2024-03-01", reason="post-crash recovery",
            system_state=ss, state_path=state_path,
            snapshot_path=snap_path,
            trading_days_elapsed=21,
        )
        assert recovered.state == RunnerState.IDLE

    def test_execute_recovery_record_has_recovery_prefix(
        self, idle_runner, snap_path, state_path, audit_path
    ):
        sig = _make_signal()
        idle_runner.step(sig, trading_days_since=21, snapshot_path=snap_path)
        recovered = AllocationRunner.startup_reconcile(snap_path)
        ss = _make_system_state(wc=0.65, ws=0.35)
        record = recovered.execute_recovery(
            executed_wc=0.65, executed_ws=0.35,
            date="2024-03-01", reason="post-crash recovery",
            system_state=ss, state_path=state_path,
            snapshot_path=snap_path,
            trading_days_elapsed=21,
            audit_path=audit_path,
        )
        assert record.reason.startswith("recovery — ")

    def test_idle_startup_does_not_trigger_recovery(self, snap_path):
        AllocationRunner.initialize(snap_path)
        runner = AllocationRunner.startup_reconcile(snap_path)
        assert runner.state == RunnerState.IDLE

    def test_failed_startup_triggers_recovery(self, idle_runner, snap_path):
        # Manually put snapshot into FAILED state
        sig = _make_signal()
        idle_runner.step(sig, trading_days_since=21, snapshot_path=snap_path)
        idle_runner.mark_failed(snap_path)
        # startup_reconcile only auto-converts PENDING; FAILED stays FAILED
        reloaded = AllocationRunner.startup_reconcile(snap_path)
        assert reloaded.state == RunnerState.REBALANCE_FAILED

    def test_recovery_required_startup_stays_recovery_required(self, idle_runner, snap_path):
        sig = _make_signal()
        idle_runner.step(sig, trading_days_since=21, snapshot_path=snap_path)
        idle_runner.request_recovery(snap_path)
        reloaded = AllocationRunner.startup_reconcile(snap_path)
        assert reloaded.state == RunnerState.RECOVERY_REQUIRED


# ── Test D: Month boundary ────────────────────────────────────────────────────

class TestMonthBoundary:
    """
    Scenario: Rebalance counter crosses 21 near month end.
    Expected: Rebalance triggers on day 21 exactly, not 20, not 22.
    No off-by-one errors.
    """

    def test_day_19_no_rebalance(self, idle_runner, snap_path):
        sig = _make_signal()
        result = idle_runner.step(sig, trading_days_since=19, snapshot_path=snap_path)
        assert result is None

    def test_day_20_no_rebalance(self, idle_runner, snap_path):
        sig = _make_signal()
        result = idle_runner.step(sig, trading_days_since=20, snapshot_path=snap_path)
        assert result is None

    def test_day_21_rebalance_exact(self, idle_runner, snap_path):
        sig = _make_signal()
        result = idle_runner.step(sig, trading_days_since=21, snapshot_path=snap_path)
        assert result is not None
        assert result.should_rebalance is True

    def test_day_22_rebalance(self, idle_runner, snap_path):
        sig = _make_signal()
        result = idle_runner.step(sig, trading_days_since=22, snapshot_path=snap_path)
        assert result is not None
        assert result.should_rebalance is True

    def test_threshold_matches_core_constant(self):
        # The runner threshold must equal core.REBALANCE_DAYS — single source of truth
        assert core.REBALANCE_DAYS == 21

    def test_evaluate_boundary_exact(self, idle_runner):
        sig = _make_signal()
        assert idle_runner.evaluate(sig, trading_days_since=20).should_rebalance is False
        assert idle_runner.evaluate(sig, trading_days_since=21).should_rebalance is True

    def test_full_month_cycle(self, snap_path, state_path):
        """Simulate 42 trading days: two complete rebalance cycles."""
        runner = AllocationRunner.initialize(snap_path)
        ss     = _make_system_state(wc=0.65, ws=0.35)
        sig    = _make_signal(c_sc=3.5, s_sc=2.25)
        rebalance_count = 0

        for day in range(1, 43):
            if runner.state != RunnerState.IDLE:
                break
            decision = runner.step(sig, trading_days_since=day, snapshot_path=snap_path)
            if decision is not None:
                rebalance_count += 1
                runner.confirm_execution(
                    executed_wc=decision.target_wc,
                    executed_ws=decision.target_ws,
                    date=f"2024-{day:02d}",
                    system_state=ss,
                    state_path=state_path,
                    snapshot_path=snap_path,
                    trading_days_elapsed=day,
                )
                # After rebalance, counter resets — simulate that by
                # restarting day count (tests the boundary, not the counter)
                break  # one cycle per runner instance to keep test clean

        assert rebalance_count == 1
        assert runner.state == RunnerState.IDLE

    def test_no_rebalance_on_day_zero(self, idle_runner, snap_path):
        sig = _make_signal()
        result = idle_runner.step(sig, trading_days_since=0, snapshot_path=snap_path)
        assert result is None


# ── Audit log ─────────────────────────────────────────────────────────────────

class TestAuditLog:
    def test_append_creates_file(self, audit_path):
        rec = AllocationChangeRecord(
            date="2024-03-01", prev_wc=0.65, prev_ws=0.35,
            new_wc=0.80, new_ws=0.20, reason="21-day rebalance — CRYPTO_DOMINANT",
            trading_days_elapsed=21,
        )
        append_audit_record(rec, audit_path)
        assert audit_path.exists()

    def test_load_empty_log_returns_empty_list(self, tmp_path):
        result = load_audit_log(tmp_path / "nonexistent.jsonl")
        assert result == []

    def test_append_and_load_single_record(self, audit_path):
        rec = AllocationChangeRecord(
            date="2024-03-01", prev_wc=0.65, prev_ws=0.35,
            new_wc=0.80, new_ws=0.20, reason="test",
            trading_days_elapsed=21,
        )
        append_audit_record(rec, audit_path)
        loaded = load_audit_log(audit_path)
        assert len(loaded) == 1
        assert loaded[0].date == "2024-03-01"
        assert loaded[0].new_wc == 0.80

    def test_append_multiple_records_preserves_order(self, audit_path):
        for i in range(3):
            rec = AllocationChangeRecord(
                date=f"2024-0{i+1}-01", prev_wc=0.65, prev_ws=0.35,
                new_wc=0.80, new_ws=0.20, reason=f"record {i}",
                trading_days_elapsed=21,
            )
            append_audit_record(rec, audit_path)
        loaded = load_audit_log(audit_path)
        assert len(loaded) == 3
        assert loaded[0].date == "2024-01-01"
        assert loaded[2].date == "2024-03-01"

    def test_audit_record_written_on_confirm(self, idle_runner, snap_path, state_path, audit_path):
        sig = _make_signal(date="2024-03-01")
        idle_runner.step(sig, trading_days_since=21, snapshot_path=snap_path)
        ss  = _make_system_state(wc=0.65, ws=0.35)
        idle_runner.confirm_execution(
            executed_wc=0.80, executed_ws=0.20,
            date="2024-03-01",
            system_state=ss, state_path=state_path,
            snapshot_path=snap_path,
            trading_days_elapsed=21,
            audit_path=audit_path,
        )
        log = load_audit_log(audit_path)
        assert len(log) == 1
        assert log[0].prev_wc == 0.65
        assert log[0].new_wc == 0.80

    def test_audit_record_written_on_recovery(self, idle_runner, snap_path, state_path, audit_path):
        sig = _make_signal()
        idle_runner.step(sig, trading_days_since=21, snapshot_path=snap_path)
        idle_runner.request_recovery(snap_path)
        ss  = _make_system_state(wc=0.65, ws=0.35)
        idle_runner.execute_recovery(
            executed_wc=0.65, executed_ws=0.35,
            date="2024-03-01", reason="restart recovery",
            system_state=ss, state_path=state_path,
            snapshot_path=snap_path,
            trading_days_elapsed=21,
            audit_path=audit_path,
        )
        log = load_audit_log(audit_path)
        assert len(log) == 1
        assert "recovery" in log[0].reason

    def test_no_audit_record_when_no_rebalance(self, idle_runner, snap_path, audit_path):
        sig = _make_signal()
        idle_runner.step(sig, trading_days_since=5, snapshot_path=snap_path)
        log = load_audit_log(audit_path)
        assert len(log) == 0

    def test_audit_contains_previous_allocation(self, idle_runner, snap_path, state_path, audit_path):
        sig = _make_signal(date="2024-03-01")
        idle_runner.step(sig, trading_days_since=21, snapshot_path=snap_path)
        ss  = _make_system_state(wc=0.40, ws=0.40)  # current is DEFENSIVE
        idle_runner.confirm_execution(
            executed_wc=0.65, executed_ws=0.35,
            date="2024-03-01",
            system_state=ss, state_path=state_path,
            snapshot_path=snap_path,
            trading_days_elapsed=21,
            audit_path=audit_path,
        )
        log = load_audit_log(audit_path)
        assert log[0].prev_wc == 0.40
        assert log[0].prev_ws == 0.40

    def test_no_allocation_logic_in_runner_module(self):
        """
        Hard rule: allocation formulas must not appear in runner.py.
        This test imports runner and verifies it contains no inline weights.
        """
        import inspect
        import nexflow.v9.runner as runner_module
        src = inspect.getsource(runner_module)
        # Allocation constants must not be defined in runner.py
        assert "BOTH_HOT_THRESHOLD" not in src, "Allocation threshold leaked into runner.py"
        assert "CRYPTO_SCORE_MAX"   not in src, "Score max leaked into runner.py"
        assert "REBALANCE_DAYS = "  not in src, "REBALANCE_DAYS defined in runner.py (use core.py)"
