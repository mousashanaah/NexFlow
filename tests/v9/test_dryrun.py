"""
Dry-run tracker tests — DryRunState, DryRunDayResult, gate enforcement.

These tests do NOT run the live daily script (which requires real data).
They verify the gate logic, counter arithmetic, serialisation, and
gate enforcement in isolation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO))

from nexflow.v9.dryrun import (
    DryRunDayResult,
    DryRunState,
    SHADOW_GATE_DAYS,
    ShadowGateNotMet,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pass_day(date: str, note: str = "") -> DryRunDayResult:
    return DryRunDayResult(
        date=date,
        parity_passed=True, reconcile_passed=True,
        audit_clean=True,   lifecycle_clean=True,
        snapshot_valid=True, note=note,
    )


def _fail_day(date: str, which: str = "parity") -> DryRunDayResult:
    kwargs = dict(
        date=date,
        parity_passed=True, reconcile_passed=True,
        audit_clean=True,   lifecycle_clean=True,
        snapshot_valid=True,
    )
    kwargs[f"{which}_passed" if which in ("parity","reconcile") else which] = False
    return DryRunDayResult(**kwargs)


def _fail_parity(date: str) -> DryRunDayResult:
    return DryRunDayResult(
        date=date,
        parity_passed=False, reconcile_passed=True,
        audit_clean=True,    lifecycle_clean=True, snapshot_valid=True,
    )


def _fill_consecutive(state: DryRunState, n: int, path: Path, start_day: int = 1) -> None:
    for i in range(n):
        state.record_day(_pass_day(f"2024-{(start_day + i):03d}"), path)


# ── DryRunDayResult ───────────────────────────────────────────────────────────

class TestDryRunDayResult:
    def test_all_pass_means_passed(self):
        r = _pass_day("2024-01-01")
        assert r.passed is True

    def test_parity_fail_means_not_passed(self):
        r = DryRunDayResult(
            date="2024-01-01",
            parity_passed=False, reconcile_passed=True,
            audit_clean=True, lifecycle_clean=True, snapshot_valid=True,
        )
        assert r.passed is False

    def test_any_check_fail_means_not_passed(self):
        for field in ("parity_passed", "reconcile_passed", "audit_clean",
                      "lifecycle_clean", "snapshot_valid"):
            kwargs = dict(
                date="2024-01-01",
                parity_passed=True, reconcile_passed=True,
                audit_clean=True, lifecycle_clean=True, snapshot_valid=True,
            )
            kwargs[field] = False
            r = DryRunDayResult(**kwargs)
            assert r.passed is False, f"{field} should cause failure"

    def test_failure_reasons_identifies_failures(self):
        r = DryRunDayResult(
            date="2024-01-01",
            parity_passed=False, reconcile_passed=True,
            audit_clean=False, lifecycle_clean=True, snapshot_valid=True,
        )
        reasons = r.failure_reasons()
        assert "parity_failed" in reasons
        assert "audit_dirty" in reasons
        assert "reconcile_failed" not in reasons

    def test_round_trip(self):
        r  = _pass_day("2024-01-01", note="all good")
        r2 = DryRunDayResult.from_dict(r.to_dict())
        assert r2.date == "2024-01-01"
        assert r2.passed is True
        assert r2.note == "all good"


# ── DryRunState counter arithmetic ────────────────────────────────────────────

class TestDryRunState:
    def test_initial_state_zero(self):
        s = DryRunState()
        assert s.consecutive_successes == 0
        assert s.gate_met is False

    def test_one_pass_increments_counter(self, tmp_path):
        s    = DryRunState()
        path = tmp_path / "dryrun.json"
        s.record_day(_pass_day("2024-01-01"), path)
        assert s.consecutive_successes == 1

    def test_one_fail_keeps_counter_at_zero(self, tmp_path):
        s    = DryRunState()
        path = tmp_path / "dryrun.json"
        s.record_day(_fail_parity("2024-01-01"), path)
        assert s.consecutive_successes == 0

    def test_fail_after_streak_resets_to_zero(self, tmp_path):
        s    = DryRunState()
        path = tmp_path / "dryrun.json"
        for i in range(15):
            s.record_day(_pass_day(f"2024-01-{i+1:02d}"), path)
        assert s.consecutive_successes == 15
        s.record_day(_fail_parity("2024-01-16"), path)
        assert s.consecutive_successes == 0

    def test_gate_met_at_exactly_30(self, tmp_path):
        s    = DryRunState()
        path = tmp_path / "dryrun.json"
        _fill_consecutive(s, SHADOW_GATE_DAYS, path)
        assert s.gate_met is True
        assert s.consecutive_successes == SHADOW_GATE_DAYS

    def test_gate_not_met_at_29(self, tmp_path):
        s    = DryRunState()
        path = tmp_path / "dryrun.json"
        _fill_consecutive(s, SHADOW_GATE_DAYS - 1, path)
        assert s.gate_met is False

    def test_gate_date_recorded(self, tmp_path):
        s    = DryRunState()
        path = tmp_path / "dryrun.json"
        _fill_consecutive(s, SHADOW_GATE_DAYS, path)
        assert s.gate_met_date is not None

    def test_gate_stays_met_after_failure(self, tmp_path):
        """Once gate is met, it stays met even if a failure occurs later."""
        s    = DryRunState()
        path = tmp_path / "dryrun.json"
        _fill_consecutive(s, SHADOW_GATE_DAYS, path)
        assert s.gate_met is True
        s.record_day(_fail_parity("2024-02-01"), path)
        assert s.gate_met is True        # gate_met is immutable once True
        assert s.consecutive_successes == 0  # counter reset

    def test_total_days_run_increments(self, tmp_path):
        s    = DryRunState()
        path = tmp_path / "dryrun.json"
        s.record_day(_pass_day("2024-01-01"), path)
        s.record_day(_fail_parity("2024-01-02"), path)
        s.record_day(_pass_day("2024-01-03"), path)
        assert s.total_days_run == 3

    def test_total_failures_counted(self, tmp_path):
        s    = DryRunState()
        path = tmp_path / "dryrun.json"
        s.record_day(_pass_day("2024-01-01"), path)
        s.record_day(_fail_parity("2024-01-02"), path)
        s.record_day(_fail_parity("2024-01-03"), path)
        assert s.total_failures == 2

    def test_start_date_set_on_first_record(self, tmp_path):
        s    = DryRunState()
        path = tmp_path / "dryrun.json"
        s.record_day(_pass_day("2024-03-15"), path)
        assert s.start_date == "2024-03-15"

    def test_start_date_not_overwritten(self, tmp_path):
        s    = DryRunState()
        path = tmp_path / "dryrun.json"
        s.record_day(_pass_day("2024-03-15"), path)
        s.record_day(_pass_day("2024-03-16"), path)
        assert s.start_date == "2024-03-15"

    def test_last_run_date_updated(self, tmp_path):
        s    = DryRunState()
        path = tmp_path / "dryrun.json"
        s.record_day(_pass_day("2024-03-15"), path)
        s.record_day(_pass_day("2024-03-16"), path)
        assert s.last_run_date == "2024-03-16"

    def test_history_appended(self, tmp_path):
        s    = DryRunState()
        path = tmp_path / "dryrun.json"
        s.record_day(_pass_day("2024-01-01"), path)
        s.record_day(_fail_parity("2024-01-02"), path)
        assert len(s.history) == 2

    def test_recent_history(self, tmp_path):
        s    = DryRunState()
        path = tmp_path / "dryrun.json"
        for i in range(15):
            s.record_day(_pass_day(f"2024-01-{i+1:02d}"), path)
        recent = s.recent_history(5)
        assert len(recent) == 5
        assert recent[-1].date == "2024-01-15"


# ── Persistence ───────────────────────────────────────────────────────────────

class TestDryRunPersistence:
    def test_save_load_round_trip(self, tmp_path):
        path = tmp_path / "dryrun.json"
        s    = DryRunState()
        s.record_day(_pass_day("2024-01-01"), path)
        s.record_day(_pass_day("2024-01-02"), path)
        loaded = DryRunState.load(path)
        assert loaded.consecutive_successes == 2
        assert loaded.total_days_run == 2

    def test_load_missing_file_returns_empty(self, tmp_path):
        path   = tmp_path / "nonexistent.json"
        loaded = DryRunState.load(path)
        assert loaded.consecutive_successes == 0
        assert loaded.gate_met is False

    def test_gate_preserved_across_save_load(self, tmp_path):
        path = tmp_path / "dryrun.json"
        s    = DryRunState()
        _fill_consecutive(s, SHADOW_GATE_DAYS, path)
        loaded = DryRunState.load(path)
        assert loaded.gate_met is True
        assert loaded.gate_met_date is not None

    def test_save_is_atomic(self, tmp_path):
        path = tmp_path / "dryrun.json"
        s    = DryRunState()
        s.record_day(_pass_day("2024-01-01"), path)
        assert list(tmp_path.glob("*.tmp")) == []

    def test_consecutive_reset_persisted(self, tmp_path):
        path = tmp_path / "dryrun.json"
        s    = DryRunState()
        for i in range(10):
            s.record_day(_pass_day(f"2024-01-{i+1:02d}"), path)
        s.record_day(_fail_parity("2024-01-11"), path)
        loaded = DryRunState.load(path)
        assert loaded.consecutive_successes == 0


# ── Gate enforcement ──────────────────────────────────────────────────────────

class TestShadowGate:
    def test_assert_module7_gate_raises_before_30_days(self, tmp_path):
        path = tmp_path / "dryrun.json"
        s    = DryRunState()
        _fill_consecutive(s, 15, path)
        with pytest.raises(ShadowGateNotMet, match="30"):
            s.assert_module7_gate()

    def test_assert_module7_gate_passes_after_30_days(self, tmp_path):
        path = tmp_path / "dryrun.json"
        s    = DryRunState()
        _fill_consecutive(s, SHADOW_GATE_DAYS, path)
        s.assert_module7_gate()  # must not raise

    def test_assert_module7_gate_raises_on_fresh_state(self):
        s = DryRunState()
        with pytest.raises(ShadowGateNotMet):
            s.assert_module7_gate()

    def test_gate_error_message_informative(self):
        s = DryRunState(consecutive_successes=12)
        try:
            s.assert_module7_gate()
            assert False, "should have raised"
        except ShadowGateNotMet as e:
            msg = str(e)
            assert "12" in msg
            assert "30" in msg
            assert "18" in msg  # remaining days

    def test_shadow_gate_days_is_30(self):
        assert SHADOW_GATE_DAYS == 30

    def test_gate_stays_met_after_subsequent_failure(self, tmp_path):
        path = tmp_path / "dryrun.json"
        s    = DryRunState()
        _fill_consecutive(s, SHADOW_GATE_DAYS, path)
        s.assert_module7_gate()  # passes
        # Failure after gate — gate remains met
        s.record_day(_fail_parity("2025-01-01"), path)
        s.assert_module7_gate()  # still passes

    def test_status_report_shows_progress(self, tmp_path):
        path = tmp_path / "dryrun.json"
        s    = DryRunState()
        _fill_consecutive(s, 15, path)
        report = s.status_report()
        assert "15" in report
        assert "30" in report
        assert "Gate met" in report

    def test_status_report_shows_gate_met(self, tmp_path):
        path = tmp_path / "dryrun.json"
        s    = DryRunState()
        _fill_consecutive(s, SHADOW_GATE_DAYS, path)
        report = s.status_report()
        assert "YES" in report


# ── pytest wrappers for VAL TEST 01 and 02 ───────────────────────────────────

class TestValidation01Integrated:
    """
    pytest wrapper for VAL TEST 01.
    Marks slow so CI can skip with: pytest -m 'not slow'
    """
    @pytest.mark.slow
    def test_integrated_replay_passes_all_gates(self):
        import sys
        sys.path.insert(0, str(_REPO / "scripts"))
        from validation_01_integrated_replay import run_validation_01
        result = run_validation_01(verbose=False)
        if result.gate_failures:
            pytest.fail(
                f"VAL TEST 01 gate failures:\n" +
                "\n".join(f"  {f}" for f in result.gate_failures)
            )
        assert result.passed
        assert result.bars_processed == 1107
        assert 48 <= result.rebalance_count <= 58

    @pytest.mark.slow
    def test_all_fixture_dates_exact(self):
        from validation_01_integrated_replay import run_validation_01, FIXTURES
        result = run_validation_01(verbose=False)
        for date, fix in FIXTURES.items():
            actual = result.fixture_results.get(date)
            assert actual is not None, f"Fixture date {date} missing"
            assert actual["crypto_score"] == fix["crypto_score"], f"{date} crypto_score"
            assert actual["stock_score"]  == fix["stock_score"],  f"{date} stock_score"
            assert actual["wc"]           == fix["wc"],           f"{date} wc"
            assert actual["ws"]           == fix["ws"],           f"{date} ws"


class TestValidation02Chaos:
    """pytest wrapper for VAL TEST 02."""
    @pytest.mark.slow
    def test_chaos_500_iterations_all_recover(self):
        from validation_02_chaos import run_validation_02
        out = run_validation_02(n_iterations=500, seed=42, verbose=False)
        if out["gate_failures"]:
            pytest.fail(
                "VAL TEST 02 gate failures:\n" +
                "\n".join(f"  {f}" for f in out["gate_failures"])
            )
        assert out["passed"]
        assert out["recovered"] == 500
        assert out["failed"] == 0

    @pytest.mark.slow
    def test_chaos_all_crash_points_covered(self):
        from validation_02_chaos import run_validation_02
        out = run_validation_02(n_iterations=200, seed=99, verbose=False)
        expected = {"A_idle", "B_pending", "C_failed", "D_recovery", "E_double"}
        missing  = expected - set(out["crash_dist"].keys())
        assert not missing, f"Crash points not exercised: {missing}"
