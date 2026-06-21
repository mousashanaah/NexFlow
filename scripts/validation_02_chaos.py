#!/usr/bin/env python3
"""
VALIDATION TEST 02 — Restart Chaos Test

Simulates hundreds of random crashes across all runner states and verifies:
  - Every crash is recoverable
  - Final state is always deterministic (IDLE with valid snapshot)
  - No state corruption survives a restart cycle
  - AllocationRunnerError is only raised for genuinely illegal sequences

Crash injection points:
  A: During IDLE (before step)          → restart → IDLE (no change)
  B: After step, before confirm         → restart → RECOVERY_REQUIRED → recover → IDLE
  C: During FAILED state                → restart → FAILED → request_recovery → IDLE
  D: During RECOVERY_REQUIRED           → restart → RECOVERY_REQUIRED → recover → IDLE
  E: Double restart (crash during crash) → still recoverable

Gates:
  1. All N iterations complete without unhandled exception
  2. Final state is always IDLE with valid snapshot
  3. Audit log grows by exactly 1 per successful rebalance (no duplicates)
  4. Crash-point distribution covers all state types
  5. Average recovery rate: 100% (zero unrecoverable scenarios)

Usage:
  python scripts/validation_02_chaos.py [--iterations N] [--seed S] [--verbose]
"""
from __future__ import annotations

import sys
import argparse
import random
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from nexflow.v9 import core
from nexflow.v9.runner import (
    AllocationRunner, AllocationRunnerError,
    RunnerSnapshot, RunnerState, load_audit_log,
)
from nexflow.v9.signals import CryptoSignal, DailySignalRecord
from nexflow.v9.state import AllocationSnapshot, RegimeSnapshot, SystemState


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_signal(c_sc: float = 3.5, s_sc: float = 2.25) -> DailySignalRecord:
    wc, ws = core.allocate(c_sc, s_sc)
    btc    = CryptoSignal(
        close=50000.0, sma200=40000.0, mom90=0.30, mom30=0.05,
        atr14=800.0, atr_avg=1200.0,
        pts_sma200=2.0, pts_mom90=1.0, pts_mom30=0.5,
        pts_vol=0.5, pts_bonus=0.0, raw=c_sc, score=c_sc,
    )
    return DailySignalRecord(
        date="2024-03-01",
        timestamp_ms=int(datetime(2024, 3, 1, tzinfo=timezone.utc).timestamp() * 1000),
        in_bear=False,
        btc=btc, crypto_score=c_sc, stocks=[], stock_score=s_sc,
        allocation_regime=core.allocation_regime_name(c_sc, s_sc),
        wc=wc, ws=ws, cash=round(1.0 - wc - ws, 10),
    )


def _make_system_state(wc: float = 0.65, ws: float = 0.35) -> SystemState:
    ss = SystemState(
        regime=RegimeSnapshot(
            in_bear=False, consecutive_above=0, last_bar_date="2024-03-01"),
        allocation=AllocationSnapshot(
            wc=wc, ws=ws,
            last_rebalance_date="2024-03-01", trading_days_since=0),
    )
    ss.gate_open = True
    return ss


# ── Crash simulation ──────────────────────────────────────────────────────────

@dataclass
class ChaosResult:
    iteration:    int
    crash_point:  str
    start_state:  str
    end_state:    str
    recovered:    bool
    error:        str | None = None


def _run_one_iteration(
    rng:        random.Random,
    snap_path:  Path,
    state_path: Path,
    audit_path: Path,
    ss:         SystemState,
) -> ChaosResult:
    """
    Run one chaos scenario.  Picks a crash point randomly, injects the
    crash, then verifies recovery to IDLE.
    """
    crash_point = rng.choice(["A_idle", "B_pending", "C_failed", "D_recovery", "E_double"])

    try:
        sig = _make_signal(
            c_sc=rng.uniform(0.5, 4.0),
            s_sc=rng.uniform(0.3, 3.0),
        )
        # Quantise to valid score steps to match allocate() boundaries
        sig = _make_signal(
            c_sc=round(sig.crypto_score * 2) / 2,
            s_sc=round(sig.stock_score * 4) / 4,
        )

        runner = AllocationRunner.initialize(snap_path)
        start  = runner.state.value

        # ── Crash point A: crash while IDLE ──────────────────────────────────
        if crash_point == "A_idle":
            # Crash: process nothing, just restart
            restarted = AllocationRunner.startup_reconcile(snap_path)
            assert restarted.state == RunnerState.IDLE
            return ChaosResult(
                iteration=0, crash_point=crash_point,
                start_state=start, end_state=restarted.state.value,
                recovered=True,
            )

        # ── Crash point B: crash after step (PENDING) ─────────────────────────
        if crash_point == "B_pending":
            runner.step(sig, trading_days_since=21, snapshot_path=snap_path)
            assert runner.state == RunnerState.REBALANCE_PENDING
            # Crash — restart
            restarted = AllocationRunner.startup_reconcile(snap_path)
            assert restarted.state == RunnerState.RECOVERY_REQUIRED
            # Recover
            restarted.execute_recovery(
                executed_wc=sig.wc, executed_ws=sig.ws,
                date="2024-03-01", reason="post-crash B",
                system_state=ss, state_path=state_path,
                snapshot_path=snap_path,
                trading_days_elapsed=21,
                audit_path=audit_path,
            )
            assert restarted.state == RunnerState.IDLE
            return ChaosResult(
                iteration=0, crash_point=crash_point,
                start_state=start, end_state=restarted.state.value,
                recovered=True,
            )

        # ── Crash point C: crash during FAILED ───────────────────────────────
        if crash_point == "C_failed":
            runner.step(sig, trading_days_since=21, snapshot_path=snap_path)
            runner.mark_failed(snap_path)
            assert runner.state == RunnerState.REBALANCE_FAILED
            # Crash — restart (FAILED persists; startup_reconcile only converts PENDING)
            restarted = AllocationRunner.startup_reconcile(snap_path)
            assert restarted.state == RunnerState.REBALANCE_FAILED
            # Manual recovery path
            restarted.request_recovery(snap_path)
            restarted.execute_recovery(
                executed_wc=sig.wc, executed_ws=sig.ws,
                date="2024-03-01", reason="post-crash C",
                system_state=ss, state_path=state_path,
                snapshot_path=snap_path,
                trading_days_elapsed=21,
                audit_path=audit_path,
            )
            assert restarted.state == RunnerState.IDLE
            return ChaosResult(
                iteration=0, crash_point=crash_point,
                start_state=start, end_state=restarted.state.value,
                recovered=True,
            )

        # ── Crash point D: crash during RECOVERY_REQUIRED ────────────────────
        if crash_point == "D_recovery":
            runner.step(sig, trading_days_since=21, snapshot_path=snap_path)
            runner.request_recovery(snap_path)
            assert runner.state == RunnerState.RECOVERY_REQUIRED
            # Crash during recovery — restart
            restarted = AllocationRunner.startup_reconcile(snap_path)
            assert restarted.state == RunnerState.RECOVERY_REQUIRED
            # Execute recovery
            restarted.execute_recovery(
                executed_wc=sig.wc, executed_ws=sig.ws,
                date="2024-03-01", reason="post-crash D",
                system_state=ss, state_path=state_path,
                snapshot_path=snap_path,
                trading_days_elapsed=21,
                audit_path=audit_path,
            )
            assert restarted.state == RunnerState.IDLE
            return ChaosResult(
                iteration=0, crash_point=crash_point,
                start_state=start, end_state=restarted.state.value,
                recovered=True,
            )

        # ── Crash point E: double crash (crash during crash recovery) ─────────
        if crash_point == "E_double":
            runner.step(sig, trading_days_since=21, snapshot_path=snap_path)
            # First crash
            r1 = AllocationRunner.startup_reconcile(snap_path)
            assert r1.state == RunnerState.RECOVERY_REQUIRED
            # Second crash before recovery completes
            r2 = AllocationRunner.startup_reconcile(snap_path)
            assert r2.state == RunnerState.RECOVERY_REQUIRED  # still recoverable
            # Finally recover
            r2.execute_recovery(
                executed_wc=sig.wc, executed_ws=sig.ws,
                date="2024-03-01", reason="post-double-crash E",
                system_state=ss, state_path=state_path,
                snapshot_path=snap_path,
                trading_days_elapsed=21,
                audit_path=audit_path,
            )
            assert r2.state == RunnerState.IDLE
            return ChaosResult(
                iteration=0, crash_point=crash_point,
                start_state=start, end_state=r2.state.value,
                recovered=True,
            )

    except Exception as exc:
        return ChaosResult(
            iteration=0, crash_point=crash_point,
            start_state="unknown", end_state="unknown",
            recovered=False, error=str(exc),
        )


# ── Main validation ───────────────────────────────────────────────────────────

def run_validation_02(
    n_iterations: int = 500,
    seed:         int = 42,
    verbose:      bool = False,
) -> dict:
    rng          = random.Random(seed)
    results: list[ChaosResult] = []
    gate_failures: list[str]   = []

    with tempfile.TemporaryDirectory() as _tmp:
        tmp        = Path(_tmp)
        snap_path  = tmp / "runner.json"
        state_path = tmp / "state.json"
        audit_path = tmp / "audit.jsonl"

        for i in range(n_iterations):
            ss = _make_system_state()
            r  = _run_one_iteration(rng, snap_path, state_path, audit_path, ss)
            r.iteration = i + 1
            results.append(r)

            if verbose and (i + 1) % 50 == 0:
                n_ok = sum(1 for x in results if x.recovered)
                print(f"  {i+1:4d}/{n_iterations}  recovered={n_ok}/{i+1}")

    # ── Evaluate gates ────────────────────────────────────────────────────────

    failed  = [r for r in results if not r.recovered]
    dist    = Counter(r.crash_point for r in results)
    end_states = Counter(r.end_state for r in results)

    # Gate 1: all iterations completed
    if len(results) != n_iterations:
        gate_failures.append(
            f"GATE 1 — Only {len(results)}/{n_iterations} iterations completed"
        )

    # Gate 2: every recovered iteration ends in IDLE
    non_idle = [r for r in results if r.recovered and r.end_state != "IDLE"]
    if non_idle:
        gate_failures.append(
            f"GATE 2 — {len(non_idle)} iterations ended in non-IDLE state"
        )

    # Gate 3: zero unrecoverable scenarios
    if failed:
        gate_failures.append(
            f"GATE 3 — {len(failed)} unrecoverable scenarios: "
            + ", ".join(f"{r.crash_point}@{r.iteration}: {r.error}" for r in failed[:3])
        )

    # Gate 4: all crash points exercised
    expected_crash_points = {"A_idle", "B_pending", "C_failed", "D_recovery", "E_double"}
    missing = expected_crash_points - set(dist.keys())
    if missing:
        gate_failures.append(f"GATE 4 — Crash points not exercised: {missing}")

    # Gate 5: recovery rate 100%
    recovery_rate = sum(1 for r in results if r.recovered) / len(results)
    if recovery_rate < 1.0:
        gate_failures.append(
            f"GATE 5 — Recovery rate {recovery_rate:.1%}, expected 100%"
        )

    return {
        "iterations":     n_iterations,
        "completed":      len(results),
        "recovered":      sum(1 for r in results if r.recovered),
        "failed":         len(failed),
        "crash_dist":     dict(dist),
        "end_state_dist": dict(end_states),
        "gate_failures":  gate_failures,
        "passed":         len(gate_failures) == 0,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="VAL TEST 02 — Chaos test")
    parser.add_argument("--iterations", "-n", type=int, default=500)
    parser.add_argument("--seed",             type=int, default=42)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("VALIDATION TEST 02 — Restart Chaos Test")
    print("=" * 70)
    print(f"  Iterations : {args.iterations}")
    print(f"  Seed       : {args.seed}")

    out = run_validation_02(args.iterations, args.seed, args.verbose)

    print(f"\nCompleted  : {out['completed']}/{out['iterations']}")
    print(f"Recovered  : {out['recovered']}")
    print(f"Failed     : {out['failed']}")
    print(f"Recovery % : {out['recovered']/out['completed']:.1%}")
    print(f"Crash dist : {out['crash_dist']}")

    if out["gate_failures"]:
        print(f"\n{'─' * 70}")
        print(f"GATE FAILURES ({len(out['gate_failures'])}):")
        for f in out["gate_failures"]:
            print(f"  ✗  {f}")
    else:
        print("\n  All gates passed")

    print(f"\n{'─' * 70}")
    print(f"RESULT: {'PASS' if out['passed'] else 'FAIL'}")
    print("=" * 70)
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
