#!/usr/bin/env python3
"""
VALIDATION TEST 03 — Daily Dry-Run Execution Script

Runs one day of the complete paper-trading workflow and records the result
in the 30-day shadow deployment gate tracker.

Run this script once per trading day.  The gate opens automatically after
30 consecutive successes.  Any failure resets the counter.

Checks performed each day:
  1. Daily parity (production == research engine for today's signals)
  2. State reconciliation (startup_gate on BTC history)
  3. Signal computation
  4. Runner step (rebalance if due)
  5. Paper execution (if rebalance)
  6. Snapshot taken
  7. Audit log integrity
  8. Runner snapshot validates cleanly

Usage:
  python scripts/run_dryrun_day.py --date YYYY-MM-DD [--verbose] [--dry]

  --date      : Date to simulate (defaults to today)
  --verbose   : Print all intermediate results
  --dry       : Run checks but do NOT update the dry-run counter

Paths (override with env vars):
  NEXFLOW_STATE_PATH      → state.json           (default /var/nexflow/state.json)
  NEXFLOW_RUNNER_PATH     → runner.json          (default /var/nexflow/runner.json)
  NEXFLOW_AUDIT_PATH      → audit.jsonl          (default /var/nexflow/audit.jsonl)
  NEXFLOW_PAPER_PATH      → paper.json           (default /var/nexflow/paper.json)
  NEXFLOW_SNAPSHOT_LOG    → snapshots.jsonl      (default /var/nexflow/snapshots.jsonl)
  NEXFLOW_EXEC_LOG        → executions.jsonl     (default /var/nexflow/executions.jsonl)
  NEXFLOW_DRYRUN_PATH     → dryrun.json          (default /var/nexflow/dryrun.json)
"""
from __future__ import annotations

import sys
import os
import argparse
from datetime import date, datetime
from pathlib import Path

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from nexflow.v9 import core
from nexflow.v9.data import load_v9_dataset
from nexflow.v9.dryrun import DryRunDayResult, DryRunState
from nexflow.v9.paper import (
    PaperTrader, append_execution_report, append_snapshot,
)
from nexflow.v9.replay import run_daily_parity
from nexflow.v9.runner import (
    AllocationRunner, AllocationRunnerError, RunnerSnapshot,
    load_audit_log,
)
from nexflow.v9.signals import compute_signals
from nexflow.v9.state import SystemState


# ── Paths ─────────────────────────────────────────────────────────────────────

STATE_PATH    = Path(os.environ.get("NEXFLOW_STATE_PATH",   "/var/nexflow/state.json"))
RUNNER_PATH   = Path(os.environ.get("NEXFLOW_RUNNER_PATH",  "/var/nexflow/runner.json"))
AUDIT_PATH    = Path(os.environ.get("NEXFLOW_AUDIT_PATH",   "/var/nexflow/audit.jsonl"))
PAPER_PATH    = Path(os.environ.get("NEXFLOW_PAPER_PATH",   "/var/nexflow/paper.json"))
SNAP_LOG      = Path(os.environ.get("NEXFLOW_SNAPSHOT_LOG", "/var/nexflow/snapshots.jsonl"))
EXEC_LOG      = Path(os.environ.get("NEXFLOW_EXEC_LOG",     "/var/nexflow/executions.jsonl"))
DRYRUN_PATH   = Path(os.environ.get("NEXFLOW_DRYRUN_PATH",  "/var/nexflow/dryrun.json"))


# ── Daily run ─────────────────────────────────────────────────────────────────

def run_day(as_of_date: str, verbose: bool = False) -> DryRunDayResult:
    """
    Execute one complete paper-trading day.

    Returns a DryRunDayResult indicating which checks passed and which failed.
    Does NOT update the dry-run state file — the caller does that.
    """
    parity_passed    = False
    reconcile_passed = False
    audit_clean      = False
    lifecycle_clean  = False
    snapshot_valid   = False
    notes: list[str] = []

    # ── CHECK 1: Daily parity ─────────────────────────────────────────────────
    try:
        parity = run_daily_parity(as_of_date, verbose=verbose, raise_on_fail=False)
        parity_passed = parity.passed
        if not parity_passed:
            notes.append(f"parity_failed: {as_of_date}")
            if verbose:
                print(f"  [FAIL] Daily parity: {parity}")
        else:
            if verbose:
                print(f"  [PASS] Daily parity")
    except Exception as exc:
        notes.append(f"parity_exception: {exc}")
        if verbose:
            print(f"  [FAIL] Daily parity exception: {exc}")

    # ── CHECK 2: Load state + reconcile ──────────────────────────────────────
    try:
        ss = SystemState.load(STATE_PATH)
        dataset = load_v9_dataset(as_of_date)
        btc = dataset.btc()
        ss.startup_gate(
            btc_closes = btc.close,
            btc_sma200 = btc.sma200,
            btc_mom30  = btc.mom30,
            bar_dates  = [
                datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")
                for ts in btc.ts
            ],
        )
        reconcile_passed = True
        if verbose:
            print(f"  [PASS] State reconciliation")
    except Exception as exc:
        notes.append(f"reconcile_failed: {exc}")
        if verbose:
            print(f"  [FAIL] State reconciliation: {exc}")
        return DryRunDayResult(
            date=as_of_date,
            parity_passed=parity_passed,
            reconcile_passed=False,
            audit_clean=False,
            lifecycle_clean=False,
            snapshot_valid=False,
            note="; ".join(notes),
        )

    # ── CHECK 3-6: Signal → runner → paper ───────────────────────────────────
    try:
        # Rebuild regime machine state from history
        from nexflow.v9.core import RegimeMachine
        btc = dataset.btc()
        regime_machine = RegimeMachine(
            in_bear           = ss.regime.in_bear,
            consecutive_above = ss.regime.consecutive_above,
        )

        # Compute today's signal
        signal = compute_signals(dataset, regime_machine)
        if verbose:
            print(f"  Signal: c_sc={signal.crypto_score:.2f}  "
                  f"s_sc={signal.stock_score:.2f}  "
                  f"regime={signal.allocation_regime}  "
                  f"bear={signal.in_bear}")

        # Load and step runner
        runner = AllocationRunner.startup_reconcile(RUNNER_PATH)

        # Increment trading days in system state
        ss.increment_trading_days(STATE_PATH)
        trading_days = ss.allocation.trading_days_since

        decision = runner.step(signal, trading_days, RUNNER_PATH)

        if decision is not None:
            if verbose:
                print(f"  Rebalance: {decision.reason}")

            # Load or initialise paper trader
            trader = (
                PaperTrader.load(PAPER_PATH)
                if PAPER_PATH.exists()
                else PaperTrader(portfolio_value=5_000.0)
            )

            orders = trader.compute_orders(decision.target_wc, decision.target_ws, signal)
            report = trader.execute(orders, signal, rebalance_reason=decision.reason)

            runner.confirm_execution(
                executed_wc          = decision.target_wc,
                executed_ws          = decision.target_ws,
                date                 = as_of_date,
                system_state         = ss,
                state_path           = STATE_PATH,
                snapshot_path        = RUNNER_PATH,
                trading_days_elapsed = trading_days,
                audit_path           = AUDIT_PATH,
                crypto_score         = signal.crypto_score,
                stock_score          = signal.stock_score,
                in_bear              = signal.in_bear,
            )

            trader.save(PAPER_PATH)
            append_execution_report(report, EXEC_LOG)

            if verbose:
                print(f"  Executed: wc={decision.target_wc}  ws={decision.target_ws}")

        else:
            # No rebalance — load trader for snapshot only
            trader = (
                PaperTrader.load(PAPER_PATH)
                if PAPER_PATH.exists()
                else PaperTrader(portfolio_value=5_000.0)
            )
            if verbose:
                print(f"  No rebalance ({trading_days} days since last)")

        # Take daily snapshot
        snap = trader.snapshot(signal, ss.allocation.wc, ss.allocation.ws)
        append_snapshot(snap, SNAP_LOG)
        lifecycle_clean = True

        if verbose:
            print(f"  [PASS] Lifecycle: signal → runner → paper → snapshot")

    except AllocationRunnerError as exc:
        notes.append(f"lifecycle_error: {exc}")
        if verbose:
            print(f"  [FAIL] AllocationRunnerError: {exc}")
    except Exception as exc:
        notes.append(f"lifecycle_exception: {exc}")
        if verbose:
            print(f"  [FAIL] Lifecycle exception: {exc}")

    # ── CHECK 7: Audit integrity ──────────────────────────────────────────────
    try:
        records = load_audit_log(AUDIT_PATH)
        # Check for duplicate dates
        dates_seen = [r.date for r in records]
        if len(dates_seen) != len(set(dates_seen)):
            notes.append("audit_duplicate_dates")
        else:
            audit_clean = True
        if verbose:
            print(f"  [PASS] Audit log: {len(records)} records, no duplicates")
    except Exception as exc:
        notes.append(f"audit_exception: {exc}")
        if verbose:
            print(f"  [FAIL] Audit exception: {exc}")

    # ── CHECK 8: Runner snapshot validates ────────────────────────────────────
    try:
        snap_loaded = RunnerSnapshot.load(RUNNER_PATH)
        snap_loaded.validate()   # raises AllocationRunnerError if corrupt
        snapshot_valid = True
        if verbose:
            print(f"  [PASS] Runner snapshot valid: {snap_loaded.runner_state}")
    except Exception as exc:
        notes.append(f"snapshot_invalid: {exc}")
        if verbose:
            print(f"  [FAIL] Snapshot validation: {exc}")

    return DryRunDayResult(
        date             = as_of_date,
        parity_passed    = parity_passed,
        reconcile_passed = reconcile_passed,
        audit_clean      = audit_clean,
        lifecycle_clean  = lifecycle_clean,
        snapshot_valid   = snapshot_valid,
        note             = "; ".join(notes),
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="VAL TEST 03 — Daily dry-run")
    parser.add_argument("--date",    default=date.today().isoformat(),
                        help="Date to run (YYYY-MM-DD, default: today)")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry",           action="store_true",
                        help="Run checks but don't update the dry-run counter")
    args = parser.parse_args()

    print("=" * 70)
    print(f"VALIDATION TEST 03 — Daily Dry-Run  [{args.date}]")
    print("=" * 70)

    result = run_day(args.date, verbose=args.verbose)

    print(f"\nDate             : {result.date}")
    print(f"Parity           : {'PASS' if result.parity_passed    else 'FAIL'}")
    print(f"Reconcile        : {'PASS' if result.reconcile_passed else 'FAIL'}")
    print(f"Audit clean      : {'PASS' if result.audit_clean      else 'FAIL'}")
    print(f"Lifecycle        : {'PASS' if result.lifecycle_clean  else 'FAIL'}")
    print(f"Snapshot valid   : {'PASS' if result.snapshot_valid   else 'FAIL'}")
    print(f"Day result       : {'PASS' if result.passed           else 'FAIL'}")
    if result.note:
        print(f"Notes            : {result.note}")

    # Update dry-run state unless --dry
    if not args.dry:
        state = DryRunState.load(DRYRUN_PATH)
        state.record_day(result, DRYRUN_PATH)
        print(f"\n{'─' * 70}")
        print(state.status_report())
    else:
        print("\n  (--dry: counter not updated)")

    print(f"\n{'─' * 70}")
    print(f"RESULT: {'PASS' if result.passed else 'FAIL'}")
    print("=" * 70)
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
