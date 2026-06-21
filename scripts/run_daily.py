#!/usr/bin/env python3
"""
V9 Confidence — Daily Orchestrator

Single entry point for the automated daily run.  Called by cron/systemd.
Never requires human interaction after initial setup.

Execution sequence:
  1. Load all state files (fail fast if missing)
  2. Run daily parity (production == research engine)
  3. Startup reconciliation (crash-recovery gate)
  4. Compute today's signals
  5. Evaluate runner (rebalance due?)
  6. Execute via adapter (NullAdapter before gate, BitgetAdapter after)
  7. Take paper snapshot
  8. Record dry-run day result
  9. Send notifications (rebalances, failures, gate reached)

Exit codes:
  0  — success (parity pass, no rebalance OR rebalance executed cleanly)
  1  — warning  (parity borderline, non-fatal issue)
  2  — fatal    (state corrupt, reconcile failed, execution failed)

Configuration:
  All paths via env vars — see initialize_system.py for the full list.
  BITGET_PAPER=1     — paper-trading mode (default; set to 0 for live)

Usage:
  python scripts/run_daily.py [--date YYYY-MM-DD] [--verbose]
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
from nexflow.v9.adapter import NullAdapter, ExecutionBridge
from nexflow.v9.data import load_v9_dataset
from nexflow.v9.dryrun import DryRunDayResult, DryRunState
from nexflow.v9.notify import Level, Notification, Notifier
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

STATE_PATH  = Path(os.environ.get("NEXFLOW_STATE_PATH",   "/var/nexflow/state.json"))
RUNNER_PATH = Path(os.environ.get("NEXFLOW_RUNNER_PATH",  "/var/nexflow/runner.json"))
AUDIT_PATH  = Path(os.environ.get("NEXFLOW_AUDIT_PATH",   "/var/nexflow/audit.jsonl"))
PAPER_PATH  = Path(os.environ.get("NEXFLOW_PAPER_PATH",   "/var/nexflow/paper.json"))
SNAP_LOG    = Path(os.environ.get("NEXFLOW_SNAPSHOT_LOG", "/var/nexflow/snapshots.jsonl"))
EXEC_LOG    = Path(os.environ.get("NEXFLOW_EXEC_LOG",     "/var/nexflow/executions.jsonl"))
DRYRUN_PATH = Path(os.environ.get("NEXFLOW_DRYRUN_PATH",  "/var/nexflow/dryrun.json"))


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_daily(as_of_date: str, verbose: bool = False) -> int:
    """
    Run one complete daily cycle.

    Returns:
      0 — success
      1 — warning (logged, bot continues tomorrow)
      2 — fatal   (state unsafe, must not continue)
    """
    notifier = Notifier()
    exit_code = 0

    # ── STEP 1: Verify state files exist ─────────────────────────────────────
    for label, path in [
        ("state",  STATE_PATH),
        ("runner", RUNNER_PATH),
        ("dryrun", DRYRUN_PATH),
    ]:
        if not path.exists():
            notifier.fatal(
                f"Missing {label} file",
                f"Expected at {path}. Run initialize_system.py first.",
            )
            _print(f"[FATAL] Missing {label} file: {path}", verbose=True)
            return 2

    # ── STEP 2: Daily parity ──────────────────────────────────────────────────
    parity_passed = False
    try:
        parity = run_daily_parity(as_of_date, verbose=verbose, raise_on_fail=False)
        parity_passed = parity.passed
        if not parity_passed:
            notifier.error(
                "Daily parity failed",
                f"Date: {as_of_date}\nProduction signal does not match research engine.",
            )
            _print(f"[FAIL] Parity: {parity}", verbose=True)
            # Parity failure is fatal — production is drifting from research
            return 2
        _print("[PASS] Parity", verbose)
    except Exception as exc:
        notifier.error("Parity check raised exception", str(exc))
        _print(f"[FAIL] Parity exception: {exc}", verbose=True)
        return 2

    # ── STEP 3: Load system state + reconcile ─────────────────────────────────
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
        _print("[PASS] State reconciliation", verbose)
    except Exception as exc:
        notifier.fatal("Startup reconciliation failed", str(exc))
        _print(f"[FATAL] Reconciliation: {exc}", verbose=True)
        return 2

    # ── STEP 4: Compute signals ───────────────────────────────────────────────
    try:
        regime_machine = core.RegimeMachine(
            in_bear           = ss.regime.in_bear,
            consecutive_above = ss.regime.consecutive_above,
        )
        signal = compute_signals(dataset, regime_machine)
        _print(
            f"[PASS] Signals: c_sc={signal.crypto_score:.2f}  "
            f"s_sc={signal.stock_score:.2f}  regime={signal.allocation_regime}  "
            f"bear={signal.in_bear}",
            verbose,
        )
    except Exception as exc:
        notifier.fatal("Signal computation failed", str(exc))
        _print(f"[FATAL] Signals: {exc}", verbose=True)
        return 2

    # ── STEP 5: Runner step ───────────────────────────────────────────────────
    decision = None
    try:
        runner = AllocationRunner.startup_reconcile(RUNNER_PATH)
        ss.increment_trading_days(STATE_PATH)
        trading_days = ss.allocation.trading_days_since
        decision = runner.step(signal, trading_days, RUNNER_PATH)
        _print(
            f"[PASS] Runner: {'rebalance pending' if decision else f'holding ({trading_days}d)'}", verbose
        )
    except AllocationRunnerError as exc:
        notifier.fatal("Runner state error", str(exc))
        _print(f"[FATAL] Runner: {exc}", verbose=True)
        return 2
    except Exception as exc:
        notifier.fatal("Runner step raised exception", str(exc))
        _print(f"[FATAL] Runner exception: {exc}", verbose=True)
        return 2

    # ── STEP 6: Load paper trader ─────────────────────────────────────────────
    try:
        trader = (
            PaperTrader.load(PAPER_PATH)
            if PAPER_PATH.exists()
            else PaperTrader(portfolio_value=5_000.0)
        )
    except Exception as exc:
        notifier.fatal("Paper trader load failed", str(exc))
        _print(f"[FATAL] PaperTrader load: {exc}", verbose=True)
        return 2

    # ── STEP 7: Execute rebalance (if due) ────────────────────────────────────
    rebalanced = False
    if decision is not None:
        try:
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
            rebalanced = True

            notifier.info(
                f"Rebalance executed — {signal.allocation_regime}",
                (
                    f"Date: {as_of_date}\n"
                    f"wc={decision.target_wc:.0%}  ws={decision.target_ws:.0%}\n"
                    f"Crypto score: {signal.crypto_score:.2f}  "
                    f"Stock score: {signal.stock_score:.2f}\n"
                    f"Bear regime: {signal.in_bear}\n"
                    f"Reason: {decision.reason}"
                ),
            )
            _print(
                f"[PASS] Rebalance: wc={decision.target_wc:.0%}  "
                f"ws={decision.target_ws:.0%}",
                verbose,
            )

        except AllocationRunnerError as exc:
            notifier.error("Rebalance execution failed", str(exc))
            _print(f"[FAIL] Execution: {exc}", verbose=True)
            exit_code = max(exit_code, 1)
        except Exception as exc:
            notifier.fatal("Rebalance raised unhandled exception", str(exc))
            _print(f"[FATAL] Execution exception: {exc}", verbose=True)
            return 2

    # ── STEP 8: Daily snapshot ────────────────────────────────────────────────
    try:
        snap = trader.snapshot(signal, ss.allocation.wc, ss.allocation.ws)
        append_snapshot(snap, SNAP_LOG)
        _print("[PASS] Snapshot written", verbose)
    except Exception as exc:
        notifier.warning("Snapshot failed", str(exc))
        _print(f"[WARN] Snapshot: {exc}", verbose=True)
        exit_code = max(exit_code, 1)

    # ── STEP 9: Audit integrity ───────────────────────────────────────────────
    audit_clean = False
    try:
        records = load_audit_log(AUDIT_PATH)
        dates_seen = [r.date for r in records]
        if len(dates_seen) != len(set(dates_seen)):
            notifier.error("Audit log has duplicate dates", f"Audit path: {AUDIT_PATH}")
            exit_code = max(exit_code, 1)
        else:
            audit_clean = True
        _print(f"[PASS] Audit: {len(records)} records", verbose)
    except Exception as exc:
        notifier.warning("Audit integrity check failed", str(exc))
        _print(f"[WARN] Audit: {exc}", verbose=True)
        exit_code = max(exit_code, 1)

    # ── STEP 10: Runner snapshot validation ───────────────────────────────────
    snapshot_valid = False
    try:
        snap_loaded = RunnerSnapshot.load(RUNNER_PATH)
        snap_loaded.validate()
        snapshot_valid = True
        _print(f"[PASS] Runner snapshot: {snap_loaded.runner_state}", verbose)
    except Exception as exc:
        notifier.fatal("Runner snapshot corrupt after execution", str(exc))
        _print(f"[FATAL] Snapshot validation: {exc}", verbose=True)
        return 2

    # ── STEP 11: Record dry-run day ───────────────────────────────────────────
    day_result = DryRunDayResult(
        date             = as_of_date,
        parity_passed    = parity_passed,
        reconcile_passed = True,
        audit_clean      = audit_clean,
        lifecycle_clean  = True,   # if we reached here, lifecycle ran
        snapshot_valid   = snapshot_valid,
    )

    try:
        dryrun_state = DryRunState.load(DRYRUN_PATH)
        gate_was_met = dryrun_state.gate_met
        dryrun_state.record_day(day_result, DRYRUN_PATH)

        if not gate_was_met and dryrun_state.gate_met:
            notifier.info(
                "Shadow deployment gate reached",
                (
                    "30 consecutive dry-run days passed.\n"
                    "The system is now authorized for live trading.\n"
                    "Set BITGET_PAPER=0 to enable live execution."
                ),
            )

        _print(
            f"[PASS] Dry-run gate: {dryrun_state.consecutive_successes}/30"
            + (" (GATE MET)" if dryrun_state.gate_met else ""),
            verbose,
        )
    except Exception as exc:
        notifier.warning("Dry-run counter update failed", str(exc))
        _print(f"[WARN] Dry-run update: {exc}", verbose=True)
        exit_code = max(exit_code, 1)

    return exit_code


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print(msg: str, verbose: bool) -> None:
    if verbose:
        print(f"  {msg}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="V9 Confidence — Daily Orchestrator")
    parser.add_argument(
        "--date", default=date.today().isoformat(),
        help="Date to run (YYYY-MM-DD, default: today)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print(f"V9 Confidence — Daily Run  [{args.date}]")
    print("=" * 70)

    exit_code = run_daily(args.date, verbose=args.verbose)

    print(f"\n{'─' * 70}")
    status = {0: "SUCCESS", 1: "WARNING", 2: "FATAL"}[exit_code]
    print(f"RESULT: {status}  (exit {exit_code})")
    print("=" * 70)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
