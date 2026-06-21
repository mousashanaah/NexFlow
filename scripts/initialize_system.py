#!/usr/bin/env python3
"""
One-time system initialization for dry-run deployment.

Run this ONCE before the first daily dry-run.
If state files already exist this will refuse to overwrite them.

Usage:
  python scripts/initialize_system.py [--date YYYY-MM-DD]
"""
from __future__ import annotations

import sys
import os
import argparse
from datetime import date
from pathlib import Path

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from nexflow.v9.adapter import NullAdapter
from nexflow.v9.dryrun import DryRunState
from nexflow.v9.paper import PaperTrader
from nexflow.v9.runner import AllocationRunner
from nexflow.v9.state import SystemState

STATE_PATH  = Path(os.environ.get("NEXFLOW_STATE_PATH",  "/var/nexflow/state.json"))
RUNNER_PATH = Path(os.environ.get("NEXFLOW_RUNNER_PATH", "/var/nexflow/runner.json"))
PAPER_PATH  = Path(os.environ.get("NEXFLOW_PAPER_PATH",  "/var/nexflow/paper.json"))
DRYRUN_PATH = Path(os.environ.get("NEXFLOW_DRYRUN_PATH", "/var/nexflow/dryrun.json"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize V9 system state")
    parser.add_argument("--date", default=date.today().isoformat(),
                        help="Initial rebalance date (YYYY-MM-DD)")
    parser.add_argument("--portfolio-value", type=float, default=5_000.0,
                        help="Paper portfolio value in USDT (default: 5000)")
    args = parser.parse_args()

    print("=" * 60)
    print("V9 Confidence — System Initialization")
    print("=" * 60)

    errors = []

    # ── state.json ────────────────────────────────────────────────────────────
    if STATE_PATH.exists():
        errors.append(f"State file already exists: {STATE_PATH}")
    else:
        SystemState.initialize(
            path           = STATE_PATH,
            wc             = 0.50,
            ws             = 0.50,
            rebalance_date = args.date,
        )
        print(f"  Created: {STATE_PATH}  (wc=0.50, ws=0.50)")

    # ── runner.json ───────────────────────────────────────────────────────────
    if RUNNER_PATH.exists():
        errors.append(f"Runner file already exists: {RUNNER_PATH}")
    else:
        AllocationRunner.initialize(RUNNER_PATH)
        print(f"  Created: {RUNNER_PATH}  (state=IDLE)")

    # ── paper.json ────────────────────────────────────────────────────────────
    if PAPER_PATH.exists():
        errors.append(f"Paper file already exists: {PAPER_PATH}")
    else:
        trader = PaperTrader(portfolio_value=args.portfolio_value)
        trader.save(PAPER_PATH)
        print(f"  Created: {PAPER_PATH}  (value={args.portfolio_value} USDT)")

    # ── dryrun.json ───────────────────────────────────────────────────────────
    if DRYRUN_PATH.exists():
        print(f"  Skipped: {DRYRUN_PATH} (already exists — preserving run history)")
    else:
        state = DryRunState()
        state.save(DRYRUN_PATH)
        print(f"  Created: {DRYRUN_PATH}  (0/30 days)")

    if errors:
        print(f"\n{'─' * 60}")
        print("ABORTED — files already exist:")
        for e in errors:
            print(f"  {e}")
        print("\nDelete them manually if you intend to reset.")
        return 1

    print(f"\n{'─' * 60}")
    print("Initialization complete.")
    print(f"Run daily: python scripts/run_dryrun_day.py --date YYYY-MM-DD")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
