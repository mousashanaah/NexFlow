#!/usr/bin/env python3
"""
V9 Confidence — Scheduler Setup

One-time installation of the daily cron job.
Run this ONCE after initialize_system.py.

The job fires at 22:00 UTC Monday–Friday.  Data downloads must happen
separately on the local machine before this time — this script schedules
the server-side orchestration only.

Usage:
  python scripts/setup_scheduler.py [--show] [--remove] [--time HH:MM]

  --show    : Print the cron entry that would be installed, then exit
  --remove  : Remove the V9 cron entry
  --time    : Run time in UTC (default: 22:00)
  --user    : User whose crontab to write (default: current user)

Requirements:
  - cron must be installed (crontab command available)
  - Python interpreter path must be correct (uses sys.executable)

Scheduling note:
  22:00 UTC is chosen because:
    - US markets close at 21:00 UTC (4pm ET) — data is final
    - Crypto data is always available (24/7)
    - Leaves 2 hours before midnight UTC for any retry window
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO     = Path(__file__).parent.parent
_SCRIPT   = _REPO / "scripts" / "run_daily.py"
_PYTHON   = sys.executable
_MARKER   = "# nexflow-v9-daily"
_LOG_PATH = "/var/nexflow/cron.log"


def build_cron_entry(run_time: str) -> str:
    hour, minute = run_time.split(":")
    # Mon–Fri only (cron: 1=Mon, 5=Fri)
    return (
        f"{minute} {hour} * * 1-5  "
        f"{_PYTHON} {_SCRIPT} >> {_LOG_PATH} 2>&1  "
        f"{_MARKER}"
    )


def read_crontab() -> str:
    try:
        result = subprocess.run(
            ["crontab", "-l"],
            capture_output=True, text=True,
        )
        # crontab -l exits 1 with "no crontab" when empty — that's fine
        return result.stdout
    except FileNotFoundError:
        print("[ERROR] crontab command not found. Install cron first.")
        sys.exit(1)


def write_crontab(content: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".crontab", delete=False) as f:
        f.write(content)
        tmp = f.name
    try:
        subprocess.run(["crontab", tmp], check=True)
    finally:
        os.unlink(tmp)


def install(run_time: str) -> None:
    entry = build_cron_entry(run_time)
    existing = read_crontab()

    if _MARKER in existing:
        print("[INFO] V9 cron entry already installed.")
        print(f"       To update, run --remove first, then re-install.")
        return

    # Ensure log directory exists
    log_dir = Path(_LOG_PATH).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    new_content = existing.rstrip("\n") + ("\n" if existing.strip() else "") + entry + "\n"
    write_crontab(new_content)

    print(f"[OK] Cron job installed.")
    print(f"     Fires: weekdays at {run_time} UTC")
    print(f"     Script: {_SCRIPT}")
    print(f"     Log: {_LOG_PATH}")
    print(f"\nCron entry:")
    print(f"  {entry}")


def remove() -> None:
    existing = read_crontab()
    if _MARKER not in existing:
        print("[INFO] No V9 cron entry found — nothing to remove.")
        return

    filtered = "\n".join(
        line for line in existing.splitlines()
        if _MARKER not in line
    ) + "\n"
    write_crontab(filtered)
    print("[OK] V9 cron entry removed.")


def show(run_time: str) -> None:
    entry = build_cron_entry(run_time)
    print("Cron entry that would be installed:")
    print(f"  {entry}")
    print()
    print("Current crontab:")
    current = read_crontab()
    if current.strip():
        for line in current.splitlines():
            print(f"  {line}")
    else:
        print("  (empty)")


def main() -> int:
    parser = argparse.ArgumentParser(description="V9 Confidence — Scheduler Setup")
    parser.add_argument("--show",   action="store_true", help="Print cron entry without installing")
    parser.add_argument("--remove", action="store_true", help="Remove V9 cron entry")
    parser.add_argument("--time",   default="22:00",     help="Run time UTC (HH:MM, default: 22:00)")
    args = parser.parse_args()

    if ":" not in args.time or len(args.time.split(":")) != 2:
        print(f"[ERROR] Invalid --time '{args.time}'. Expected HH:MM.")
        return 1

    print("=" * 60)
    print("V9 Confidence — Scheduler Setup")
    print("=" * 60)

    if args.show:
        show(args.time)
    elif args.remove:
        remove()
    else:
        print(f"\nInstalling cron job for {_PYTHON}")
        print(f"Script: {_SCRIPT}")
        print(f"Time:   {args.time} UTC (Mon–Fri)\n")
        install(args.time)

    return 0


if __name__ == "__main__":
    sys.exit(main())
