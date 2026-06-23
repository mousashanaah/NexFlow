#!/usr/bin/env python3
"""
NexFlow Alpha — DB flag migration: fix corrupt TOP10_HOLDS percentages.

The first batch of RugCheck risk checks stored TOP10_HOLDS flags with values
on the 0-100 scale formatted as if they were 0-1 (e.g. 5714% instead of 57.1%).
This script patches risk_flags JSON in both risk_results and alpha_memory tables.

Usage:
  python scripts/alpha_migrate_flags.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

_DB_PATH = Path(os.environ.get("NEXFLOW_ALPHA_DB", "/var/nexflow/alpha.db"))

# Matches TOP10_HOLDS:NNNNN% where NNNNN > 100 (corrupt values)
_BAD_TOP10 = re.compile(r"TOP10_HOLDS:(\d+(?:\.\d+)?)%")


def _fix_flags_json(raw: str | None) -> tuple[str | None, bool]:
    """Return (fixed_json, changed). Divides corrupt TOP10_HOLDS values by 100."""
    if not raw:
        return raw, False
    try:
        flags = json.loads(raw)
    except Exception:
        return raw, False

    changed = False
    new_flags = []
    for flag in flags:
        m = _BAD_TOP10.match(flag)
        if m:
            val = float(m.group(1))
            if val > 100:           # clearly corrupt (e.g. 5714 = 57.14 * 100)
                fixed = val / 100
                new_flag = f"TOP10_HOLDS:{fixed:.1f}%"
                new_flags.append(new_flag)
                changed = True
                continue
        new_flags.append(flag)

    return (json.dumps(new_flags) if changed else raw), changed


def run_migration(dry_run: bool) -> None:
    if not _DB_PATH.exists():
        print(f"DB not found: {_DB_PATH}")
        return

    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row

    tables = [
        ("risk_results", "token_address", "risk_flags"),
        ("alpha_memory",  "discovery_id",  "risk_flags"),
    ]

    total_fixed = 0

    for table, pk_col, flags_col in tables:
        # Check table exists
        exists = conn.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,)
        ).fetchone()
        if not exists:
            continue

        rows = conn.execute(f"SELECT {pk_col}, {flags_col} FROM {table}").fetchall()
        fixed_rows = []
        for row in rows:
            new_val, changed = _fix_flags_json(row[flags_col])
            if changed:
                fixed_rows.append((new_val, row[pk_col]))

        print(f"  {table}: {len(rows)} rows checked, {len(fixed_rows)} to fix")
        if fixed_rows and not dry_run:
            conn.executemany(
                f"UPDATE {table} SET {flags_col}=? WHERE {pk_col}=?",
                fixed_rows,
            )
            conn.commit()
        total_fixed += len(fixed_rows)

        if fixed_rows:
            print(f"  Examples fixed:")
            for new_val, pk in fixed_rows[:3]:
                print(f"    {pk[:20]}...  →  {new_val}")

    conn.close()
    prefix = "[DRY RUN] " if dry_run else ""
    print(f"\n{prefix}Total rows fixed: {total_fixed}")
    if dry_run and total_fixed:
        print("  Re-run without --dry-run to apply.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fix corrupt TOP10_HOLDS flag values")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"Auditing flag data in: {_DB_PATH}")
    run_migration(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
