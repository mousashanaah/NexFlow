#!/usr/bin/env python3
"""
NexFlow Alpha — Narrative Tagger

Retroactively tags all tokens in Alpha Memory with narrative categories,
and prints a breakdown of the current narrative distribution.

Run this once after upgrading to add narrative tags to existing discoveries.
Subsequent runs are idempotent (ON CONFLICT UPDATE).

Usage:
  python scripts/alpha_tag_narratives.py [--verbose]
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from nexflow.alpha.narrative.categorizer import categorize
from nexflow.alpha.narrative.store import (
    init_narrative_store, upsert_narrative, narrative_stats, narrative_win_rates,
)
from nexflow.alpha.store.memory import init_memory

_DB_PATH = Path(os.environ.get("NEXFLOW_ALPHA_DB", "/var/nexflow/alpha.db"))


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def run(verbose: bool) -> None:
    with _connect(_DB_PATH) as conn:
        rows = conn.execute(
            "SELECT DISTINCT token_address, token_name, token_symbol FROM alpha_memory"
        ).fetchall()

    tokens = [dict(r) for r in rows]
    print(f"Tagging {len(tokens)} tokens from Alpha Memory...")

    counts: dict[str, int] = {}
    for t in tokens:
        result = categorize(
            token_name    = t.get("token_name", ""),
            token_symbol  = t.get("token_symbol", ""),
            token_address = t.get("token_address", ""),
        )
        upsert_narrative(result, _DB_PATH)
        counts[result.category] = counts.get(result.category, 0) + 1

        if verbose:
            sig = result.matched_signals[:2]
            print(f"  {t.get('token_symbol', '?'):<12}  {result.category:<14}  "
                  f"conf={result.confidence:.2f}  signals={sig}")

    print(f"\nNarrative distribution:")
    for cat, n in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * n
        print(f"  {cat:<14}  {n:>4}  {bar}")

    # Win rates (will be empty until outcomes arrive, shows the future value)
    wr_rows = narrative_win_rates(_DB_PATH)
    classified_any = any((r["classified"] or 0) > 0 for r in wr_rows)
    if classified_any:
        print(f"\nNarrative win rates:")
        print(f"  {'CATEGORY':<14}  {'TOTAL':>6}  {'WIN%':>6}  {'AVG 7D':>8}")
        for r in wr_rows:
            if (r["classified"] or 0) == 0:
                continue
            wp = f"{r['win_rate']*100:.1f}%" if r.get("win_rate") is not None else "—"
            ar = f"{r['avg_ret_7d']:.2f}x" if r.get("avg_ret_7d") is not None else "—"
            print(f"  {r['category']:<14}  {r['total']:>6}  {wp:>6}  {ar:>8}")
    else:
        print("\nWin rates: no classified outcomes yet — check back after 7d.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Retroactively tag narrative categories")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    init_memory(_DB_PATH)
    init_narrative_store(_DB_PATH)
    run(verbose=args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
