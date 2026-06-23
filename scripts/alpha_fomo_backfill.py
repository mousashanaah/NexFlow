#!/usr/bin/env python3
"""
NexFlow Alpha — FOMO tradability diagnostic + backfill

Step 1: Diagnoses why FOMO shows 0 tradable
Step 2: Tests Jupiter API against your 5 most liquid pools
Step 3: Backfills fomo_available into signal_snapshots for all existing pools

Run this after pulling the latest code:
  python scripts/alpha_fomo_backfill.py --diagnose    # diagnosis only
  python scripts/alpha_fomo_backfill.py               # diagnose + backfill
  python scripts/alpha_fomo_backfill.py --verbose     # show all API results
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from nexflow.alpha.tradability.fomo import is_fomo_tradable
from nexflow.alpha.store.attribution import init_attribution, record_snapshot, update_fomo_availability
from nexflow.alpha.narrative.categorizer import categorize
from nexflow.alpha.narrative.store import init_narrative_store, upsert_narrative, narrative_win_rates
from nexflow.alpha.wallets.registry import init_wallet_registry, token_wallet_summary
from nexflow.alpha.store.memory import init_memory

_DB_PATH = Path(os.environ.get("NEXFLOW_ALPHA_DB", "/var/nexflow/alpha.db"))


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def diagnose() -> dict:
    conn = _connect(_DB_PATH)
    pools_total = conn.execute("SELECT COUNT(*) FROM pools").fetchone()[0]
    solana_pools = conn.execute(
        "SELECT COUNT(*) FROM pools WHERE chain_id='solana'"
    ).fetchone()[0]

    snap_table = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='signal_snapshots'"
    ).fetchone()
    snaps_total = 0
    fomo_flagged = 0
    if snap_table:
        snaps_total  = conn.execute("SELECT COUNT(*) FROM signal_snapshots").fetchone()[0]
        fomo_flagged = conn.execute(
            "SELECT COUNT(*) FROM signal_snapshots WHERE fomo_available=1"
        ).fetchone()[0]

    top5 = conn.execute("""
        SELECT p.token_address, p.chain_id, p.token_symbol,
               p.liquidity_usd, p.volume_24h,
               ss.fomo_available, ss.snapshot_id
        FROM pools p
        LEFT JOIN signal_snapshots ss ON p.pair_address = ss.pair_address
        WHERE p.chain_id = 'solana'
        ORDER BY p.liquidity_usd DESC NULLS LAST
        LIMIT 5
    """).fetchall()
    conn.close()

    print(f"\n{'─'*60}")
    print("FOMO TRADABILITY DIAGNOSIS")
    print(f"{'─'*60}")
    print(f"  Total pools:              {pools_total}")
    print(f"  Solana pools:             {solana_pools}")
    print(f"  signal_snapshots rows:    {snaps_total}")
    print(f"  Pools with FOMO checked:  {fomo_flagged}")
    print()

    if snaps_total == 0:
        print("  ROOT CAUSE: signal_snapshots table is empty.")
        print("  The table was added in the latest code update but only gets")
        print("  populated when --refresh is run with the new code.")
        print("  Fix: run this script to backfill, then re-run --refresh.")
    elif fomo_flagged == 0 and snaps_total > 0:
        print("  ROOT CAUSE: Snapshots exist but all show fomo_available=0.")
        print("  Either Jupiter returned no_route for all tokens,")
        print("  or all pools are EVM (not Solana).")
    print()

    print("  Top 5 Solana pools (snapshot status):")
    print(f"  {'SYMBOL':<12}  {'LIQUIDITY':>12}  {'SNAPSHOT':>10}  {'FOMO_AVAIL':>10}  TOKEN")
    for r in top5:
        has_snap = "yes" if r["snapshot_id"] else "no"
        fomo_v   = str(r["fomo_available"]) if r["snapshot_id"] else "—"
        print(f"  {(r['token_symbol'] or '?'):<12}  ${(r['liquidity_usd'] or 0):>10,.0f}"
              f"  {has_snap:>10}  {fomo_v:>10}  {r['token_address'][:20]}...")

    return {
        "pools_total":    pools_total,
        "solana_pools":   solana_pools,
        "snaps_total":    snaps_total,
        "fomo_flagged":   fomo_flagged,
        "top5":           [dict(r) for r in top5],
    }


def backfill(verbose: bool) -> None:
    """
    For every Solana pool that lacks a signal_snapshot (or has one without
    FOMO checked), run the Jupiter tradability check and upsert the result.
    """
    conn = _connect(_DB_PATH)
    pools = conn.execute("""
        SELECT p.pair_address, p.token_address, p.chain_id, p.token_symbol,
               p.liquidity_usd, p.volume_24h, p.market_cap, p.age_hours,
               p.first_seen_at,
               r.risk_score, r.risk_label, r.passed AS risk_passed,
               ss.snapshot_id, ss.fomo_available AS current_fomo
        FROM pools p
        LEFT JOIN risk_results r    ON p.token_address = r.token_address
        LEFT JOIN signal_snapshots ss ON p.pair_address = ss.pair_address
        WHERE p.chain_id = 'solana'
        ORDER BY p.liquidity_usd DESC NULLS LAST
    """).fetchall()
    conn.close()

    print(f"\nBackfilling FOMO tradability for {len(pools)} Solana pools...")
    print(f"  (Checking Jupiter Quote API for each token)\n")

    tradable = 0
    not_tradable = 0
    errors = 0

    for pool in pools:
        sym   = (pool["token_symbol"] or pool["token_address"][:8])
        liq   = pool["liquidity_usd"] or 0
        token = pool["token_address"]

        fomo_ok, fomo_err = is_fomo_tradable(token, "solana")

        if fomo_err and fomo_err not in ("no_route",):
            status = f"ERROR({fomo_err})"
            errors += 1
        elif fomo_ok:
            status = "TRADABLE ✓"
            tradable += 1
        else:
            status = "no route"
            not_tradable += 1

        if verbose or fomo_ok:
            print(f"  {sym:<12}  liq=${liq:>10,.0f}  {status}")

        disc_ts = pool["first_seen_at"] or datetime.now(timezone.utc).isoformat()

        if not pool["snapshot_id"]:
            # No snapshot at all — create one
            try:
                wsummary = token_wallet_summary(token, _DB_PATH)
                wscore   = wsummary["wallet_score"]
                wbacked  = wsummary.get("outcome_backed", False)
                wtracked = wsummary.get("known_wallets", 0)
            except Exception:
                wscore = None; wbacked = False; wtracked = 0

            narrative = categorize(pool["token_symbol"] or "", pool["token_symbol"] or "")
            try:
                upsert_narrative(narrative, _DB_PATH)
            except Exception:
                pass

            try:
                nwr_rows   = narrative_win_rates(_DB_PATH)
                nwr_lookup = {r["category"]: r.get("win_rate") for r in nwr_rows}
                nwr        = nwr_lookup.get(narrative.category)
            except Exception:
                nwr = None

            record_snapshot(
                pair_address          = pool["pair_address"],
                token_address         = token,
                token_symbol          = pool["token_symbol"] or "",
                chain_id              = "solana",
                discovery_ts          = disc_ts,
                risk_score            = pool["risk_score"],
                risk_label            = pool["risk_label"],
                risk_passed           = bool(pool["risk_passed"]),
                liquidity_usd         = pool["liquidity_usd"],
                volume_24h            = pool["volume_24h"],
                market_cap            = pool["market_cap"],
                age_hours             = pool["age_hours"],
                wallet_score          = wscore,
                wallet_outcome_backed = wbacked,
                wallets_tracked       = wtracked,
                narrative_category    = narrative.category,
                narrative_confidence  = narrative.confidence,
                narrative_win_rate    = nwr,
                fomo_available        = fomo_ok,
                path                  = _DB_PATH,
            )
        else:
            # Snapshot exists — just update FOMO field
            update_fomo_availability(token, fomo_ok, disc_ts if fomo_ok else None, _DB_PATH)

        time.sleep(0.2)

    print(f"\n  Results: {tradable} tradable  |  {not_tradable} no route  |  {errors} errors")
    print(f"  Tradable tokens are now shown in ACTIONABLE section of the board.")
    if errors:
        print(f"  {errors} errors may indicate network issues — re-run if needed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="FOMO tradability diagnostic + backfill")
    parser.add_argument("--diagnose", action="store_true",
                        help="Only show diagnosis, skip backfill")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    init_memory(_DB_PATH)
    init_wallet_registry(_DB_PATH)
    init_narrative_store(_DB_PATH)
    init_attribution(_DB_PATH)

    diag = diagnose()

    if not args.diagnose:
        backfill(verbose=args.verbose)
        print()
        diagnose()   # re-run to show updated counts

    return 0


if __name__ == "__main__":
    sys.exit(main())
