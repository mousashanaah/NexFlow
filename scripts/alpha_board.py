#!/usr/bin/env python3
"""
NexFlow Alpha — Board

Discovers new DEX pools, runs the risk gate, writes to Alpha Memory,
and prints a ranked list.

Usage:
  python scripts/alpha_board.py [--refresh] [--all] [--verbose]
  python scripts/alpha_board.py [--chains solana,ethereum,bsc,base]
  python scripts/alpha_board.py [--max-age 48] [--min-liquidity 1000]

  --refresh        : Fetch new pools from DexScreener
  --all            : Show blocked/unverified pools too
  --chains         : Comma-separated chain filter
  --max-age        : Max pool age in hours (default: 48)
  --min-liquidity  : Min liquidity USD (default: 1000)
  --verbose        : Show detailed fetch progress

Risk labels:
  CLEAN      — passed all checks
  CAUTION    — passed but has minor flags
  RISKY      — passed but significant flags
  BLOCKED    — failed risk gate
  UNVERIFIED — risk API unavailable

Config:
  NEXFLOW_ALPHA_DB : SQLite path (default: /var/nexflow/alpha.db)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from nexflow.alpha.discovery.dexscreener import fetch_new_pools
from nexflow.alpha.discovery.risk import check_risk
from nexflow.alpha.store.db import init_db, load_board, upsert_pool, upsert_risk
from nexflow.alpha.store.memory import init_memory, record_discovery, summary_stats

_DB_PATH = Path(os.environ.get("NEXFLOW_ALPHA_DB", "/var/nexflow/alpha.db"))

_RISK_COLOR = {
    "CLEAN":      "\033[92m",   # green
    "CAUTION":    "\033[93m",   # yellow
    "RISKY":      "\033[91m",   # red
    "BLOCKED":    "\033[90m",   # grey
    "UNVERIFIED": "\033[35m",   # magenta — never showed as CLEAN without a real check
    None:         "\033[0m",
}
_RESET = "\033[0m"


def _fmt_usd(val) -> str:
    if val is None:
        return "—"
    if val >= 1_000_000:
        return f"${val/1_000_000:.1f}M"
    if val >= 1_000:
        return f"${val/1_000:.0f}K"
    return f"${val:.0f}"


def _color(text: str, label: str | None) -> str:
    c = _RISK_COLOR.get(label, "")
    return f"{c}{text}{_RESET}" if c else text


def refresh(
    max_age_hours: float,
    min_liquidity: float,
    chains:        list[str] | None,
    verbose:       bool = False,
) -> int:
    """Fetch new pools, run risk gate, write to Alpha Memory."""
    print("Fetching new pools from DexScreener...")
    pools = fetch_new_pools(
        max_age_hours    = max_age_hours,
        min_liquidity    = min_liquidity,
        supported_chains = chains,
        verbose          = verbose,
    )
    print(f"  Found {len(pools)} pools")

    if not pools:
        return 0

    print("Running risk gate...")
    for i, pool in enumerate(pools, 1):
        upsert_pool(pool, _DB_PATH)

        print(f"  [{i}/{len(pools)}] {pool.token_symbol or pool.token_address[:8]}"
              f"  ({pool.chain_id})  liq={_fmt_usd(pool.liquidity_usd)}"
              f"  age={pool.age_label}", end="  ", flush=True)

        risk = check_risk(pool.token_address, pool.chain_id)
        upsert_risk(risk, _DB_PATH)

        # Write to Alpha Memory (permanent record)
        record_discovery(pool, risk, _DB_PATH)

        print(_color(risk.risk_label, risk.risk_label))
        time.sleep(0.3)

    return len(pools)


def display_board(passed_only: bool, max_age_hours: float) -> None:
    rows = load_board(
        path          = _DB_PATH,
        passed_only   = passed_only,
        max_age_hours = max_age_hours,
        limit         = 100,
    )

    if not rows:
        print("\nNo pools found. Run with --refresh to fetch new pools.")
        return

    w = 115
    print(f"\n{'─'*w}")
    print(f"{'NEXFLOW ALPHA BOARD':^{w}}")
    print(f"{'─'*w}")
    print(
        f"{'#':<3}  {'SYMBOL':<10}  {'CHAIN':<9}  {'AGE':<7}  "
        f"{'LIQUIDITY':<11}  {'VOLUME 24H':<11}  {'MCAP':<11}  "
        f"{'RISK':<11}  FLAGS"
    )
    print(f"{'─'*w}")

    for i, row in enumerate(rows, 1):
        risk_label = row.get("risk_label") or "UNVERIFIED"
        flags_raw  = row.get("risk_flags")
        flags_list = json.loads(flags_raw) if flags_raw else []
        flags_str  = ", ".join(flags_list[:3]) if flags_list else ""

        age_h = row.get("age_hours")
        if age_h is not None:
            if age_h < 1:
                age_str = f"{int(age_h*60)}m"
            elif age_h < 24:
                age_str = f"{age_h:.1f}h"
            else:
                age_str = f"{age_h/24:.1f}d"
        else:
            age_str = "—"

        symbol = (row.get("token_symbol") or "")[:9]
        chain  = (row.get("chain_id")     or "")[:9]
        line = (
            f"{i:<3}  {symbol:<10}  {chain:<9}  {age_str:<7}  "
            f"{_fmt_usd(row.get('liquidity_usd')):<11}  "
            f"{_fmt_usd(row.get('volume_24h')):<11}  "
            f"{_fmt_usd(row.get('market_cap')):<11}  "
            f"{risk_label:<11}  {flags_str}"
        )
        print(_color(line, risk_label))

    passed     = sum(1 for r in rows if r.get("passed"))
    blocked    = sum(1 for r in rows if r.get("risk_label") == "BLOCKED")
    unverified = sum(1 for r in rows if r.get("risk_label") in ("UNVERIFIED", None))

    print(f"{'─'*w}")
    print(
        f"  {len(rows)} pools  |  {passed} clean/caution/risky  |  "
        f"{blocked} blocked  |  {unverified} unverified"
    )

    # Alpha Memory summary
    try:
        stats = summary_stats(_DB_PATH)
        total = stats["total_discoveries"]
        by_c  = stats["by_classification"]
        print(f"  Alpha Memory: {total} total discoveries recorded", end="")
        if by_c:
            parts = [f"{k}={v}" for k, v in by_c.items() if k != "unclassified"]
            if parts:
                print(f"  ({', '.join(parts)})", end="")
        print()
    except Exception:
        pass

    print(f"{'─'*w}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="NexFlow Alpha Board")
    parser.add_argument("--refresh",       action="store_true")
    parser.add_argument("--all",           action="store_true",
                        help="Show blocked/unverified pools")
    parser.add_argument("--chains",        default=None)
    parser.add_argument("--max-age",       type=float, default=48.0)
    parser.add_argument("--min-liquidity", type=float, default=1_000.0)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    chains = [c.strip() for c in args.chains.split(",")] if args.chains else None

    init_db(_DB_PATH)
    init_memory(_DB_PATH)

    if args.refresh:
        count = refresh(args.max_age, args.min_liquidity, chains, verbose=args.verbose)
        print(f"\n  Processed {count} pools\n")

    display_board(passed_only=not args.all, max_age_hours=args.max_age)
    return 0


if __name__ == "__main__":
    sys.exit(main())
