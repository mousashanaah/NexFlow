#!/usr/bin/env python3
"""
NexFlow Alpha — Week 1 Board

Discovers new DEX pools, runs the risk gate, and prints a ranked list.

Usage:
  python scripts/alpha_board.py [--refresh] [--all] [--chains ETH,BSC,BASE]

  --refresh        : Fetch new pools from DexScreener (otherwise show cached)
  --all            : Show all pools including blocked ones
  --chains         : Comma-separated chain filter (default: all supported)
  --max-age        : Max pool age in hours to show (default: 48)
  --min-liquidity  : Min liquidity USD to include (default: 5000)

Output:
  Ranked list of new pools with risk classification.
  CLEAN / CAUTION / RISKY / BLOCKED

Config:
  NEXFLOW_ALPHA_DB   : SQLite database path (default: /var/nexflow/alpha.db)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from nexflow.alpha.discovery.dexscreener import fetch_new_pools
from nexflow.alpha.discovery.risk import check_risk
from nexflow.alpha.store.db import init_db, load_board, upsert_pool, upsert_risk

_DB_PATH = Path(os.environ.get("NEXFLOW_ALPHA_DB", "/var/nexflow/alpha.db"))

_RISK_COLOR = {
    "CLEAN":   "\033[92m",   # green
    "CAUTION": "\033[93m",   # yellow
    "RISKY":   "\033[91m",   # red
    "BLOCKED": "\033[90m",   # grey
    None:      "\033[0m",
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
    max_age_hours:   float,
    min_liquidity:   float,
    chains:          list[str] | None,
) -> int:
    """Fetch new pools, run risk gate, store results. Returns count of new pools."""
    print("Fetching new pools from DexScreener...")
    pools = fetch_new_pools(
        max_age_hours    = max_age_hours,
        min_liquidity    = min_liquidity,
        supported_chains = chains,
    )
    print(f"  Found {len(pools)} pools")

    if not pools:
        return 0

    print("Running risk gate...")
    new_count = 0
    for i, pool in enumerate(pools, 1):
        upsert_pool(pool, _DB_PATH)

        print(f"  [{i}/{len(pools)}] {pool.token_symbol or pool.token_address[:8]}  "
              f"({pool.chain_id})  liq={_fmt_usd(pool.liquidity_usd)}  "
              f"age={pool.age_label}", end="  ", flush=True)

        risk = check_risk(pool.token_address, pool.chain_id)
        upsert_risk(risk, _DB_PATH)
        print(_color(risk.risk_label, risk.risk_label))

        time.sleep(0.3)
        new_count += 1

    return new_count


def display_board(
    passed_only:   bool,
    max_age_hours: float,
) -> None:
    rows = load_board(
        path          = _DB_PATH,
        passed_only   = passed_only,
        max_age_hours = max_age_hours,
        limit         = 100,
    )

    if not rows:
        print("\nNo pools in database. Run with --refresh to fetch new pools.")
        return

    import json

    header = (
        f"\n{'─'*110}\n"
        f"{'NEXFLOW ALPHA — WEEK 1 BOARD':^110}\n"
        f"{'─'*110}\n"
        f"{'#':<3}  {'SYMBOL':<10}  {'CHAIN':<9}  {'AGE':<7}  "
        f"{'LIQUIDITY':<11}  {'VOLUME 24H':<11}  {'MCAP':<11}  "
        f"{'RISK':<8}  {'FLAGS'}\n"
        f"{'─'*110}"
    )
    print(header)

    for i, row in enumerate(rows, 1):
        risk_label = row.get("risk_label") or "UNKNOWN"
        flags_raw  = row.get("risk_flags")
        flags_list = json.loads(flags_raw) if flags_raw else []
        flags_str  = ", ".join(flags_list[:3]) if flags_list else ""

        age_hours  = row.get("age_hours")
        if age_hours is not None:
            if age_hours < 1:
                age_str = f"{int(age_hours*60)}m"
            elif age_hours < 24:
                age_str = f"{age_hours:.1f}h"
            else:
                age_str = f"{age_hours/24:.1f}d"
        else:
            age_str = "—"

        symbol = (row.get("token_symbol") or "")[:9]
        chain  = (row.get("chain_id") or "")[:9]

        line = (
            f"{i:<3}  {symbol:<10}  {chain:<9}  {age_str:<7}  "
            f"{_fmt_usd(row.get('liquidity_usd')):<11}  "
            f"{_fmt_usd(row.get('volume_24h')):<11}  "
            f"{_fmt_usd(row.get('market_cap')):<11}  "
            f"{risk_label:<8}  {flags_str}"
        )
        print(_color(line, risk_label))

    passed  = sum(1 for r in rows if r.get("passed"))
    blocked = len(rows) - passed
    print(f"{'─'*110}")
    print(f"  {len(rows)} pools  |  {passed} passed risk gate  |  {blocked} blocked")
    print(f"{'─'*110}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="NexFlow Alpha — Week 1 Board")
    parser.add_argument("--refresh",       action="store_true",
                        help="Fetch new pools from DexScreener")
    parser.add_argument("--all",           action="store_true",
                        help="Show blocked pools too")
    parser.add_argument("--chains",        default=None,
                        help="Comma-separated chains, e.g. ethereum,bsc,base")
    parser.add_argument("--max-age",       type=float, default=48.0,
                        help="Max pool age in hours (default: 48)")
    parser.add_argument("--min-liquidity", type=float, default=5_000.0,
                        help="Min liquidity USD (default: 5000)")
    args = parser.parse_args()

    chains = [c.strip() for c in args.chains.split(",")] if args.chains else None

    init_db(_DB_PATH)

    if args.refresh:
        count = refresh(args.max_age, args.min_liquidity, chains)
        print(f"\n  Processed {count} pools\n")

    display_board(
        passed_only   = not args.all,
        max_age_hours = args.max_age,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
