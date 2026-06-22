"""
DexScreener client — new pool discovery.

Two sources:
  - /token-profiles/latest/v1   : recently created token profiles (30 items)
  - /token-boosts/latest/v1     : recently boosted/promoted tokens (30 items)

Both are free, no auth required.
Rate limit: ~1 req/sec safe.
"""
from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional


_BASE    = "https://api.dexscreener.com"
_TIMEOUT = 15
_HEADERS = {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}


@dataclass
class DexPool:
    chain_id:        str
    dex_id:          str
    pair_address:    str
    token_address:   str
    token_name:      str
    token_symbol:    str
    price_usd:       Optional[float]
    liquidity_usd:   Optional[float]
    volume_24h:      Optional[float]
    market_cap:      Optional[float]
    pair_created_at: Optional[int]    # unix ms
    age_hours:       Optional[float]
    url:             str
    source:          str              # "profiles" or "boosts"

    @property
    def age_label(self) -> str:
        if self.age_hours is None:
            return "unknown"
        if self.age_hours < 1:
            return f"{int(self.age_hours * 60)}m"
        if self.age_hours < 24:
            return f"{self.age_hours:.1f}h"
        return f"{self.age_hours / 24:.1f}d"


def _get(path: str) -> dict | list:
    url = f"{_BASE}{path}" if path.startswith("/") else path
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode())


def _parse_pair(pair: dict, source: str) -> Optional[DexPool]:
    try:
        base = pair.get("baseToken") or {}
        liq  = pair.get("liquidity") or {}
        vol  = pair.get("volume") or {}

        created_at = pair.get("pairCreatedAt")
        age_hours  = None
        if created_at:
            age_hours = (time.time() * 1000 - created_at) / 3_600_000

        price_str = pair.get("priceUsd")
        price     = float(price_str) if price_str else None

        return DexPool(
            chain_id        = pair.get("chainId", ""),
            dex_id          = pair.get("dexId", ""),
            pair_address    = pair.get("pairAddress", ""),
            token_address   = base.get("address", ""),
            token_name      = base.get("name", ""),
            token_symbol    = base.get("symbol", ""),
            price_usd       = price,
            liquidity_usd   = liq.get("usd"),
            volume_24h      = vol.get("h24"),
            market_cap      = pair.get("marketCap"),
            pair_created_at = created_at,
            age_hours       = age_hours,
            url             = pair.get("url", ""),
            source          = source,
        )
    except Exception:
        return None


def _fetch_pairs_for_token(token_address: str, chain_id: str) -> list[dict]:
    """Fetch all pairs for a token address. Returns raw pair dicts."""
    try:
        data  = _get(f"/latest/dex/tokens/{token_address}")
        pairs = data.get("pairs") or []
        # Keep only pairs on the expected chain
        return [p for p in pairs if p.get("chainId") == chain_id]
    except Exception:
        return []


def fetch_new_profiles(
    max_age_hours: float = 48.0,
    verbose:       bool  = False,
) -> list[DexPool]:
    """Fetch pools from token-profiles endpoint."""
    pools: list[DexPool] = []
    errors: list[str]    = []

    try:
        items = _get("/token-profiles/latest/v1")
        if not isinstance(items, list):
            items = []
    except Exception as exc:
        if verbose:
            print(f"  [profiles] fetch failed: {exc}")
        return []

    if verbose:
        print(f"  [profiles] {len(items)} profile items")

    for item in items:
        token_addr = item.get("tokenAddress", "")
        chain_id   = item.get("chainId", "")
        if not token_addr or not chain_id:
            continue

        pairs = _fetch_pairs_for_token(token_addr, chain_id)
        if not pairs:
            errors.append(f"no_pairs:{token_addr[:12]}")
            time.sleep(0.2)
            continue

        # Best pair = highest liquidity
        best = max(pairs, key=lambda p: (p.get("liquidity") or {}).get("usd") or 0)
        pool = _parse_pair(best, "profiles")
        if pool:
            if pool.age_hours is None or pool.age_hours <= max_age_hours:
                pools.append(pool)
            elif verbose:
                print(f"  [profiles] skip {pool.token_symbol}: age {pool.age_hours:.0f}h > {max_age_hours}h")
        time.sleep(0.25)

    if verbose and errors:
        print(f"  [profiles] {len(errors)} tokens had no pairs")
    return pools


def fetch_boosted_pools(
    max_age_hours: float = 48.0,
    verbose:       bool  = False,
) -> list[DexPool]:
    """Fetch pools from token-boosts endpoint."""
    pools: list[DexPool] = []
    errors: list[str]    = []

    try:
        items = _get("/token-boosts/latest/v1")
        if not isinstance(items, list):
            items = []
    except Exception as exc:
        if verbose:
            print(f"  [boosts] fetch failed: {exc}")
        return []

    if verbose:
        print(f"  [boosts] {len(items)} boost items")

    for item in items:
        token_addr = item.get("tokenAddress", "")
        chain_id   = item.get("chainId", "")
        if not token_addr or not chain_id:
            continue

        pairs = _fetch_pairs_for_token(token_addr, chain_id)
        if not pairs:
            errors.append(f"no_pairs:{token_addr[:12]}")
            time.sleep(0.2)
            continue

        best = max(pairs, key=lambda p: (p.get("liquidity") or {}).get("usd") or 0)
        pool = _parse_pair(best, "boosts")
        if pool:
            if pool.age_hours is None or pool.age_hours <= max_age_hours:
                pools.append(pool)
        time.sleep(0.25)

    if verbose and errors:
        print(f"  [boosts] {len(errors)} tokens had no pairs")
    return pools


def fetch_new_pools(
    max_age_hours:    float = 48.0,
    min_liquidity:    float = 1_000.0,
    supported_chains: list[str] | None = None,
    verbose:          bool = False,
) -> list[DexPool]:
    """
    Unified new pool discovery from profiles + boosts.
    Deduplicates by pair_address, filters by age and liquidity.
    """
    if supported_chains is None:
        supported_chains = [
            "ethereum", "bsc", "base", "solana", "polygon",
            "arbitrum", "optimism", "avalanche",
        ]

    seen:  set[str]      = set()
    pools: list[DexPool] = []

    all_pools = (
        fetch_new_profiles(max_age_hours, verbose=verbose)
        + fetch_boosted_pools(max_age_hours, verbose=verbose)
    )

    if verbose:
        print(f"  [merge] {len(all_pools)} total before dedup/filter")

    for pool in all_pools:
        if pool.pair_address in seen:
            continue
        if pool.chain_id not in supported_chains:
            if verbose:
                print(f"  [filter] skip {pool.token_symbol}: chain {pool.chain_id} not in supported list")
            continue
        if pool.liquidity_usd is not None and pool.liquidity_usd < min_liquidity:
            if verbose:
                print(f"  [filter] skip {pool.token_symbol}: liq ${pool.liquidity_usd:.0f} < ${min_liquidity:.0f}")
            continue
        seen.add(pool.pair_address)
        pools.append(pool)

    pools.sort(key=lambda p: p.pair_created_at or 0, reverse=True)

    if verbose:
        print(f"  [merge] {len(pools)} pools after dedup/filter")

    return pools
