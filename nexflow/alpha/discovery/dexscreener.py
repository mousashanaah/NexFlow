"""
DexScreener client — new pool discovery.

Two sources:
  - /token-profiles/latest/v1   : recently created token profiles
  - /token-boosts/latest/v1     : recently boosted/promoted tokens

Both are free, no auth required.
Rate limit: ~1 req/sec safe; we sleep between calls.
"""
from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


_BASE = "https://api.dexscreener.com"
_TIMEOUT = 10


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
    pair_created_at: Optional[int]   # unix ms
    age_hours:       Optional[float]
    url:             str
    source:          str             # "profiles" or "boosts"

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
    url = f"{_BASE}{path}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode())


def _parse_pair(pair: dict, source: str) -> Optional[DexPool]:
    """Extract a DexPool from a DexScreener pair object."""
    try:
        base = pair.get("baseToken", {})
        liq  = pair.get("liquidity", {})
        vol  = pair.get("volume", {})

        created_at = pair.get("pairCreatedAt")
        age_hours  = None
        if created_at:
            now_ms    = time.time() * 1000
            age_hours = (now_ms - created_at) / 3_600_000

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


def fetch_new_profiles(max_age_hours: float = 48.0) -> list[DexPool]:
    """
    Fetch recently created token profiles from DexScreener.
    Returns pools younger than max_age_hours.
    """
    pools: list[DexPool] = []
    try:
        data = _get("/token-profiles/latest/v1")
        items = data if isinstance(data, list) else data.get("pairs", [])
        for item in items:
            # token-profiles returns profile objects, not pair objects
            # we need to fetch the actual pair data for each
            token_addr = item.get("tokenAddress", "")
            chain_id   = item.get("chainId", "")
            if not token_addr or not chain_id:
                continue
            try:
                pair_data = _get(f"/latest/dex/tokens/{token_addr}")
                pairs     = pair_data.get("pairs") or []
                # take the pair with highest liquidity
                pairs = [p for p in pairs if p.get("chainId") == chain_id]
                if not pairs:
                    continue
                best = max(pairs, key=lambda p: p.get("liquidity", {}).get("usd") or 0)
                pool = _parse_pair(best, "profiles")
                if pool and (pool.age_hours is None or pool.age_hours <= max_age_hours):
                    pools.append(pool)
                time.sleep(0.3)
            except Exception:
                continue
    except Exception:
        pass
    return pools


def fetch_boosted_pools(max_age_hours: float = 48.0) -> list[DexPool]:
    """
    Fetch recently boosted tokens from DexScreener.
    """
    pools: list[DexPool] = []
    try:
        data  = _get("/token-boosts/latest/v1")
        items = data if isinstance(data, list) else []
        for item in items:
            token_addr = item.get("tokenAddress", "")
            chain_id   = item.get("chainId", "")
            if not token_addr or not chain_id:
                continue
            try:
                pair_data = _get(f"/latest/dex/tokens/{token_addr}")
                pairs     = pair_data.get("pairs") or []
                pairs     = [p for p in pairs if p.get("chainId") == chain_id]
                if not pairs:
                    continue
                best = max(pairs, key=lambda p: p.get("liquidity", {}).get("usd") or 0)
                pool = _parse_pair(best, "boosts")
                if pool and (pool.age_hours is None or pool.age_hours <= max_age_hours):
                    pools.append(pool)
                time.sleep(0.3)
            except Exception:
                continue
    except Exception:
        pass
    return pools


def fetch_new_pools(
    max_age_hours:   float = 48.0,
    min_liquidity:   float = 5_000.0,
    supported_chains: list[str] | None = None,
) -> list[DexPool]:
    """
    Unified new pool discovery.

    Fetches from both profiles and boosts endpoints, deduplicates by
    pair_address, filters by age and minimum liquidity.
    """
    if supported_chains is None:
        supported_chains = ["ethereum", "bsc", "base", "solana", "polygon",
                            "arbitrum", "optimism", "avalanche"]

    seen:  set[str]    = set()
    pools: list[DexPool] = []

    for pool in fetch_new_profiles(max_age_hours) + fetch_boosted_pools(max_age_hours):
        if pool.pair_address in seen:
            continue
        if pool.chain_id not in supported_chains:
            continue
        if pool.liquidity_usd is not None and pool.liquidity_usd < min_liquidity:
            continue
        seen.add(pool.pair_address)
        pools.append(pool)

    # Sort: newest first
    pools.sort(key=lambda p: p.pair_created_at or 0, reverse=True)
    return pools
