"""
FOMO Tradability — fomo.family integration

FOMO (fomo.family) is a social trading app that routes through Jupiter
aggregator on Solana.  It does NOT curate listings — any token with
active liquidity on Jupiter/Raydium/Orca is tradable on FOMO immediately.

Therefore: FOMO tradability = Jupiter tradability.

Verification approach (Jupiter Quote API):
  GET {endpoint}?inputMint=SOL&outputMint={token}&amount=1000000&slippageBps=5000
  A positive outAmount → token is tradable.
  HTTP 400/404 / no route → not currently tradable.

Endpoint history:
  - Legacy `quote-api.jup.ag/v6/quote` was DEPRECATED in 2025.
  - Current free endpoint is `lite-api.jup.ag/swap/v1/quote`.
  We try the current endpoint first and fall back to legacy, so a single
  dead host cannot silently turn every token into an "error".

EVM chains (Base, BSC) are NOT on FOMO/Jupiter — they return False.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Optional

# Current free endpoint first, legacy last. First one to answer wins.
_JUPITER_ENDPOINTS = [
    "https://lite-api.jup.ag/swap/v1/quote",   # current free tier (2025+)
    "https://quote-api.jup.ag/v6/quote",       # legacy fallback
]
_SOL_MINT = "So11111111111111111111111111111111111111112"
_AMOUNT   = "1000000"   # 0.001 SOL in lamports
_TIMEOUT  = 12
_HEADERS  = {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}


def _extract_out_amount(data: dict) -> Optional[int]:
    """Pull outAmount from either v1 or v6 response shapes."""
    out = data.get("outAmount")
    if out is None and isinstance(data.get("data"), dict):
        out = data["data"].get("outAmount")
    try:
        return int(out) if out is not None else None
    except (TypeError, ValueError):
        return None


def _query_endpoint(base: str, token_address: str) -> tuple[Optional[bool], Optional[str]]:
    """
    Query one Jupiter endpoint.
    Returns (tradable, error):
      (True,  None)        — valid quote, tradable
      (False, "no_route")  — endpoint answered but no route (definitive)
      (None,  error_msg)   — endpoint failed; caller should try the next one
    """
    url = (
        f"{base}"
        f"?inputMint={_SOL_MINT}"
        f"&outputMint={token_address}"
        f"&amount={_AMOUNT}"
        f"&slippageBps=5000"
    )
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
            out  = _extract_out_amount(data)
            if out and out > 0:
                return True, None
            return False, "no_route"          # definitive negative
    except urllib.error.HTTPError as e:
        if e.code in (400, 404):
            return False, "no_route"          # Jupiter says no route — definitive
        return None, f"http_{e.code}"         # transient/other — try next endpoint
    except urllib.error.URLError as e:
        return None, f"urlerror:{e.reason}"   # host unreachable — try next endpoint
    except Exception as exc:
        return None, f"{type(exc).__name__}:{str(exc)[:50]}"


def is_fomo_tradable(token_address: str, chain_id: str) -> tuple[bool, Optional[str]]:
    """
    Check if a token is tradable on FOMO (via Jupiter).

    Returns (tradable: bool, reason: str | None).
      (True,  None)            tradable
      (False, "no_route")      endpoint reachable, no liquidity route
      (False, "evm_not_supported")
      (False, "all_endpoints_failed:...")  network/endpoint problem (unknown, not blocked)
    """
    if chain_id != "solana":
        return False, "evm_not_supported"

    last_errors = []
    for base in _JUPITER_ENDPOINTS:
        tradable, info = _query_endpoint(base, token_address)
        if tradable is True:
            return True, None
        if tradable is False:                 # definitive answer (no_route)
            return False, info
        last_errors.append(f"{base.split('//')[1].split('/')[0]}={info}")

    # Every endpoint failed transiently — this is "unknown", surfaced explicitly
    return False, "all_endpoints_failed:" + "; ".join(last_errors)


def check_fomo_batch(
    tokens: list[tuple[str, str]],   # [(token_address, chain_id), ...]
    delay_s: float = 0.2,
) -> dict[str, bool]:
    """
    Check FOMO tradability for multiple tokens.
    Returns dict: {token_address: tradable_bool}
    Only Solana tokens are checked; EVM always False.
    """
    import time
    results = {}
    for token_address, chain_id in tokens:
        tradable, _ = is_fomo_tradable(token_address, chain_id)
        results[token_address] = tradable
        if chain_id == "solana":
            time.sleep(delay_s)
    return results
