"""
FOMO Tradability — fomo.family integration

FOMO (fomo.family) is a social trading app that routes through Jupiter
aggregator on Solana.  It does NOT curate listings — any token with
active liquidity on Jupiter/Raydium/Orca is tradable on FOMO immediately.

Therefore: FOMO tradability = Jupiter tradability.

Verification approach:
  Jupiter Quote API: GET https://quote-api.jup.ag/v6/quote
    ?inputMint=So11111111111111111111111111111111111111112  (SOL)
    &outputMint={token_address}
    &amount=1000000  (0.001 SOL in lamports)

  If a valid quote is returned → token is tradable on FOMO.
  If 400/no routes found → not currently tradable.

EVM chains (Base, BSC) are NOT on FOMO/Jupiter.  Only Solana tokens
are checked here.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Optional

_JUPITER_QUOTE = "https://quote-api.jup.ag/v6/quote"
_SOL_MINT      = "So11111111111111111111111111111111111111112"
_AMOUNT        = "1000000"   # 0.001 SOL in lamports
_TIMEOUT       = 10
_HEADERS       = {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}


def is_fomo_tradable(token_address: str, chain_id: str) -> tuple[bool, Optional[str]]:
    """
    Check if a token is tradable on FOMO (via Jupiter).

    Returns (tradable: bool, error_msg: str | None).
    Non-Solana tokens always return (False, "evm_not_supported").
    Network errors return (False, error_message) — treat as unknown, not blocked.
    """
    if chain_id != "solana":
        return False, "evm_not_supported"

    url = (
        f"{_JUPITER_QUOTE}"
        f"?inputMint={_SOL_MINT}"
        f"&outputMint={token_address}"
        f"&amount={_AMOUNT}"
        f"&slippageBps=5000"
    )
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
            # A valid quote has outAmount > 0
            out = data.get("outAmount")
            if out and int(out) > 0:
                return True, None
            return False, "no_route"
    except urllib.error.HTTPError as e:
        if e.code in (400, 404):
            return False, "no_route"
        return False, f"http_{e.code}"
    except Exception as exc:
        return False, str(exc)[:60]


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
