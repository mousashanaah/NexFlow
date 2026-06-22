"""
Risk gate — contract safety screening.

Two free APIs:
  GoPlus Security:  contract flags (mint, blacklist, honeypot, ownership)
  Honeypot.is:      buy/sell tax simulation, honeypot detection (EVM only)

A token must pass the risk gate before appearing on the Alpha Board.
Failing the gate does not raise — it returns a RiskResult with passed=False.
"""
from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Optional


_GOPLUS_BASE   = "https://api.gopluslabs.io/api/v1"
_HONEYPOT_BASE = "https://api.honeypot.is/v2"
_TIMEOUT       = 10

# GoPlus chain IDs
_CHAIN_MAP = {
    "ethereum":  "1",
    "bsc":       "56",
    "base":      "8453",
    "polygon":   "137",
    "arbitrum":  "42161",
    "optimism":  "10",
    "avalanche": "43114",
    "solana":    "solana",
}


@dataclass
class RiskResult:
    token_address:      str
    chain_id:           str
    passed:             bool

    # GoPlus flags
    is_honeypot:        Optional[bool]  = None
    has_mint_function:  Optional[bool]  = None
    owner_not_renounced:Optional[bool]  = None
    has_blacklist:      Optional[bool]  = None
    lp_locked:          Optional[bool]  = None
    buy_tax:            Optional[float] = None
    sell_tax:           Optional[float] = None
    creator_percent:    Optional[float] = None   # % supply held by creator

    # Honeypot.is flags
    honeypot_is_flag:   Optional[bool]  = None
    simulate_buy_tax:   Optional[float] = None
    simulate_sell_tax:  Optional[float] = None

    # Risk score (0–20, higher = safer)
    risk_score:         int   = 0
    risk_flags:         list  = field(default_factory=list)
    source_errors:      list  = field(default_factory=list)

    @property
    def risk_label(self) -> str:
        if not self.passed:
            return "BLOCKED"
        if self.risk_score >= 16:
            return "CLEAN"
        if self.risk_score >= 10:
            return "CAUTION"
        return "RISKY"


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode())


def _goplus_check(token_address: str, chain_id: str) -> dict:
    goplus_chain = _CHAIN_MAP.get(chain_id, chain_id)
    url = (
        f"{_GOPLUS_BASE}/token_security/{goplus_chain}"
        f"?contract_addresses={token_address}"
    )
    data = _get(url)
    result = data.get("result", {})
    # GoPlus returns address-keyed dict
    key  = token_address.lower()
    info = result.get(key) or result.get(list(result.keys())[0]) if result else {}
    return info


def _honeypot_check(token_address: str, chain_id: str) -> dict:
    if chain_id not in _CHAIN_MAP or chain_id == "solana":
        return {}
    goplus_chain = _CHAIN_MAP[chain_id]
    url = f"{_HONEYPOT_BASE}/IsHoneypot?address={token_address}&chainID={goplus_chain}"
    return _get(url)


def _safe_bool(val) -> Optional[bool]:
    if val is None:
        return None
    return str(val) == "1"


def _safe_float(val) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def check_risk(token_address: str, chain_id: str) -> RiskResult:
    """
    Run GoPlus + Honeypot.is checks and return a RiskResult.

    Never raises. Errors are recorded in source_errors.
    """
    result = RiskResult(
        token_address = token_address,
        chain_id      = chain_id,
        passed        = False,
    )
    flags:  list[str] = []
    errors: list[str] = []
    score = 20  # start clean, deduct for risk factors

    # ── GoPlus ────────────────────────────────────────────────────────────────
    gp = {}
    try:
        gp = _goplus_check(token_address, chain_id)
    except Exception as exc:
        errors.append(f"goplus_error: {exc}")

    if gp:
        is_hp = _safe_bool(gp.get("is_honeypot"))
        if is_hp:
            result.is_honeypot = True
            flags.append("HONEYPOT")
            score = 0  # instant disqualification

        has_mint = _safe_bool(gp.get("is_mintable"))
        result.has_mint_function = has_mint
        if has_mint:
            flags.append("MINTABLE")
            score -= 4

        not_renounced = not _safe_bool(gp.get("owner_address") == "0x0000000000000000000000000000000000000000")
        owner_addr = gp.get("owner_address", "")
        is_renounced = (
            owner_addr in ("0x0000000000000000000000000000000000000000", "", "0x")
            or gp.get("owner_address") is None
        )
        result.owner_not_renounced = not is_renounced
        if not is_renounced is False:  # owner still holds
            flags.append("OWNER_NOT_RENOUNCED")
            score -= 3

        has_bl = _safe_bool(gp.get("is_blacklisted"))
        result.has_blacklist = has_bl
        if has_bl:
            flags.append("HAS_BLACKLIST")
            score -= 2

        buy_tax  = _safe_float(gp.get("buy_tax"))
        sell_tax = _safe_float(gp.get("sell_tax"))
        result.buy_tax  = buy_tax
        result.sell_tax = sell_tax
        if sell_tax is not None and sell_tax > 0.10:
            flags.append(f"HIGH_SELL_TAX:{sell_tax:.0%}")
            score -= 4
        elif sell_tax is not None and sell_tax > 0.05:
            flags.append(f"SELL_TAX:{sell_tax:.0%}")
            score -= 2

        creator_pct = _safe_float(gp.get("creator_percent"))
        result.creator_percent = creator_pct
        if creator_pct is not None and creator_pct > 0.10:
            flags.append(f"CREATOR_HOLDS:{creator_pct:.0%}")
            score -= 3

    # ── Honeypot.is ───────────────────────────────────────────────────────────
    hp = {}
    if chain_id != "solana":
        try:
            time.sleep(0.2)
            hp = _honeypot_check(token_address, chain_id)
        except Exception as exc:
            errors.append(f"honeypot_error: {exc}")

    if hp:
        hp_flag = hp.get("isHoneypot") or hp.get("honeypotResult", {}).get("isHoneypot")
        result.honeypot_is_flag = bool(hp_flag)
        if hp_flag:
            flags.append("HONEYPOT_IS")
            score = 0

        sim_buy  = _safe_float(hp.get("simulationResult", {}).get("buyTax"))
        sim_sell = _safe_float(hp.get("simulationResult", {}).get("sellTax"))
        result.simulate_buy_tax  = sim_buy
        result.simulate_sell_tax = sim_sell
        if sim_sell is not None and sim_sell > 10 and "HIGH_SELL_TAX" not in str(flags):
            flags.append(f"SIM_SELL_TAX:{sim_sell:.1f}%")
            score -= 3

    score = max(0, score)
    result.risk_score  = score
    result.risk_flags  = flags
    result.source_errors = errors

    # Pass if: not honeypot, risk score >= 8, no instant-kill flags
    instant_kill = {"HONEYPOT", "HONEYPOT_IS"}
    result.passed = (
        score >= 8
        and not instant_kill.intersection(set(flags))
        and is_hp is not True
    )

    return result


def batch_check_risk(
    tokens: list[tuple[str, str]],   # [(token_address, chain_id), ...]
    delay_s: float = 0.5,
) -> dict[str, RiskResult]:
    """Check multiple tokens. Returns {token_address: RiskResult}."""
    results = {}
    for token_address, chain_id in tokens:
        results[token_address] = check_risk(token_address, chain_id)
        time.sleep(delay_s)
    return results
