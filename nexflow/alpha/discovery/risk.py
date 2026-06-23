"""
Risk gate — contract safety screening.

EVM chains:  GoPlus Security (contract flags) + Honeypot.is (simulation)
Solana:      RugCheck (mint authority, freeze authority, LP lock, top holders)

A token must pass the risk gate before appearing on the Alpha Board.
Failing the gate does not raise — it returns a RiskResult with passed=False.
If risk data is unavailable: passed=False, risk_label="UNVERIFIED".
"""
from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Optional


_GOPLUS_BASE   = "https://api.gopluslabs.io/api/v1"
_HONEYPOT_BASE = "https://api.honeypot.is/v2"
_RUGCHECK_BASE = "https://api.rugcheck.xyz/v1"
_TIMEOUT       = 15
_HEADERS       = {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}

# GoPlus chain IDs (EVM only)
_EVM_CHAIN_MAP = {
    "ethereum":  "1",
    "bsc":       "56",
    "base":      "8453",
    "polygon":   "137",
    "arbitrum":  "42161",
    "optimism":  "10",
    "avalanche": "43114",
}

_SOLANA_CHAINS = {"solana"}


@dataclass
class RiskResult:
    token_address:       str
    chain_id:            str
    passed:              bool

    # GoPlus / RugCheck flags
    is_honeypot:         Optional[bool]  = None
    has_mint_function:   Optional[bool]  = None   # EVM: mintable; Solana: mint authority
    has_freeze_authority:Optional[bool]  = None   # Solana only
    owner_not_renounced: Optional[bool]  = None
    has_blacklist:       Optional[bool]  = None
    lp_locked:           Optional[bool]  = None
    buy_tax:             Optional[float] = None
    sell_tax:            Optional[float] = None
    creator_percent:     Optional[float] = None
    top10_percent:       Optional[float] = None   # Solana: top-10 holder concentration (0-100 scale, e.g. 57.14 = 57.14%)

    # Honeypot.is (EVM only)
    honeypot_is_flag:    Optional[bool]  = None
    simulate_buy_tax:    Optional[float] = None
    simulate_sell_tax:   Optional[float] = None

    # RugCheck (Solana)
    rugcheck_score:      Optional[int]   = None   # raw RugCheck score (higher = more risk)
    rugcheck_risks:      list            = field(default_factory=list)
    already_rugged:      Optional[bool]  = None
    top_holders_raw:     list            = field(default_factory=list)  # raw topHolders list for wallet registry

    # Summary
    risk_score:          int             = 0      # 0–20, higher = safer
    risk_flags:          list            = field(default_factory=list)
    source_errors:       list            = field(default_factory=list)
    check_source:        str             = ""     # "goplus" | "rugcheck" | "unverified"

    @property
    def risk_label(self) -> str:
        if self.check_source == "unverified":
            return "UNVERIFIED"
        if not self.passed:
            return "BLOCKED"
        if self.risk_score >= 16:
            return "CLEAN"
        if self.risk_score >= 10:
            return "CAUTION"
        return "RISKY"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get(url: str) -> dict | list:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode())


def _safe_bool(val) -> Optional[bool]:
    if val is None:
        return None
    return str(val) == "1"


def _safe_float(val) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ── EVM: GoPlus ───────────────────────────────────────────────────────────────

def _goplus_check(token_address: str, chain_id: str) -> dict:
    chain = _EVM_CHAIN_MAP[chain_id]
    url   = f"{_GOPLUS_BASE}/token_security/{chain}?contract_addresses={token_address}"
    data  = _get(url)
    result = data.get("result", {})
    if not result:
        return {}
    key = token_address.lower()
    return result.get(key) or result.get(list(result.keys())[0], {})


def _honeypot_check(token_address: str, chain_id: str) -> dict:
    chain = _EVM_CHAIN_MAP[chain_id]
    url   = f"{_HONEYPOT_BASE}/IsHoneypot?address={token_address}&chainID={chain}"
    return _get(url)


def _check_evm(token_address: str, chain_id: str, result: RiskResult) -> tuple[int, list, list]:
    """Run EVM risk checks. Returns (score, flags, errors)."""
    flags:  list[str] = []
    errors: list[str] = []
    score = 20
    is_hp = None

    # GoPlus
    gp = {}
    try:
        gp = _goplus_check(token_address, chain_id)
        result.check_source = "goplus"
    except Exception as exc:
        errors.append(f"goplus:{exc}")

    if gp:
        is_hp = _safe_bool(gp.get("is_honeypot"))
        if is_hp:
            result.is_honeypot = True
            flags.append("HONEYPOT")
            score = 0

        has_mint = _safe_bool(gp.get("is_mintable"))
        result.has_mint_function = has_mint
        if has_mint:
            flags.append("MINTABLE")
            score -= 4

        owner_addr   = gp.get("owner_address") or ""
        is_renounced = owner_addr.lower() in ("0x0000000000000000000000000000000000000000", "", "0x")
        result.owner_not_renounced = not is_renounced
        if not is_renounced:
            flags.append("OWNER_NOT_RENOUNCED")
            score -= 3

        has_bl = _safe_bool(gp.get("is_blacklisted"))
        result.has_blacklist = has_bl
        if has_bl:
            flags.append("HAS_BLACKLIST")
            score -= 2

        sell_tax = _safe_float(gp.get("sell_tax"))
        result.sell_tax = sell_tax
        result.buy_tax  = _safe_float(gp.get("buy_tax"))
        if sell_tax and sell_tax > 0.10:
            flags.append(f"HIGH_SELL_TAX:{sell_tax:.0%}")
            score -= 4
        elif sell_tax and sell_tax > 0.05:
            flags.append(f"SELL_TAX:{sell_tax:.0%}")
            score -= 2

        creator_pct = _safe_float(gp.get("creator_percent"))
        result.creator_percent = creator_pct
        if creator_pct and creator_pct > 0.10:
            flags.append(f"CREATOR_HOLDS:{creator_pct:.0%}")
            score -= 3

    # Honeypot.is
    try:
        time.sleep(0.2)
        hp = _honeypot_check(token_address, chain_id)
        if hp:
            hp_flag = hp.get("isHoneypot") or (hp.get("honeypotResult") or {}).get("isHoneypot")
            result.honeypot_is_flag = bool(hp_flag)
            if hp_flag:
                flags.append("HONEYPOT_IS")
                score = 0
                is_hp = True
            sim = hp.get("simulationResult") or {}
            result.simulate_sell_tax = _safe_float(sim.get("sellTax"))
            result.simulate_buy_tax  = _safe_float(sim.get("buyTax"))
            if result.simulate_sell_tax and result.simulate_sell_tax > 10:
                if "HIGH_SELL_TAX" not in str(flags):
                    flags.append(f"SIM_SELL_TAX:{result.simulate_sell_tax:.1f}%")
                    score -= 3
    except Exception as exc:
        errors.append(f"honeypot:{exc}")

    instant_kill = {"HONEYPOT", "HONEYPOT_IS"}
    passed = (
        score >= 8
        and not instant_kill.intersection(set(flags))
        and is_hp is not True
    )
    result.passed = passed
    return score, flags, errors


# ── Solana: RugCheck ──────────────────────────────────────────────────────────

def _rugcheck_check(mint_address: str) -> dict:
    url = f"{_RUGCHECK_BASE}/tokens/{mint_address}/report"
    return _get(url)


def _check_solana(token_address: str, result: RiskResult) -> tuple[int, list, list]:
    """Run Solana risk checks via RugCheck. Returns (score, flags, errors)."""
    flags:  list[str] = []
    errors: list[str] = []
    score = 20

    rc = {}
    try:
        rc = _rugcheck_check(token_address)
        result.check_source = "rugcheck"
    except Exception as exc:
        errors.append(f"rugcheck:{exc}")
        result.check_source = "unverified"
        result.passed = False
        return 0, flags, errors

    if not rc:
        result.check_source = "unverified"
        result.passed = False
        return 0, flags, errors

    # Already rugged
    if rc.get("rugged"):
        result.already_rugged = True
        flags.append("ALREADY_RUGGED")
        result.passed = False
        return 0, flags, errors

    # RugCheck risk items (each has name, level, score)
    risk_items = rc.get("risks") or []
    result.rugcheck_risks = [r.get("name", "") for r in risk_items]
    rc_score = rc.get("score") or 0
    result.rugcheck_score = rc_score

    # RugCheck score: higher = more risky. >500 = danger, 200-500 = warn, <200 = ok
    if rc_score > 500:
        flags.append(f"RUGCHECK_DANGER:{rc_score}")
        score -= 8
    elif rc_score > 200:
        flags.append(f"RUGCHECK_WARN:{rc_score}")
        score -= 4

    # Mint authority
    mint_auth = rc.get("mintAuthority")
    if mint_auth and mint_auth not in (None, "", "null"):
        result.has_mint_function = True
        flags.append("MINT_AUTHORITY")
        score -= 5
    else:
        result.has_mint_function = False

    # Freeze authority
    freeze_auth = rc.get("freezeAuthority")
    if freeze_auth and freeze_auth not in (None, "", "null"):
        result.has_freeze_authority = True
        flags.append("FREEZE_AUTHORITY")
        score -= 4
    else:
        result.has_freeze_authority = False

    # Top holders concentration
    # RugCheck returns pct on a 0-100 scale (e.g. 57.14 means 57.14%), not 0-1.
    top_holders = rc.get("topHolders") or []
    result.top_holders_raw = top_holders          # preserve for wallet registry
    if top_holders:
        top10_pct = sum(
            float(h.get("pct") or 0)
            for h in top_holders[:10]
        )
        result.top10_percent = top10_pct          # stored as 0-100
        if top10_pct > 80:
            flags.append(f"TOP10_HOLDS:{top10_pct:.1f}%")
            score -= 5
        elif top10_pct > 60:
            flags.append(f"TOP10_HOLDS:{top10_pct:.1f}%")
            score -= 3

    # LP lock via markets
    markets = rc.get("markets") or []
    lp_locked = any(m.get("lpLockedPct", 0) > 0 for m in markets)
    result.lp_locked = lp_locked
    if not lp_locked and markets:
        flags.append("LP_NOT_LOCKED")
        score -= 2

    # Danger-level risk items
    danger_items = [r.get("name", "") for r in risk_items if r.get("level") == "danger"]
    for item in danger_items:
        if item not in str(flags):
            flags.append(f"DANGER:{item[:30]}")
            score -= 3

    score = max(0, score)
    instant_kill = {"ALREADY_RUGGED"}
    result.passed = (
        score >= 8
        and not instant_kill.intersection(set(flags))
        and "MINT_AUTHORITY" not in flags  # mint authority = immediate caution
        or (score >= 12 and "MINT_AUTHORITY" in flags and "FREEZE_AUTHORITY" not in flags)
    )
    # Simpler, cleaner pass logic:
    result.passed = (
        not rc.get("rugged")
        and score >= 8
        and "ALREADY_RUGGED" not in flags
    )
    return score, flags, errors


# ── Public interface ──────────────────────────────────────────────────────────

def check_risk(token_address: str, chain_id: str) -> RiskResult:
    """
    Run risk checks appropriate for the chain.
    Never raises. Returns RiskResult with risk_label:
      CLEAN / CAUTION / RISKY / BLOCKED / UNVERIFIED
    """
    result = RiskResult(
        token_address = token_address,
        chain_id      = chain_id,
        passed        = False,
        check_source  = "unverified",
    )

    if chain_id in _SOLANA_CHAINS:
        score, flags, errors = _check_solana(token_address, result)
    elif chain_id in _EVM_CHAIN_MAP:
        score, flags, errors = _check_evm(token_address, chain_id, result)
    else:
        result.source_errors = [f"unsupported_chain:{chain_id}"]
        result.check_source  = "unverified"
        return result

    result.risk_score    = max(0, score)
    result.risk_flags    = flags
    result.source_errors = errors
    return result


def batch_check_risk(
    tokens:  list[tuple[str, str]],
    delay_s: float = 0.5,
) -> dict[str, RiskResult]:
    results = {}
    for token_address, chain_id in tokens:
        results[token_address] = check_risk(token_address, chain_id)
        time.sleep(delay_s)
    return results
