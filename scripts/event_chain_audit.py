#!/usr/bin/env python3
"""Event chain auditor.

For every trade, reconstructs the complete lifecycle from raw journal events
and verifies mathematical consistency at every stage.

Stage checks
  SIGNAL   — geometry, ATR multiples, stop recalculation cross-check
  FILL     — slippage direction/magnitude, entry fee formula
  PARTIAL_TP — fill vs expected TP price, fraction size, net PnL, maker fee
  STOP_HIT — adverse-slippage direction, taker fee, implied remaining size

Sequence checks
  A) All recorded TPs logged before STOP_HIT (timestamp order)
  B) If stop and TP both triggered on same bar, flag intra-bar ambiguity
  C) Remaining size after each PARTIAL_TP matches TP_FRACTIONS = [0.50,0.25,0.25]
  D) Remaining size implied by stop fee matches cumulative TP reductions
  E) Reconstructed PnL (from raw arithmetic) equals journal PnL

Bug location assessment
  strategy        — stop/TP geometry violations in the SIGNAL itself
  execution_engine — fill prices deviate from simulation formulas
  journal         — event PnL/fee fields inconsistent with position arithmetic
  analysis_layer  — reconstructed PnL diverges from journal (our math is wrong)
  ok              — no violations

Usage
  python scripts/event_chain_audit.py
  python scripts/event_chain_audit.py --journal-dir logs/paper --verbose
  python scripts/event_chain_audit.py --csv event_chain_audit.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Strategy constants — must match live_signal_router.py / paper_execution.py
# ---------------------------------------------------------------------------
ATR_STOP_MULT   = 1.5
ATR_TP_MULTS    = [1.0, 2.0, 3.0]          # TP1, TP2, TP3
TP_FRACTIONS    = [0.50, 0.25, 0.25]        # fraction of total size per TP
# Entry slippage:  ATR × (0.05 + 0.5 × 0.005) = ATR × 0.0525
ENTRY_SLIP_MULT = 0.05 + 0.5 * 0.005        # 0.0525
# Stop slippage:   stop_dist × 0.05 × 0.5
STOP_SLIP_FRAC  = 0.05 * 0.5                # 0.025
TAKER_FEE       = 0.0006
MAKER_FEE       = 0.0002

# Comparison tolerances
TOL_PRICE = 0.01      # absolute — handles floating-point accumulation
TOL_FEE   = 0.01      # quote currency
TOL_PNL   = 0.05      # quote currency — allow small rounding chains
TOL_SIZE  = 1e-6      # contract quantity


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Check:
    name: str
    passed: bool
    expected: str = ""
    actual: str = ""
    note: str = ""
    layer: str = "ok"   # strategy | execution_engine | journal | analysis | ok


@dataclass
class TradeChain:
    trade_id: int
    symbol: str
    direction: str        # "long" | "short"
    signal: dict
    fill: dict
    partials: list[dict]  # PARTIAL_TP events sorted by tp_idx
    stop: Optional[dict]  # STOP_HIT event or None
    is_force_close: bool


@dataclass
class AuditResult:
    chain: TradeChain
    checks: list[Check]   = field(default_factory=list)
    missing_events: list[str] = field(default_factory=list)
    incorrect_events: list[str] = field(default_factory=list)
    reconstructed_pnl: float = 0.0
    journal_pnl: float = 0.0
    integrity_score: int = 0
    layers_hit: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Journal loading
# ---------------------------------------------------------------------------

def load_journals(journal_dir: Path) -> list[dict]:
    events: list[dict] = []
    paths = sorted(journal_dir.rglob("*.jsonl"))
    if not paths:
        print(f"[WARN] No .jsonl files found in {journal_dir}", file=sys.stderr)
        return events
    for p in paths:
        with p.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    events.sort(key=lambda e: e.get("ts_epoch", 0))
    return events


def build_trade_chains(events: list[dict]) -> list[TradeChain]:
    """Group events into per-symbol trade chains.

    A new chain starts on every FILL event for a symbol.
    A chain closes on STOP_HIT, FORCE_CLOSE, or the last PARTIAL_TP that
    exhausts the position (all 3 TPs hit → remaining = 0).
    """
    # Track open chains: symbol → (signal, fill, partials)
    open_sigs: dict[str, dict] = {}
    open_fills: dict[str, dict] = {}
    open_partials: dict[str, list] = {}

    chains: list[TradeChain] = []
    trade_id = 0

    for ev in events:
        sym = ev.get("symbol")
        if sym is None:
            continue
        etype = ev.get("event", "")

        if etype == "SIGNAL":
            # If there's already an open signal for this symbol, overwrite
            # (shouldn't happen but be robust)
            open_sigs[sym] = ev

        elif etype == "FILL":
            open_fills[sym] = ev
            open_partials[sym] = []

        elif etype == "PARTIAL_TP":
            if sym in open_fills:
                open_partials.setdefault(sym, []).append(ev)
                # Check if all 3 TPs hit → chain closes without STOP_HIT
                total_size = open_fills[sym].get("size", 0.0)
                closed = sum(p.get("size", 0.0) for p in open_partials[sym])
                if total_size > 0 and abs(closed - total_size) < TOL_SIZE:
                    sig = open_sigs.pop(sym, {})
                    fill = open_fills.pop(sym)
                    partials = sorted(open_partials.pop(sym, []),
                                      key=lambda p: p.get("tp_idx", 0))
                    chains.append(TradeChain(
                        trade_id=trade_id,
                        symbol=sym,
                        direction=fill.get("direction", "?"),
                        signal=sig,
                        fill=fill,
                        partials=partials,
                        stop=None,
                        is_force_close=False,
                    ))
                    trade_id += 1

        elif etype in ("STOP_HIT", "FORCE_CLOSE"):
            if sym in open_fills:
                sig = open_sigs.pop(sym, {})
                fill = open_fills.pop(sym)
                partials = sorted(open_partials.pop(sym, []),
                                  key=lambda p: p.get("tp_idx", 0))
                chains.append(TradeChain(
                    trade_id=trade_id,
                    symbol=sym,
                    direction=fill.get("direction", "?"),
                    signal=sig,
                    fill=fill,
                    partials=partials,
                    stop=ev,
                    is_force_close=(etype == "FORCE_CLOSE"),
                ))
                trade_id += 1
            else:
                # Stop without a matching fill — orphaned event
                pass

    return chains


# ---------------------------------------------------------------------------
# Individual stage checks
# ---------------------------------------------------------------------------

def _f(v: float, dp: int = 4) -> str:
    return f"{v:.{dp}f}"


def check_signal(chain: TradeChain) -> list[Check]:
    checks: list[Check] = []
    sig = chain.signal
    if not sig:
        checks.append(Check("signal.present", False, "SIGNAL event",
                             "missing", layer="journal"))
        return checks

    entry   = sig.get("entry_price", float("nan"))
    stop    = sig.get("stop_price",  float("nan"))
    atr     = sig.get("atr",         float("nan"))
    tps     = sig.get("tp_prices",   [])
    is_long = chain.direction.lower() == "long"

    # --- Geometry: stop on correct side of entry ---
    if is_long:
        geom_ok = stop < entry
        checks.append(Check(
            "signal.stop_geometry",
            geom_ok,
            f"stop({_f(stop)}) < entry({_f(entry)})",
            "✓" if geom_ok else f"FAIL: stop={_f(stop)} >= entry={_f(entry)}",
            layer="ok" if geom_ok else "strategy",
        ))
    else:
        geom_ok = stop > entry
        checks.append(Check(
            "signal.stop_geometry",
            geom_ok,
            f"stop({_f(stop)}) > entry({_f(entry)})",
            "✓" if geom_ok else f"FAIL: stop={_f(stop)} <= entry={_f(entry)}",
            layer="ok" if geom_ok else "strategy",
        ))

    # --- ATR × 1.5 = stop distance ---
    if not math.isnan(atr) and atr > 0:
        expected_dist = atr * ATR_STOP_MULT
        actual_dist   = abs(entry - stop)
        ok = abs(actual_dist - expected_dist) < TOL_PRICE
        checks.append(Check(
            "signal.stop_atr_dist",
            ok,
            f"{_f(expected_dist)} (ATR×{ATR_STOP_MULT})",
            _f(actual_dist),
            note=f"diff={_f(actual_dist - expected_dist)}",
            layer="ok" if ok else "strategy",
        ))

    # --- TP order (ascending for LONG, descending for SHORT) ---
    if len(tps) >= 2:
        for i in range(len(tps) - 1):
            if is_long:
                ok = tps[i] < tps[i + 1]
            else:
                ok = tps[i] > tps[i + 1]
            checks.append(Check(
                f"signal.tp{i+1}_tp{i+2}_order",
                ok,
                f"tp{i+1} {'<' if is_long else '>'} tp{i+2}",
                f"tp{i+1}={_f(tps[i])} tp{i+2}={_f(tps[i+1])}",
                layer="ok" if ok else "strategy",
            ))

    # --- TP distances match ATR multiples ---
    if not math.isnan(atr) and atr > 0:
        for i, (tp, mult) in enumerate(zip(tps, ATR_TP_MULTS)):
            expected = entry + (1 if is_long else -1) * atr * mult
            ok = abs(tp - expected) < TOL_PRICE
            checks.append(Check(
                f"signal.tp{i+1}_atr_dist",
                ok,
                f"{_f(expected)} (entry ± ATR×{mult})",
                _f(tp),
                note=f"diff={_f(tp - expected)}",
                layer="ok" if ok else "strategy",
            ))

    # --- Recalculate stop from signal entry (should match exactly) ---
    if not math.isnan(atr) and atr > 0:
        recomputed = entry + (-1 if is_long else 1) * atr * ATR_STOP_MULT
        ok = abs(recomputed - stop) < TOL_PRICE
        checks.append(Check(
            "signal.stop_recompute_match",
            ok,
            _f(recomputed),
            _f(stop),
            note=f"diff={_f(recomputed - stop)}",
            layer="ok" if ok else "strategy",
        ))

    return checks


def check_fill(chain: TradeChain) -> list[Check]:
    checks: list[Check] = []
    sig  = chain.signal
    fill = chain.fill
    if not fill:
        checks.append(Check("fill.present", False, "FILL event",
                             "missing", layer="journal"))
        return checks

    is_long   = chain.direction.lower() == "long"
    atr       = sig.get("atr", float("nan")) if sig else float("nan")
    sig_entry = sig.get("entry_price", float("nan")) if sig else float("nan")
    fill_px   = fill.get("fill_price", float("nan"))
    size      = fill.get("size", 0.0)
    fee       = fill.get("fee", 0.0)
    slippage  = fill.get("slippage", float("nan"))

    # --- Slippage direction ---
    if not math.isnan(sig_entry):
        if is_long:
            slip_dir_ok = fill_px >= sig_entry
        else:
            slip_dir_ok = fill_px <= sig_entry
        checks.append(Check(
            "fill.slippage_direction",
            slip_dir_ok,
            f"fill {'>' if is_long else '<'} signal_entry={_f(sig_entry)}",
            _f(fill_px),
            layer="ok" if slip_dir_ok else "execution_engine",
        ))

    # --- Slippage magnitude ---
    if not math.isnan(atr) and atr > 0 and not math.isnan(sig_entry):
        expected_slip = atr * ENTRY_SLIP_MULT
        actual_slip   = abs(fill_px - sig_entry)
        ok = abs(actual_slip - expected_slip) < TOL_PRICE
        checks.append(Check(
            "fill.slippage_magnitude",
            ok,
            f"{_f(expected_slip)} (ATR×{ENTRY_SLIP_MULT})",
            _f(actual_slip),
            note=f"diff={_f(actual_slip - expected_slip)}",
            layer="ok" if ok else "execution_engine",
        ))

    # --- Entry fee formula: fill_price × size × TAKER ---
    if size > 0 and not math.isnan(fill_px):
        expected_fee = fill_px * size * TAKER_FEE
        ok = abs(fee - expected_fee) < TOL_FEE
        checks.append(Check(
            "fill.fee_formula",
            ok,
            f"{_f(expected_fee)} (fill×size×{TAKER_FEE})",
            _f(fee),
            note=f"diff={_f(fee - expected_fee)}",
            layer="ok" if ok else "journal",
        ))

    # --- Slippage field consistency (slippage = |fill - sig_entry|) ---
    if not math.isnan(sig_entry) and not math.isnan(slippage):
        expected_slip_field = abs(fill_px - sig_entry)
        ok = abs(slippage - expected_slip_field) < TOL_PRICE
        checks.append(Check(
            "fill.slippage_field",
            ok,
            _f(expected_slip_field),
            _f(slippage),
            note=f"diff={_f(slippage - expected_slip_field)}",
            layer="ok" if ok else "journal",
        ))

    return checks


def check_partial_tps(chain: TradeChain) -> list[Check]:
    checks: list[Check] = []
    sig     = chain.signal
    fill    = chain.fill
    if not fill:
        return checks

    total_size = fill.get("size", 0.0)
    fill_px    = fill.get("fill_price", float("nan"))
    is_long    = chain.direction.lower() == "long"
    atr        = sig.get("atr", float("nan")) if sig else float("nan")
    sig_tps    = sig.get("tp_prices", []) if sig else []
    sig_entry  = sig.get("entry_price", float("nan")) if sig else float("nan")

    # Expected remaining after each consecutive TP
    # After k TPs: remaining = total × (1 − sum(TP_FRACTIONS[:k]))
    cumulative_frac = 0.0
    remaining_after = {}
    for i, f in enumerate(TP_FRACTIONS):
        cumulative_frac += f
        remaining_after[i] = round(total_size * (1.0 - cumulative_frac), 8)

    for pt in chain.partials:
        idx      = pt.get("tp_idx", -1)
        pt_fill  = pt.get("fill_price", float("nan"))
        pt_size  = pt.get("size", 0.0)
        pt_pnl   = pt.get("pnl", float("nan"))   # net-of-fee
        pt_fee   = pt.get("fee", float("nan"))

        prefix = f"partial_tp{idx+1}"

        # --- Fill price matches expected TP level ---
        if idx < len(sig_tps) and not math.isnan(sig_tps[idx]):
            exp_price = sig_tps[idx]
            ok = abs(pt_fill - exp_price) < TOL_PRICE
            checks.append(Check(
                f"{prefix}.fill_price",
                ok,
                _f(exp_price),
                _f(pt_fill),
                note=f"diff={_f(pt_fill - exp_price)}",
                layer="ok" if ok else "execution_engine",
            ))

        # --- TP fill price matches ATR formula ---
        if not math.isnan(atr) and atr > 0 and idx < len(ATR_TP_MULTS):
            exp_formula = sig_entry + (1 if is_long else -1) * atr * ATR_TP_MULTS[idx]
            ok = abs(pt_fill - exp_formula) < TOL_PRICE
            checks.append(Check(
                f"{prefix}.fill_vs_atr_formula",
                ok,
                f"{_f(exp_formula)} (entry±ATR×{ATR_TP_MULTS[idx]})",
                _f(pt_fill),
                note=f"diff={_f(pt_fill - exp_formula)}",
                layer="ok" if ok else "strategy",
            ))

        # --- Size fraction ---
        if idx < len(TP_FRACTIONS) and total_size > 0:
            exp_size = total_size * TP_FRACTIONS[idx]
            ok = abs(pt_size - exp_size) < TOL_SIZE
            checks.append(Check(
                f"{prefix}.size_fraction",
                ok,
                f"{_f(exp_size)} ({TP_FRACTIONS[idx]*100:.0f}% of {_f(total_size)})",
                _f(pt_size),
                note=f"diff={_f(pt_size - exp_size)}",
                layer="ok" if ok else "execution_engine",
            ))

        # --- Fee: fill × size × MAKER ---
        if not math.isnan(pt_fill) and pt_size > 0 and not math.isnan(pt_fee):
            exp_fee = pt_fill * pt_size * MAKER_FEE
            ok = abs(pt_fee - exp_fee) < TOL_FEE
            checks.append(Check(
                f"{prefix}.fee_formula",
                ok,
                f"{_f(exp_fee)} (fill×size×{MAKER_FEE})",
                _f(pt_fee),
                note=f"diff={_f(pt_fee - exp_fee)}",
                layer="ok" if ok else "journal",
            ))

        # --- Net PnL: (fill − fill_price) × size × sign − fee ---
        # pos.entry_price = fill_price (fill price, not signal entry)
        if not math.isnan(fill_px) and not math.isnan(pt_fill) and pt_size > 0 and not math.isnan(pt_fee):
            sign = 1.0 if is_long else -1.0
            exp_pnl = (pt_fill - fill_px) * pt_size * sign - pt_fee
            ok = abs(pt_pnl - exp_pnl) < TOL_PNL
            checks.append(Check(
                f"{prefix}.net_pnl",
                ok,
                _f(exp_pnl),
                _f(pt_pnl),
                note=f"diff={_f(pt_pnl - exp_pnl)}",
                layer="ok" if ok else "journal",
            ))

    return checks


def check_stop(chain: TradeChain) -> list[Check]:
    checks: list[Check] = []
    fill = chain.fill
    stop = chain.stop
    if stop is None or chain.is_force_close:
        return checks  # all-TP-exit or force close — different rules
    if not fill:
        return checks

    is_long    = chain.direction.lower() == "long"
    fill_px    = fill.get("fill_price", float("nan"))
    total_size = fill.get("size", 0.0)
    entry_fee  = fill.get("fee", 0.0)

    stop_fill  = stop.get("fill_price", float("nan"))
    stop_fee   = stop.get("fee", float("nan"))
    stop_pnl   = stop.get("pnl", float("nan"))   # total trade net PnL

    # Compute stop_price used at exit
    # After TP1: stop moves to break-even = fill_price
    tp1_hit = any(p.get("tp_idx", -1) == 0 for p in chain.partials)
    if tp1_hit:
        effective_stop = fill_px   # moved to BE
    else:
        sig = chain.signal
        effective_stop = sig.get("stop_price", float("nan")) if sig else float("nan")

    # --- Stop fill direction ---
    if not math.isnan(effective_stop):
        if is_long:
            ok = stop_fill <= effective_stop
        else:
            ok = stop_fill >= effective_stop
        checks.append(Check(
            "stop.fill_direction",
            ok,
            f"stop_fill {'<=' if is_long else '>='} stop_level({_f(effective_stop)})",
            _f(stop_fill),
            layer="ok" if ok else "execution_engine",
        ))

    # --- Stop slippage magnitude ---
    # stop_dist = abs(pos.entry_price − pos.stop_price)
    # After BE: stop_dist = 0 → slip = 0 → fill = entry_fill
    if not math.isnan(effective_stop) and not math.isnan(fill_px):
        stop_dist    = abs(fill_px - effective_stop)
        expected_slip = stop_dist * STOP_SLIP_FRAC
        actual_slip  = abs(stop_fill - effective_stop)
        ok = abs(actual_slip - expected_slip) < TOL_PRICE
        checks.append(Check(
            "stop.slippage_magnitude",
            ok,
            f"{_f(expected_slip)} (stop_dist×{STOP_SLIP_FRAC})",
            _f(actual_slip),
            note=f"diff={_f(actual_slip - expected_slip)}",
            layer="ok" if ok else "execution_engine",
        ))

    # --- Remaining size at stop (inferred from fee) ---
    # fee = stop_fill × remaining × TAKER  →  remaining ≈ fee / (stop_fill × TAKER)
    tp_sizes_hit = sum(p.get("size", 0.0) for p in chain.partials)
    expected_remaining = total_size - tp_sizes_hit
    if not math.isnan(stop_fill) and stop_fill > 0 and not math.isnan(stop_fee):
        implied_remaining = stop_fee / (stop_fill * TAKER_FEE)
        ok = abs(implied_remaining - expected_remaining) < TOL_SIZE
        checks.append(Check(
            "stop.implied_remaining_size",
            ok,
            f"{_f(expected_remaining)} (total − Σtp_sizes)",
            f"{_f(implied_remaining)} (fee÷(fill×{TAKER_FEE}))",
            note=f"diff={_f(implied_remaining - expected_remaining)}",
            layer="ok" if ok else "journal",
        ))

    # --- Stop fee formula ---
    if expected_remaining > 0 and not math.isnan(stop_fill) and not math.isnan(stop_fee):
        exp_fee = stop_fill * expected_remaining * TAKER_FEE
        ok = abs(stop_fee - exp_fee) < TOL_FEE
        checks.append(Check(
            "stop.fee_formula",
            ok,
            f"{_f(exp_fee)} (fill×remaining×{TAKER_FEE})",
            _f(stop_fee),
            note=f"diff={_f(stop_fee - exp_fee)}",
            layer="ok" if ok else "journal",
        ))

    return checks


def check_sequence(chain: TradeChain) -> list[Check]:
    checks: list[Check] = []
    stop = chain.stop
    if not stop or chain.is_force_close:
        return checks

    stop_ts = stop.get("ts_epoch", float("inf"))

    # A) All TPs logged before STOP_HIT
    for pt in chain.partials:
        idx  = pt.get("tp_idx", -1)
        pt_ts = pt.get("ts_epoch", float("inf"))
        ok = pt_ts <= stop_ts
        checks.append(Check(
            f"sequence.tp{idx+1}_before_stop",
            ok,
            f"tp{idx+1}_ts ({_f(pt_ts,1)}) ≤ stop_ts ({_f(stop_ts,1)})",
            "✓" if ok else f"FAIL tp_ts={_f(pt_ts,1)} > stop_ts={_f(stop_ts,1)}",
            layer="ok" if ok else "journal",
        ))

    return checks


def check_size_reductions(chain: TradeChain) -> list[Check]:
    """Check C/D: size reduces correctly after each TP."""
    checks: list[Check] = []
    fill = chain.fill
    if not fill:
        return checks

    total   = fill.get("size", 0.0)
    is_long = chain.direction.lower() == "long"

    cumulative = 0.0
    for i, pt in enumerate(chain.partials):
        idx     = pt.get("tp_idx", -1)
        pt_size = pt.get("size", 0.0)

        if idx < len(TP_FRACTIONS):
            exp_size   = total * TP_FRACTIONS[idx]
            cumulative += pt_size
            exp_remaining = total - cumulative
            actual_remaining = total - cumulative  # we need to track...

            # Check this TP's size
            ok = abs(pt_size - exp_size) < TOL_SIZE
            checks.append(Check(
                f"size.tp{idx+1}_size",
                ok,
                f"{_f(exp_size)} ({TP_FRACTIONS[idx]*100:.0f}%×{_f(total)})",
                _f(pt_size),
                note=f"diff={_f(pt_size - exp_size)}",
                layer="ok" if ok else "execution_engine",
            ))

    # D) Remaining at stop = total − Σ(tp_sizes)
    if chain.stop and not chain.is_force_close:
        tp_sizes_hit  = sum(p.get("size", 0.0) for p in chain.partials)
        expected_rem  = total - tp_sizes_hit
        n_tps_hit     = len(chain.partials)
        exp_by_frac   = total * (1 - sum(TP_FRACTIONS[:n_tps_hit]))
        ok = abs(expected_rem - exp_by_frac) < TOL_SIZE
        checks.append(Check(
            "size.remaining_at_stop",
            ok,
            f"{_f(exp_by_frac)} (frac-based)",
            f"{_f(expected_rem)} (total−Σtps)",
            note=f"diff={_f(expected_rem - exp_by_frac)}",
            layer="ok" if ok else "execution_engine",
        ))

    return checks


def reconstruct_pnl(chain: TradeChain) -> tuple[float, float]:
    """Return (reconstructed_pnl, journal_pnl).

    Reconstructed:
      - start with −entry_fee
      - add (tp_fill − fill_px) × tp_size × sign − tp_fee for each PARTIAL_TP
      - add (stop_fill − fill_px) × remaining × sign − stop_fee for STOP_HIT
    Journal:
      - STOP_HIT.pnl is the authoritative total-net PnL
      - If no stop (all TPs): sum of PARTIAL_TP.pnl − entry_fee
    """
    fill = chain.fill
    if not fill:
        return 0.0, 0.0

    is_long   = chain.direction.lower() == "long"
    sign      = 1.0 if is_long else -1.0
    fill_px   = fill.get("fill_price", float("nan"))
    total_size = fill.get("size", 0.0)
    entry_fee = fill.get("fee", 0.0)

    recon = -entry_fee

    for pt in chain.partials:
        tp_fill = pt.get("fill_price", float("nan"))
        tp_size = pt.get("size", 0.0)
        tp_fee  = pt.get("fee", 0.0)
        if math.isnan(tp_fill) or math.isnan(fill_px):
            continue
        recon += (tp_fill - fill_px) * tp_size * sign - tp_fee

    stop = chain.stop
    if stop and not chain.is_force_close:
        stop_fill = stop.get("fill_price", float("nan"))
        stop_fee  = stop.get("fee", 0.0)
        tp_sizes_hit = sum(p.get("size", 0.0) for p in chain.partials)
        remaining    = total_size - tp_sizes_hit
        if not math.isnan(stop_fill) and not math.isnan(fill_px):
            recon += (stop_fill - fill_px) * remaining * sign - stop_fee

        journal_pnl = stop.get("pnl", float("nan"))
    elif chain.is_force_close and stop:
        journal_pnl = stop.get("pnl", float("nan"))
    else:
        # All TPs — no authoritative single-number; sum partials − entry_fee
        journal_pnl = -entry_fee + sum(p.get("pnl", 0.0) for p in chain.partials)

    return recon, journal_pnl


def check_pnl_reconciliation(chain: TradeChain) -> list[Check]:
    recon, journal = reconstruct_pnl(chain)
    if math.isnan(recon) or math.isnan(journal):
        return []
    ok = abs(recon - journal) < TOL_PNL
    return [Check(
        "pnl.reconciliation",
        ok,
        f"recon={_f(recon,4)}",
        f"journal={_f(journal,4)}",
        note=f"diff={_f(recon - journal,4)}",
        layer="ok" if ok else "analysis",
    )]


# ---------------------------------------------------------------------------
# Per-trade audit
# ---------------------------------------------------------------------------

def audit_trade(chain: TradeChain) -> AuditResult:
    checks: list[Check] = []
    checks += check_signal(chain)
    checks += check_fill(chain)
    checks += check_partial_tps(chain)
    checks += check_sequence(chain)
    checks += check_size_reductions(chain)
    checks += check_stop(chain)
    checks += check_pnl_reconciliation(chain)

    recon, journal = reconstruct_pnl(chain)

    missing: list[str] = []
    incorrect: list[str] = []
    for c in checks:
        if not c.passed:
            if "missing" in c.actual.lower():
                missing.append(c.name)
            else:
                incorrect.append(c.name)

    total = len(checks)
    passed = sum(1 for c in checks if c.passed)
    score = int(100 * passed / total) if total > 0 else 100

    layers: dict[str, int] = {}
    for c in checks:
        if not c.passed and c.layer != "ok":
            layers[c.layer] = layers.get(c.layer, 0) + 1

    return AuditResult(
        chain=chain,
        checks=checks,
        missing_events=missing,
        incorrect_events=incorrect,
        reconstructed_pnl=recon,
        journal_pnl=journal,
        integrity_score=score,
        layers_hit=layers,
    )


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

_W = 70


def _sep(ch: str = "─") -> str:
    return ch * _W


def _pass(ok: bool) -> str:
    return "✓" if ok else "✗"


def print_trade(result: AuditResult, verbose: bool) -> None:
    chain  = result.chain
    sig    = chain.signal
    fill   = chain.fill
    is_long = chain.direction.lower() == "long"

    print(_sep("═"))
    direction_str = chain.direction.upper()
    ts_str = fill.get("ts", "?")[:19] if fill else "?"
    print(f"TRADE #{chain.trade_id + 1}  {chain.symbol}  {direction_str}  {ts_str}")
    print(_sep("═"))

    # --- SIGNAL section ---
    print("\nSIGNAL")
    if sig:
        entry = sig.get("entry_price", float("nan"))
        stop  = sig.get("stop_price",  float("nan"))
        atr   = sig.get("atr",         float("nan"))
        tps   = sig.get("tp_prices",   [])
        print(f"  entry_price    {_f(entry)}")
        print(f"  stop_price     {_f(stop)}    (expected {_f(entry + (-1 if is_long else 1)*atr*ATR_STOP_MULT)} = entry±{ATR_STOP_MULT}×ATR)")
        for i, tp in enumerate(tps):
            sign  = 1 if is_long else -1
            exp   = entry + sign * atr * ATR_TP_MULTS[i]
            print(f"  tp{i+1}            {_f(tp)}    (expected {_f(exp)} = entry±{ATR_TP_MULTS[i]}×ATR)")
        print(f"  atr            {_f(atr)}")
    else:
        print("  [MISSING — no SIGNAL event found]")

    # --- FILL section ---
    print("\nFILL")
    if fill:
        fp   = fill.get("fill_price", float("nan"))
        sz   = fill.get("size",       float("nan"))
        fee  = fill.get("fee",        float("nan"))
        slip = fill.get("slippage",   float("nan"))
        atr  = sig.get("atr", float("nan")) if sig else float("nan")
        exp_slip = atr * ENTRY_SLIP_MULT if not math.isnan(atr) else float("nan")
        print(f"  fill_price     {_f(fp)}")
        print(f"  size           {_f(sz,6)}")
        print(f"  slippage       {_f(slip)}    (expected {_f(exp_slip)} = ATR×{ENTRY_SLIP_MULT})")
        print(f"  fee            {_f(fee)}    (expected {_f(fp*sz*TAKER_FEE,4)} = fill×size×{TAKER_FEE})")
    else:
        print("  [MISSING]")

    # --- PARTIAL_TP sections ---
    if chain.partials:
        for pt in chain.partials:
            idx     = pt.get("tp_idx", -1)
            pt_fill = pt.get("fill_price", float("nan"))
            pt_size = pt.get("size",       float("nan"))
            pt_pnl  = pt.get("pnl",        float("nan"))
            pt_fee  = pt.get("fee",        float("nan"))
            fill_px = fill.get("fill_price", float("nan")) if fill else float("nan")
            sign    = 1.0 if is_long else -1.0
            exp_pnl = (pt_fill - fill_px) * pt_size * sign - pt_fee
            exp_fee = pt_fill * pt_size * MAKER_FEE
            total   = fill.get("size", 0.0) if fill else 0.0
            exp_sz  = total * TP_FRACTIONS[idx] if idx < len(TP_FRACTIONS) else float("nan")
            cumulative_frac = sum(TP_FRACTIONS[:idx+1])
            exp_rem = round(total * (1 - cumulative_frac), 8)
            actual_rem = total - sum(p.get("size",0) for p in chain.partials if p.get("tp_idx",-1) <= idx)

            sig_tps = sig.get("tp_prices", []) if sig else []
            exp_price = sig_tps[idx] if idx < len(sig_tps) else float("nan")
            print(f"\nPARTIAL_TP[{idx}]  (TP{idx+1})")
            print(f"  expected_price {_f(exp_price)}")
            print(f"  fill_price     {_f(pt_fill)}"
                  f"    diff={_f(pt_fill - exp_price) if not math.isnan(exp_price) else '?'}")
            print(f"  size           {_f(pt_size,6)}    (expected {_f(exp_sz,6)} = {TP_FRACTIONS[idx]*100:.0f}% × total)")
            print(f"  fee            {_f(pt_fee,4)}    (expected {_f(exp_fee,4)} maker)")
            print(f"  pnl (net)      {_f(pt_pnl,4)}    (recon {_f(exp_pnl,4)})")
            remaining_after = total - sum(p.get("size",0) for p in chain.partials if p.get("tp_idx",-1) <= idx)
            print(f"  remaining      {_f(remaining_after,6)}    (expected {_f(exp_rem,6)})")
    else:
        print("\nPARTIAL_TP  [none]")

    # --- STOP_HIT / FORCE_CLOSE section ---
    print("\nSTOP_HIT" if not chain.is_force_close else "\nFORCE_CLOSE")
    if chain.stop:
        sv    = chain.stop
        sfill = sv.get("fill_price", float("nan"))
        ssz   = sv.get("size",       float("nan"))
        spnl  = sv.get("pnl",        float("nan"))
        sfee  = sv.get("fee",        float("nan"))
        # effective stop
        tp1_hit = any(p.get("tp_idx",-1) == 0 for p in chain.partials)
        fill_px = fill.get("fill_price", float("nan")) if fill else float("nan")
        if tp1_hit:
            eff_stop = fill_px
            stop_note = "(BE — moved to fill_price after TP1)"
        else:
            eff_stop = chain.signal.get("stop_price", float("nan")) if chain.signal else float("nan")
            stop_note = "(original)"
        total   = fill.get("size", 0.0) if fill else 0.0
        tp_sum  = sum(p.get("size", 0.0) for p in chain.partials)
        exp_rem = total - tp_sum
        exp_fee = sfill * exp_rem * TAKER_FEE if not math.isnan(sfill) else float("nan")
        stop_dist = abs(fill_px - eff_stop) if not math.isnan(fill_px) and not math.isnan(eff_stop) else float("nan")
        exp_slip  = stop_dist * STOP_SLIP_FRAC if not math.isnan(stop_dist) else float("nan")
        exp_sfill = (eff_stop - exp_slip) if is_long else (eff_stop + exp_slip)

        print(f"  stop_level     {_f(eff_stop)}    {stop_note}")
        print(f"  fill_price     {_f(sfill)}    (expected {_f(exp_sfill)}, slip={_f(exp_slip)})")
        print(f"  size(journal)  {_f(ssz,6)}    [NOTE: journal logs total_size, not remaining]")
        print(f"  remaining      {_f(exp_rem,6)}    (total − Σtp_sizes)")
        print(f"  fee            {_f(sfee,4)}    (expected {_f(exp_fee,4)} taker)")
        print(f"  pnl (total net){_f(spnl,4)}")
    else:
        print("  [no exit event — position closed via all 3 TPs]")

    # --- Verification section ---
    print(f"\n{'─'*40}  VERIFICATION")
    check_groups = {
        "A) TP-before-stop order": [c for c in result.checks if c.name.startswith("sequence.")],
        "B) Candle crossing":      [],  # skipped (no candle data here)
        "C) Size after each TP":   [c for c in result.checks if c.name.startswith("size.")],
        "D) Remaining at stop":    [c for c in result.checks if "remaining_at_stop" in c.name],
        "E) PnL reconciliation":   [c for c in result.checks if "pnl.reconciliation" in c.name],
    }
    for label, grp in check_groups.items():
        if label.startswith("B)"):
            print(f"  {label:35s} ✗ skipped (no candle data)")
            continue
        if not grp:
            print(f"  {label:35s} n/a")
            continue
        all_ok = all(c.passed for c in grp)
        status = "✓ ok" if all_ok else f"✗ FAIL ({sum(1 for c in grp if not c.passed)} issue(s))"
        print(f"  {label:35s} {status}")

    # --- All check detail if verbose or any failure ---
    has_failure = any(not c.passed for c in result.checks)
    if verbose or has_failure:
        print(f"\n  Detailed checks:")
        for c in result.checks:
            sym = _pass(c.passed)
            row = f"    {sym} {c.name:<45s}"
            if not c.passed:
                row += f"  expected={c.expected}  got={c.actual}"
                if c.note:
                    row += f"  [{c.note}]"
                row += f"  [LAYER: {c.layer}]"
            print(row)

    # --- Summary line ---
    print(f"\n  INTEGRITY SCORE: {result.integrity_score}/100"
          f"  |  missing={len(result.missing_events)}"
          f"  |  incorrect={len(result.incorrect_events)}")
    print(f"  recon_pnl={_f(result.reconstructed_pnl,4)}"
          f"  journal_pnl={_f(result.journal_pnl,4)}"
          f"  diff={_f(result.reconstructed_pnl - result.journal_pnl,4)}")
    if result.layers_hit:
        layer_str = "  |  ".join(f"{k}:{v}" for k, v in sorted(result.layers_hit.items()))
        print(f"  Bug layers: {layer_str}")


def print_session_summary(results: list[AuditResult]) -> None:
    print("\n" + "═" * _W)
    print("SESSION SUMMARY")
    print("═" * _W)

    n = len(results)
    if n == 0:
        print("No trades found.")
        return

    perfect  = sum(1 for r in results if r.integrity_score == 100)
    minor    = sum(1 for r in results if 80 <= r.integrity_score < 100)
    critical = sum(1 for r in results if r.integrity_score < 80)
    total_missing   = sum(len(r.missing_events)   for r in results)
    total_incorrect = sum(len(r.incorrect_events) for r in results)
    total_journal   = sum(r.journal_pnl           for r in results)
    total_recon     = sum(r.reconstructed_pnl     for r in results)
    pnl_diff        = total_recon - total_journal

    print(f"  trades_total     : {n}")
    print(f"  perfect  (100)   : {perfect}  ({100*perfect/n:.1f}%)")
    print(f"  minor  (80-99)   : {minor}  ({100*minor/n:.1f}%)")
    print(f"  critical (<80)   : {critical}  ({100*critical/n:.1f}%)")
    print(f"  missing_events   : {total_missing}")
    print(f"  incorrect_events : {total_incorrect}")
    print(f"  total journal_pnl: {_f(total_journal,2)}")
    print(f"  total recon_pnl  : {_f(total_recon,2)}")
    print(f"  pnl_discrepancy  : {_f(pnl_diff,4)}")

    # Aggregate layer violations
    all_layers: dict[str, int] = {}
    for r in results:
        for k, v in r.layers_hit.items():
            all_layers[k] = all_layers.get(k, 0) + v

    print(f"\n{'─'*40}  BUG LOCATION ASSESSMENT")
    if not all_layers:
        print("  ✓ No violations detected. Journal is internally consistent.")
        print("    → Performance loss is real and not an accounting artefact.")
    else:
        for layer, count in sorted(all_layers.items(), key=lambda x: -x[1]):
            label = {
                "strategy":         "Strategy (signal geometry / ATR multiples wrong)",
                "execution_engine": "Execution engine (fill prices deviate from formulas)",
                "journal":          "Journal (event PnL/fee fields inconsistent)",
                "analysis":         "Analysis layer (our reconstruction math is wrong)",
            }.get(layer, layer)
            print(f"  {'✗':2s} {label}")
            print(f"       {count} violation(s) across {n} trades")

        # Decision table
        print(f"\n{'─'*40}  VERDICT")
        has_strategy = "strategy" in all_layers
        has_engine   = "execution_engine" in all_layers
        has_journal  = "journal" in all_layers
        has_analysis = "analysis" in all_layers

        if has_analysis and not has_journal and not has_engine and not has_strategy:
            print("  The reconstruction math in this script is wrong.")
            print("  Journal and engine appear correct. Fix analysis_layer code.")
        elif has_journal and not has_engine and not has_strategy:
            print("  Journal event fields are inconsistent with position arithmetic.")
            print("  Check execution_journal.py write methods and log_* call sites.")
        elif has_engine and not has_strategy:
            print("  Execution engine fills deviate from simulation formulas.")
            print("  Check paper_execution.py simulate_entry/simulate_stop/simulate_tp.")
        elif has_strategy:
            print("  Signal geometry or ATR multiples are wrong at source.")
            print("  Check MomentumStrategy.generate_signal() and MomentumConfig.")
        else:
            print("  Mixed violations — inspect per-trade detail above.")
            if abs(pnl_diff) < TOL_PNL * n:
                print("  PnL discrepancy is within tolerance despite field errors.")
                print("  Performance loss is likely real, not an accounting bug.")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def write_csv(results: list[AuditResult], path: Path) -> None:
    rows = []
    for r in results:
        c   = r.chain
        row = {
            "trade_id":          c.trade_id + 1,
            "symbol":            c.symbol,
            "direction":         c.direction,
            "ts":                c.fill.get("ts", "") if c.fill else "",
            "fill_price":        c.fill.get("fill_price", "") if c.fill else "",
            "total_size":        c.fill.get("size", "") if c.fill else "",
            "n_partials":        len(c.partials),
            "has_stop_hit":      int(c.stop is not None and not c.is_force_close),
            "is_force_close":    int(c.is_force_close),
            "integrity_score":   r.integrity_score,
            "missing_events":    len(r.missing_events),
            "incorrect_events":  len(r.incorrect_events),
            "reconstructed_pnl": round(r.reconstructed_pnl, 4),
            "journal_pnl":       round(r.journal_pnl, 4),
            "pnl_diff":          round(r.reconstructed_pnl - r.journal_pnl, 6),
            "bug_strategy":      r.layers_hit.get("strategy", 0),
            "bug_engine":        r.layers_hit.get("execution_engine", 0),
            "bug_journal":       r.layers_hit.get("journal", 0),
            "bug_analysis":      r.layers_hit.get("analysis", 0),
            "violation_details": "; ".join(
                f"{ch.name}(exp={ch.expected},got={ch.actual})"
                for ch in r.checks if not ch.passed
            ),
        }
        rows.append(row)
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV written → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Event chain auditor")
    parser.add_argument("--journal-dir", default="logs/paper_validation",
                        help="Directory containing .jsonl journal files")
    parser.add_argument("--csv", default="event_chain_audit.csv",
                        help="Output CSV path (empty to skip)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print all check lines, not just failures")
    parser.add_argument("--trade", type=int, default=None,
                        help="Audit only trade N (1-based index)")
    args = parser.parse_args()

    journal_dir = _REPO_ROOT / args.journal_dir
    if not journal_dir.exists():
        # Try relative to CWD
        journal_dir = Path(args.journal_dir)
    if not journal_dir.exists():
        print(f"[ERROR] Journal directory not found: {journal_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading journals from {journal_dir} …")
    events = load_journals(journal_dir)
    print(f"  {len(events)} events loaded")

    chains = build_trade_chains(events)
    print(f"  {len(chains)} trade chains reconstructed")

    if not chains:
        print("Nothing to audit.")
        return

    results: list[AuditResult] = []
    for chain in chains:
        if args.trade is not None and (chain.trade_id + 1) != args.trade:
            continue
        result = audit_trade(chain)
        results.append(result)
        print_trade(result, verbose=args.verbose)

    print_session_summary(results)

    if args.csv and args.trade is None:
        write_csv(results, _REPO_ROOT / args.csv)


if __name__ == "__main__":
    main()
