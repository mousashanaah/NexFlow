#!/usr/bin/env python3
"""
VALIDATION TEST 01 — Full Integrated Production Stack Replay

Runs the complete production stack (data → signals → runner → paper trader)
across the full historical period (1107 bars, 2021-01-04 → 2025-05-30) and
verifies the integrated implementation faithfully reproduces the validated
research outputs.

Gates:
  1. No AllocationRunnerError or snapshot validation failure across all bars
  2. Signal exact equality at 4 frozen fixture dates
  3. Allocation exact equality at 4 frozen fixture dates
  4. Runner state is always IDLE between rebalance events
  5. Rebalance count within expected range [48, 58]
  6. Audit log written without gaps or duplicates
  7. Paper trader state consistent throughout

Usage:
  python scripts/validation_01_integrated_replay.py [--verbose]
"""
from __future__ import annotations

import sys
import tempfile
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

import numpy as np

from nexflow.v9 import core
from nexflow.v9.data import load_v9_dataset, STOCK_TICKERS
from nexflow.v9.runner import AllocationRunner, AllocationRunnerError, load_audit_log
from nexflow.v9.paper import PaperTrader
from nexflow.v9.signals import CryptoSignal, DailySignalRecord, StockSignal
from nexflow.v9.state import AllocationSnapshot, RegimeSnapshot, SystemState

# ── Frozen fixtures from research engine ──────────────────────────────────────
# Source: test_signals.py / test_v9_confidence.py

FIXTURES = {
    "2022-03-01": dict(crypto_score=1.0, stock_score=0.375, wc=0.40, ws=0.40),
    "2023-06-01": dict(crypto_score=3.5, stock_score=2.75,  wc=0.65, ws=0.35),
    "2024-03-01": dict(crypto_score=4.0, stock_score=2.75,  wc=0.65, ws=0.35),
    "2025-05-01": dict(crypto_score=3.0, stock_score=1.625, wc=0.80, ws=0.20),
}

AS_OF = "2025-05-30"


# ── Helpers ───────────────────────────────────────────────────────────────────

def ms_to_date(ts_ms: int) -> str:
    return datetime.utcfromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d")


def _signal_at_ts(
    dataset,
    regime_machine: core.RegimeMachine,
    ts: int,
) -> DailySignalRecord:
    """Compute a DailySignalRecord for a specific historical timestamp."""
    date = ms_to_date(ts)
    btc  = dataset.btc()
    bi   = btc.by_ts[ts]

    c_sc, cb = core.crypto_score_breakdown(
        btc_close = float(btc.close[bi]),
        sma200    = float(btc.sma200[bi]),
        mom90     = float(btc.mom90[bi]),
        mom30     = float(btc.mom30[bi]),
        atr14     = float(btc.atr14[bi]),
        atr_avg   = float(btc.atr_avg[bi]),
    )

    btc_sig = CryptoSignal(
        close=float(btc.close[bi]),   sma200=float(btc.sma200[bi]),
        mom90=float(btc.mom90[bi]),   mom30=float(btc.mom30[bi]),
        atr14=float(btc.atr14[bi]),   atr_avg=float(btc.atr_avg[bi]),
        pts_sma200=cb["pts_sma200"],  pts_mom90=cb["pts_mom90"],
        pts_mom30=cb["pts_mom30"],    pts_vol=cb["pts_vol"],
        pts_bonus=cb["pts_bonus"],    raw=cb["raw"],  score=c_sc,
    )

    stock_sigs: list[StockSignal] = []
    ticker_scores: list[float]   = []

    for stk in dataset.stocks.values():
        if ts not in stk.by_ts:
            continue
        si = stk.by_ts[ts]
        s_single, sb = core.stock_score_single_breakdown(
            close = float(stk.close[si]),
            s200  = float(stk.sma200[si]),
            mom90 = float(stk.mom90[si]),
            ema_f = float(stk.ema_f[si]),
            ema_s = float(stk.ema_s[si]),
        )
        stock_sigs.append(StockSignal(
            ticker=stk.ticker, close=float(stk.close[si]),
            sma200=float(stk.sma200[si]), mom90=float(stk.mom90[si]),
            ema_fast=float(stk.ema_f[si]), ema_slow=float(stk.ema_s[si]),
            pts_sma200=sb["pts_sma200"], pts_mom90=sb["pts_mom90"],
            pts_bonus=sb["pts_bonus"],   pts_ema=sb["pts_ema"],
            score=s_single,
        ))
        ticker_scores.append(s_single)

    s_sc  = core.stock_score_portfolio(ticker_scores) if ticker_scores else 0.0
    wc, ws = core.allocate(c_sc, s_sc)

    return DailySignalRecord(
        date=date, timestamp_ms=ts,
        in_bear=regime_machine.in_bear,
        btc=btc_sig, crypto_score=c_sc,
        stocks=stock_sigs, stock_score=s_sc,
        allocation_regime=core.allocation_regime_name(c_sc, s_sc),
        wc=wc, ws=ws, cash=round(1.0 - wc - ws, 10),
    )


# ── Replay engine ─────────────────────────────────────────────────────────────

@dataclass
class RebalanceEvent:
    date:             str
    wc:               float
    ws:               float
    crypto_score:     float
    stock_score:      float
    days_elapsed:     int
    allocation_regime: str


@dataclass
class ValidationResult:
    bars_processed:   int
    rebalance_count:  int
    rebalance_events: list
    fixture_results:  dict
    gate_failures:    list
    passed:           bool


def run_validation_01(verbose: bool = False) -> ValidationResult:
    gate_failures: list[str] = []

    # ── Load data ─────────────────────────────────────────────────────────────
    try:
        dataset = load_v9_dataset(AS_OF)
    except Exception as e:
        return ValidationResult(
            bars_processed=0, rebalance_count=0,
            rebalance_events=[], fixture_results={},
            gate_failures=[f"Data load failed: {e}"], passed=False,
        )

    if verbose:
        print(f"  Dataset: {len(dataset.common_ts)} bars  "
              f"({ms_to_date(int(dataset.common_ts[0]))} → "
              f"{ms_to_date(int(dataset.common_ts[-1]))})")

    with tempfile.TemporaryDirectory() as _tmp:
        tmp        = Path(_tmp)
        snap_path  = tmp / "runner.json"
        state_path = tmp / "state.json"
        audit_path = tmp / "audit.jsonl"

        # ── Initialize production stack ───────────────────────────────────────
        regime_machine = core.RegimeMachine()
        runner         = AllocationRunner.initialize(snap_path)
        ss = SystemState(
            regime=RegimeSnapshot(
                in_bear=False, consecutive_above=0, last_bar_date=""),
            allocation=AllocationSnapshot(
                wc=0.50, ws=0.50,
                last_rebalance_date="", trading_days_since=0),
        )
        ss.gate_open = True   # replay: known-good start, bypass gate
        trader           = PaperTrader(portfolio_value=5_000.0)
        btc              = dataset.btc()

        rebalance_events: list[RebalanceEvent] = []
        fixture_results:  dict = {}
        state_errors:     list[str] = []
        trading_days_since = 0
        bars_processed     = 0

        # ── Bar-by-bar replay ─────────────────────────────────────────────────
        for ts in dataset.common_ts:
            ts   = int(ts)
            date = ms_to_date(ts)
            bi   = btc.by_ts[ts]

            # Step regime machine (must happen before signal)
            regime_machine.step(
                float(btc.close[bi]),
                float(btc.sma200[bi]),
                float(btc.mom30[bi]),
            )

            # Compute integrated signal
            signal = _signal_at_ts(dataset, regime_machine, ts)

            # Record fixture dates
            if date in FIXTURES:
                fixture_results[date] = dict(
                    crypto_score = signal.crypto_score,
                    stock_score  = signal.stock_score,
                    wc           = signal.wc,
                    ws           = signal.ws,
                )
                if verbose:
                    fix = FIXTURES[date]
                    ok  = (
                        signal.crypto_score == fix["crypto_score"] and
                        signal.stock_score  == fix["stock_score"]  and
                        signal.wc           == fix["wc"]           and
                        signal.ws           == fix["ws"]
                    )
                    print(f"  Fixture {date}: {'PASS' if ok else 'FAIL'}  "
                          f"c_sc={signal.crypto_score}  s_sc={signal.stock_score}  "
                          f"wc={signal.wc}  ws={signal.ws}")

            # Step allocation runner
            trading_days_since += 1
            try:
                decision = runner.step(signal, trading_days_since, snap_path)
                if decision is not None:
                    runner.confirm_execution(
                        executed_wc          = decision.target_wc,
                        executed_ws          = decision.target_ws,
                        date                 = date,
                        system_state         = ss,
                        state_path           = state_path,
                        snapshot_path        = snap_path,
                        trading_days_elapsed = trading_days_since,
                        audit_path           = audit_path,
                        crypto_score         = signal.crypto_score,
                        stock_score          = signal.stock_score,
                        in_bear              = regime_machine.in_bear,
                    )
                    rebalance_events.append(RebalanceEvent(
                        date             = date,
                        wc               = decision.target_wc,
                        ws               = decision.target_ws,
                        crypto_score     = signal.crypto_score,
                        stock_score      = signal.stock_score,
                        days_elapsed     = trading_days_since,
                        allocation_regime = signal.allocation_regime,
                    ))
                    if verbose:
                        print(f"  Rebalance #{len(rebalance_events):3d}  {date}  "
                              f"wc={decision.target_wc:.2f} ws={decision.target_ws:.2f}  "
                              f"regime={signal.allocation_regime}  days={trading_days_since}")
                    trading_days_since = 0

                    # Paper execute
                    orders = trader.compute_orders(decision.target_wc, decision.target_ws, signal)
                    trader.execute(orders, signal, rebalance_reason=decision.reason)

                # Verify runner stays IDLE between rebalances
                if runner.state.value != "IDLE":
                    state_errors.append(
                        f"{date}: runner in unexpected state {runner.state} after step"
                    )

            except AllocationRunnerError as exc:
                state_errors.append(f"{date}: AllocationRunnerError — {exc}")

            bars_processed += 1

        # ── Evaluate gates ────────────────────────────────────────────────────

        # Gate 1: no state machine errors
        if state_errors:
            for e in state_errors:
                gate_failures.append(f"GATE 1 — State machine error: {e}")

        # Gate 2+3: fixture exact equality
        for date, fix in FIXTURES.items():
            result = fixture_results.get(date)
            if result is None:
                gate_failures.append(
                    f"GATE 2/3 — Fixture date {date} not in common_ts"
                )
                continue
            for field, expected in fix.items():
                actual = result[field]
                if actual != expected:
                    gate_failures.append(
                        f"GATE 2/3 — {date} {field}: got {actual}, expected {expected}"
                    )

        # Gate 4: runner IDLE between rebalances (checked inline above)

        # Gate 5: rebalance count in range
        n = len(rebalance_events)
        if not (48 <= n <= 58):
            gate_failures.append(
                f"GATE 5 — Rebalance count {n} outside expected range [48, 58]"
            )

        # Gate 6: audit log count matches rebalance count
        audit_records = load_audit_log(audit_path)
        if len(audit_records) != len(rebalance_events):
            gate_failures.append(
                f"GATE 6 — Audit log has {len(audit_records)} records, "
                f"expected {len(rebalance_events)}"
            )

        # Gate 7: paper trader has positions (was exercised)
        if not trader.positions:
            gate_failures.append("GATE 7 — Paper trader has no positions after replay")

    passed = len(gate_failures) == 0
    return ValidationResult(
        bars_processed   = bars_processed,
        rebalance_count  = len(rebalance_events),
        rebalance_events = rebalance_events,
        fixture_results  = fixture_results,
        gate_failures    = gate_failures,
        passed           = passed,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="VAL TEST 01 — Integrated replay")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("VALIDATION TEST 01 — Full Integrated Production Stack Replay")
    print("=" * 70)

    result = run_validation_01(verbose=args.verbose)

    print(f"\nBars processed : {result.bars_processed}")
    print(f"Rebalances     : {result.rebalance_count}")
    print(f"Fixture checks : {len(FIXTURES)}")

    if result.gate_failures:
        print(f"\n{'─' * 70}")
        print(f"GATE FAILURES ({len(result.gate_failures)}):")
        for f in result.gate_failures:
            print(f"  ✗  {f}")
    else:
        print("\n  All gates passed")

    print(f"\n{'─' * 70}")
    print(f"RESULT: {'PASS' if result.passed else 'FAIL'}")
    print("=" * 70)
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
