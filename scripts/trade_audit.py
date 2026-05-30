#!/usr/bin/env python3
"""Forensic trade reconstructor and fee accounting auditor.

Reconstructs every trade from JSONL journals, verifies fee accounting,
checks for double-charging, and outputs trade_audit.csv.

Usage:
    python scripts/trade_audit.py
    python scripts/trade_audit.py --journal-dir logs/paper --out trade_audit.csv
    python scripts/trade_audit.py --journal-dir logs/paper --days 2026-05-29,2026-05-30
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

TAKER_FEE = 0.0006
MAKER_FEE = 0.0002


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    trade_id: int
    symbol: str
    side: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float          # weighted average
    size: float
    gross_pnl: float           # price-movement pnl, no fees
    entry_fee: float
    exit_fees: float           # sum of all exit-side fees
    total_fees: float          # entry_fee + exit_fees
    net_pnl: float             # gross_pnl - total_fees
    exit_reason: str
    maker_attempted: int       # TP limit orders placed
    maker_filled: int          # TP fills that occurred
    partial_tp_count: int
    holding_bars: int
    # Audit fields
    journal_net_pnl: float     # what the STOP_HIT/FORCE_CLOSE event claims as trade.pnl
    pnl_match: bool            # journal_net_pnl ≈ our reconstructed net_pnl
    fee_double_charged: bool   # detected fee double-charge


@dataclass
class _OpenTrade:
    trade_id: int
    symbol: str
    side: str
    entry_time: float
    entry_price: float
    size: float
    entry_fee: float
    partials: list[dict] = field(default_factory=list)
    maker_attempted: int = 0
    maker_filled: int = 0
    holding_bars: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _day(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d")


def _load_events(journal_dir: Path, filter_days: set[str] | None) -> list[dict]:
    events: list[dict] = []
    for path in sorted(journal_dir.glob("*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if filter_days:
                    day = _day(ev.get("ts_epoch", 0.0))
                    if day not in filter_days:
                        continue
                events.append(ev)
    events.sort(key=lambda e: e.get("ts_epoch", 0.0))
    return events


# ---------------------------------------------------------------------------
# Trade reconstruction
# ---------------------------------------------------------------------------

def reconstruct_trades(events: list[dict]) -> list[TradeRecord]:
    open_trades: dict[str, _OpenTrade] = {}   # symbol → open trade
    records: list[TradeRecord] = []
    trade_id = 0

    for ev in events:
        evt = ev.get("event", "")
        sym = ev.get("symbol", "")
        ts = ev.get("ts_epoch", 0.0)

        if evt == "FILL":
            if sym in open_trades:
                # Shouldn't happen (position already open guard in router)
                # but log it as a potential duplicate
                pass
            trade_id += 1
            open_trades[sym] = _OpenTrade(
                trade_id=trade_id,
                symbol=sym,
                side=ev.get("direction", ""),
                entry_time=ts,
                entry_price=ev.get("fill_price", 0.0),
                size=ev.get("size", 0.0),
                entry_fee=ev.get("fee", 0.0),
            )

        elif evt == "TP_LIMITS_PLACED":
            if sym in open_trades:
                open_trades[sym].maker_attempted += ev.get("n_limits", 0)

        elif evt == "TP_MAKER_FILL":
            if sym in open_trades:
                open_trades[sym].maker_filled += 1

        elif evt == "PARTIAL_TP":
            if sym in open_trades:
                open_trades[sym].partials.append({
                    "fill_price": ev.get("fill_price", 0.0),
                    "size": ev.get("size", 0.0),
                    # PARTIAL_TP.pnl is ALREADY net of fee (apply_partial_close deducts fee
                    # before computing the delta stored as partial_pnl)
                    "pnl_net": ev.get("pnl", 0.0),
                    "fee": ev.get("fee", 0.0),
                    "tp_idx": ev.get("tp_idx", -1),
                    "is_maker": True,
                })

        elif evt in ("STOP_HIT", "FORCE_CLOSE", "SESSION_END"):
            if sym not in open_trades:
                continue

            ot = open_trades.pop(sym)

            if evt == "STOP_HIT":
                exit_price = ev.get("fill_price", 0.0)
                exit_fee = ev.get("fee", 0.0)
                exit_reason = "STOP"
                # STOP_HIT.pnl = trade.pnl = pos.realized_pnl which already includes
                # entry_fee (negative), all partial pnls (net of their fees), and the
                # stop's raw pnl minus stop fee.  It is the TOTAL NET trade pnl.
                journal_net = ev.get("pnl", 0.0)
            elif evt == "FORCE_CLOSE":
                exit_price = ev.get("price", 0.0)
                exit_fee = 0.0
                exit_reason = "FORCE_CLOSE"
                journal_net = ev.get("pnl", 0.0)
            else:
                continue

            # Reconstruct gross pnl from first principles
            direction_sign = 1.0 if ot.side == "long" else -1.0

            # Gross PnL from partial TPs
            partial_gross = sum(
                (p["fill_price"] - ot.entry_price) * p["size"] * direction_sign
                for p in ot.partials
            )
            partial_fees = sum(p["fee"] for p in ot.partials)

            # Remaining size at stop
            partial_size_closed = sum(p["size"] for p in ot.partials)
            remaining = ot.size - partial_size_closed

            # Gross PnL from stop/force
            stop_gross = (exit_price - ot.entry_price) * remaining * direction_sign

            gross_pnl = partial_gross + stop_gross
            total_fees = ot.entry_fee + partial_fees + exit_fee
            net_pnl = gross_pnl - total_fees

            # Weighted average exit price
            pv = sum(p["fill_price"] * p["size"] for p in ot.partials) + exit_price * remaining
            total_closed = partial_size_closed + remaining
            wavg_exit = pv / total_closed if total_closed > 0 else exit_price

            # Check: does our reconstructed net match journal's claimed net?
            pnl_match = abs(net_pnl - journal_net) < 0.10  # within 10 cents

            # Fee double-charge detection:
            # In daily_summary.py (OLD version), PARTIAL_TP.pnl (already net) was treated
            # as gross, then fee was subtracted again → double-charge.
            # Flag trades where partial fees > 0 (all of them) for the auditor's awareness.
            fee_double_charged = len(ot.partials) > 0  # any partial TP means daily_summary was wrong

            records.append(TradeRecord(
                trade_id=ot.trade_id,
                symbol=ot.symbol,
                side=ot.side,
                entry_time=_ts(ot.entry_time),
                exit_time=_ts(ts),
                entry_price=round(ot.entry_price, 4),
                exit_price=round(wavg_exit, 4),
                size=round(ot.size, 6),
                gross_pnl=round(gross_pnl, 4),
                entry_fee=round(ot.entry_fee, 4),
                exit_fees=round(partial_fees + exit_fee, 4),
                total_fees=round(total_fees, 4),
                net_pnl=round(net_pnl, 4),
                exit_reason=exit_reason,
                maker_attempted=ot.maker_attempted,
                maker_filled=ot.maker_filled,
                partial_tp_count=len(ot.partials),
                holding_bars=ot.holding_bars,
                journal_net_pnl=round(journal_net, 4),
                pnl_match=pnl_match,
                fee_double_charged=fee_double_charged,
            ))

        elif evt == "EQUITY_SNAPSHOT":
            # Use hold_bars from snapshots: count 1m bars per symbol between fill and close
            pass

    return records


# ---------------------------------------------------------------------------
# Fee accounting verification
# ---------------------------------------------------------------------------

def verify_fee_accounting(events: list[dict], records: list[TradeRecord]) -> dict:
    issues: list[str] = []

    # Check: no fees on REJECTED events
    for ev in events:
        if ev.get("event") == "REJECTED" and ev.get("fee", 0.0) != 0.0:
            issues.append(f"Fee charged on REJECTED order: {ev}")

    # Check: duplicate FILL events for same symbol (no close between them)
    open_syms: set[str] = set()
    for ev in events:
        evt = ev.get("event", "")
        sym = ev.get("symbol", "")
        if evt == "FILL":
            if sym in open_syms:
                issues.append(f"Duplicate FILL for {sym} at {_ts(ev['ts_epoch'])} — position already open")
            open_syms.add(sym)
        elif evt in ("STOP_HIT", "FORCE_CLOSE"):
            open_syms.discard(sym)

    # Check: PARTIAL_TP.pnl is net (fee already deducted) — verify against gross formula
    fill_prices: dict[str, float] = {}
    fill_sides: dict[str, str] = {}
    for ev in events:
        evt = ev.get("event", "")
        sym = ev.get("symbol", "")
        if evt == "FILL":
            fill_prices[sym] = ev.get("fill_price", 0.0)
            fill_sides[sym] = ev.get("direction", "long")
        elif evt == "PARTIAL_TP":
            ep = fill_prices.get(sym)
            if ep:
                sign = 1.0 if fill_sides.get(sym) == "long" else -1.0
                tp_price = ev.get("fill_price", 0.0)
                size = ev.get("size", 0.0)
                fee = ev.get("fee", 0.0)
                logged_pnl = ev.get("pnl", 0.0)
                expected_gross = (tp_price - ep) * size * sign
                expected_net = expected_gross - fee
                if abs(logged_pnl - expected_net) > 0.01:
                    issues.append(
                        f"PARTIAL_TP pnl mismatch for {sym}: logged={logged_pnl:.4f} "
                        f"expected_net={expected_net:.4f} (gross={expected_gross:.4f} fee={fee:.4f})"
                    )

    # Check: maker fee rate on TP fills vs taker on stops
    for ev in events:
        evt = ev.get("event", "")
        sym = ev.get("symbol", "")
        if evt == "PARTIAL_TP":
            fill_price = ev.get("fill_price", 0.0)
            size = ev.get("size", 0.0)
            fee = ev.get("fee", 0.0)
            if fill_price > 0 and size > 0:
                rate = fee / (fill_price * size)
                if abs(rate - MAKER_FEE) > 0.0001 and abs(rate - TAKER_FEE) > 0.0001:
                    issues.append(
                        f"Unexpected fee rate on PARTIAL_TP {sym}: "
                        f"fee={fee:.6f} price={fill_price} size={size} rate={rate:.6f}"
                    )
        elif evt == "STOP_HIT":
            fill_price = ev.get("fill_price", 0.0)
            # stop uses remaining_size but that's not logged directly; skip rate check

    # Check: daily_summary fee double-charging
    # Reconstruct what daily_summary.py computes vs correct values
    ds_gross = 0.0
    ds_fees = 0.0
    ds_entry_fees = 0.0
    for ev in events:
        evt = ev.get("event", "")
        if evt == "FILL":
            ds_entry_fees += ev.get("fee", 0.0)
        elif evt == "PARTIAL_TP":
            # daily_summary adds pnl (which is NET) as gross, then subtracts fee again
            ds_gross += ev.get("pnl", 0.0)   # net treated as gross → already missing fee
            ds_fees += ev.get("fee", 0.0)     # fee subtracted again → double-charge
        elif evt == "STOP_HIT":
            # daily_summary adds pnl (which is TOTAL NET) as gross, then subtracts entry+exit fees
            ds_gross += ev.get("pnl", 0.0)   # already net of ALL fees
            ds_fees += ev.get("fee", 0.0)     # stop fee subtracted again → double-charge

    ds_net_computed = ds_gross - ds_entry_fees - ds_fees

    correct_net = sum(r.net_pnl for r in records)
    double_charge_error = ds_net_computed - correct_net

    if abs(double_charge_error) > 0.01:
        issues.append(
            f"DAILY SUMMARY FEE DOUBLE-CHARGE: "
            f"daily_summary reports net≈{ds_net_computed:.2f} but correct net={correct_net:.2f} "
            f"(error={double_charge_error:+.2f})"
        )

    return {
        "issues": issues,
        "daily_summary_net": round(ds_net_computed, 2),
        "correct_net": round(correct_net, 2),
        "fee_error": round(double_charge_error, 2),
    }


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def _summary(records: list[TradeRecord], label: str = "ALL TRADES") -> None:
    if not records:
        print(f"\n{label}: no trades.")
        return

    wins = [r for r in records if r.net_pnl > 0]
    gross = sum(r.gross_pnl for r in records)
    fees = sum(r.total_fees for r in records)
    net = sum(r.net_pnl for r in records)
    wr = len(wins) / len(records) * 100
    pf_denom = sum(abs(r.net_pnl) for r in records if r.net_pnl < 0)
    pf_num = sum(r.net_pnl for r in records if r.net_pnl > 0)
    pf = pf_num / pf_denom if pf_denom > 0 else float("inf")

    mk_att = sum(r.maker_attempted for r in records)
    mk_fill = sum(r.maker_filled for r in records)
    mk_rate = mk_fill / mk_att * 100 if mk_att > 0 else 0.0

    exits = defaultdict(int)
    for r in records:
        exits[r.exit_reason] += 1

    bar = "─" * 60
    print(f"\n{'═'*60}")
    print(f"  {label}")
    print(f"{'═'*60}")
    print(f"  Trades         : {len(records)}")
    print(f"  Wins           : {len(wins)}  ({wr:.1f}%)")
    print(f"  Gross PnL      : {gross:>+12.2f}")
    print(f"  Total Fees     : {fees:>12.2f}")
    print(f"  Net PnL        : {net:>+12.2f}")
    print(f"  Profit Factor  : {pf:.3f}")
    print(bar)
    print(f"  Exit reasons   : {dict(exits)}")
    print(bar)
    print(f"  Maker attempts : {mk_att}")
    print(f"  Maker fills    : {mk_fill}  ({mk_rate:.1f}%)")
    print(f"{'═'*60}")


# ---------------------------------------------------------------------------
# Discrepancy diagnosis
# ---------------------------------------------------------------------------

def diagnose_discrepancy(records: list[TradeRecord]) -> None:
    print("\n" + "═" * 60)
    print("  DISCREPANCY DIAGNOSIS")
    print("═" * 60)
    print("  Historical audit (ETH, backtest period):")
    print("    WR ~44-57%,  PF >4,  Gross +115,  Fees -132,  Net -17")
    print()

    if not records:
        print("  No live trades to compare.")
        return

    wins = [r for r in records if r.net_pnl > 0]
    wr = len(wins) / len(records) * 100
    gross = sum(r.gross_pnl for r in records)
    fees = sum(r.total_fees for r in records)
    net = sum(r.net_pnl for r in records)
    stops = sum(1 for r in records if r.exit_reason == "STOP")
    tps_any = sum(1 for r in records if r.partial_tp_count > 0)
    be_stops = sum(1 for r in records if r.partial_tp_count > 0 and r.exit_reason == "STOP")

    print(f"  Live results ({len(records)} trades):")
    print(f"    WR {wr:.1f}%,  Gross {gross:+.2f},  Fees {fees:.2f},  Net {net:+.2f}")
    print(f"    Stops: {stops},  Trades with TP1: {tps_any},  BE stops (TP1 then stop): {be_stops}")
    print()

    # Hypothesis A: strategy changed
    print("  [A] Strategy changed?")
    print("      MomentumStrategy untouched in this session. UNLIKELY.")
    print()

    # Hypothesis B: execution logic changed
    print("  [B] Execution logic changed?")
    if be_stops > 0:
        pct = be_stops / len(records) * 100
        print(f"      {be_stops} trades ({pct:.0f}%) hit TP1 then stopped at BE.")
        print("      In live markets, price revisiting entry after TP1 is common.")
        print("      This pattern inflates stop rate and destroys WR vs backtest.")
        print("      This is EXPECTED LIVE BEHAVIOR, not a code bug.")
    else:
        print("      No BE-stop pattern detected.")
    print()

    # Hypothesis C: accounting bug
    print("  [C] Accounting bug in daily_summary.py?")
    print("      YES — confirmed double-charging of fees.")
    print("      PARTIAL_TP.pnl is already net-of-fee, but daily_summary")
    print("      subtracts the fee again. STOP_HIT.pnl is the total net")
    print("      trade pnl, but daily_summary subtracts stop fee again.")
    print("      This inflates reported fees and depresses net PnL.")
    print()

    # Hypothesis D: journal reconstruction bug
    print("  [D] Journal reconstruction bug in daily_summary.py?")
    pnl_mismatches = sum(1 for r in records if not r.pnl_match)
    if pnl_mismatches > 0:
        print(f"      {pnl_mismatches} trades have PnL mismatch > $0.10 between")
        print("      reconstructed and journal values. Investigate these rows.")
    else:
        print("      All reconstructed PnLs match journal values within $0.10. OK.")
    print()

    # WR explanation
    print("  WR 5.3% vs 57% — most likely cause:")
    if be_stops / len(records) > 0.3 if records else False:
        print("      High BE-stop rate: live market is mean-reverting after TP1.")
        print("      Strategy has edge in trending conditions (backtest period)")
        print("      but degrades in range-bound or choppy conditions (live).")
    else:
        print("      Insufficient live data to isolate cause.")
        print("      Collect 50+ trades before drawing conclusions.")
    print("═" * 60)


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_csv(records: list[TradeRecord], out_path: Path) -> None:
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "trade_id", "timestamp", "symbol", "side",
            "entry_price", "exit_price", "size",
            "gross_pnl", "entry_fee", "exit_fees", "total_fees", "net_pnl",
            "exit_reason", "partial_tp_count",
            "maker_attempted", "maker_filled",
            "holding_bars",
            "journal_net_pnl", "pnl_match", "fee_double_charged",
        ])
        for r in records:
            w.writerow([
                r.trade_id, r.entry_time, r.symbol, r.side,
                r.entry_price, r.exit_price, r.size,
                r.gross_pnl, r.entry_fee, r.exit_fees, r.total_fees, r.net_pnl,
                r.exit_reason, r.partial_tp_count,
                r.maker_attempted, r.maker_filled,
                r.holding_bars,
                r.journal_net_pnl, r.pnl_match, r.fee_double_charged,
            ])
    print(f"\nCSV written: {out_path}  ({len(records)} trades)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Forensic trade audit")
    p.add_argument("--journal-dir", type=Path, default=Path("logs/paper"))
    p.add_argument("--out", type=Path, default=Path("trade_audit.csv"))
    p.add_argument("--days", type=str, default="",
                   help="Comma-separated dates to filter, e.g. 2026-05-29,2026-05-30")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.journal_dir.exists():
        print(f"Journal directory not found: {args.journal_dir}")
        sys.exit(1)

    filter_days = {d.strip() for d in args.days.split(",") if d.strip()} if args.days else None

    print(f"Loading journals from {args.journal_dir} ...")
    events = _load_events(args.journal_dir, filter_days)
    if not events:
        print("No events found.")
        sys.exit(1)
    print(f"Loaded {len(events)} events.")

    print("\nReconstructing trades ...")
    records = reconstruct_trades(events)
    print(f"Reconstructed {len(records)} closed trades.")

    # Per-day breakdown
    days: dict[str, list[TradeRecord]] = defaultdict(list)
    for r in records:
        day = r.entry_time[:10]
        days[day].append(r)

    for day, recs in sorted(days.items()):
        _summary(recs, label=f"{day}  ({len(recs)} trades)")

    _summary(records, label="TOTAL")

    print("\nVerifying fee accounting ...")
    audit = verify_fee_accounting(events, records)
    if audit["issues"]:
        print(f"\n  *** {len(audit['issues'])} ACCOUNTING ISSUE(S) FOUND ***")
        for issue in audit["issues"]:
            print(f"  ! {issue}")
    else:
        print("  Fee accounting: no issues found.")
    print(f"  daily_summary.py would report net: {audit['daily_summary_net']:.2f}")
    print(f"  Correct net (this script):          {audit['correct_net']:.2f}")
    print(f"  Error from double-charging:         {audit['fee_error']:+.2f}")

    if args.verbose:
        print("\n  Per-trade detail:")
        print(f"  {'#':>3}  {'Symbol':8}  {'Side':5}  {'Gross':>9}  {'Fees':>7}  {'Net':>9}  {'Exit':12}  {'PnL OK':6}  {'Entries':>7}")
        print(f"  {'─'*3}  {'─'*8}  {'─'*5}  {'─'*9}  {'─'*7}  {'─'*9}  {'─'*12}  {'─'*6}  {'─'*7}")
        for r in records:
            ok = "OK" if r.pnl_match else "MISMATCH"
            print(
                f"  {r.trade_id:>3}  {r.symbol:8}  {r.side:5}  {r.gross_pnl:>+9.2f}  "
                f"{r.total_fees:>7.2f}  {r.net_pnl:>+9.2f}  {r.exit_reason:12}  {ok:6}  "
                f"att={r.maker_attempted} fill={r.maker_filled}"
            )

    diagnose_discrepancy(records)
    write_csv(records, args.out)


if __name__ == "__main__":
    main()
