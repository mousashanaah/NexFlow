#!/usr/bin/env python3
"""ETH-only strategy audit: per-trade excursion + 4 fee/execution counterfactuals.

For every ETHUSDT trade this script reports:
  entry timestamp · direction · gross PnL · net PnL · fees paid
  MFE (max favorable excursion) · MAE (max adverse excursion)
  ATR at entry · hold time · exit reason

Then computes 4 counterfactuals on the same entries to answer:
  CF1  Zero fees         → is ETH profitable before fees?
  CF2  Maker fees only   → does the fee *rate* explain the drag?
  CF3  20-bar fixed exit → does holding longer change the outcome?
  CF4  No TP structure   → stop-only hold: does the TP split hurt?

Diagnosis
---------
  If CF1 net PnL > 0: ETH engine is working. Execution is the remaining problem.
  If CF1 net PnL <= 0: Entry logic still needs work regardless of fees.

Usage
-----
  python scripts/eth_strategy_audit.py --journal-dir logs/paper --candle-dir data/candles
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Reuse shared primitives from the exit model experiment
_SCRIPTS = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS))

from exit_model_experiment import (  # noqa: E402
    EntryRecord,
    _bars_after,
    _fee,
    _load_candles_1m,
    _load_events,
    _extract_entries,
    _match_entries_to_candles,
    _raw_pnl,
    _rolling_atr,
    _stop_fill,
    MAX_BARS,
    TAKER_FEE,
    MAKER_FEE,
)

from nexflow.services.candles.candle_engine import Candle  # noqa: E402

FIXED_EXIT_BARS = 20    # CF3 hold duration
ETH_SYMBOL      = "ETHUSDT"

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class AuditRecord:
    seq: int
    entry_ts: str           # ISO-8601 of the entry candle close
    direction: str
    entry_price: float
    exit_price: float       # weighted average
    total_size: float
    gross_pnl: float        # from Model-A simulation (no fees)
    net_pnl: float          # Model-A with actual fee rates
    fees_paid: float        # entry + exit fees
    entry_fee: float        # entry-side fee only
    mfe: float              # max favorable excursion vs entry (price units)
    mae: float              # max adverse excursion vs entry (price units)
    mfe_r: float            # MFE / initial_risk_R
    mae_r: float            # MAE / initial_risk_R
    atr: float
    risk_r: float           # |entry - stop|
    hold_bars: int
    exit_reason: str
    # Counterfactual net PnLs
    cf1_zero_fees: float = 0.0
    cf2_maker_fees: float = 0.0
    cf3_20bar: float = 0.0
    cf4_no_tp: float = 0.0


# ── Model-A simulation (current system) used to set the audit baseline ────────

def _run_model_A(
    entry: EntryRecord,
    bars: list[Candle],
) -> tuple[float, float, float, float, int, str]:
    """Simulate current TP1/TP2/TP3 exit mechanics.

    Returns (gross_pnl, net_pnl, fees, wavg_exit_price, hold_bars, reason).
    """
    is_long = entry.direction == "long"
    stop    = entry.stop_price
    tp1, tp2, tp3 = entry.tp_prices[0], entry.tp_prices[1], entry.tp_prices[2]
    s1, s2, s3 = entry.total_size * 0.50, entry.total_size * 0.25, entry.total_size * 0.25

    fills: list[tuple[float, float]] = []   # (price, size)
    fees_out = 0.0
    tp1_hit = tp2_hit = tp3_hit = False
    hold_bars = len(bars)
    reason = "EOD"

    for i, bar in enumerate(bars):
        stop_hit = (is_long and bar.low <= stop) or (not is_long and bar.high >= stop)
        if stop_hit:
            fp = _stop_fill(stop, entry.direction, entry.fill_price)
            rem = entry.total_size - sum(s for _, s in fills)
            fills.append((fp, rem))
            fees_out += _fee(fp, rem, maker=False)
            hold_bars = i + 1
            reason = "STOP"
            break

        if not tp1_hit:
            if (is_long and bar.high >= tp1) or (not is_long and bar.low <= tp1):
                tp1_hit = True
                fills.append((tp1, s1))
                fees_out += _fee(tp1, s1, maker=True)
                stop = entry.fill_price   # break-even

        if tp1_hit and not tp2_hit:
            if (is_long and bar.high >= tp2) or (not is_long and bar.low <= tp2):
                tp2_hit = True
                fills.append((tp2, s2))
                fees_out += _fee(tp2, s2, maker=True)

        if tp2_hit and not tp3_hit:
            if (is_long and bar.high >= tp3) or (not is_long and bar.low <= tp3):
                tp3_hit = True
                fills.append((tp3, s3))
                fees_out += _fee(tp3, s3, maker=True)
                hold_bars = i + 1
                reason = "TP3"
                break

    if not fills or reason == "EOD":
        rem = entry.total_size - sum(s for _, s in fills)
        if rem > 0:
            last = bars[-1].close if bars else entry.fill_price
            fills.append((last, rem))
            fees_out += _fee(last, rem, maker=False)
            reason = ("TP1" if tp1_hit and not tp2_hit else
                      "TP2" if tp2_hit and not tp3_hit else
                      "EOD")

    total_s = sum(s for _, s in fills) or 1.0
    wavg = sum(p * s for p, s in fills) / total_s
    gross = sum(_raw_pnl(entry.direction, entry.fill_price, p, s) for p, s in fills)
    net = gross - entry.entry_fee - fees_out
    return gross, net, entry.entry_fee + fees_out, wavg, hold_bars, reason


# ── MFE / MAE ─────────────────────────────────────────────────────────────────

def _mfe_mae(
    direction: str, entry_price: float, bars: list[Candle]
) -> tuple[float, float]:
    """Return (MFE, MAE) in price units over the given bar window."""
    mfe = 0.0
    mae = 0.0
    for bar in bars:
        if direction == "long":
            mfe = max(mfe, bar.high  - entry_price)
            mae = max(mae, entry_price - bar.low)
        else:
            mfe = max(mfe, entry_price - bar.low)
            mae = max(mae, bar.high  - entry_price)
    return mfe, mae


# ── Counterfactuals ───────────────────────────────────────────────────────────

def _cf1_zero_fees(gross_pnl: float) -> float:
    """CF1: same gross PnL, zero fees."""
    return gross_pnl


def _cf2_maker_fees(
    entry: EntryRecord,
    bars: list[Candle],
    hold_bars: int,
) -> float:
    """CF2: replay the exact same Model-A exit but apply maker rate everywhere."""
    is_long = entry.direction == "long"
    stop = entry.stop_price
    tp1, tp2, tp3 = entry.tp_prices[0], entry.tp_prices[1], entry.tp_prices[2]
    s1, s2, s3 = entry.total_size * 0.50, entry.total_size * 0.25, entry.total_size * 0.25

    fills: list[tuple[float, float]] = []
    fees_out = 0.0
    tp1_hit = tp2_hit = tp3_hit = False

    for i, bar in enumerate(bars[:hold_bars]):
        stop_hit = (is_long and bar.low <= stop) or (not is_long and bar.high >= stop)
        if stop_hit:
            fp = _stop_fill(stop, entry.direction, entry.fill_price)
            rem = entry.total_size - sum(s for _, s in fills)
            fills.append((fp, rem))
            fees_out += _fee(fp, rem, maker=True)   # ← maker everywhere
            break

        if not tp1_hit:
            if (is_long and bar.high >= tp1) or (not is_long and bar.low <= tp1):
                tp1_hit = True
                fills.append((tp1, s1))
                fees_out += _fee(tp1, s1, maker=True)
                stop = entry.fill_price

        if tp1_hit and not tp2_hit:
            if (is_long and bar.high >= tp2) or (not is_long and bar.low <= tp2):
                tp2_hit = True
                fills.append((tp2, s2))
                fees_out += _fee(tp2, s2, maker=True)

        if tp2_hit and not tp3_hit:
            if (is_long and bar.high >= tp3) or (not is_long and bar.low <= tp3):
                tp3_hit = True
                fills.append((tp3, s3))
                fees_out += _fee(tp3, s3, maker=True)
                break

    if not fills:
        last = bars[hold_bars - 1].close if hold_bars <= len(bars) else (bars[-1].close if bars else entry.fill_price)
        fills.append((last, entry.total_size))
        fees_out += _fee(last, entry.total_size, maker=True)
    else:
        rem = entry.total_size - sum(s for _, s in fills)
        if rem > 1e-9:
            last = bars[hold_bars - 1].close if hold_bars <= len(bars) else (bars[-1].close if bars else entry.fill_price)
            fills.append((last, rem))
            fees_out += _fee(last, rem, maker=True)

    entry_fee_maker = entry.fill_price * entry.total_size * MAKER_FEE
    gross = sum(_raw_pnl(entry.direction, entry.fill_price, p, s) for p, s in fills)
    return gross - entry_fee_maker - fees_out


def _cf3_20bar(entry: EntryRecord, bars: list[Candle]) -> float:
    """CF3: hold full position for 20 bars (stop still kills early). Taker fees."""
    is_long  = entry.direction == "long"
    stop     = entry.stop_price
    cap      = min(FIXED_EXIT_BARS, len(bars))

    for i, bar in enumerate(bars[:cap]):
        stop_hit = (is_long and bar.low <= stop) or (not is_long and bar.high >= stop)
        if stop_hit:
            fp       = _stop_fill(stop, entry.direction, entry.fill_price)
            fees_out = _fee(fp, entry.total_size, maker=False)
            gross    = _raw_pnl(entry.direction, entry.fill_price, fp, entry.total_size)
            return gross - entry.entry_fee - fees_out

    exit_price = bars[cap - 1].close if cap > 0 else entry.fill_price
    fees_out   = _fee(exit_price, entry.total_size, maker=False)
    gross      = _raw_pnl(entry.direction, entry.fill_price, exit_price, entry.total_size)
    return gross - entry.entry_fee - fees_out


def _cf4_no_tp(entry: EntryRecord, bars: list[Candle]) -> float:
    """CF4: full position, fixed stop, no TP structure. Taker fees at exit."""
    is_long = entry.direction == "long"
    stop    = entry.stop_price   # fixed, no movement

    for i, bar in enumerate(bars):
        stop_hit = (is_long and bar.low <= stop) or (not is_long and bar.high >= stop)
        if stop_hit:
            fp       = _stop_fill(stop, entry.direction, entry.fill_price)
            fees_out = _fee(fp, entry.total_size, maker=False)
            gross    = _raw_pnl(entry.direction, entry.fill_price, fp, entry.total_size)
            return gross - entry.entry_fee - fees_out

    last     = bars[-1].close if bars else entry.fill_price
    fees_out = _fee(last, entry.total_size, maker=False)
    gross    = _raw_pnl(entry.direction, entry.fill_price, last, entry.total_size)
    return gross - entry.entry_fee - fees_out


# ── Build audit records ───────────────────────────────────────────────────────

def _build_audit(
    entries: list[EntryRecord],
    candles_1m: dict[str, list[Candle]],
) -> list[AuditRecord]:
    records: list[AuditRecord] = []

    eth_entries = [e for e in entries if e.symbol == ETH_SYMBOL and e.candle_idx >= 0]
    if not eth_entries:
        return records

    eth_candles = candles_1m.get(ETH_SYMBOL, [])

    for entry in eth_entries:
        bars = _bars_after(eth_candles, entry.candle_idx)
        if not bars:
            continue

        # Entry candle timestamp
        entry_candle = eth_candles[entry.candle_idx]
        entry_ts = datetime.fromtimestamp(
            entry_candle.close_time / 1000, tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M")

        # Run Model-A to get holding period and baseline PnL
        gross, net, fees, wavg_exit, hold_bars, reason = _run_model_A(entry, bars)

        # MFE / MAE over the actual holding period
        hold_window = bars[:hold_bars]
        mfe, mae = _mfe_mae(entry.direction, entry.fill_price, hold_window)
        risk_r = abs(entry.fill_price - entry.stop_price)
        mfe_r  = mfe / risk_r if risk_r > 0 else 0.0
        mae_r  = mae / risk_r if risk_r > 0 else 0.0

        # Counterfactuals
        cf1 = _cf1_zero_fees(gross)
        cf2 = _cf2_maker_fees(entry, bars, hold_bars)
        cf3 = _cf3_20bar(entry, bars)
        cf4 = _cf4_no_tp(entry, bars)

        records.append(AuditRecord(
            seq=entry.seq,
            entry_ts=entry_ts,
            direction=entry.direction,
            entry_price=entry.fill_price,
            exit_price=wavg_exit,
            total_size=entry.total_size,
            gross_pnl=gross,
            net_pnl=net,
            fees_paid=fees,
            entry_fee=entry.entry_fee,
            mfe=mfe,
            mae=mae,
            mfe_r=mfe_r,
            mae_r=mae_r,
            atr=entry.atr,
            risk_r=risk_r,
            hold_bars=hold_bars,
            exit_reason=reason,
            cf1_zero_fees=cf1,
            cf2_maker_fees=cf2,
            cf3_20bar=cf3,
            cf4_no_tp=cf4,
        ))

    return records


# ── Printing ──────────────────────────────────────────────────────────────────

_G   = "\033[92m"
_R   = "\033[91m"
_Y   = "\033[93m"
_C   = "\033[96m"
_B   = "\033[1m"
_DIM = "\033[2m"
_RST = "\033[0m"


def _c(v: float) -> str:
    return _G if v > 0 else (_R if v < 0 else "")


def _col(v: float, w: int) -> str:
    return f"{_c(v)}{v:>+{w}.2f}{_RST}"


def _print_per_trade(records: list[AuditRecord]) -> None:
    bar = "─" * 120
    print(f"\n{_B}ETHUSDT — PER-TRADE AUDIT{_RST}")
    print(bar)
    hdr = (f"  {'#':>3}  {'Entry TS':>16}  {'Dir':>5}  {'Entry':>9}  "
           f"{'Gross':>8}  {'Net':>8}  {'Fees':>7}  "
           f"{'MFE':>6}  {'MAE':>6}  {'MFE(R)':>6}  {'MAE(R)':>6}  "
           f"{'ATR':>7}  {'Bars':>4}  {'Exit'}")
    print(hdr)
    print(bar)

    for r in records:
        gross_s = f"{_c(r.gross_pnl)}{r.gross_pnl:>+8.2f}{_RST}"
        net_s   = f"{_c(r.net_pnl)}{r.net_pnl:>+8.2f}{_RST}"
        fees_s  = f"{_R}{r.fees_paid:>7.2f}{_RST}"
        mfe_s   = f"{_G}{r.mfe:>6.2f}{_RST}"
        mae_s   = f"{_R}{r.mae:>6.2f}{_RST}"

        mfe_r_s = f"{_G}{r.mfe_r:>5.2f}R{_RST}"
        mae_r_s = f"{_R}{r.mae_r:>5.2f}R{_RST}"

        direction_s = f"{_G}LONG{_RST}" if r.direction == "long" else f"{_R}SHORT{_RST}"

        print(
            f"  {r.seq:>3}  {r.entry_ts:>16}  {direction_s}  {r.entry_price:>9.4f}  "
            f"{gross_s}  {net_s}  {fees_s}  "
            f"{mfe_s}  {mae_s}  {mfe_r_s}  {mae_r_s}  "
            f"{r.atr:>7.4f}  {r.hold_bars:>4}  {r.exit_reason}"
        )

    print(bar)

    n = len(records)
    tot_gross = sum(r.gross_pnl for r in records)
    tot_net   = sum(r.net_pnl   for r in records)
    tot_fees  = sum(r.fees_paid for r in records)
    avg_mfe_r = sum(r.mfe_r for r in records) / n
    avg_mae_r = sum(r.mae_r for r in records) / n
    avg_hold  = sum(r.hold_bars for r in records) / n
    wins      = sum(1 for r in records if r.net_pnl > 0)
    gross_wins = sum(r.gross_pnl for r in records if r.gross_pnl > 0)
    gross_loss = sum(abs(r.gross_pnl) for r in records if r.gross_pnl < 0)

    print(f"\n  {_B}Totals / Averages  ({n} ETH trades){_RST}")
    print(f"  {'Gross PnL':<20}: {_c(tot_gross)}{tot_gross:>+10.2f}{_RST}  USDT")
    print(f"  {'Net PnL':<20}: {_c(tot_net)}{tot_net:>+10.2f}{_RST}  USDT")
    print(f"  {'Total Fees':<20}: {_R}{tot_fees:>10.2f}{_RST}  USDT")
    print(f"  {'Fee Drag':<20}: {_R}{tot_fees/abs(tot_gross)*100 if tot_gross else 0:>10.1f}%{_RST}")
    print(f"  {'Win Rate (net)':<20}: {wins/n*100:>10.1f}%  ({wins}/{n})")
    print(f"  {'Profit Factor':<20}: {gross_wins/gross_loss if gross_loss else float('inf'):>10.3f}")
    print(f"  {'Avg MFE':<20}: {avg_mfe_r:>10.2f}R")
    print(f"  {'Avg MAE':<20}: {avg_mae_r:>10.2f}R")
    print(f"  {'Avg Hold':<20}: {avg_hold:>10.1f} bars")
    print()


def _print_counterfactuals(records: list[AuditRecord]) -> None:
    bar = "─" * 72
    n = len(records)

    actual_gross = sum(r.gross_pnl     for r in records)
    actual_net   = sum(r.net_pnl       for r in records)
    actual_fees  = sum(r.fees_paid     for r in records)
    cf1_total    = sum(r.cf1_zero_fees for r in records)
    cf2_total    = sum(r.cf2_maker_fees for r in records)
    cf3_total    = sum(r.cf3_20bar     for r in records)
    cf4_total    = sum(r.cf4_no_tp     for r in records)

    cf1_fees = 0.0
    cf2_fees = actual_gross - cf2_total

    print(f"{_B}COUNTERFACTUAL ANALYSIS  ({n} ETH trades){_RST}")
    print(bar)
    print(f"  {'':30}  {'Net PnL':>10}  {'Gross PnL':>10}  {'Fees':>8}  {'Δ vs Actual':>11}")
    print(bar)

    rows = [
        ("Actual (current system)",  actual_net,  actual_gross,  actual_fees),
        ("CF1  Zero fees",           cf1_total,   actual_gross,  0.0),
        ("CF2  Maker fees (0.02%)",  cf2_total,   actual_gross,  cf2_fees),
        ("CF3  20-bar fixed exit",   cf3_total,   None,          None),
        ("CF4  No TP structure",     cf4_total,   None,          None),
    ]

    for label, net, gross, fees in rows:
        delta = (net - actual_net) if label != "Actual (current system)" else None
        net_s   = f"{_c(net)}{net:>+10.2f}{_RST}"
        gross_s = f"{gross:>10.2f}" if gross is not None else f"{'—':>10}"
        fees_s  = f"{_R}{fees:>8.2f}{_RST}" if fees is not None else f"{'—':>8}"
        delta_s = (f"{_c(delta)}{delta:>+11.2f}{_RST}" if delta is not None else f"{'—':>11}")
        print(f"  {label:<30}  {net_s}  {gross_s}  {fees_s}  {delta_s}")

    print(bar)
    print()

    # ── Per-trade CF comparison ───────────────────────────────────────────────
    print(f"  {'Per-trade  ':>12}  {'Actual':>8}  {'CF1 0-fee':>9}  {'CF2 Maker':>9}  "
          f"{'CF3 20bar':>9}  {'CF4 NoTP':>9}")
    print(f"  {'─'*10}  {'─'*8}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}")
    for r in records:
        print(
            f"  {r.seq:>3} {r.direction[:1].upper()} {r.entry_ts[5:16]}  "
            f"{_c(r.net_pnl)}{r.net_pnl:>+8.2f}{_RST}  "
            f"{_c(r.cf1_zero_fees)}{r.cf1_zero_fees:>+9.2f}{_RST}  "
            f"{_c(r.cf2_maker_fees)}{r.cf2_maker_fees:>+9.2f}{_RST}  "
            f"{_c(r.cf3_20bar)}{r.cf3_20bar:>+9.2f}{_RST}  "
            f"{_c(r.cf4_no_tp)}{r.cf4_no_tp:>+9.2f}{_RST}"
        )
    print()


def _print_diagnosis(records: list[AuditRecord]) -> None:
    bar = "─" * 72
    n   = len(records)

    gross_total   = sum(r.gross_pnl       for r in records)
    net_total     = sum(r.net_pnl         for r in records)
    fees_total    = sum(r.fees_paid       for r in records)
    cf1_total     = sum(r.cf1_zero_fees   for r in records)
    cf2_total     = sum(r.cf2_maker_fees  for r in records)
    cf3_total     = sum(r.cf3_20bar       for r in records)
    cf4_total     = sum(r.cf4_no_tp       for r in records)

    avg_mfe_r = sum(r.mfe_r for r in records) / n
    avg_mae_r = sum(r.mae_r for r in records) / n
    wins_gross = sum(1 for r in records if r.gross_pnl > 0)

    print(f"{_B}DIAGNOSIS{_RST}")
    print(bar)

    # ── Primary verdict ───────────────────────────────────────────────────────
    print(f"\n  {_B}Primary question: is ETH profitable before fees?{_RST}")
    if cf1_total > 0:
        print(f"  {_G}▸ YES.{_RST}  CF1 (zero fees) net = {cf1_total:+.2f} USDT  "
              f"(actual gross = {gross_total:+.2f})")
        print(f"  {_G}  The strategy engine is capturing edge.{_RST}")
        print(f"  {_G}  Execution friction is the remaining problem — not signal quality.{_RST}")
    elif gross_total > 0:
        print(f"  {_Y}▸ MARGINAL.{_RST}  Gross = {gross_total:+.2f} but"
              f" fees ({fees_total:.2f}) flip it to {net_total:+.2f}.")
        print(f"  {_Y}  The engine has positive expectancy but fees erase it entirely.{_RST}")
    else:
        print(f"  {_R}▸ NO.{_RST}  Gross = {gross_total:+.2f} — "
              f"ETH loses even before fees are applied.")
        print(f"  {_R}  Entry logic still needs work. Fixing fees will not solve this.{_RST}")

    print()

    # ── Fee drag analysis ─────────────────────────────────────────────────────
    print(f"  {_B}Fee drag detail{_RST}")
    fee_drag = fees_total / abs(gross_total) * 100 if gross_total != 0 else 0
    maker_saving = cf2_total - net_total
    fee_saved_pct = (fees_total - (fees_total * MAKER_FEE / TAKER_FEE)) / fees_total * 100 if fees_total else 0
    print(f"  Total fees       : {fees_total:>8.2f} USDT  ({fee_drag:.1f}% of gross)")
    print(f"  CF2 maker saving : {maker_saving:>+8.2f} USDT  "
          f"(switching entry+exit to 0.02% saves {maker_saving:+.2f})")
    if cf2_total > 0 and net_total <= 0:
        print(f"  {_Y}  ▸ Maker fees alone make ETH net-positive ({cf2_total:+.2f}).{_RST}")
        print(f"  {_Y}    Fee rate (taker vs maker) is the decisive margin.{_RST}")
    elif cf2_total > net_total:
        print(f"  {_C}  ▸ Maker fees improve ETH by {maker_saving:+.2f} USDT but "
              f"don't flip sign.{_RST}")

    print()

    # ── Excursion analysis ────────────────────────────────────────────────────
    print(f"  {_B}Excursion (MFE / MAE){_RST}")
    print(f"  Avg MFE : {avg_mfe_r:.2f}R  — how far trades move in your favour before exit")
    print(f"  Avg MAE : {avg_mae_r:.2f}R  — how far trades move against you before reversing")

    if avg_mfe_r > 2.0:
        print(f"  {_G}  ▸ MFE > 2R on average — the market is offering more than the TP captures.{_RST}")
    if avg_mae_r > 0.8:
        print(f"  {_Y}  ▸ MAE > 0.8R — stops are frequently tested before reversals occur.{_RST}")
    if avg_mfe_r > avg_mae_r * 1.5:
        print(f"  {_G}  ▸ MFE/MAE ratio {avg_mfe_r/avg_mae_r:.2f} — favourable trade shape.{_RST}")
    elif avg_mfe_r < avg_mae_r:
        print(f"  {_R}  ▸ MAE > MFE — trades move more against than in favour."
              f"  Entry timing may be off.{_RST}")

    print()

    # ── Hold time / execution ─────────────────────────────────────────────────
    print(f"  {_B}Execution structure (CF3 vs CF4 vs actual){_RST}")
    print(f"  CF3 20-bar exit : {cf3_total:>+8.2f}  (hold fixed 20 bars, stop guards)")
    print(f"  CF4 no TP       : {cf4_total:>+8.2f}  (full position to stop, no partial exits)")
    print(f"  Actual          : {net_total:>+8.2f}")
    best_exec = max(("CF3", cf3_total), ("CF4", cf4_total), ("Actual", net_total),
                    key=lambda x: x[1])
    if best_exec[0] != "Actual":
        print(f"  {_Y}  ▸ {best_exec[0]} outperforms actual by "
              f"{best_exec[1]-net_total:+.2f} USDT — "
              f"{'longer hold' if best_exec[0]=='CF3' else 'simpler stop-only exit'} "
              f"may suit ETH better.{_RST}")
    else:
        print(f"  {_G}  ▸ Current execution is the best of the three structures.{_RST}")

    print()

    # ── Concise summary ───────────────────────────────────────────────────────
    print(bar)
    print(f"  {_B}SUMMARY{_RST}")
    print(bar)
    verdicts = []
    if cf1_total > 0:
        verdicts.append(f"{_G}Engine has edge pre-fees{_RST}")
    else:
        verdicts.append(f"{_R}No pre-fee edge — entry logic is the primary issue{_RST}")
    if cf2_total > 0 and net_total <= 0:
        verdicts.append(f"{_Y}Maker fee switch alone fixes profitability{_RST}")
    elif fees_total > abs(net_total):
        verdicts.append(f"{_Y}Fee drag exceeds total loss — fee reduction is high priority{_RST}")
    if avg_mfe_r > 1.5:
        verdicts.append(f"{_C}MFE > 1.5R avg — TP placement may be cutting gains short{_RST}")

    for v in verdicts:
        print(f"  ▸ {v}")
    print(bar)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ETH-only strategy audit with excursion + counterfactuals")
    p.add_argument("--journal-dir", type=Path, default=Path("logs/paper"))
    p.add_argument("--candle-dir",  type=Path, default=Path("data/candles"))
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.journal_dir.exists():
        print(f"Journal directory not found: {args.journal_dir}")
        sys.exit(1)

    events  = _load_events(args.journal_dir)
    entries = _extract_entries(events)

    if not entries:
        print("No FILL entries found. Run the paper trader first.")
        sys.exit(1)

    eth_count = sum(1 for e in entries if e.symbol == ETH_SYMBOL)
    print(f"Loaded {len(entries)} entries ({eth_count} ETHUSDT) from {args.journal_dir}")

    if eth_count == 0:
        print(f"No ETHUSDT entries found in journal.")
        sys.exit(1)

    if not args.candle_dir.exists():
        print(f"Candle directory not found: {args.candle_dir}")
        sys.exit(1)

    candles_1m = _load_candles_1m(args.candle_dir, [ETH_SYMBOL])
    n_matched  = _match_entries_to_candles(entries, candles_1m)
    print(f"Candle match: {n_matched}/{len(entries)} entries → 1m bars\n")

    records = _build_audit(entries, candles_1m)

    if not records:
        print("No ETHUSDT trades could be matched to candle data.")
        print("Ensure data/candles/ETHUSDT_1m.parquet covers the same session as the journal.")
        sys.exit(1)

    print(f"Auditing {len(records)} ETHUSDT trades\n")

    _print_per_trade(records)
    _print_counterfactuals(records)
    _print_diagnosis(records)


if __name__ == "__main__":
    main()
