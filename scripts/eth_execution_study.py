#!/usr/bin/env python3
"""ETH execution study: 4 taker/maker entry×exit combinations on identical entries.

Entries are FROZEN.  Signal generation is UNCHANGED.

Fee combinations
----------------
  TT  Taker entry + Taker exit   baseline   (entry fills always, all fees 0.06%)
  MT  Maker entry + Taker exit              (entry requires price reversal)
  TM  Taker entry + Maker exit              (entry fills always, exits 0.02%)
  MM  Maker entry + Maker exit              (limit both sides)

Maker fill semantics
--------------------
  Entry maker limit is placed at signal_entry_price (= candle.close of signal bar).
  It fills on bar[i] if price TRADES THROUGH the limit within MAKER_ENTRY_TIMEOUT bars:
    LONG  : bar.low  < signal_entry   (price dips below our bid)
    SHORT : bar.high > signal_entry   (price rises above our ask)
  No fill within timeout → entry is MISSED.

  Exit maker limits (TPs, stops) fill at exact limit price with MAKER_FEE (0.02%).
  Stop exits in maker mode: exact price, no adverse slippage (limit order, exact fill).
  Stop exits in taker mode: adverse slippage applied (market order).

Fill / miss accounting
----------------------
  Each missed entry is classified using the TT (taker baseline) simulation outcome:
    Missed winner : the TT result was net_pnl > 0
    Missed loser  : the TT result was net_pnl <= 0

Key question
------------
  Does lower fee drag from maker fills offset profitability lost from missed entries?

Usage
-----
  python scripts/eth_execution_study.py --journal-dir logs/paper --candle-dir data/candles
  python scripts/eth_execution_study.py --journal-dir logs/paper --candle-dir data/candles --timeout 5
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

_SCRIPTS = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS))

from exit_model_experiment import (  # noqa: E402
    EntryRecord,
    _bars_after,
    _load_candles_1m,
    _load_events,
    _extract_entries,
    _match_entries_to_candles,
    _raw_pnl,
    _stop_fill,
    MAX_BARS,
    TAKER_FEE,
    MAKER_FEE,
)
from nexflow.services.candles.candle_engine import Candle  # noqa: E402

MAKER_ENTRY_TIMEOUT_DEFAULT = 3   # bars; configurable via --timeout
ETH_SYMBOL = "ETHUSDT"

VARIANT_DESCRIPTIONS = {
    "TT": "Taker entry + Taker exit  (current baseline)",
    "MT": "Maker entry + Taker exit",
    "TM": "Taker entry + Maker exit",
    "MM": "Maker entry + Maker exit  (ideal)",
}

# ── Trade result per variant ──────────────────────────────────────────────────

@dataclass
class ExecResult:
    seq: int
    variant: str
    filled: bool
    # If filled:
    entry_price: float = 0.0
    entry_fee: float = 0.0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    total_fees: float = 0.0
    exit_reason: str = ""
    hold_bars: int = 0
    fill_bar: int = 0         # which bar (0-based) the entry filled on
    # If missed:
    tt_net_pnl: float = 0.0   # what TT would have made (for classification)


@dataclass
class VariantSummary:
    variant: str
    n_attempted: int
    n_filled: int
    n_missed: int
    n_missed_winners: int
    n_missed_losers: int
    net_pnl: float
    gross_pnl: float
    total_fees: float
    fee_drag_pct: float
    profit_factor: float
    fill_pct: float
    missed_winner_pct: float   # of all missed, how many were winners
    results: list[ExecResult] = field(default_factory=list)


# ── Maker entry: try to fill a limit at signal_entry ─────────────────────────

def _try_maker_entry(
    entry: EntryRecord,
    bars: list[Candle],
    timeout: int,
) -> tuple[int, float]:
    """Return (fill_bar_index, fill_price) or (-1, 0.0) if missed.

    fill_bar_index: index in bars[] where the limit was triggered (0-based).
    fill_price    : signal_entry_price (exact, no slippage).
    """
    is_long = entry.direction == "long"
    limit   = entry.signal_entry

    for i, bar in enumerate(bars[:timeout]):
        # Price must trade THROUGH the limit, not just touch it
        if is_long  and bar.low  < limit:
            return i, limit
        if not is_long and bar.high > limit:
            return i, limit

    return -1, 0.0


# ── Parameterised Model-A simulation ─────────────────────────────────────────

def _run_model_A(
    entry: EntryRecord,
    bars: list[Candle],
    actual_entry: float,
    actual_entry_fee: float,
    exit_fee_rate: float,
    stop_exact: bool,           # True = no slippage on stop (maker); False = adverse slip
) -> tuple[float, float, float, float, int, str]:
    """Model-A (TP1/TP2/TP3 + BE) with explicit entry price and fee rates.

    Returns (gross_pnl, net_pnl, total_fees, wavg_exit, hold_bars, reason).
    """
    is_long = entry.direction == "long"
    stop    = entry.stop_price
    tp1, tp2, tp3 = entry.tp_prices[0], entry.tp_prices[1], entry.tp_prices[2]
    s1, s2, s3 = entry.total_size * 0.50, entry.total_size * 0.25, entry.total_size * 0.25

    fills: list[tuple[float, float]] = []
    fees_out    = 0.0
    tp1_hit = tp2_hit = tp3_hit = False
    hold_bars   = len(bars)
    reason      = "EOD"

    for i, bar in enumerate(bars):
        stop_hit = (is_long and bar.low <= stop) or (not is_long and bar.high >= stop)
        if stop_hit:
            fp  = stop if stop_exact else _stop_fill(stop, entry.direction, actual_entry)
            rem = entry.total_size - sum(s for _, s in fills)
            fills.append((fp, rem))
            fees_out  += fp * rem * exit_fee_rate
            hold_bars  = i + 1
            reason     = "STOP"
            break

        if not tp1_hit:
            if (is_long and bar.high >= tp1) or (not is_long and bar.low <= tp1):
                tp1_hit = True
                fills.append((tp1, s1))
                fees_out += tp1 * s1 * exit_fee_rate
                stop = actual_entry   # break-even at actual entry (not signal entry)

        if tp1_hit and not tp2_hit:
            if (is_long and bar.high >= tp2) or (not is_long and bar.low <= tp2):
                tp2_hit = True
                fills.append((tp2, s2))
                fees_out += tp2 * s2 * exit_fee_rate

        if tp2_hit and not tp3_hit:
            if (is_long and bar.high >= tp3) or (not is_long and bar.low <= tp3):
                tp3_hit = True
                fills.append((tp3, s3))
                fees_out += tp3 * s3 * exit_fee_rate
                hold_bars = i + 1
                reason    = "TP3"
                break

    if not fills or reason == "EOD":
        rem = entry.total_size - sum(s for _, s in fills)
        if rem > 0:
            last = bars[-1].close if bars else actual_entry
            fills.append((last, rem))
            fees_out += last * rem * exit_fee_rate
            reason = ("TP1" if tp1_hit and not tp2_hit else
                      "TP2" if tp2_hit and not tp3_hit else
                      "EOD")

    total_s   = sum(s for _, s in fills) or 1.0
    wavg_exit = sum(p * s for p, s in fills) / total_s
    gross     = sum(_raw_pnl(entry.direction, actual_entry, p, s) for p, s in fills)
    total_fees = actual_entry_fee + fees_out
    net       = gross - total_fees
    return gross, net, total_fees, wavg_exit, hold_bars, reason


# ── Simulate one entry under a given variant ──────────────────────────────────

def _simulate(
    entry: EntryRecord,
    bars: list[Candle],
    variant: str,
    maker_timeout: int,
    tt_net: float,       # TT simulation net_pnl (for missed-trade classification)
) -> ExecResult:
    entry_mode, exit_mode = variant[0], variant[1]   # "T" or "M"

    # ── Entry fill ────────────────────────────────────────────────────────────
    if entry_mode == "T":
        actual_entry    = entry.fill_price
        actual_entry_fee = entry.entry_fee
        fill_bar        = 0
        exit_bars       = bars
    else:
        fill_bar, fill_price = _try_maker_entry(entry, bars, maker_timeout)
        if fill_bar < 0:
            return ExecResult(seq=entry.seq, variant=variant, filled=False, tt_net_pnl=tt_net)
        actual_entry     = fill_price
        actual_entry_fee = fill_price * entry.total_size * MAKER_FEE
        exit_bars        = bars[fill_bar + 1:]   # bars after the fill bar

    if not exit_bars:
        # Filled on the last available bar — forced flat at fill price
        return ExecResult(
            seq=entry.seq, variant=variant, filled=True,
            entry_price=actual_entry, entry_fee=actual_entry_fee,
            gross_pnl=0.0, net_pnl=-actual_entry_fee,
            total_fees=actual_entry_fee,
            exit_reason="EOD", hold_bars=0, fill_bar=fill_bar,
        )

    # ── Exit simulation ───────────────────────────────────────────────────────
    exit_fee_rate = MAKER_FEE if exit_mode == "M" else TAKER_FEE
    stop_exact    = (exit_mode == "M")

    gross, net, total_fees, _, hold_bars, reason = _run_model_A(
        entry, exit_bars, actual_entry, actual_entry_fee,
        exit_fee_rate, stop_exact,
    )

    return ExecResult(
        seq=entry.seq, variant=variant, filled=True,
        entry_price=actual_entry, entry_fee=actual_entry_fee,
        gross_pnl=gross, net_pnl=net, total_fees=total_fees,
        exit_reason=reason, hold_bars=hold_bars, fill_bar=fill_bar,
    )


# ── Aggregate metrics ─────────────────────────────────────────────────────────

def _aggregate(variant: str, results: list[ExecResult], n_attempted: int) -> VariantSummary:
    filled  = [r for r in results if r.filled]
    missed  = [r for r in results if not r.filled]
    n_f = len(filled)
    n_m = len(missed)

    missed_winners = sum(1 for r in missed if r.tt_net_pnl > 0)
    missed_losers  = n_m - missed_winners

    gross     = sum(r.gross_pnl  for r in filled)
    net       = sum(r.net_pnl    for r in filled)
    fees      = sum(r.total_fees for r in filled)
    gw        = sum(r.gross_pnl  for r in filled if r.gross_pnl > 0)
    gl        = sum(abs(r.gross_pnl) for r in filled if r.gross_pnl < 0)
    pf        = gw / gl if gl > 0 else float("inf")
    drag      = fees / abs(gross) * 100 if gross != 0 else 0.0
    fill_pct  = n_f / n_attempted * 100 if n_attempted else 0
    mw_pct    = missed_winners / n_m * 100 if n_m else 0.0

    return VariantSummary(
        variant=variant, n_attempted=n_attempted,
        n_filled=n_f, n_missed=n_m,
        n_missed_winners=missed_winners, n_missed_losers=missed_losers,
        net_pnl=net, gross_pnl=gross, total_fees=fees,
        fee_drag_pct=drag, profit_factor=pf,
        fill_pct=fill_pct, missed_winner_pct=mw_pct,
        results=results,
    )


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


def _print_per_trade(
    eth_entries: list[EntryRecord],
    all_results: dict[str, list[ExecResult]],
) -> None:
    bar = "─" * 100
    print(f"\n{_B}PER-TRADE BREAKDOWN{_RST}")
    print(bar)
    hdr = (f"  {'#':>3}  {'Dir':>5}  {'Entry':>10}  "
           f"{'TT Net':>9}  {'MT Net':>9}  {'TM Net':>9}  {'MM Net':>9}  "
           f"{'TT Exit':>7}  {'MT Fill@':>8}  {'TM Exit':>7}  {'MM Fill@':>8}")
    print(hdr)
    print(bar)

    by_seq: dict[str, dict[int, ExecResult]] = {}
    for v, results in all_results.items():
        by_seq[v] = {r.seq: r for r in results}

    for entry in eth_entries:
        direction_s = f"{_G}LONG {_RST}" if entry.direction == "long" else f"{_R}SHORT{_RST}"
        row = f"  {entry.seq:>3}  {direction_s}  {entry.fill_price:>10.4f}"

        for v in ["TT", "MT", "TM", "MM"]:
            r = by_seq[v].get(entry.seq)
            if r is None:
                row += f"  {'N/A':>9}"
            elif not r.filled:
                row += f"  {_DIM}{'MISSED':>9}{_RST}"
            else:
                row += f"  {_col(r.net_pnl, 9)}"

        # Exit reasons / fill bars
        for v in ["TT", "MT", "TM", "MM"]:
            r = by_seq[v].get(entry.seq)
            w = 8 if v in ("MT", "MM") else 7
            if r is None:
                row += f"  {'N/A':>{w}}"
            elif not r.filled:
                row += f"  {_DIM}{'MISS':>{w}}{_RST}"
            elif v in ("MT", "MM"):
                row += f"  bar{r.fill_bar}:{r.exit_reason:>3}"
            else:
                row += f"  {r.exit_reason:>{w}}"

        print(row)

    print(bar)
    print()


def _print_summary(summaries: list[VariantSummary], maker_timeout: int) -> None:
    bar = "─" * 88
    print(f"\n{_B}╔══ ETH EXECUTION STUDY ══════════════════════════════════════════════════════════╗{_RST}")
    print(f"  Entries frozen.  Maker entry timeout: {maker_timeout} bar(s).  ETHUSDT only.")
    print(f"  Maker entry fill condition: price must TRADE THROUGH limit within {maker_timeout} bar(s).")
    print(bar)

    col = 14

    def _hdr(label: str) -> None:
        print(f"  {label:<24}", end="")
        for s in summaries:
            v_lbl = f"{_C}{s.variant}{_RST}"
            print(f"  {v_lbl:>{col + 9}}", end="")
        print()

    def _row(label: str, getter, fmt_fn) -> None:
        print(f"  {label:<24}", end="")
        for s in summaries:
            v = getter(s)
            print(f"  {fmt_fn(v, col)}", end="")
        print()

    def _net(v, w): return _col(v, w)
    def _gross(v, w): return _col(v, w)
    def _fees(v, w): return f"{_R}{v:>{w}.2f}{_RST}"
    def _drag(v, w):
        c = _G if v < 50 else (_Y if v < 100 else _R)
        return f"{c}{v:>{w-1}.1f}%{_RST}"
    def _pf(v, w):
        c = _G if v >= 1.5 else (_Y if v >= 1.0 else _R)
        s = f"{v:.3f}" if v != float("inf") else "∞"
        return f"{c}{s:>{w}}{_RST}"
    def _fill(v, w):
        c = _G if v >= 90 else (_Y if v >= 60 else _R)
        return f"{c}{v:>{w-1}.1f}%{_RST}"
    def _mw(v, w):
        # High missed-winner % is bad; low is good (missed the losers)
        c = _R if v > 60 else (_Y if v > 30 else _G)
        return f"{c}{v:>{w-1}.1f}%{_RST}"

    _hdr("")
    print(f"  {'':24}", end="")
    for s in summaries:
        label = VARIANT_DESCRIPTIONS[s.variant][:col]
        print(f"  {_DIM}{label:>{col}}{_RST}", end="")
    print()
    print(bar)

    _row("Net PnL (USDT)",      lambda s: s.net_pnl,           _net)
    _row("Gross PnL (USDT)",    lambda s: s.gross_pnl,         _gross)
    _row("Total Fees (USDT)",   lambda s: s.total_fees,        _fees)
    _row("Fee Drag %",          lambda s: s.fee_drag_pct,      _drag)
    print(bar)
    _row("Profit Factor",       lambda s: s.profit_factor,     _pf)
    print(bar)
    _row("Fill %",              lambda s: s.fill_pct,          _fill)
    _row("Missed (count)",      lambda s: s.n_missed,
         lambda v, w: f"{v:>{w}}")
    _row("Missed Winners",      lambda s: s.n_missed_winners,
         lambda v, w: f"{v:>{w}}")
    _row("Missed Losers",       lambda s: s.n_missed_losers,
         lambda v, w: f"{v:>{w}}")
    _row("Missed Winner %",     lambda s: s.missed_winner_pct, _mw)

    print(f"{_B}╚{'═'*86}╝{_RST}")
    print()


def _print_fee_table(summaries: list[VariantSummary]) -> None:
    """Show exactly how much each transition saves in fees."""
    bar = "─" * 72
    tt = next(s for s in summaries if s.variant == "TT")
    print(f"{_B}FEE DECOMPOSITION{_RST}")
    print(bar)
    print(f"  {'':30}  {'Total Fees':>10}  {'Fee Saving vs TT':>16}  {'Net PnL':>9}")
    print(bar)
    for s in summaries:
        saving  = tt.total_fees - s.total_fees
        saving_s = f"{_G}{saving:>+16.2f}{_RST}" if saving > 0 else f"{saving:>16.2f}"
        print(f"  {VARIANT_DESCRIPTIONS[s.variant]:<30}  {s.total_fees:>10.2f}  "
              f"{saving_s}  {_col(s.net_pnl, 9)}")
    print(bar)
    print()


def _print_diagnosis(summaries: list[VariantSummary]) -> None:
    bar = "─" * 72
    tt = next(s for s in summaries if s.variant == "TT")
    tm = next((s for s in summaries if s.variant == "TM"), None)
    mt = next((s for s in summaries if s.variant == "MT"), None)
    mm = next((s for s in summaries if s.variant == "MM"), None)

    print(f"{_B}DIAGNOSIS{_RST}")
    print(bar)

    # Q1: does taker→maker on EXITS help?
    if tm:
        exit_saving = tm.net_pnl - tt.net_pnl
        exit_fee_diff = tt.total_fees - tm.total_fees
        print(f"\n  {_B}Q1  Taker→Maker exits (TM vs TT){_RST}")
        print(f"  Fee saving : {_G}{exit_fee_diff:>+.2f} USDT{_RST}  "
              f"({tt.total_fees:.2f} → {tm.total_fees:.2f})")
        print(f"  Net Δ      : {_col(exit_saving, 8)}  USDT")
        if exit_saving > 0:
            print(f"  {_G}  ▸ Maker exits add {exit_saving:+.2f} USDT.{_RST}  "
                  f"{'Flips net-positive.' if tm.net_pnl > 0 and tt.net_pnl <= 0 else 'Improves but not decisive.'}")
        else:
            print(f"  {_Y}  ▸ Maker exits save fees but TM net is still {tm.net_pnl:+.2f}.{_RST}")

    # Q2: does taker→maker on ENTRY help or miss too much?
    if mt:
        print(f"\n  {_B}Q2  Taker→Maker entry (MT vs TT){_RST}")
        fill_loss = (1 - mt.fill_pct / 100) * tt.n_attempted
        print(f"  Fill rate  : {mt.fill_pct:.1f}%  ({mt.n_filled}/{mt.n_attempted} filled)")
        print(f"  Missed winners: {mt.n_missed_winners}  "
              f"Missed losers: {mt.n_missed_losers}  "
              f"(winner-miss rate: {mt.missed_winner_pct:.1f}%)")
        entry_net_diff = mt.net_pnl - tt.net_pnl
        if mt.fill_pct < 80:
            print(f"  {_R}  ▸ High miss rate ({100-mt.fill_pct:.0f}% missed) — maker entry unsuitable for breakout signals.{_RST}")
        elif mt.net_pnl > tt.net_pnl:
            print(f"  {_G}  ▸ Maker entry improves net PnL by {entry_net_diff:+.2f}.{_RST}  "
                  f"{'Missed mostly losers — favourable.' if mt.missed_winner_pct < 40 else 'Narrow advantage.'}")
        else:
            print(f"  {_Y}  ▸ Maker entry hurts net by {entry_net_diff:+.2f} despite "
                  f"lower fees — missed fills outweigh savings.{_RST}")

    # Q3: best combo?
    best = max(summaries, key=lambda s: s.net_pnl)
    print(f"\n  {_B}Q3  Best variant by net PnL{_RST}")
    print(f"  {_G}{best.variant} ({VARIANT_DESCRIPTIONS[best.variant].strip()}){_RST}")
    print(f"  Net PnL: {_col(best.net_pnl, 8)}")
    if best.variant == "TT":
        print(f"  {_Y}  ▸ Current execution is already optimal for this signal set.{_RST}")
    elif best.variant == "TM":
        print(f"  {_G}  ▸ Switching exits to maker (limit orders) is the highest-impact change.{_RST}")
        print(f"  {_G}    Entries can remain as market orders — no fill-miss risk.{_RST}")
    elif best.variant == "MM":
        filled_pct = best.fill_pct
        if filled_pct >= 80:
            print(f"  {_G}  ▸ Full maker achievable: fill rate {filled_pct:.0f}% is acceptable.{_RST}")
        else:
            print(f"  {_Y}  ▸ Full maker wins on filled trades but {100-filled_pct:.0f}% miss rate "
                  f"is a live-trading risk.{_RST}")
    elif best.variant == "MT":
        print(f"  {_C}  ▸ Maker entry + taker exit best — entry limit orders suit this signal.{_RST}")

    # Q4: core question — fees vs misses
    print(f"\n  {_B}Q4  Does lower fee drag offset missed fills?{_RST}")
    if tm and tm.net_pnl > tt.net_pnl:
        print(f"  {_G}  ▸ YES for exits (TM).{_RST}  No fill risk, pure fee improvement.")
    if mt and mt.fill_pct < 70:
        print(f"  {_R}  ▸ NO for entries (MT).{_RST}  "
              f"Breakout signals rarely retrace; fill-miss destroys value.")
    elif mt and mt.net_pnl > tt.net_pnl:
        print(f"  {_G}  ▸ YES for entries (MT).{_RST}  ETH retraces enough to fill maker entries profitably.")

    print()
    print(bar)
    # One-line verdict
    print(f"  {_B}VERDICT{_RST}")
    print(bar)
    if tm and tm.net_pnl > 0 and tt.net_pnl <= 0:
        print(f"  {_G}▸ Maker exits ALONE flip ETH net-positive.{_RST}")
        print(f"    Action: target limit-order exits (post-TP limit, stop-limit orders).")
    elif best.net_pnl > 0 and tt.net_pnl <= 0:
        print(f"  {_G}▸ Best variant ({best.variant}) flips net-positive.  "
              f"Recommended path: {VARIANT_DESCRIPTIONS[best.variant].strip()}.{_RST}")
    elif best.net_pnl > tt.net_pnl:
        print(f"  {_Y}▸ Best variant ({best.variant}) improves net PnL by "
              f"{best.net_pnl - tt.net_pnl:+.2f} but does not flip sign.{_RST}")
        print(f"    Fee reduction helps but is not the only required change.")
    else:
        print(f"  {_R}▸ No execution variant improves on TT baseline.{_RST}")
        print(f"    Signal quality or holding structure may need reassessment.")
    print(bar)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ETH execution study — 4 taker/maker fee combinations")
    p.add_argument("--journal-dir", type=Path, default=Path("logs/paper"))
    p.add_argument("--candle-dir",  type=Path, default=Path("data/candles"))
    p.add_argument("--timeout", type=int, default=MAKER_ENTRY_TIMEOUT_DEFAULT,
                   help=f"Maker entry fill timeout in bars (default {MAKER_ENTRY_TIMEOUT_DEFAULT})")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.journal_dir.exists():
        print(f"Journal directory not found: {args.journal_dir}")
        sys.exit(1)

    events  = _load_events(args.journal_dir)
    entries = _extract_entries(events)

    eth_entries = [e for e in entries if e.symbol == ETH_SYMBOL]
    if not eth_entries:
        print(f"No ETHUSDT entries found in {args.journal_dir}")
        sys.exit(1)

    print(f"Loaded {len(entries)} entries total, {len(eth_entries)} ETHUSDT")

    if not args.candle_dir.exists():
        print(f"Candle directory not found: {args.candle_dir}")
        sys.exit(1)

    candles_1m = _load_candles_1m(args.candle_dir, [ETH_SYMBOL])
    n_matched  = _match_entries_to_candles(eth_entries, candles_1m)
    print(f"Candle match: {n_matched}/{len(eth_entries)} ETH entries → 1m bars")

    if n_matched == 0:
        print("No candle matches.  Ensure ETHUSDT_1m.parquet covers the same session.")
        sys.exit(1)

    eth_with_candles = [e for e in eth_entries if e.candle_idx >= 0]
    n_eth = len(eth_with_candles)
    print(f"Simulating {n_eth} ETH trades × 4 variants "
          f"(maker timeout = {args.timeout} bar{'s' if args.timeout != 1 else ''})\n")

    all_candles = candles_1m[ETH_SYMBOL]

    # ── Step 1: simulate TT for all entries (classification baseline) ─────────
    tt_by_seq: dict[int, ExecResult] = {}
    for entry in eth_with_candles:
        bars = _bars_after(all_candles, entry.candle_idx)
        r    = _simulate(entry, bars, "TT", args.timeout, tt_net=0.0)
        tt_by_seq[entry.seq] = r

    # ── Step 2: simulate all 4 variants ──────────────────────────────────────
    all_results: dict[str, list[ExecResult]] = {v: [] for v in ("TT", "MT", "TM", "MM")}

    for entry in eth_with_candles:
        bars   = _bars_after(all_candles, entry.candle_idx)
        tt_net = tt_by_seq[entry.seq].net_pnl

        for variant in ("TT", "MT", "TM", "MM"):
            r = _simulate(entry, bars, variant, args.timeout, tt_net=tt_net)
            all_results[variant].append(r)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    summaries = [
        _aggregate(v, all_results[v], n_eth)
        for v in ("TT", "MT", "TM", "MM")
    ]

    # ── Print ─────────────────────────────────────────────────────────────────
    _print_summary(summaries, args.timeout)
    _print_fee_table(summaries)
    _print_per_trade(eth_with_candles, all_results)
    _print_diagnosis(summaries)


if __name__ == "__main__":
    main()
