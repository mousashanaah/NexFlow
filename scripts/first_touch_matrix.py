#!/usr/bin/env python3
"""First-touch matrix — quantify exactly where the volatility edge leaks.

For every signal entry, walks subsequent bar high/low to determine whether
the TP or stop is touched first, at every combination of:
    stop distance : 1.0, 1.5, 2.0 × ATR
    TP distance   : 1.0, 2.0, 3.0, 4.0 × ATR

Tested in four modes:
    SIGNAL     — original signal direction (+1 long, -1 short)
    INVERTED   — opposite direction (reversal hypothesis)
    ORACLE     — hindsight-optimal direction: whichever side reaches a
                 2×ATR excursion first. Upper bound on directional edge.
    RANDOM     — coin-flip direction on same bars (noise floor)

For each (stop, TP, direction_mode) cell:
    P(TP)      — fraction of trades where TP touched before stop
    P(stop)    — fraction where stop touched before TP
    P(neither) — neither target touched within horizon (force-close)
    E(R) zero-fee  — P(TP)×(TP/stop) − P(stop)×1 + P(neither)×E(force-close/stop)
    E(R) fee-adj   — same but entry + exit fees deducted in R units

Leak attribution:
    Direction cost  = E(R, ORACLE) − E(R, SIGNAL)   [cost of wrong direction]
    Stop cost       = best E across stop rows − worst
    TP cost         = best E across TP columns − worst
    Fee cost        = E(R, zero-fee) − E(R, fee-adj)
    Residual        = what remains even with oracle direction + optimal levels

Max horizon for force-close: 50 bars (≈ 50 minutes).

Usage:
    python scripts/first_touch_matrix.py --candle-dir data/candles --start 2023-01
    python scripts/first_touch_matrix.py --candle-dir data/candles --horizon 20
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    import pyarrow.parquet as pq
except ImportError:
    print("[ERROR] pyarrow required: pip install pyarrow")
    sys.exit(1)

from nexflow.services.candles.candle_engine import Candle
from nexflow.services.strategy.momentum_strategy import (
    MomentumConfig, MomentumStrategy, compute_atr,
)

TAKER    = 0.0006
MAKER    = 0.0002
_W       = 90

STOP_MULTS = [1.0, 1.5, 2.0]
TP_MULTS   = [1.0, 2.0, 3.0, 4.0]

# Direction modes
MODE_SIGNAL   = "signal"
MODE_INVERTED = "inverted"
MODE_ORACLE   = "oracle"
MODE_RANDOM   = "random"
MODES = [MODE_SIGNAL, MODE_INVERTED, MODE_ORACLE, MODE_RANDOM]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TradeEntry:
    bar_idx:         int
    close_at_entry:  float
    atr:             float
    signal_direction: int   # +1 or -1 (original signal)
    oracle_direction: int   # +1 or -1 (direction of first 2×ATR touch)


@dataclass
class CellResult:
    """Result for one (stop_mult, tp_mult, direction_mode) combination."""
    n:              int
    n_tp:           int
    n_stop:         int
    n_neither:      int
    sum_neither_r:  float   # sum of force-close returns in R units (no fee)
    sum_fees_r:     float   # total fee drag in R units

    @property
    def p_tp(self) -> float:
        return self.n_tp / self.n if self.n else float("nan")

    @property
    def p_stop(self) -> float:
        return self.n_stop / self.n if self.n else float("nan")

    @property
    def p_neither(self) -> float:
        return self.n_neither / self.n if self.n else float("nan")

    def expectancy_nofee(self, stop_mult: float, tp_mult: float) -> float:
        if self.n == 0:
            return float("nan")
        r_tp    = self.p_tp    * (tp_mult / stop_mult)
        r_stop  = self.p_stop  * (-1.0)
        r_nei   = self.sum_neither_r / self.n
        return r_tp + r_stop + r_nei

    def expectancy_fee(self, stop_mult: float, tp_mult: float) -> float:
        e = self.expectancy_nofee(stop_mult, tp_mult)
        if math.isnan(e):
            return float("nan")
        return e - (self.sum_fees_r / self.n if self.n else 0.0)


# ---------------------------------------------------------------------------
# Candle loading
# ---------------------------------------------------------------------------

def _rows_to_candles(rows: list[dict], start_s: int, end_s: int) -> list[Candle]:
    return [
        Candle(
            symbol=r["symbol"], timeframe=r["timeframe"],
            open_time=r["open_time"], close_time=r["close_time"],
            open=r["open"], high=r["high"], low=r["low"], close=r["close"],
            volume=r["volume"], buy_volume=r["buy_volume"],
            sell_volume=r["sell_volume"], trade_count=r["trade_count"],
            vwap=r["vwap"], spread_avg=r["spread_avg"],
            spread_max=r["spread_max"],
            volatility_estimate=r["volatility_estimate"],
            is_final=True,
        )
        for r in rows
        if r.get("is_final") and start_s <= r.get("close_time", 0) <= end_s
    ]


def _aggregate_5m(c1m: list[Candle]) -> list[Candle]:
    buckets: dict[int, list[Candle]] = defaultdict(list)
    for c in c1m:
        buckets[(c.open_time // 300) * 300].append(c)
    result: list[Candle] = []
    for key in sorted(buckets):
        bars = sorted(buckets[key], key=lambda c: c.open_time)
        sym  = bars[0].symbol
        result.append(Candle(
            symbol=sym, timeframe="5m",
            open_time=bars[0].open_time, close_time=bars[-1].close_time,
            open=bars[0].open, high=max(b.high for b in bars),
            low=min(b.low for b in bars), close=bars[-1].close,
            volume=sum(b.volume for b in bars),
            buy_volume=sum(b.buy_volume for b in bars),
            sell_volume=sum(b.sell_volume for b in bars),
            trade_count=sum(b.trade_count for b in bars),
            vwap=bars[-1].close, spread_avg=0.0, spread_max=0.0,
            volatility_estimate=0.0, is_final=True,
        ))
    return result


def _load(candle_dir: Path, symbol: str,
          start_s: int, end_s: int) -> tuple[list[Candle], list[Candle]]:
    p1 = candle_dir / f"{symbol}_1m.parquet"
    p5 = candle_dir / f"{symbol}_5m.parquet"
    if not p1.exists():
        print(f"[ERROR] {p1} not found.")
        sys.exit(1)
    c1m = sorted(_rows_to_candles(pq.read_table(p1).to_pylist(), start_s, end_s),
                 key=lambda c: c.close_time)
    c5m = (sorted(_rows_to_candles(pq.read_table(p5).to_pylist(), start_s, end_s),
                  key=lambda c: c.close_time)
           if p5.exists() else _aggregate_5m(c1m))
    return c1m, sorted(c1m + c5m, key=lambda c: c.close_time)


# ---------------------------------------------------------------------------
# Oracle direction: first side to reach 2×ATR within the study horizon
# ---------------------------------------------------------------------------

def _oracle_direction(i: int, c1m: list[Candle], atr: float,
                       horizon: int, oracle_mult: float = 2.0) -> int:
    """Return +1 if upside target reached first, -1 if downside, 0 if neither."""
    entry = c1m[i].close
    up_t  = entry + oracle_mult * atr
    dn_t  = entry - oracle_mult * atr
    n     = len(c1m)
    for bar in c1m[i + 1 : min(i + horizon + 1, n)]:
        up_hit = bar.high >= up_t
        dn_hit = bar.low  <= dn_t
        if up_hit and dn_hit:
            return 0    # same-bar, ambiguous → conservative: treat as neither
        if up_hit:
            return +1
        if dn_hit:
            return -1
    return 0   # neither within horizon → no oracle signal


# ---------------------------------------------------------------------------
# Entry collection
# ---------------------------------------------------------------------------

def collect_entries(c1m: list[Candle], merged: list[Candle],
                    horizon: int) -> list[TradeEntry]:
    idx_by_close: dict[int, int] = {c.close_time: i for i, c in enumerate(c1m)}
    strategy = MomentumStrategy()
    entries: list[TradeEntry] = []
    for c in merged:
        sig = strategy.on_candle(c)
        if sig is None or c.timeframe != "1m":
            continue
        i = idx_by_close.get(c.close_time, -1)
        if i < 16 or i + horizon + 1 >= len(c1m):
            continue
        d_sig    = 1 if sig.direction.value == "long" else -1
        d_oracle = _oracle_direction(i, c1m, sig.atr, horizon)
        entries.append(TradeEntry(
            bar_idx=i,
            close_at_entry=c.close,
            atr=sig.atr,
            signal_direction=d_sig,
            oracle_direction=d_oracle,
        ))
    return entries


# ---------------------------------------------------------------------------
# First-touch evaluation (core loop)
# ---------------------------------------------------------------------------

def _first_touch(entry_price: float, atr: float, direction: int,
                  bars: list[Candle],
                  stop_mult: float, tp_mult: float,
                  ) -> tuple[str, float]:
    """Walk bars to find first touch. Returns (outcome, force_close_return_R).

    outcome ∈ {"tp", "stop", "neither"}
    force_close_return_R is non-zero only for "neither".
    """
    if atr <= 0 or direction == 0:
        return "neither", 0.0

    tp_target   = entry_price + direction * tp_mult   * atr
    stop_target = entry_price - direction * stop_mult * atr

    for bar in bars:
        tp_hit   = (direction == +1 and bar.high >= tp_target) or \
                   (direction == -1 and bar.low  <= tp_target)
        stop_hit = (direction == +1 and bar.low  <= stop_target) or \
                   (direction == -1 and bar.high >= stop_target)

        if stop_hit:             # stop priority (conservative)
            return "stop", 0.0
        if tp_hit:
            return "tp", 0.0

    # Neither: force-close at last bar close
    last_close = bars[-1].close if bars else entry_price
    fwd_ret_r  = direction * (last_close - entry_price) / (stop_mult * atr)
    return "neither", fwd_ret_r


def _fee_in_r(entry_price: float, atr: float, stop_mult: float,
               outcome: str) -> float:
    """Fee drag in R units for one trade leg."""
    if atr <= 0 or stop_mult <= 0:
        return 0.0
    stop_dist = stop_mult * atr
    entry_fee = TAKER * entry_price
    exit_fee  = (MAKER if outcome == "tp" else TAKER) * entry_price
    return (entry_fee + exit_fee) / stop_dist


# ---------------------------------------------------------------------------
# Compute the full results grid
# ---------------------------------------------------------------------------

def compute_grid(entries: list[TradeEntry],
                 c1m: list[Candle],
                 horizon: int,
                 rng_seed: int = 42,
                 ) -> dict[tuple[str, float, float], CellResult]:
    """Returns {(mode, stop_mult, tp_mult): CellResult}."""
    rng = random.Random(rng_seed)
    n   = len(c1m)

    results: dict[tuple[str, float, float], CellResult] = {}
    for mode in MODES:
        for sm in STOP_MULTS:
            for tm in TP_MULTS:
                results[(mode, sm, tm)] = CellResult(0, 0, 0, 0, 0.0, 0.0)

    for e in entries:
        i     = e.bar_idx
        bars  = c1m[i + 1 : min(i + horizon + 1, n)]
        if not bars:
            continue
        ep    = e.close_at_entry
        atr   = e.atr

        d_map = {
            MODE_SIGNAL:   e.signal_direction,
            MODE_INVERTED: -e.signal_direction,
            MODE_ORACLE:   e.oracle_direction if e.oracle_direction != 0 else e.signal_direction,
            MODE_RANDOM:   rng.choice([+1, -1]),
        }

        for mode, direction in d_map.items():
            for sm in STOP_MULTS:
                for tm in TP_MULTS:
                    outcome, fc_r = _first_touch(ep, atr, direction, bars, sm, tm)
                    fee_r = _fee_in_r(ep, atr, sm, outcome)
                    cell  = results[(mode, sm, tm)]
                    cell.n += 1
                    if outcome == "tp":
                        cell.n_tp   += 1
                    elif outcome == "stop":
                        cell.n_stop += 1
                    else:
                        cell.n_neither  += 1
                        cell.sum_neither_r += fc_r
                    cell.sum_fees_r += fee_r

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _ef(v: float, dp: int = 3) -> str:
    """Format expectancy value."""
    if math.isnan(v) or math.isinf(v):
        return "   —   "
    return f"{v:>+{dp+5}.{dp}f}"


def _pf(v: float) -> str:
    """Format fraction as percent."""
    if math.isnan(v):
        return "  —  "
    return f"{v*100:.1f}%"


def _print_matrix(results: dict[tuple[str, float, float], CellResult],
                   mode: str, metric: str,
                   stop_mults: list[float] = STOP_MULTS,
                   tp_mults:   list[float] = TP_MULTS) -> None:
    """Print a stop × TP matrix for one mode and one metric."""
    col_w = 10
    tp_labels = "".join(f"  {'TP='+str(tm)+'x':>{col_w}}" for tm in tp_mults)
    header = f"    {'Stop / TP':<14}" + tp_labels
    print(header)
    print("    " + "─" * (14 + (col_w + 2) * len(tp_mults)))
    for sm in stop_mults:
        row = f"    stop={sm}×ATR    "
        for tm in tp_mults:
            cell = results.get((mode, sm, tm))
            if cell is None:
                row += f"  {'—':>{col_w}}"
                continue
            if metric == "e_nofee":
                v = cell.expectancy_nofee(sm, tm)
                row += f"  {_ef(v):>{col_w}}"
            elif metric == "e_fee":
                v = cell.expectancy_fee(sm, tm)
                row += f"  {_ef(v):>{col_w}}"
            elif metric == "p_tp":
                row += f"  {_pf(cell.p_tp):>{col_w}}"
            elif metric == "p_stop":
                row += f"  {_pf(cell.p_stop):>{col_w}}"
        print(row)


def print_report(entries: list[TradeEntry],
                 results: dict[tuple[str, float, float], CellResult],
                 c1m: list[Candle],
                 symbol: str,
                 horizon: int) -> None:

    period_start = datetime.fromtimestamp(c1m[0].close_time,  tz=timezone.utc).strftime("%Y-%m-%d")
    period_end   = datetime.fromtimestamp(c1m[-1].close_time, tz=timezone.utc).strftime("%Y-%m-%d")

    oracle_valid = sum(1 for e in entries if e.oracle_direction != 0)
    pct_dir_agree = (sum(1 for e in entries
                         if e.oracle_direction != 0
                         and e.signal_direction == e.oracle_direction)
                     / oracle_valid * 100) if oracle_valid else 0.0

    print()
    print("═" * _W)
    print(f"  FIRST-TOUCH MATRIX — {symbol}")
    print(f"  Period: {period_start} → {period_end}   ({len(c1m):,} bars)")
    print(f"  Signal entries: {len(entries):,}   Max horizon: {horizon} bars")
    print(f"  Oracle entries with clear direction: {oracle_valid:,}  "
          f"({oracle_valid/len(entries)*100:.0f}%)")
    print(f"  Signal agrees with oracle: {pct_dir_agree:.1f}% of oracle-valid entries")
    print(f"  Fees: entry={TAKER*100:.3f}% taker  |  TP exit={MAKER*100:.3f}% maker  "
          f"|  stop exit={TAKER*100:.3f}% taker")
    print("═" * _W)

    # -----------------------------------------------------------------------
    # Section 1: Win rate (P_TP) matrix — all four modes
    # -----------------------------------------------------------------------
    print()
    print("  1. P(TP TOUCHED FIRST)  — win rate matrix by stop × TP distance")
    print("     50% = coin flip.  > 50% = edge in this direction at these levels.")
    print()
    for mode in MODES:
        lbl = {"signal": "SIGNAL direction", "inverted": "INVERTED direction",
               "oracle": "ORACLE direction (hindsight best)", "random": "RANDOM direction"}[mode]
        print(f"  {lbl}:")
        _print_matrix(results, mode, "p_tp")
        print()

    # -----------------------------------------------------------------------
    # Section 2: Expectancy — zero fees
    # -----------------------------------------------------------------------
    print()
    print("  2. EXPECTANCY IN R  (zero fees, force-close at horizon)")
    print("     Positive = profitable even before fees.")
    print("     This is the ceiling; fees will reduce every number.")
    print()
    for mode in MODES:
        lbl = {"signal": "SIGNAL direction", "inverted": "INVERTED direction",
               "oracle": "ORACLE direction (theoretical maximum)", "random": "RANDOM direction"}[mode]
        print(f"  {lbl}:")
        _print_matrix(results, mode, "e_nofee")
        print()

    # -----------------------------------------------------------------------
    # Section 3: Expectancy — with fees
    # -----------------------------------------------------------------------
    print()
    print("  3. EXPECTANCY IN R  (with fees: taker entry, maker/taker exit)")
    print("     This is the real achievable expectancy.")
    print()
    for mode in MODES:
        lbl = {"signal": "SIGNAL direction", "inverted": "INVERTED direction",
               "oracle": "ORACLE direction (fee-adjusted theoretical max)", "random": "RANDOM direction"}[mode]
        print(f"  {lbl}:")
        _print_matrix(results, mode, "e_fee")
        print()

    # -----------------------------------------------------------------------
    # Section 4: Leak attribution table
    # -----------------------------------------------------------------------
    print()
    print("  4. EDGE LEAK ATTRIBUTION")
    print("     Measures how much each factor destroys vs the oracle ceiling.")
    print()

    # Best cells across all (stop, tp) combinations
    def best_e(mode: str, fee: bool) -> tuple[float, float, float]:
        """Return (best_expectancy, best_stop, best_tp)."""
        best = float("-inf")
        bs, bt = float("nan"), float("nan")
        for sm in STOP_MULTS:
            for tm in TP_MULTS:
                cell = results.get((mode, sm, tm))
                if cell is None:
                    continue
                e = cell.expectancy_fee(sm, tm) if fee else cell.expectancy_nofee(sm, tm)
                if not math.isnan(e) and e > best:
                    best, bs, bt = e, sm, tm
        return best, bs, bt

    oracle_best_nofee, obs, obt = best_e(MODE_ORACLE,   fee=False)
    oracle_best_fee,   _,   _   = best_e(MODE_ORACLE,   fee=True)
    signal_best_fee,   sbs, sbt = best_e(MODE_SIGNAL,   fee=True)
    inv_best_fee,      ibs, ibt = best_e(MODE_INVERTED, fee=True)
    ran_best_fee,      _,   _   = best_e(MODE_RANDOM,   fee=True)

    # Fee cost = oracle_nofee − oracle_fee (at same cell)
    ocell = results.get((MODE_ORACLE, obs, obt))
    oracle_at_best_fee = ocell.expectancy_fee(obs, obt) if ocell else float("nan")
    fee_cost = oracle_best_nofee - oracle_at_best_fee

    # Direction cost = oracle_fee − signal_fee (at their own best cells)
    direction_cost = oracle_best_fee - signal_best_fee

    # Stop cost: for SIGNAL mode, difference between best and worst stop at fixed TP
    def stop_range(mode: str, tp_mult: float) -> float:
        vals = [results[(mode, sm, tp_mult)].expectancy_fee(sm, tp_mult)
                for sm in STOP_MULTS
                if (mode, sm, tp_mult) in results]
        vals = [v for v in vals if not math.isnan(v)]
        return max(vals) - min(vals) if len(vals) >= 2 else float("nan")

    def tp_range(mode: str, stop_mult: float) -> float:
        vals = [results[(mode, stop_mult, tm)].expectancy_fee(stop_mult, tm)
                for tm in TP_MULTS
                if (mode, stop_mult, tm) in results]
        vals = [v for v in vals if not math.isnan(v)]
        return max(vals) - min(vals) if len(vals) >= 2 else float("nan")

    # Use signal mode to measure stop and TP sensitivity
    stop_sens  = max((stop_range(MODE_SIGNAL, tm) for tm in TP_MULTS
                      if not math.isnan(stop_range(MODE_SIGNAL, tm))), default=float("nan"))
    tp_sens    = max((tp_range(MODE_SIGNAL, sm) for sm in STOP_MULTS
                      if not math.isnan(tp_range(MODE_SIGNAL, sm))), default=float("nan"))
    inv_edge   = inv_best_fee - signal_best_fee  # how much inversion helps

    print(f"  Oracle ceiling (no fees, best cell stop={obs}×, TP={obt}×):  "
          f"{_ef(oracle_best_nofee)}")
    print(f"  Oracle ceiling (with fees, same cell):                       "
          f"{_ef(oracle_at_best_fee)}")
    print(f"  Signal best (with fees, stop={sbs}×, TP={sbt}×):             "
          f"{_ef(signal_best_fee)}")
    print(f"  Inverted best (with fees, stop={ibs}×, TP={ibt}×):           "
          f"{_ef(inv_best_fee)}")
    print(f"  Random best (with fees):                                     "
          f"{_ef(ran_best_fee)}")
    print()
    print("  ┌───────────────────────────────────────────────────────────────────┐")
    print(f"  │ A) Direction cost    oracle_fee − signal_best :  {_ef(direction_cost):>10}  R │")
    print(f"  │    (inversion gain   inv_best − signal_best):    {_ef(inv_edge):>10}  R │")
    print(f"  │ B) Stop sensitivity  max E-range across TPs  :  {_ef(stop_sens):>10}  R │")
    print(f"  │ C) TP sensitivity    max E-range across stops:  {_ef(tp_sens):>10}  R │")
    print(f"  │ D) Fee cost          oracle_nofee − oracle_fee:  {_ef(fee_cost):>10}  R │")
    print(f"  │ E) Residual ceiling  oracle_fee (best):          {_ef(oracle_best_fee):>10}  R │")
    print("  └───────────────────────────────────────────────────────────────────┘")
    print()
    print("  Note: A+B+C+D do not sum to oracle ceiling — they are independent")
    print("  sensitivities, not additive decomposition. Residual (E) is the")
    print("  theoretical max if you could solve A+B+C+D perfectly.")

    # -----------------------------------------------------------------------
    # Section 5: Signal direction agreement with oracle by year
    # -----------------------------------------------------------------------
    print()
    print("  5. SIGNAL DIRECTION AGREEMENT WITH ORACLE  (year-by-year)")
    print("     Shows whether the directional signal was ever useful.")
    print()
    by_year: dict[int, list[TradeEntry]] = defaultdict(list)
    for e in entries:
        dt = datetime.fromtimestamp(c1m[e.bar_idx].close_time, tz=timezone.utc)
        by_year[dt.year].append(e)

    print(f"  {'Year':<8}  {'N':>6}  {'Oracle=±':>10}  {'Sig=oracle':>12}  "
          f"{'Best sig E':>12}  {'Best ora E':>12}")
    print("  " + "─" * 66)
    for yr in sorted(by_year):
        yr_ents  = by_year[yr]
        valid    = [e for e in yr_ents if e.oracle_direction != 0]
        agree    = sum(1 for e in valid if e.signal_direction == e.oracle_direction)
        agree_pct = agree / len(valid) * 100 if valid else float("nan")
        flag = " ▲" if agree_pct > 52 else (" ▼" if agree_pct < 48 else "")

        # Approximate per-year best expectancy: use the 1.5×stop / 2×TP cell
        def yr_cell_e(mode: str, fee: bool) -> float:
            sm, tm = 1.5, 2.0
            yr_sub = [e for e in yr_ents
                      if e.oracle_direction != 0 or mode != MODE_ORACLE]
            n_local = len(yr_sub)
            if n_local == 0:
                return float("nan")
            n_tp = n_stop = n_nei = 0
            sum_fc = sum_fee = 0.0
            for e in yr_sub:
                i     = e.bar_idx
                bars  = c1m[i + 1 : min(i + horizon + 1, len(c1m))]
                d     = (e.signal_direction if mode == MODE_SIGNAL
                         else (e.oracle_direction if e.oracle_direction != 0
                               else e.signal_direction))
                ep, atr_ = e.close_at_entry, e.atr
                oc, fc_r = _first_touch(ep, atr_, d, bars, sm, tm)
                fee_r = _fee_in_r(ep, atr_, sm, oc)
                if oc == "tp":    n_tp   += 1
                elif oc == "stop": n_stop += 1
                else:              n_nei += 1; sum_fc += fc_r
                sum_fee += fee_r
            p_tp_  = n_tp   / n_local
            p_stp_ = n_stop / n_local
            e_base = p_tp_ * (tm / sm) + p_stp_ * (-1.0) + sum_fc / n_local
            return (e_base - sum_fee / n_local) if fee else e_base

        sig_e = yr_cell_e(MODE_SIGNAL,  fee=True)
        ora_e = yr_cell_e(MODE_ORACLE,  fee=True)
        print(f"  {yr:<8}  {len(yr_ents):>6,}  {len(valid):>9,}  "
              f"{agree_pct:>11.1f}%  {_ef(sig_e):>12}  {_ef(ora_e):>12}{flag}")

    # -----------------------------------------------------------------------
    # Section 6: P(TP) vs P(stop) for signal entries at key cells
    # -----------------------------------------------------------------------
    print()
    print("  6. DETAILED WIN/STOP/NEITHER BREAKDOWN  (key cells, signal direction)")
    print()
    key_cells = [(1.5, 1.0), (1.5, 2.0), (1.5, 3.0), (2.0, 2.0), (2.0, 3.0)]
    print(f"  {'(stop, TP)':<16}  {'P(TP)':>8}  {'P(stop)':>8}  {'P(nei)':>8}  "
          f"{'E(R) 0fee':>12}  {'E(R) fee':>12}  {'oracle E':>12}")
    print("  " + "─" * 84)
    for sm, tm in key_cells:
        cs = results.get((MODE_SIGNAL, sm, tm))
        co = results.get((MODE_ORACLE, sm, tm))
        if cs is None:
            continue
        print(f"  ({sm}×, {tm}×)           "
              f"  {_pf(cs.p_tp):>8}  {_pf(cs.p_stop):>8}  {_pf(cs.p_neither):>8}"
              f"  {_ef(cs.expectancy_nofee(sm, tm)):>12}"
              f"  {_ef(cs.expectancy_fee(sm, tm)):>12}"
              f"  {_ef(co.expectancy_fee(sm, tm) if co else float('nan')):>12}")

    # -----------------------------------------------------------------------
    # Section 7: Verdict
    # -----------------------------------------------------------------------
    print()
    print("═" * _W)
    print("  VERDICT: WHERE DOES THE EDGE LEAK?")
    print("═" * _W)
    print()

    # Label each factor
    factors: list[tuple[str, float, str]] = []

    factors.append(("A  Direction",
                     direction_cost,
                     f"Signal agrees with oracle {pct_dir_agree:.1f}% of the time. "
                     f"Inversion gains {_ef(inv_edge)} R over signal."))

    factors.append(("B  Stop placement",
                     stop_sens,
                     f"Max expectancy range across stop distances: {_ef(stop_sens)} R."))

    factors.append(("C  TP placement",
                     tp_sens,
                     f"Max expectancy range across TP distances: {_ef(tp_sens)} R."))

    factors.append(("D  Fees",
                     fee_cost,
                     f"Fee drag on oracle at best cell: {_ef(fee_cost)} R."))

    factors.append(("E  Residual ceiling",
                     oracle_best_fee,
                     f"Best achievable E(R) with hindsight direction + optimal levels."))

    factors.sort(key=lambda x: abs(x[1]) if not math.isnan(x[1]) else 0, reverse=True)

    print("  Ranked by magnitude:")
    print()
    for label, mag, desc in factors:
        bar_len = max(0, min(30, int(abs(mag) * 15))) if not math.isnan(mag) else 0
        bar_str = "█" * bar_len
        sign    = "+" if not math.isnan(mag) and mag > 0 else ""
        print(f"  {label:<22}  {sign}{_ef(mag)} R  {bar_str}")
        print(f"    {desc}")
        print()

    print()
    # Overall conclusion
    any_positive = oracle_best_fee > 0.0

    if not any_positive:
        print("  CONCLUSION: NO RECOVERABLE EDGE UNDER ANY COMBINATION")
        print()
        print("  Even with hindsight-optimal direction, the oracle ceiling is negative.")
        print("  This means the volatility edge identified in the straddle audit cannot")
        print("  be monetized by a stop/TP structure at any tested level.")
        print()
        print("  The fundamental constraint is the stop-takes-priority rule: large")
        print("  volatility bars move in both directions, and the stop is hit on the")
        print("  losing leg before the TP is reached on the winning leg.")
    elif signal_best_fee > 0.0:
        print("  CONCLUSION: EDGE IS PRESENT AND PARTIALLY CAPTURED")
        print()
        print(f"  Best signal expectancy {_ef(signal_best_fee)} R at "
              f"stop={sbs}×, TP={sbt}×.")
        print(f"  Gap to oracle ceiling: {_ef(oracle_best_fee - signal_best_fee)} R")
        print("  Reducing this gap requires improving direction accuracy.")
    else:
        print("  CONCLUSION: EDGE EXISTS IN ORACLE BUT NOT IN SIGNAL")
        print()
        print(f"  Oracle ceiling: {_ef(oracle_best_fee)} R (with fees)")
        print(f"  Signal best:    {_ef(signal_best_fee)} R (with fees)")
        print()
        gap = oracle_best_fee - signal_best_fee
        if direction_cost > fee_cost and direction_cost > stop_sens:
            print("  PRIMARY LEAK: DIRECTION (A)")
            print("  The volatility edge is real but the directional signal is")
            print("  anti-predictive. The strategy enters in the wrong direction")
            print(f"  approximately {100-pct_dir_agree:.0f}% of the time on trades")
            print("  where a clear oracle direction exists.")
        elif fee_cost > direction_cost and fee_cost > stop_sens:
            print("  PRIMARY LEAK: FEES (D)")
            print("  Even with correct direction, fee drag consumes most of the edge.")
            print("  Maker entries and exits would recover this.")
        else:
            print("  PRIMARY LEAK: STOP/TP PLACEMENT (B/C)")
            print("  Stop distances and TP targets are mis-calibrated for the")
            print("  volatility regime the signal operates in.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="First-touch matrix: quantify edge leakage")
    p.add_argument("--symbol",     default="ETHUSDT")
    p.add_argument("--candle-dir", default="data/candles", type=Path)
    p.add_argument("--start",      default="2023-01", help="YYYY-MM")
    p.add_argument("--end",        default=None,      help="YYYY-MM")
    p.add_argument("--horizon",    default=50,        type=int,
                   help="Max bars before force-close (default 50)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    candle_dir = _REPO_ROOT / args.candle_dir
    if not candle_dir.exists():
        candle_dir = Path(args.candle_dir)

    sy, sm = [int(x) for x in args.start.split("-")]
    if args.end:
        ey, em = [int(x) for x in args.end.split("-")]
    else:
        ey, em = 2099, 12

    from calendar import monthrange
    _, last_day = monthrange(min(ey, 2099), min(em, 12))
    start_s = int(datetime(sy, sm, 1, tzinfo=timezone.utc).timestamp())
    end_s   = int(datetime(min(ey, 2099), min(em, 12), last_day,
                            23, 59, 59, tzinfo=timezone.utc).timestamp())

    print(f"\nFirst-touch matrix — {args.symbol}  horizon={args.horizon} bars")
    print(f"Loading candles from {candle_dir} …")
    c1m, merged = _load(candle_dir, args.symbol, start_s, end_s)
    print(f"  {len(c1m):,} × 1m bars")

    if len(c1m) < 1000:
        print("[ERROR] Too few bars.")
        sys.exit(1)

    print("Collecting signal entries + oracle directions …")
    entries = collect_entries(c1m, merged, args.horizon)
    print(f"  {len(entries):,} entries collected")

    if len(entries) == 0:
        print("[ERROR] Zero entries.")
        sys.exit(1)

    print("Computing first-touch grid (all modes × stops × TPs) …")
    results = compute_grid(entries, c1m, args.horizon)

    print_report(entries, results, c1m, args.symbol, args.horizon)


if __name__ == "__main__":
    main()
