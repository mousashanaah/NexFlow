#!/usr/bin/env python3
"""Signal quality study.

Ranks every trade by composite signal strength, bins into quintiles,
and measures PnL / PF / WR / MFE / MAE per bin.

Signal strength score (0–100) is built from five normalised sub-scores:
    1. rel_vol            — higher is stronger
    2. range_expansion    — higher is stronger
    3. buy_sell_imbalance — distance from 0.5 (more directional = stronger)
    4. spread_regime      — lower is stronger (cheaper to cross)
    5. momentum_5m        — absolute magnitude (stronger trend = stronger)

Breakout distance (entry vs rolling high/low / ATR) is computed from
journal fields and reported per trade but not included in the composite
(it's a consequence of the entry, not a predictor of quality).

If the top quintile contains most of the edge, a selective filter table
is printed showing projected metrics at each selectivity cutoff.

MFE / MAE are approximated from candle data (parquet or Bitget REST).

Usage:
    python scripts/signal_quality_study.py
    python scripts/signal_quality_study.py --journal-dir logs/paper --candle-dir data/candles
    python scripts/signal_quality_study.py --verbose
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    import pyarrow.parquet as pq
    _HAS_PARQUET = True
except ImportError:
    _HAS_PARQUET = False

TAKER = 0.0006
MAKER = 0.0002


# ---------------------------------------------------------------------------
# Bar struct
# ---------------------------------------------------------------------------

class Bar(NamedTuple):
    ts_ms: int
    open: float
    high: float
    low: float
    close: float


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    # Identity
    trade_num: int
    symbol: str
    side: str
    entry_time: float        # epoch seconds
    exit_time: float

    # Prices
    entry_price: float
    exit_price: float        # weighted average

    # Sizing
    size: float
    entry_fee: float

    # PnL (reconstructed from journal, same method as trade_audit.py)
    gross_pnl: float
    total_fees: float
    net_pnl: float
    exit_reason: str

    # Raw signal features (from SIGNAL event)
    atr: float
    rel_vol: float
    range_expansion: float
    spread_regime: float
    buy_sell_imbalance: float
    momentum_5m: float
    rolling_high: float
    rolling_low: float
    vwap_deviation: float

    # Derived
    breakout_dist_r: float   # abs(entry - rolling_level) / atr
    imbalance_strength: float  # abs(buy_sell_imbalance - 0.5) * 2  → [0, 1]
    momentum_abs: float        # abs(momentum_5m)

    # Composite score (set after normalisation)
    signal_score: float = 0.0
    quintile: int = 0          # 1=top 20%, 5=bottom 20%

    # MFE / MAE (set from candle scan)
    mfe_price: float = 0.0     # max price move in favour
    mae_price: float = 0.0     # max adverse price move
    mfe_r: float = 0.0         # MFE in ATR units
    mae_r: float = 0.0         # MAE in ATR units


# ---------------------------------------------------------------------------
# Journal parsing — reconstruct closed trades with full signal features
# ---------------------------------------------------------------------------

def _load_events(journal_dir: Path) -> list[dict]:
    evts: list[dict] = []
    for path in sorted(journal_dir.glob("*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    evts.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    evts.sort(key=lambda e: e.get("ts_epoch", 0.0))
    return evts


def reconstruct_trades(events: list[dict]) -> list[TradeRecord]:
    last_signal: dict[str, dict] = {}
    open_fills: dict[str, dict] = {}
    open_fees: dict[str, float] = {}    # symbol → accumulated fees
    open_partials: dict[str, list[dict]] = defaultdict(list)
    entry_sides: dict[str, str] = {}

    records: list[TradeRecord] = []
    trade_num = 0

    for ev in events:
        evt = ev.get("event", "")
        sym = ev.get("symbol", "")

        if evt == "SIGNAL":
            last_signal[sym] = ev

        elif evt == "FILL":
            trade_num += 1
            open_fills[sym] = ev
            open_fees[sym] = ev.get("fee", 0.0)
            open_partials[sym] = []
            entry_sides[sym] = ev.get("direction", "long")

        elif evt == "PARTIAL_TP":
            if sym in open_fills:
                open_fees[sym] += ev.get("fee", 0.0)
                open_partials[sym].append(ev)

        elif evt in ("STOP_HIT", "FORCE_CLOSE"):
            fill_ev = open_fills.pop(sym, None)
            if fill_ev is None:
                continue

            sig = last_signal.get(sym, {})
            feats = sig.get("features", {})
            atr = feats.get("atr", sig.get("atr", 0.0))
            if atr <= 0:
                open_fees.pop(sym, None)
                open_partials.pop(sym, [])
                continue

            side = entry_sides.get(sym, "long")
            is_long = side == "long"
            sign = 1.0 if is_long else -1.0
            entry_price = fill_ev.get("fill_price", 0.0)
            entry_fee = fill_ev.get("fee", 0.0)
            entry_size = fill_ev.get("size", 0.0)

            # Reconstruct PnL
            partials = open_partials.pop(sym, [])
            partial_gross = sum(
                (p["fill_price"] - entry_price) * p["size"] * sign
                for p in partials
            )
            partial_size = sum(p["size"] for p in partials)
            remaining = entry_size - partial_size

            if evt == "STOP_HIT":
                exit_price = ev.get("fill_price", 0.0)
                stop_fee = ev.get("fee", 0.0)
                stop_gross = (exit_price - entry_price) * remaining * sign
            else:
                exit_price = ev.get("price", 0.0)
                stop_fee = 0.0
                stop_gross = (exit_price - entry_price) * remaining * sign

            gross = partial_gross + stop_gross
            total_fees = open_fees.pop(sym, 0.0) + stop_fee
            net = gross - total_fees

            # Weighted avg exit price
            pv = sum(p["fill_price"] * p["size"] for p in partials) + exit_price * remaining
            total_closed = partial_size + remaining
            wavg_exit = pv / total_closed if total_closed > 0 else exit_price

            # Feature extraction
            rel_vol = feats.get("rel_vol", 0.0)
            range_exp = feats.get("range_expansion", 0.0)
            spread_reg = feats.get("spread_regime", 0.0)
            imbalance = feats.get("buy_sell_imbalance", 0.5)
            momentum = feats.get("momentum_5m", 0.0)
            rolling_high = feats.get("rolling_high", entry_price)
            rolling_low = feats.get("rolling_low", entry_price)
            vwap_dev = feats.get("vwap_deviation", 0.0)

            # Breakout distance: how far past the rolling level did price break?
            if is_long:
                breakout_dist_r = (entry_price - rolling_high) / atr if atr > 0 else 0.0
            else:
                breakout_dist_r = (rolling_low - entry_price) / atr if atr > 0 else 0.0

            records.append(TradeRecord(
                trade_num=trade_num,
                symbol=sym,
                side=side,
                entry_time=fill_ev.get("ts_epoch", 0.0),
                exit_time=ev.get("ts_epoch", 0.0),
                entry_price=entry_price,
                exit_price=round(wavg_exit, 6),
                size=entry_size,
                entry_fee=entry_fee,
                gross_pnl=round(gross, 4),
                total_fees=round(total_fees, 4),
                net_pnl=round(net, 4),
                exit_reason=evt,
                atr=atr,
                rel_vol=rel_vol,
                range_expansion=range_exp,
                spread_regime=spread_reg,
                buy_sell_imbalance=imbalance,
                momentum_5m=momentum,
                rolling_high=rolling_high,
                rolling_low=rolling_low,
                vwap_deviation=vwap_dev,
                breakout_dist_r=round(max(0.0, breakout_dist_r), 4),
                imbalance_strength=round(abs(imbalance - 0.5) * 2.0, 4),
                momentum_abs=round(abs(momentum), 6),
            ))

    return records


# ---------------------------------------------------------------------------
# Signal scoring
# ---------------------------------------------------------------------------

def _percentile_rank(value: float, all_values: list[float]) -> float:
    """Return value's rank within all_values as a fraction 0–1."""
    if not all_values:
        return 0.5
    below = sum(1 for v in all_values if v < value)
    return below / len(all_values)


def score_signals(records: list[TradeRecord]) -> None:
    """Compute normalised composite score (0–100) and quintile for each record."""
    if not records:
        return

    rel_vols     = [r.rel_vol for r in records]
    range_exps   = [r.range_expansion for r in records]
    spreads      = [r.spread_regime for r in records]
    imbalances   = [r.imbalance_strength for r in records]
    momenta      = [r.momentum_abs for r in records]

    for r in records:
        # Higher = better for all except spread (lower spread = better)
        s_rv  = _percentile_rank(r.rel_vol, rel_vols)
        s_re  = _percentile_rank(r.range_expansion, range_exps)
        s_sp  = 1.0 - _percentile_rank(r.spread_regime, spreads)  # invert
        s_im  = _percentile_rank(r.imbalance_strength, imbalances)
        s_mo  = _percentile_rank(r.momentum_abs, momenta)

        # Equal weights across five dimensions
        composite = (s_rv + s_re + s_sp + s_im + s_mo) / 5.0
        r.signal_score = round(composite * 100, 2)

    # Sort descending, assign quintiles 1=top … 5=bottom
    sorted_scores = sorted((r.signal_score for r in records), reverse=True)
    n = len(sorted_scores)
    cutoffs = [sorted_scores[int(n * q / 5)] for q in range(1, 5)]
    # cutoffs[0]=80th pct, cutoffs[1]=60th, cutoffs[2]=40th, cutoffs[3]=20th

    for r in records:
        if r.signal_score >= cutoffs[0]:
            r.quintile = 1
        elif r.signal_score >= cutoffs[1]:
            r.quintile = 2
        elif r.signal_score >= cutoffs[2]:
            r.quintile = 3
        elif r.signal_score >= cutoffs[3]:
            r.quintile = 4
        else:
            r.quintile = 5


# ---------------------------------------------------------------------------
# MFE / MAE via candles
# ---------------------------------------------------------------------------

def _load_parquet(candle_dir: Path, symbol: str) -> list[Bar]:
    if not _HAS_PARQUET:
        return []
    path = candle_dir / f"{symbol}_1m.parquet"
    if not path.exists():
        return []
    rows = pq.read_table(path).to_pylist()
    bars = [
        Bar(ts_ms=r["close_time"], open=r["open"], high=r["high"],
            low=r["low"], close=r["close"])
        for r in rows if r.get("is_final")
    ]
    return sorted(bars, key=lambda b: b.ts_ms)


def _fetch_rest(symbol: str, start_ms: int, end_ms: int) -> list[Bar]:
    url = (
        f"https://api.bitget.com/api/v2/mix/market/candles"
        f"?symbol={symbol}&productType=USDT-FUTURES&granularity=1m"
        f"&startTime={start_ms}&endTime={end_ms}&limit=1000"
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read())
        if data.get("code") != "00000":
            return []
        return sorted([
            Bar(ts_ms=int(r[0]), open=float(r[1]), high=float(r[2]),
                low=float(r[3]), close=float(r[4]))
            for r in data.get("data", [])
        ], key=lambda b: b.ts_ms)
    except Exception as exc:
        print(f"  [WARN] REST fetch failed for {symbol}: {exc}")
        return []


def compute_mfe_mae(
    records: list[TradeRecord],
    candle_dir: Path,
    no_rest: bool,
) -> None:
    """Fill in mfe_price / mae_price / mfe_r / mae_r for each record."""
    parquet_cache: dict[str, list[Bar]] = {}
    rest_cache: dict[str, list[Bar]] = {}

    for r in records:
        sym = r.symbol
        if sym not in parquet_cache:
            parquet_cache[sym] = _load_parquet(candle_dir, sym)

        start_ms = int(r.entry_time * 1000)
        end_ms   = int(r.exit_time * 1000) + 60_000

        window = [b for b in parquet_cache[sym] if start_ms <= b.ts_ms <= end_ms]
        if not window and not no_rest:
            key = f"{sym}_{start_ms}_{end_ms}"
            if key not in rest_cache:
                rest_cache[key] = _fetch_rest(sym, start_ms, end_ms)
                time.sleep(0.15)
            window = rest_cache[key]

        if not window:
            continue

        is_long = r.side == "long"
        sign = 1.0 if is_long else -1.0
        mfe = r.entry_price
        mae = r.entry_price

        for bar in window:
            if is_long:
                mfe = max(mfe, bar.high)
                mae = min(mae, bar.low)
            else:
                mfe = min(mfe, bar.low)
                mae = max(mae, bar.high)

        mfe_price = abs(mfe - r.entry_price)
        mae_price = abs(mae - r.entry_price)
        r.mfe_price = round(mfe_price, 6)
        r.mae_price = round(mae_price, 6)
        r.mfe_r = round(mfe_price / r.atr, 3) if r.atr > 0 else 0.0
        r.mae_r = round(mae_price / r.atr, 3) if r.atr > 0 else 0.0


# ---------------------------------------------------------------------------
# Per-bin metrics
# ---------------------------------------------------------------------------

def _bin_metrics(recs: list[TradeRecord]) -> dict:
    if not recs:
        return {
            "count": 0, "gross": 0.0, "fees": 0.0, "net": 0.0,
            "pf": 0.0, "wr": 0.0,
            "avg_mfe_r": 0.0, "avg_mae_r": 0.0,
            "avg_score": 0.0,
        }
    wins   = [r for r in recs if r.net_pnl > 0]
    losses = [r for r in recs if r.net_pnl <= 0]
    gp = sum(r.net_pnl for r in wins)
    gl = sum(abs(r.net_pnl) for r in losses)
    gross = sum(r.gross_pnl for r in recs)
    fees  = sum(r.total_fees for r in recs)
    net   = sum(r.net_pnl for r in recs)
    pf    = gp / gl if gl > 0 else (float("inf") if gp > 0 else 0.0)
    wr    = len(wins) / len(recs)
    avg_mfe_r = sum(r.mfe_r for r in recs) / len(recs)
    avg_mae_r = sum(r.mae_r for r in recs) / len(recs)
    avg_score = sum(r.signal_score for r in recs) / len(recs)
    return {
        "count": len(recs), "gross": gross, "fees": fees, "net": net,
        "pf": pf, "wr": wr,
        "avg_mfe_r": avg_mfe_r, "avg_mae_r": avg_mae_r,
        "avg_score": avg_score,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

_QNAMES = {1: "Top 20%", 2: "20–40%", 3: "40–60%", 4: "60–80%", 5: "Bot 20%"}

_GREEN = "\033[92m"
_RED   = "\033[91m"
_BOLD  = "\033[1m"
_RST   = "\033[0m"


def _c(v: float) -> str:
    return _GREEN if v > 0 else (_RED if v < 0 else "")


def _print_quintile_table(bins: dict[int, dict]) -> None:
    bar = "─" * 92
    print(f"\n{'═'*92}")
    print(f"  {'Signal Quality Study — Quintile Breakdown':^90}")
    print(f"{'═'*92}")
    print(
        f"  {'Quintile':10}  {'N':>4}  {'Score':>6}  "
        f"{'Gross':>9}  {'Fees':>7}  {'Net':>9}  "
        f"{'PF':>6}  {'WR':>6}  {'MFE(R)':>7}  {'MAE(R)':>7}"
    )
    print(f"  {bar}")

    total_net = sum(b["net"] for b in bins.values())

    for q in range(1, 6):
        b = bins[q]
        if b["count"] == 0:
            print(f"  {_QNAMES[q]:10}  {'0':>4}  {'—':>6}  {'—':>9}  {'—':>7}  {'—':>9}  {'—':>6}  {'—':>6}  {'—':>7}  {'—':>7}")
            continue
        pf_s = f"{b['pf']:.2f}" if b["pf"] != float("inf") else " inf"
        share = b["net"] / total_net * 100 if total_net != 0 else 0.0
        net_col = f"{_c(b['net'])}{b['net']:>+9.2f}{_RST}"
        gross_col = f"{b['gross']:>+9.2f}"
        print(
            f"  {_QNAMES[q]:10}  {b['count']:>4}  {b['avg_score']:>6.1f}  "
            f"{gross_col}  {b['fees']:>7.2f}  {net_col}  "
            f"{pf_s:>6}  {b['wr']*100:>5.1f}%  {b['avg_mfe_r']:>7.2f}  {b['avg_mae_r']:>7.2f}"
            f"   ({share:+.0f}% of total edge)"
        )

    # Totals
    all_recs_flat: list = []  # will fill later — placeholder
    total_count = sum(b["count"] for b in bins.values())
    t_gross = sum(b["gross"] for b in bins.values())
    t_fees  = sum(b["fees"]  for b in bins.values())
    t_net   = total_net
    t_wins  = sum(b["count"] * b["wr"] for b in bins.values())
    t_wr    = t_wins / total_count if total_count > 0 else 0.0
    print(f"  {bar}")
    print(
        f"  {'TOTAL':10}  {total_count:>4}  {'':>6}  "
        f"{t_gross:>+9.2f}  {t_fees:>7.2f}  {_c(t_net)}{t_net:>+9.2f}{_RST}  "
        f"{'':>6}  {t_wr*100:>5.1f}%"
    )
    print(f"{'═'*92}")


def _print_feature_table(bins: dict[int, list[TradeRecord]]) -> None:
    """Average feature values per quintile — shows what 'strong' looks like."""
    print(f"\n{'═'*80}")
    print(f"  {'Feature Averages by Quintile':^78}")
    print(f"{'═'*80}")
    print(
        f"  {'Quintile':10}  {'RelVol':>7}  {'RngExp':>7}  {'Spread':>7}  "
        f"{'Imbalnc':>8}  {'Mom5m':>8}  {'BrkDist':>8}"
    )
    print(f"  {'─'*78}")
    for q in range(1, 6):
        recs = bins[q]
        if not recs:
            continue
        n = len(recs)
        avg = lambda attr: sum(getattr(r, attr) for r in recs) / n
        print(
            f"  {_QNAMES[q]:10}  {avg('rel_vol'):>7.3f}  {avg('range_expansion'):>7.3f}  "
            f"{avg('spread_regime'):>7.4f}  {avg('imbalance_strength'):>8.3f}  "
            f"{avg('momentum_abs'):>8.5f}  {avg('breakout_dist_r'):>8.4f}"
        )
    print(f"{'═'*80}")


def _print_selectivity_table(records: list[TradeRecord]) -> None:
    """Show cumulative metrics as you tighten the score threshold."""
    if not records:
        return
    sorted_recs = sorted(records, key=lambda r: r.signal_score, reverse=True)
    n = len(sorted_recs)

    print(f"\n{'═'*72}")
    print(f"  {'Selectivity Curve — cumulative from top score':^70}")
    print(f"{'═'*72}")
    print(f"  {'Cutoff':>8}  {'Trades':>6}  {'WR%':>6}  {'Net PnL':>9}  {'PF':>6}  {'Avg net/tr':>11}")
    print(f"  {'─'*70}")

    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00]
    for pct in thresholds:
        k = max(1, int(n * pct))
        subset = sorted_recs[:k]
        wins = [r for r in subset if r.net_pnl > 0]
        losses = [r for r in subset if r.net_pnl <= 0]
        net = sum(r.net_pnl for r in subset)
        wr = len(wins) / len(subset) * 100
        gp = sum(r.net_pnl for r in wins)
        gl = sum(abs(r.net_pnl) for r in losses)
        pf = gp / gl if gl > 0 else float("inf")
        pf_s = f"{pf:.2f}" if pf != float("inf") else "  inf"
        avg_net = net / len(subset)
        col = _c(net)
        print(
            f"  {'Top '+f'{pct*100:.0f}%':>8}  {len(subset):>6}  {wr:>5.1f}%  "
            f"{col}{net:>+9.2f}{_RST}  {pf_s:>6}  {col}{avg_net:>+11.2f}{_RST}"
        )
    print(f"{'═'*72}")


def _print_verdict(bins: dict[int, dict], records: list[TradeRecord]) -> None:
    top_net = bins[1]["net"]
    all_net = sum(b["net"] for b in bins.values())

    print(f"\n{'═'*60}")
    print(f"  VERDICT")
    print(f"{'═'*60}")

    if all_net == 0:
        print("  Insufficient data.")
        print(f"{'═'*60}")
        return

    top_share = top_net / all_net * 100 if all_net != 0 else 0.0
    top_pf = bins[1]["pf"]
    top_wr = bins[1]["wr"]
    bot_pf = bins[5]["pf"]

    if top_share > 80 and top_net > 0 and all_net < top_net:
        print(f"  Top 20% generates {top_share:.0f}% of edge while full set")
        print(f"  is net negative. STRONG FILTER CASE.")
        print(f"  Recommendation: Trade top quintile only.")
    elif top_share > 100 and top_net > 0:
        print(f"  Top quintile is the only profitable group ({top_share:.0f}% of edge).")
        print(f"  Lower quintiles destroy capital. FILTER is warranted.")
        print(f"  Recommendation: Trade top quintile only (score >= cutoff).")
    elif top_share > 60 and top_pf > 1.5:
        print(f"  Top 20% concentrates {top_share:.0f}% of edge (PF {top_pf:.2f}).")
        print(f"  Evidence of signal selectivity. Selective trading likely improves PF.")
        print(f"  Recommendation: Test top-40% filter before committing to top-20% only.")
    elif top_share > 0 and all_net > 0:
        print(f"  Top quintile has {top_share:.0f}% of edge but all quintiles contribute.")
        print(f"  Edge is distributed across signal strengths.")
        print(f"  Recommendation: Do not filter — collect more data first.")
    else:
        print(f"  No clear edge concentration. Insufficient data or strategy has no edge.")
        print(f"  Recommendation: Collect more trades before filtering.")

    print(f"\n  Top-20% score cutoff : {sorted(r.signal_score for r in records)[int(len(records)*0.80)-1]:.1f}")
    print(f"  Top-20% WR           : {top_wr*100:.1f}%")
    print(f"  Top-20% PF           : {top_pf:.2f}")
    print(f"  Bottom-20% PF        : {bot_pf:.2f}")
    print(f"{'═'*60}\n")


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def _write_csv(records: list[TradeRecord], out: Path) -> None:
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "trade_num", "entry_time", "symbol", "side",
            "entry_price", "exit_price", "size",
            "gross_pnl", "fees", "net_pnl", "exit_reason",
            "atr", "rel_vol", "range_expansion", "spread_regime",
            "buy_sell_imbalance", "momentum_5m", "vwap_deviation",
            "breakout_dist_r", "imbalance_strength", "momentum_abs",
            "signal_score", "quintile",
            "mfe_r", "mae_r",
        ])
        for r in records:
            ts = datetime.fromtimestamp(r.entry_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            w.writerow([
                r.trade_num, ts, r.symbol, r.side,
                r.entry_price, r.exit_price, r.size,
                r.gross_pnl, r.total_fees, r.net_pnl, r.exit_reason,
                r.atr, r.rel_vol, r.range_expansion, r.spread_regime,
                r.buy_sell_imbalance, r.momentum_5m, r.vwap_deviation,
                r.breakout_dist_r, r.imbalance_strength, r.momentum_abs,
                r.signal_score, r.quintile,
                r.mfe_r, r.mae_r,
            ])
    print(f"CSV written: {out}  ({len(records)} trades)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Signal quality study")
    p.add_argument("--journal-dir", type=Path, default=Path("logs/paper"))
    p.add_argument("--candle-dir",  type=Path, default=Path("data/candles"))
    p.add_argument("--out",         type=Path, default=Path("signal_quality_study.csv"))
    p.add_argument("--no-rest",     action="store_true", help="Disable Bitget REST fallback")
    p.add_argument("--skip-mfe",    action="store_true", help="Skip MFE/MAE (faster, no candles needed)")
    p.add_argument("--verbose",     action="store_true", help="Print per-trade detail")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.journal_dir.exists():
        print(f"Journal directory not found: {args.journal_dir}")
        sys.exit(1)

    print("Loading journal events ...")
    events = _load_events(args.journal_dir)
    print(f"  {len(events)} events")

    print("Reconstructing trades ...")
    records = reconstruct_trades(events)
    print(f"  {len(records)} closed trades")

    if not records:
        print("No closed trades found.")
        sys.exit(1)

    if not args.skip_mfe:
        print("Computing MFE / MAE from candles ...")
        compute_mfe_mae(records, args.candle_dir, args.no_rest)

    print("Scoring signals ...")
    score_signals(records)

    # Bin by quintile
    bins_recs: dict[int, list[TradeRecord]] = {q: [] for q in range(1, 6)}
    for r in records:
        bins_recs[r.quintile].append(r)
    bins_metrics: dict[int, dict] = {q: _bin_metrics(bins_recs[q]) for q in range(1, 6)}

    _print_quintile_table(bins_metrics)
    _print_feature_table(bins_recs)
    _print_selectivity_table(records)
    _print_verdict(bins_metrics, records)

    if args.verbose:
        print(f"\n  {'#':>3}  {'Time':16}  {'Sym':8}  {'Side':5}  "
              f"{'Score':>6}  {'Q':>2}  {'Net':>9}  {'MFE_R':>6}  {'MAE_R':>6}  {'Exit':12}")
        print(f"  {'─'*100}")
        for r in sorted(records, key=lambda x: x.signal_score, reverse=True):
            col = _c(r.net_pnl)
            print(
                f"  {r.trade_num:>3}  "
                f"{datetime.fromtimestamp(r.entry_time, tz=timezone.utc).strftime('%Y-%m-%d %H:%M'):16}  "
                f"{r.symbol:8}  {r.side:5}  {r.signal_score:>6.1f}  {r.quintile:>2}  "
                f"{col}{r.net_pnl:>+9.2f}{_RST}  {r.mfe_r:>6.2f}  {r.mae_r:>6.2f}  {r.exit_reason:12}"
            )

    _write_csv(records, args.out)


if __name__ == "__main__":
    main()
