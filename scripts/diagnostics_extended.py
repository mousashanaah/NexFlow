#!/usr/bin/env python3
"""
NexFlow Extended Diagnostics
Runs two analyses on JSONL journals + candle parquet files:

  2. Post-exit MFE study   — how much did price move favourably after each exit?
  3. Signal quality study  — do high-strength signals outperform weak ones?

Usage:
    python scripts/diagnostics_extended.py --journal-dir logs/paper --candle-dir data/candles
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_DIM    = "\033[2m"

def _c(val: float, w: int = 9) -> str:
    s = f"{val:+{w}.2f}"
    return (_GREEN + s + _RESET) if val > 0 else (_RED + s + _RESET) if val < 0 else s

BAR  = "=" * 70
BAR2 = "─" * 70

# ---------------------------------------------------------------------------
# Journal loading
# ---------------------------------------------------------------------------

@dataclass
class _Open:
    symbol: str
    direction: str
    fill_price: float
    fill_time: float
    size: float
    fee: float
    stop_price: float
    signal_features: dict = field(default_factory=dict)
    partial_pnl: float = 0.0
    partial_fees: float = 0.0
    partial_count: int = 0
    last_partial_ts: float = 0.0
    last_tp_idx: int = -1

@dataclass
class ClosedTrade:
    symbol: str
    direction: str
    entry: float
    exit_price: float
    size: float
    net_pnl: float
    fees: float
    hold_min: float
    exit_reason: str
    entry_ts: float
    exit_ts: float
    signal_features: dict = field(default_factory=dict)  # from matching SIGNAL event

@dataclass
class SignalRecord:
    symbol: str
    direction: str
    entry_price: float
    stop_price: float
    atr: float
    features: dict
    ts: float
    outcome: str = "unknown"   # "filled", "rejected", "killed"
    trade_net_pnl: float = 0.0
    reject_reason: str = ""


def _load(journal_dir: Path) -> tuple[list[ClosedTrade], list[SignalRecord]]:
    files = sorted(journal_dir.glob("*.jsonl"))
    if not files:
        sys.exit(f"No .jsonl files in {journal_dir}")

    events: list[dict] = []
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    events.sort(key=lambda e: e.get("ts_epoch", 0))

    open_trades: dict[str, _Open] = {}
    pending_signals: dict[str, dict] = {}
    trades: list[ClosedTrade] = []
    signals: list[SignalRecord] = []

    for ev in events:
        et = ev.get("event", "")
        sym = ev.get("symbol", "")

        if et == "SIGNAL":
            pending_signals[sym] = ev
            signals.append(SignalRecord(
                symbol=sym,
                direction=ev.get("direction", ""),
                entry_price=ev.get("entry_price", 0.0),
                stop_price=ev.get("stop_price", 0.0),
                atr=ev.get("atr", 0.0),
                features=ev.get("features", {}),
                ts=ev.get("ts_epoch", 0.0),
            ))

        elif et == "REJECTED":
            reason = ev.get("reason", "")
            # Mark the most recent signal for this symbol
            for sr in reversed(signals):
                if sr.symbol == sym and sr.outcome == "unknown":
                    sr.outcome = "killed" if "kill_switch" in reason else "rejected"
                    sr.reject_reason = reason
                    break

        elif et == "FILL":
            existing = open_trades.pop(sym, None)
            if existing and existing.partial_count > 0:
                _flush_trade(trades, existing, signals)
            sig = pending_signals.get(sym, {})
            open_trades[sym] = _Open(
                symbol=sym,
                direction=ev.get("direction", "").upper(),
                fill_price=ev.get("fill_price", 0.0),
                fill_time=ev.get("ts_epoch", 0.0),
                size=ev.get("size", 0.0),
                fee=ev.get("fee", 0.0),
                stop_price=sig.get("stop_price", 0.0),
                signal_features=sig.get("features", {}),
            )
            # Mark signal as filled
            for sr in reversed(signals):
                if sr.symbol == sym and sr.outcome == "unknown":
                    sr.outcome = "filled"
                    break

        elif et == "PARTIAL_TP":
            t = open_trades.get(sym)
            if t:
                t.partial_pnl  += ev.get("pnl", 0.0)
                t.partial_fees += ev.get("fee", 0.0)
                t.partial_count += 1
                t.last_partial_ts = ev.get("ts_epoch", t.fill_time)
                t.last_tp_idx = max(t.last_tp_idx, ev.get("tp_idx", 0))

        elif et in ("STOP_HIT", "FORCE_CLOSE"):
            t = open_trades.pop(sym, None)
            if t:
                close_pnl  = ev.get("pnl", 0.0)
                close_fee  = ev.get("fee", 0.0)
                gross      = t.partial_pnl + close_pnl
                total_fees = t.partial_fees + t.fee + close_fee
                exit_time  = ev.get("ts_epoch", t.fill_time)
                exit_px    = ev.get("fill_price", ev.get("price", t.fill_price))
                tr = ClosedTrade(
                    symbol=sym, direction=t.direction,
                    entry=t.fill_price, exit_price=exit_px,
                    size=t.size, net_pnl=gross, fees=total_fees,
                    hold_min=max(0.0, (exit_time - t.fill_time) / 60.0),
                    exit_reason="STOP" if et == "STOP_HIT" else "FORCED",
                    entry_ts=t.fill_time, exit_ts=exit_time,
                    signal_features=t.signal_features,
                )
                trades.append(tr)
                # back-fill signal outcome
                for sr in reversed(signals):
                    if sr.symbol == sym and sr.outcome == "filled" and sr.trade_net_pnl == 0.0:
                        sr.trade_net_pnl = gross
                        break

    for t in open_trades.values():
        if t.partial_count >= 3:
            _flush_trade(trades, t, signals)

    trades.sort(key=lambda x: x.entry_ts)
    return trades, signals


def _flush_trade(trades: list[ClosedTrade], t: _Open, signals: list[SignalRecord]) -> None:
    tp_labels = ["TP1", "TP2", "TP3"]
    label = tp_labels[min(t.last_tp_idx, 2)] if t.partial_count >= 3 else "PARTIAL"
    total_fees = t.partial_fees + t.fee
    close_ts = t.last_partial_ts or t.fill_time
    tr = ClosedTrade(
        symbol=t.symbol, direction=t.direction,
        entry=t.fill_price, exit_price=t.fill_price,
        size=t.size, net_pnl=t.partial_pnl, fees=total_fees,
        hold_min=max(0.0, (close_ts - t.fill_time) / 60.0),
        exit_reason=label, entry_ts=t.fill_time, exit_ts=close_ts,
        signal_features=t.signal_features,
    )
    trades.append(tr)
    for sr in reversed(signals):
        if sr.symbol == t.symbol and sr.outcome == "filled" and sr.trade_net_pnl == 0.0:
            sr.trade_net_pnl = t.partial_pnl
            break


# ---------------------------------------------------------------------------
# Candle loading (parquet)
# ---------------------------------------------------------------------------

@dataclass
class _Candle:
    open_time: int
    high: float
    low: float
    close: float

def _load_candles(candle_dir: Path, symbol: str, timeframe: str = "1m") -> list[_Candle]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return []
    path = candle_dir / f"{symbol}_{timeframe}.parquet"
    if not path.exists():
        return []
    try:
        tbl = pq.read_table(path, columns=["open_time", "high", "low", "close"])
        rows = tbl.to_pylist()
        return [_Candle(open_time=r["open_time"], high=r["high"],
                        low=r["low"], close=r["close"]) for r in rows]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# 2. Post-exit MFE study
# ---------------------------------------------------------------------------

def post_exit_mfe(trades: list[ClosedTrade], candle_dir: Path) -> None:
    print(f"\n{_BOLD}{BAR}{_RESET}")
    print(f"{_BOLD}  2. POST-EXIT MFE STUDY{_RESET}")
    print(f"     (How much did price move favourably AFTER each exit?){_RESET}")
    print(f"{_BOLD}{BAR}{_RESET}")
    print(f"  MFE = Maximum Favorable Excursion after exit.")
    print(f"  Positive = price continued in your direction after you exited.")
    print(f"  Negative = price reversed; exit was correct.\n")

    candle_cache: dict[str, list[_Candle]] = {}

    WINDOWS = [1, 3, 5]  # candle bars after exit

    rows: list[dict] = []

    for t in trades:
        sym = t.symbol
        if sym not in candle_cache:
            candle_cache[sym] = _load_candles(candle_dir, sym, "1m")
        candles = candle_cache[sym]

        # Find candles AFTER exit (open_time >= exit bar start)
        exit_bar = (int(t.exit_ts) // 60) * 60
        post = [c for c in candles if c.open_time >= exit_bar]

        mfe: dict[int, float] = {}
        for w in WINDOWS:
            window = post[:w]
            if not window:
                mfe[w] = float("nan")
                continue
            if t.direction == "LONG":
                best_price = max(c.high for c in window)
                mfe[w] = (best_price - t.exit_price) * t.size
            else:
                best_price = min(c.low for c in window)
                mfe[w] = (t.exit_price - best_price) * t.size

        rows.append({"trade": t, "mfe": mfe})

    if not rows:
        print("  No trades with candle data found.")
        return

    # Per-trade table
    hdr = (f"  {'#':>3}  {'SYM':6} {'DIR':5} {'EXIT':>10} {'REASON':8} "
           f"{'NET PnL':>9}  {'MFE+1m':>9}  {'MFE+3m':>9}  {'MFE+5m':>9}")
    print(hdr)
    print(f"  {BAR2}")

    any_data = False
    for i, row in enumerate(rows, 1):
        t = row["trade"]
        mfe = row["mfe"]
        def _mf(v: float) -> str:
            if math.isnan(v):
                return f"{'N/A':>9}"
            return _c(v, 9)
        print(f"  {i:>3}  {t.symbol:6} {t.direction[:5]:5}  {t.exit_price:>10.2f}  "
              f"{t.exit_reason:8}  {_c(t.net_pnl, 9)}  "
              f"{_mf(mfe.get(1,float('nan')))}  "
              f"{_mf(mfe.get(3,float('nan')))}  "
              f"{_mf(mfe.get(5,float('nan')))}")
        if not math.isnan(mfe.get(1, float("nan"))):
            any_data = True

    print(f"  {BAR2}")

    if not any_data:
        print(f"\n  {_YELLOW}No post-exit candle data available.{_RESET}")
        print(f"  The candle engine only stores candles from the current session.")
        print(f"  To use this study, run the paper trader for a full session,")
        print(f"  then run diagnostics WITHOUT restarting (data/candles/ must be populated).")
        print(f"  Candle dir checked: {candle_dir.resolve()}")
        return

    # Aggregate: missed profit per window
    print(f"\n  Aggregate missed profit (price continued after exit):")
    print(f"  {'Window':10}  {'Avg MFE/trade':>14}  {'Total missed':>14}  "
          f"{'% trades net-positive MFE':>26}")
    for w in WINDOWS:
        vals = [row["mfe"][w] for row in rows if not math.isnan(row["mfe"].get(w, float("nan")))]
        if not vals:
            continue
        avg = sum(vals) / len(vals)
        total = sum(vals)
        pos_pct = sum(1 for v in vals if v > 0) / len(vals) * 100
        print(f"  +{w}m        {_c(avg, 14)}  {_c(total, 14)}  {pos_pct:>25.1f}%")

    # Split: exits that left most on the table vs those that were correct
    for w in [3]:
        vals_with_trades = [(row["mfe"][w], row["trade"]) for row in rows
                            if not math.isnan(row["mfe"].get(w, float("nan")))]
        if len(vals_with_trades) < 4:
            continue
        vals_with_trades.sort(key=lambda x: -x[0])
        top_half = vals_with_trades[:len(vals_with_trades)//2]
        bot_half = vals_with_trades[len(vals_with_trades)//2:]

        print(f"\n  Trades that left MOST on the table (top half by MFE+{w}m):")
        for mfe_val, t in top_half[:5]:
            print(f"    {t.symbol} {t.direction[:5]} {t.exit_reason:8}  "
                  f"net {_c(t.net_pnl, 8)}  MFE+{w}m {_c(mfe_val, 8)}")

        print(f"\n  Trades where exit was CORRECT (bottom half, price reversed):")
        for mfe_val, t in bot_half[:5]:
            print(f"    {t.symbol} {t.direction[:5]} {t.exit_reason:8}  "
                  f"net {_c(t.net_pnl, 8)}  MFE+{w}m {_c(mfe_val, 8)}")


# ---------------------------------------------------------------------------
# 3. Signal quality analysis
# ---------------------------------------------------------------------------

def signal_quality(signals: list[SignalRecord], trades: list[ClosedTrade]) -> None:
    print(f"\n{_BOLD}{BAR}{_RESET}")
    print(f"{_BOLD}  3. SIGNAL QUALITY ANALYSIS{_RESET}")
    print(f"{_BOLD}{BAR}{_RESET}")

    filled = [s for s in signals if s.outcome == "filled"]
    rejected = [s for s in signals if s.outcome in ("rejected", "killed")]

    print(f"\n  Signal summary:")
    print(f"    Total signals      : {len(signals)}")
    print(f"    Filled (traded)    : {len(filled)}")
    print(f"    Rejected / killed  : {len(rejected)}")

    if not filled:
        print(f"\n  No filled signals to analyse.")
        return

    # Print all signal features for reference
    sample_features = filled[0].features if filled else {}
    if sample_features:
        print(f"\n  Available signal features: {list(sample_features.keys())}")

    # -----------------------------------------------------------------------
    # Breakout strength composite score
    # Higher is stronger:
    #   rel_vol (higher = more volume surge)
    #   range_expansion (higher = wider breakout bar)
    #   abs(buy_sell_imbalance - 0.5) * 2 (higher = more one-sided)
    #   abs(momentum_5m) normalised (directional alignment)
    # -----------------------------------------------------------------------

    def _strength(s: SignalRecord) -> float:
        f = s.features
        rv     = f.get("rel_vol", 1.0)
        re     = f.get("range_expansion", 0.0)
        imb    = f.get("buy_sell_imbalance", 0.5)
        imb_s  = abs(imb - 0.5) * 2.0          # 0 = neutral, 1 = fully one-sided
        mom5   = abs(f.get("momentum_5m", 0.0))
        # Normalise momentum by ATR to make it dimensionless
        atr    = s.atr if s.atr > 0 else 1.0
        mom_n  = min(mom5 / atr, 3.0) / 3.0    # cap at 3× ATR, scale 0-1
        # Composite: equal-weight four factors
        return (rv + re + imb_s + mom_n) / 4.0

    for s in filled:
        s._strength = _strength(s)  # type: ignore[attr-defined]

    filled_sorted = sorted(filled, key=lambda s: s._strength, reverse=True)  # type: ignore[attr-defined]

    n = len(filled_sorted)
    q_size = max(1, n // 4)
    q1 = filled_sorted[:q_size]           # strongest quartile
    q4 = filled_sorted[n - q_size:]       # weakest quartile

    print(f"\n  {_BOLD}Signal strength ranking (all filled signals):{_RESET}")
    print(f"  {'#':>3}  {'SYM':6} {'DIR':5} {'STRENGTH':>9}  "
          f"{'REL_VOL':>8}  {'RANGE_EXP':>10}  {'IMBALANCE':>10}  {'MOM5_N':>8}  {'NET PnL':>9}")
    print(f"  {BAR2}")

    for i, s in enumerate(filled_sorted, 1):
        f = s.features
        rv   = f.get("rel_vol", 0.0)
        re   = f.get("range_expansion", 0.0)
        imb  = f.get("buy_sell_imbalance", 0.5)
        mom5 = abs(f.get("momentum_5m", 0.0))
        atr  = s.atr if s.atr > 0 else 1.0
        mom_n = min(mom5 / atr, 3.0) / 3.0
        pnl_str = _c(s.trade_net_pnl) if s.trade_net_pnl != 0 else "  pending"
        tier = ""
        if i <= q_size:
            tier = f"{_GREEN}Q1{_RESET}"
        elif i > n - q_size:
            tier = f"{_RED}Q4{_RESET}"
        print(f"  {i:>3}  {s.symbol:6} {s.direction[:5]:5}  "
              f"{s._strength:>9.3f}  {rv:>8.3f}  {re:>10.3f}  {imb:>10.3f}  "  # type: ignore[attr-defined]
              f"{mom_n:>8.3f}  {pnl_str}  {tier}")

    def _quartile_stats(qs: list[SignalRecord], label: str) -> None:
        if not qs:
            return
        pnls = [s.trade_net_pnl for s in qs]
        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p <= 0)
        net = sum(pnls)
        gp  = sum(p for p in pnls if p > 0)
        gl  = abs(sum(p for p in pnls if p <= 0))
        pf  = (gp / gl) if gl > 0 else float("inf")
        avg_str = [s._strength for s in qs]  # type: ignore[attr-defined]
        print(f"\n  {_BOLD}{label}{_RESET}")
        print(f"  {BAR2}")
        print(f"    Signals          : {len(qs)}")
        print(f"    Avg strength     : {sum(avg_str)/len(avg_str):.3f}")
        print(f"    Wins / Losses    : {wins}W / {losses}L   WR {wins/len(qs)*100:.1f}%")
        print(f"    Net PnL          : {_c(net)}")
        pf_s = f"{pf:.3f}" if pf != float("inf") else "∞"
        print(f"    Profit factor    : {pf_s}")
        print(f"    Best / Worst     : {_c(max(pnls))} / {_c(min(pnls))}")

    _quartile_stats(q1, f"Strongest quartile (Q1) — top {q_size} signal(s)")
    _quartile_stats(q4, f"Weakest quartile (Q4) — bottom {q_size} signal(s)")

    # Verdict
    if n >= 4:
        q1_net = sum(s.trade_net_pnl for s in q1)
        q4_net = sum(s.trade_net_pnl for s in q4)
        q1_wr  = sum(1 for s in q1 if s.trade_net_pnl > 0) / len(q1) * 100
        q4_wr  = sum(1 for s in q4 if s.trade_net_pnl > 0) / len(q4) * 100
        print(f"\n  {_BOLD}Verdict:{_RESET}")
        if q1_net > q4_net and q1_wr > q4_wr:
            print(f"    {_GREEN}Strong signals outperform weak ones.{_RESET}")
            print(f"    Q1 net {_c(q1_net)}  WR {q1_wr:.0f}%  vs  Q4 net {_c(q4_net)}  WR {q4_wr:.0f}%")
        elif q1_net < q4_net:
            print(f"    {_YELLOW}Weak signals outperformed strong ones in this session.{_RESET}")
            print(f"    Q1 net {_c(q1_net)}  vs  Q4 net {_c(q4_net)}")
            print(f"    Small sample — inconclusive without more sessions.")
        else:
            print(f"    No clear relationship between signal strength and outcome.")
            print(f"    Sample too small ({n} trades) for a definitive conclusion.")
    else:
        print(f"\n  {_YELLOW}Only {n} filled signal(s) — need ≥ 4 to split into quartiles.{_RESET}")

    # Feature correlation table
    print(f"\n  {_BOLD}Feature vs outcome correlation (filled signals):{_RESET}")
    print(f"  {'Feature':20s}  {'Winners avg':>12}  {'Losers avg':>12}  {'Δ':>8}")
    print(f"  {BAR2}")
    winners = [s for s in filled if s.trade_net_pnl > 0]
    losers  = [s for s in filled if s.trade_net_pnl <= 0]
    feature_names = list(filled[0].features.keys()) if filled else []
    for fname in feature_names:
        w_vals = [s.features.get(fname, 0.0) for s in winners]
        l_vals = [s.features.get(fname, 0.0) for s in losers]
        w_avg  = sum(w_vals) / len(w_vals) if w_vals else float("nan")
        l_avg  = sum(l_vals) / len(l_vals) if l_vals else float("nan")
        if math.isnan(w_avg) or math.isnan(l_avg):
            continue
        delta  = w_avg - l_avg
        delta_s = (_GREEN if delta > 0 else _RED) + f"{delta:+.4f}" + _RESET
        print(f"  {fname:20s}  {w_avg:>12.4f}  {l_avg:>12.4f}  {delta_s:>8}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="NexFlow extended diagnostics")
    ap.add_argument("--journal-dir", type=Path, default=Path("logs/paper"))
    ap.add_argument("--candle-dir",  type=Path, default=Path("data/candles"))
    args = ap.parse_args()

    trades, signals = _load(args.journal_dir)

    print(f"\n{_BOLD}{BAR}{_RESET}")
    print(f"{_BOLD}  NEXFLOW EXTENDED DIAGNOSTICS{_RESET}")
    print(f"  Journal : {args.journal_dir}")
    print(f"  Candles : {args.candle_dir}")
    print(f"  Trades  : {len(trades)}   Signals: {len(signals)}")
    print(f"{_BOLD}{BAR}{_RESET}")

    post_exit_mfe(trades, args.candle_dir)
    signal_quality(signals, trades)

    print(f"\n{_BOLD}{BAR}{_RESET}\n")


if __name__ == "__main__":
    main()
