#!/usr/bin/env python3
"""
NexFlow Paper Trading Diagnostics
Produces trade-by-trade ledger and all requested statistics from JSONL journals.
Usage: python scripts/paper_diagnostics.py --journal-dir logs/paper
"""
from __future__ import annotations
import argparse
import json
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

def _c(val: float, width: int = 9) -> str:
    s = f"{val:+{width}.2f}"
    return (_GREEN + s + _RESET) if val > 0 else (_RED + s + _RESET) if val < 0 else s

def _pct(val: float) -> str:
    s = f"{val:+.1f}%"
    return (_GREEN + s + _RESET) if val > 0 else (_RED + s + _RESET) if val < 0 else s

# ---------------------------------------------------------------------------
# Journal loading  (mirrors analyze_paper_results.py reconstruction)
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
    partial_pnl: float = 0.0
    partial_fees: float = 0.0
    partial_count: int = 0
    last_partial_ts: float = 0.0
    last_tp_idx: int = -1

@dataclass
class Trade:
    idx: int
    symbol: str
    direction: str
    entry: float
    exit_price: float
    size: float
    gross_pnl: float   # pnl before fees
    fees: float
    net_pnl: float
    hold_min: float
    exit_reason: str
    entry_ts: float
    exit_ts: float

@dataclass
class KillBlock:
    ts: float
    reason: str
    symbol: str
    direction: str

def _load(journal_dir: Path) -> tuple[list[Trade], list[KillBlock], list[dict]]:
    files = sorted(journal_dir.glob("*.jsonl"))
    if not files:
        sys.exit(f"No .jsonl files found in {journal_dir}")

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
    trades: list[Trade] = []
    kill_blocks: list[KillBlock] = []
    idx = 0

    for ev in events:
        et = ev.get("event", "")
        sym = ev.get("symbol", "")

        if et == "SIGNAL":
            pending_signals[sym] = ev

        elif et == "REJECTED":
            # Capture kill-blocked entries (direction available)
            reason = ev.get("reason", "")
            if "kill_switch" in reason:
                kill_blocks.append(KillBlock(
                    ts=ev.get("ts_epoch", 0),
                    reason=reason,
                    symbol=sym,
                    direction=ev.get("direction", ""),
                ))

        elif et == "FILL":
            existing = open_trades.pop(sym, None)
            if existing and existing.partial_count > 0:
                # flush all-TP-closed trade
                _flush(trades, existing, idx)
                idx += 1
            sig = pending_signals.get(sym, {})
            open_trades[sym] = _Open(
                symbol=sym,
                direction=ev.get("direction", "").upper(),
                fill_price=ev.get("fill_price", 0.0),
                fill_time=ev.get("ts_epoch", 0.0),
                size=ev.get("size", 0.0),
                fee=ev.get("fee", 0.0),
                stop_price=sig.get("stop_price", 0.0),
            )

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
                exit_label = "STOP" if et == "STOP_HIT" else "FORCED"
                idx += 1
                trades.append(Trade(
                    idx=idx, symbol=sym, direction=t.direction,
                    entry=t.fill_price, exit_price=exit_px,
                    size=t.size, gross_pnl=gross + total_fees,
                    fees=total_fees, net_pnl=gross,
                    hold_min=max(0.0, (exit_time - t.fill_time) / 60.0),
                    exit_reason=exit_label,
                    entry_ts=t.fill_time, exit_ts=exit_time,
                ))

    # End-of-journal: flush all-TP trades
    for t in open_trades.values():
        if t.partial_count >= 3:
            _flush(trades, t, idx + 1)
            idx += 1

    trades.sort(key=lambda x: x.entry_ts)
    for i, tr in enumerate(trades, 1):
        tr.idx = i

    return trades, kill_blocks, events


def _flush(trades: list[Trade], t: _Open, idx: int) -> None:
    tp_labels = ["TP1", "TP2", "TP3"]
    label = tp_labels[min(t.last_tp_idx, 2)] if t.partial_count >= 3 else "PARTIAL"
    close_ts = t.last_partial_ts or t.fill_time
    total_fees = t.partial_fees + t.fee
    trades.append(Trade(
        idx=idx, symbol=t.symbol, direction=t.direction,
        entry=t.fill_price, exit_price=t.fill_price,
        size=t.size, gross_pnl=t.partial_pnl + total_fees,
        fees=total_fees, net_pnl=t.partial_pnl,
        hold_min=max(0.0, (close_ts - t.fill_time) / 60.0),
        exit_reason=label,
        entry_ts=t.fill_time, exit_ts=close_ts,
    ))


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _stats(ts: list[Trade]) -> dict:
    if not ts:
        return dict(n=0, wins=0, losses=0, wr=0.0, gross=0.0, fees=0.0,
                    net=0.0, pf=0.0, avg_hold=0.0, best=0.0, worst=0.0,
                    exp_r=0.0, stop=0, tp=0, forced=0)
    wins    = [t for t in ts if t.net_pnl > 0]
    losses  = [t for t in ts if t.net_pnl <= 0]
    gross   = sum(t.gross_pnl for t in ts)
    fees    = sum(t.fees for t in ts)
    net     = sum(t.net_pnl for t in ts)
    gp      = sum(t.net_pnl for t in wins)
    gl      = abs(sum(t.net_pnl for t in losses))
    pf      = (gp / gl) if gl > 0 else float("inf")
    r_vals  = []
    for t in ts:
        risk = abs(t.entry) * t.size * 0.005  # rough 0.5% proxy since stop not on Trade
        if risk > 1e-6:
            r_vals.append(t.net_pnl / risk)
    exp_r   = sum(r_vals) / len(r_vals) if r_vals else float("nan")
    return dict(
        n=len(ts), wins=len(wins), losses=len(losses),
        wr=len(wins) / len(ts) * 100,
        gross=gross, fees=fees, net=net, pf=pf,
        avg_hold=sum(t.hold_min for t in ts) / len(ts),
        best=max(t.net_pnl for t in ts),
        worst=min(t.net_pnl for t in ts),
        exp_r=exp_r,
        stop=sum(1 for t in ts if t.exit_reason == "STOP"),
        tp=sum(1 for t in ts if t.exit_reason.startswith("TP")),
        forced=sum(1 for t in ts if t.exit_reason == "FORCED"),
    )


def _print_stats_block(label: str, ts: list[Trade]) -> None:
    s = _stats(ts)
    bar = "─" * 65
    print(f"\n{_BOLD}{label}{_RESET}")
    print(bar)
    if s["n"] == 0:
        print("  No trades.")
        return
    print(f"  {'Trades':25s}: {s['n']}   ({s['wins']}W / {s['losses']}L)   WR {s['wr']:.1f}%")
    print(f"  {'Gross PnL (before fees)':25s}: {_c(s['gross'])}")
    print(f"  {'Total fees':25s}: {_c(-s['fees'])}")
    print(f"  {'Net PnL':25s}: {_c(s['net'])}")
    pf_str = f"{s['pf']:.3f}" if s["pf"] != float("inf") else "∞"
    print(f"  {'Profit factor':25s}: {pf_str}")
    print(f"  {'Avg hold time':25s}: {s['avg_hold']:.1f} min")
    print(f"  {'Best / Worst':25s}: {_c(s['best'])} / {_c(s['worst'])}")
    print(f"  {'TP / Stop / Forced':25s}: {s['tp']} / {s['stop']} / {s['forced']}")
    print(bar)


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal-dir", type=Path, default=Path("logs/paper"))
    args = ap.parse_args()

    trades, kill_blocks, events = _load(args.journal_dir)

    bar  = "=" * 65
    bar2 = "─" * 65

    # -----------------------------------------------------------------------
    # 1. Trade-by-trade ledger
    # -----------------------------------------------------------------------
    print(f"\n{_BOLD}{bar}{_RESET}")
    print(f"{_BOLD}  1. TRADE-BY-TRADE LEDGER{_RESET}")
    print(f"{_BOLD}{bar}{_RESET}")
    hdr = (f"  {'#':>3}  {'SYM':6} {'DIR':5} {'ENTRY':>10} {'EXIT':>10} "
           f"{'NET PNL':>9} {'FEES':>7} {'HOLD':>6} {'REASON'}")
    print(hdr)
    print(f"  {bar2}")
    cum = 0.0
    for t in trades:
        cum += t.net_pnl
        dir_col = (_GREEN + t.direction[:5] + _RESET) if t.direction == "LONG" else (_RED + t.direction[:5] + _RESET)
        reason_col = (_GREEN + t.exit_reason + _RESET) if t.exit_reason.startswith("TP") else t.exit_reason
        gross_str = f"{t.gross_pnl:+.2f}"
        print(f"  {t.idx:>3}  {t.symbol:6} {dir_col:} "
              f"  {t.entry:>10.2f}  {t.exit_price:>10.2f} "
              f"  {_c(t.net_pnl, 8)}  {-t.fees:>7.2f}  {t.hold_min:>5.1f}m  {reason_col}  "
              f"{_DIM}(gross {gross_str}){_RESET}")

    print(f"  {bar2}")
    print(f"  {'Cumulative net PnL':40s}: {_c(cum)}")

    # -----------------------------------------------------------------------
    # 2 & 3. Gross / Net totals
    # -----------------------------------------------------------------------
    gross_total = sum(t.gross_pnl for t in trades)
    fees_total  = sum(t.fees for t in trades)
    net_total   = sum(t.net_pnl for t in trades)

    print(f"\n{_BOLD}{bar}{_RESET}")
    print(f"{_BOLD}  2 & 3. GROSS PNL vs NET PNL{_RESET}")
    print(f"{_BOLD}{bar}{_RESET}")
    print(f"  {'Gross PnL (before fees)':40s}: {_c(gross_total)}")
    print(f"  {'Total fees paid':40s}: {_c(-fees_total)}")
    print(f"  {'Net PnL (after fees)':40s}: {_c(net_total)}")
    fee_drag = (fees_total / gross_total * 100) if gross_total > 0 else float("nan")
    print(f"  {'Fee drag on gross profit':40s}: {fee_drag:.1f}%"
          if not (fee_drag != fee_drag) else
          f"  {'Fee drag':40s}: N/A (gross is negative)")

    # -----------------------------------------------------------------------
    # 4 & 5. Per-symbol statistics
    # -----------------------------------------------------------------------
    btc = [t for t in trades if t.symbol == "BTCUSDT"]
    eth = [t for t in trades if t.symbol == "ETHUSDT"]

    _print_stats_block("4. BTCUSDT STATISTICS", btc)
    _print_stats_block("5. ETHUSDT STATISTICS", eth)

    # -----------------------------------------------------------------------
    # 6. Kill-switch analysis
    # -----------------------------------------------------------------------
    print(f"\n{_BOLD}{bar}{_RESET}")
    print(f"{_BOLD}  6. KILL-SWITCH ANALYSIS{_RESET}")
    print(f"{_BOLD}{bar}{_RESET}")
    print(f"  Trades that were BLOCKED (REJECTED with kill_switch reason): {len(kill_blocks)}")

    if kill_blocks:
        reasons: dict[str, int] = {}
        by_sym: dict[str, int] = {}
        for kb in kill_blocks:
            r = kb.reason.replace("kill_switch:", "")
            reasons[r] = reasons.get(r, 0) + 1
            by_sym[kb.symbol] = by_sym.get(kb.symbol, 0) + 1
        print(f"\n  By kill reason:")
        for r, cnt in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {r:35s}: {cnt}×")
        print(f"\n  By symbol:")
        for sym, cnt in sorted(by_sym.items()):
            print(f"    {sym:10s}: {cnt} blocked entries")

    # Counterfactual: would blocked trades have improved things?
    # We can't know actual PnL of blocked trades, but we can measure
    # whether trades that DID execute immediately after a kill window
    # were winners or losers.
    print(f"\n  Kill switch firing summary:")
    kill_events = [(e.get("ts_epoch",0), e.get("reason",""), e.get("symbol",""))
                   for e in events if e.get("event") == "KILL_SWITCH"]
    kill_cleared = [(e.get("ts_epoch",0), e.get("reason",""), e.get("symbol",""))
                    for e in events if e.get("event") == "KILL_CLEARED"]
    print(f"    KILL_SWITCH events   : {len(kill_events)}")
    print(f"    KILL_CLEARED events  : {len(kill_cleared)}")
    if kill_events:
        print(f"\n  Kill activations by reason:")
        k_reasons: dict[str, int] = {}
        for _, r, _ in kill_events:
            k_reasons[r] = k_reasons.get(r, 0) + 1
        for r, cnt in sorted(k_reasons.items(), key=lambda x: -x[1]):
            print(f"    {r:35s}: {cnt}×")

    # Trades that executed AFTER the kill window cleared
    print(f"\n  Performance of trades that fired AFTER a kill window cleared:")
    if kill_cleared:
        # For each cleared event, find the next trade for that symbol
        post_kill: list[Trade] = []
        for clear_ts, _, sym in kill_cleared:
            next_trade = next(
                (t for t in trades if t.symbol == sym and t.entry_ts >= clear_ts),
                None
            )
            if next_trade and next_trade not in post_kill:
                post_kill.append(next_trade)
        if post_kill:
            pk_wins = sum(1 for t in post_kill if t.net_pnl > 0)
            pk_net  = sum(t.net_pnl for t in post_kill)
            print(f"    Post-kill trades : {len(post_kill)}  ({pk_wins}W / {len(post_kill)-pk_wins}L)")
            print(f"    Post-kill net PnL: {_c(pk_net)}")
            print(f"    Post-kill WR     : {pk_wins/len(post_kill)*100:.1f}%")
        else:
            print("    No trades found immediately after kill clears.")

    # Kill switch verdict
    print(f"\n  {'Verdict':}")
    losing_streak_trades = [t for t in trades
                            if t.exit_reason == "STOP"]
    total_stop_loss_pnl = sum(t.net_pnl for t in losing_streak_trades)
    if net_total < 0 and len(kill_events) > 0:
        print(f"    Kill switch fired {len(kill_events)} times. Net session is negative.")
        print(f"    Cannot confirm improvement without blocked-trade outcomes,")
        print(f"    but kill switch prevented entries during established loss runs.")
    else:
        print(f"    Kill switch fired {len(kill_events)} times. Further sessions needed for verdict.")

    # -----------------------------------------------------------------------
    # 7. Counterfactual: BTC-only and ETH-only
    # -----------------------------------------------------------------------
    print(f"\n{_BOLD}{bar}{_RESET}")
    print(f"{_BOLD}  7. COUNTERFACTUAL REPORT{_RESET}")
    print(f"{_BOLD}{bar}{_RESET}")

    print(f"\n  {_BOLD}If only ETHUSDT had been traded (BTC disabled):{_RESET}")
    _print_stats_block("  ETH-only session", eth)

    print(f"\n  {_BOLD}If only BTCUSDT had been traded (ETH disabled):{_RESET}")
    _print_stats_block("  BTC-only session", btc)

    print(f"\n  {_BOLD}Combined (actual):{_RESET}")
    _print_stats_block("  Combined session", trades)

    # -----------------------------------------------------------------------
    # 8. Hold time distribution
    # -----------------------------------------------------------------------
    print(f"\n{_BOLD}{bar}{_RESET}")
    print(f"{_BOLD}  8. HOLD TIME DISTRIBUTION{_RESET}")
    print(f"{_BOLD}{bar}{_RESET}")

    hold_times = [t.hold_min for t in trades]
    if hold_times:
        buckets = [
            ("< 1 min",   [t for t in trades if t.hold_min < 1]),
            ("1 – 2 min", [t for t in trades if 1 <= t.hold_min < 2]),
            ("2 – 5 min", [t for t in trades if 2 <= t.hold_min < 5]),
            ("5 – 10 min",[t for t in trades if 5 <= t.hold_min < 10]),
            ("10 – 30 min",[t for t in trades if 10 <= t.hold_min < 30]),
            ("> 30 min",  [t for t in trades if t.hold_min >= 30]),
        ]
        print(f"\n  {'Bucket':15s}  {'Count':>5}  {'Net PnL':>10}  {'WR':>7}  {'Avg hold':>9}")
        print(f"  {bar2}")
        for label, bucket in buckets:
            if not bucket:
                continue
            bnet = sum(t.net_pnl for t in bucket)
            bwr  = sum(1 for t in bucket if t.net_pnl > 0) / len(bucket) * 100
            bavg = sum(t.hold_min for t in bucket) / len(bucket)
            print(f"  {label:15s}  {len(bucket):>5}  {_c(bnet, 10)}  {bwr:>6.1f}%  {bavg:>8.1f}m")
        print(f"  {bar2}")

        # Winners vs losers by hold time
        w_holds = [t.hold_min for t in trades if t.net_pnl > 0]
        l_holds = [t.hold_min for t in trades if t.net_pnl <= 0]
        print(f"\n  Avg hold — winners  : {sum(w_holds)/len(w_holds):.1f} min" if w_holds else "")
        print(f"  Avg hold — losers   : {sum(l_holds)/len(l_holds):.1f} min" if l_holds else "")

        # Trade-level detail
        print(f"\n  {'#':>3}  {'SYM':6} {'DIR':5} {'HOLD':>7} {'NET PnL':>9} {'REASON'}")
        print(f"  {bar2}")
        for t in sorted(trades, key=lambda x: x.hold_min):
            print(f"  {t.idx:>3}  {t.symbol:6} {t.direction[:5]:5} {t.hold_min:>6.1f}m  {_c(t.net_pnl, 8)}  {t.exit_reason}")

    print(f"\n{_BOLD}{bar}{_RESET}\n")


if __name__ == "__main__":
    main()
