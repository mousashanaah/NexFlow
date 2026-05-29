#!/usr/bin/env python3
"""Generate a self-contained HTML report from paper trading execution journals.

Usage:
    python scripts/generate_paper_report.py
    python scripts/generate_paper_report.py --journal-dir logs/paper --output reports/report.html
    python scripts/generate_paper_report.py --journal-dir logs/paper --no-open
    python scripts/generate_paper_report.py --journal-dir logs/paper --summary

The report is a single HTML file — no external dependencies, works offline.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from nexflow.analysis.analyze_paper_results import PaperAnalyzer
from nexflow.analysis.report_generator import generate_html_report
from nexflow.logging import configure_logging, get_logger


def _print_summary(result) -> None:
    """Print a compact text summary to stdout alongside the HTML report."""
    from nexflow.analysis.analyze_paper_results import AnalysisResult
    r: AnalysisResult = result
    if not r.has_data:
        print("No paper trading data found. Run the paper trader first.")
        return

    ts_ = r.trade_stats
    rs_ = r.risk_stats
    ex_ = r.execution_stats
    mk_ = r.market_stats
    net_pnl = r.final_equity - r.initial_equity

    bar = "=" * 65
    print(bar)
    print("NEXFLOW PAPER TRADING REPORT — SUMMARY")
    print(bar)
    print(f"  Period          : {r.date_range[0]}  →  {r.date_range[1]}")
    print(f"  Sessions        : {r.session_count}  |  Files: {r.files_loaded}  |  Events: {r.total_events:,}")
    print(bar)
    print("  TRADE STATISTICS")
    print(f"  {'Total trades':25s}: {ts_.total}")
    print(f"  {'Win rate':25s}: {ts_.win_rate * 100:.1f}%  ({ts_.wins}W / {ts_.losses}L)")
    pf_str = f"{ts_.profit_factor:.3f}" if ts_.profit_factor != float("inf") else "∞"
    print(f"  {'Profit factor':25s}: {pf_str}")
    print(f"  {'Expectancy (R)':25s}: {ts_.expectancy_r:+.4f}")
    print(f"  {'Avg R multiple':25s}: {ts_.avg_r:+.4f}")
    print(f"  {'Avg hold time':25s}: {ts_.avg_hold_minutes:.1f} min")
    print(f"  {'Best / Worst trade':25s}: {ts_.best_trade:+.2f} / {ts_.worst_trade:+.2f} USD")
    print(f"  {'TP / Stop / Forced':25s}: {ts_.tp_exits} / {ts_.stop_exits} / {ts_.forced_exits}")
    print(bar)
    print("  RISK STATISTICS")
    print(f"  {'Max drawdown':25s}: {rs_.max_drawdown * 100:.3f}%")
    print(f"  {'Avg drawdown':25s}: {rs_.avg_drawdown * 100:.3f}%")
    print(f"  {'Kill switch events':25s}: {rs_.kill_count}")
    print(f"  {'Max consec. losses':25s}: {rs_.max_consec_losses}")
    print(f"  {'Rejected signals':25s}: {rs_.rejected_count}")
    print(bar)
    print("  EXECUTION STATISTICS")
    print(f"  {'Avg slippage':25s}: {ex_.avg_slippage_pct * 100:.4f}%")
    print(f"  {'Total fees':25s}: {ex_.total_fees:.2f} USD")
    print(f"  {'Fee drag (gross)':25s}: {ex_.fee_drag_gross_pct * 100:.2f}%")
    print(f"  {'Avg fee / trade':25s}: {ex_.avg_fee_per_trade:.2f} USD")
    print(f"  {'Spread anomalies':25s}: {ex_.spread_anomaly_events}")
    print(bar)
    print("  MARKET BREAKDOWN")
    print(f"  {'Long':25s}: {mk_.long_trades} trades  WR {mk_.long_win_rate*100:.1f}%  PnL {mk_.long_pnl:+.2f}")
    print(f"  {'Short':25s}: {mk_.short_trades} trades  WR {mk_.short_win_rate*100:.1f}%  PnL {mk_.short_pnl:+.2f}")
    for sym, ss in sorted(mk_.by_symbol.items()):
        print(f"  {sym:25s}: {ss.trades} trades  WR {ss.win_rate*100:.1f}%  PnL {ss.net_pnl:+.2f}  AvgR {ss.avg_r:+.3f}")
    print(bar)
    print("  MONTHLY PnL")
    cum = 0.0
    for month, pnl in r.monthly_pnl.items():
        cum += pnl
        sign = "+" if pnl >= 0 else ""
        print(f"  {month:25s}: {sign}{pnl:9.2f} USD   cumulative: {cum:+.2f}")
    if not r.monthly_pnl:
        print("  No closed trades")
    print(bar)
    print(f"  Net PnL         : {net_pnl:+.2f} USD  ({net_pnl / r.initial_equity * 100:+.2f}%)")
    print(f"  Initial equity  : {r.initial_equity:,.2f} USD")
    print(f"  Final equity    : {r.final_equity:,.2f} USD")
    print(bar)
    if rs_.kill_reasons:
        print("  KILL SWITCH REASONS")
        for reason, count in sorted(rs_.kill_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"  {'  ' + reason:25s}: {count}×")
        print(bar)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate NexFlow paper trading HTML report")
    p.add_argument("--journal-dir", type=Path, default=Path("logs/paper"),
                   help="Directory containing .jsonl execution journals")
    p.add_argument("--output", type=Path, default=None,
                   help="Output HTML file path (default: <journal-dir>/report.html)")
    p.add_argument("--no-open", action="store_true",
                   help="Do not attempt to open the report in a browser")
    p.add_argument("--summary", action="store_true",
                   help="Print text summary to stdout in addition to generating HTML")
    p.add_argument("--log-level", default="WARNING",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    configure_logging(args.log_level)
    log = get_logger(__name__)

    journal_dir = args.journal_dir
    output_path = args.output or (journal_dir / "report.html")

    log.info("report.analyzing", journal_dir=str(journal_dir))

    analyzer = PaperAnalyzer()
    result = analyzer.load_and_analyze(journal_dir)

    if args.summary or not result.has_data:
        _print_summary(result)

    if not result.has_data:
        print(f"\nGenerating empty-state report → {output_path}")

    html = generate_html_report(result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"\nReport written → {output_path.resolve()}")

    if not args.no_open:
        try:
            import webbrowser
            webbrowser.open(output_path.resolve().as_uri())
        except Exception:
            pass   # non-interactive environment; silently skip


if __name__ == "__main__":
    main()
