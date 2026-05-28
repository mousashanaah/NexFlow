#!/usr/bin/env python3
"""Run walk-forward robustness harness + Monte Carlo analysis.

Usage:
    python scripts/run_walk_forward.py --data-dir data/candles
    python scripts/run_walk_forward.py --data-dir data/candles --mode anchored --train 3000 --test 750
    python scripts/run_walk_forward.py --data-dir data/candles --sweep --mc-sims 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import pyarrow.parquet as pq

from nexflow.config import get_config
from nexflow.logging import configure_logging, get_logger
from nexflow.services.candles.candle_engine import Candle, TIMEFRAMES
from nexflow.services.strategy.backtest_runner import BacktestConfig, BacktestRunner
from nexflow.services.strategy.momentum_strategy import MomentumConfig, MomentumStrategy
from nexflow.services.strategy.paper_execution import ExecutionConfig
from nexflow.services.strategy.risk_engine import RiskConfig
from nexflow.research.walk_forward import WFConfig, WalkForwardEngine, WalkForwardResult
from nexflow.research.parameter_sweeper import ParamRange, ParameterSweeper
from nexflow.research.monte_carlo import MCConfig, MonteCarloEngine
from nexflow.research.equity_curve_analysis import EquityCurveAnalysis


def _load_candles(data_dir: Path, symbols: list[str]) -> dict[str, dict[str, list[Candle]]]:
    all_candles: dict[str, dict[str, list[Candle]]] = {}
    for symbol in symbols:
        all_candles[symbol] = {}
        for tf in TIMEFRAMES:
            path = data_dir / f"{symbol}_{tf}.parquet"
            if not path.exists():
                continue
            rows = pq.read_table(path).to_pylist()
            candles = [
                Candle(
                    symbol=r["symbol"], timeframe=r["timeframe"],
                    open_time=r["open_time"], close_time=r["close_time"],
                    open=r["open"], high=r["high"], low=r["low"], close=r["close"],
                    volume=r["volume"], buy_volume=r["buy_volume"],
                    sell_volume=r["sell_volume"], trade_count=r["trade_count"],
                    vwap=r["vwap"], spread_avg=r["spread_avg"],
                    spread_max=r["spread_max"],
                    volatility_estimate=r["volatility_estimate"],
                    is_final=r["is_final"],
                )
                for r in rows if r.get("is_final")
            ]
            if candles:
                all_candles[symbol][tf] = sorted(candles, key=lambda c: c.close_time)
    return all_candles


def _print_wf_results(result: WalkForwardResult) -> None:
    bar = "=" * 65
    print(bar)
    print("WALK-FORWARD RESULTS")
    print(bar)
    print(f"  Total folds          : {len(result.folds)}")
    conclusive = [f for f in result.folds if f.is_conclusive]
    print(f"  Conclusive folds     : {len(conclusive)}")
    print(f"  Total OOS trades     : {result.total_oos_trades}")
    print(f"  Mean OOS win rate    : {result.mean_oos_win_rate * 100:.1f}%")
    print(f"  Mean OOS PnL         : {result.mean_oos_pnl:+.2f} USD")
    print(f"  Mean OOS max DD      : {result.mean_oos_drawdown * 100:.2f}%")
    print(f"  Mean OOS Sharpe      : {result.mean_oos_sharpe:.2f}")
    print(f"  Degradation ratio    : {result.degradation_ratio:.3f}  (1.0 = no degradation)")
    print(f"  Consistency score    : {result.consistency_score * 100:.1f}%  (% folds with +PnL OOS)")
    print(bar)
    print(f"  {'Fold':>4}  {'IS WR':>6}  {'OOS WR':>6}  {'OOS PnL':>10}  {'OOS DD':>7}  {'OOS Sh':>7}  {'Trades':>6}")
    print(f"  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*7}  {'-'*7}  {'-'*6}")
    for f in result.folds:
        conclusive_mark = "" if f.is_conclusive else "*"
        print(
            f"  {f.window.fold_idx:>4}  "
            f"{f.train_metrics.win_rate*100:>5.1f}%  "
            f"{f.test_metrics.win_rate*100:>5.1f}%  "
            f"{f.test_metrics.net_pnl:>+10.2f}  "
            f"{f.test_metrics.max_drawdown*100:>6.2f}%  "
            f"{f.test_metrics.sharpe:>7.2f}  "
            f"{f.test_metrics.total_trades:>5}{conclusive_mark}"
        )
    if any(not f.is_conclusive for f in result.folds):
        print("  * inconclusive (too few OOS trades)")
    print(bar)


def _print_mc_results(mc, initial_equity: float) -> None:
    bar = "=" * 65
    print(bar)
    print("MONTE CARLO RESULTS")
    print(bar)
    print(f"  Simulations          : {mc.n_simulations}")
    print(f"  Ruin probability     : {mc.ruin_probability * 100:.2f}%")
    print(f"  Profitable paths     : {mc.pct_profitable * 100:.1f}%")
    print(f"  Mean max drawdown    : {mc.mean_max_drawdown * 100:.2f}%")
    print(f"  Worst drawdown       : {mc.worst_drawdown * 100:.2f}%")
    print()
    print("  Final equity percentiles:")
    for pct, val in sorted(mc.final_equity_percentiles.items()):
        pnl = val - initial_equity
        print(f"    p{int(pct*100):>2}  {val:>12.2f}  ({pnl:+.2f})")
    print()
    print("  Max drawdown percentiles:")
    for pct, val in sorted(mc.drawdown_percentiles.items()):
        print(f"    p{int(pct*100):>2}  {val*100:>6.2f}%")
    print(bar)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NexFlow walk-forward robustness harness")
    p.add_argument("--data-dir", type=Path, default=Path("data/candles"))
    p.add_argument("--symbols", type=str, default=None)
    p.add_argument("--equity", type=float, default=100_000.0)
    p.add_argument("--risk", type=float, default=0.005)
    p.add_argument("--mode", choices=["rolling", "anchored"], default="rolling")
    p.add_argument("--train", type=int, default=2_000, help="Training bars per fold")
    p.add_argument("--test", type=int, default=500, help="OOS bars per fold")
    p.add_argument("--step", type=int, default=500, help="Step size between folds (rolling)")
    p.add_argument("--sweep", action="store_true", help="Run parameter stability sweep")
    p.add_argument("--mc-sims", type=int, default=1_000, help="Monte Carlo simulations")
    p.add_argument("--mc-seed", type=int, default=42)
    p.add_argument("--log-level", type=str, default="WARNING",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    configure_logging(args.log_level)
    log = get_logger(__name__)

    cfg = get_config()
    symbols = (
        [s.strip() for s in args.symbols.split(",") if s.strip()]
        if args.symbols else cfg.market_data.symbols
    )

    log.info("wf.loading", data_dir=str(args.data_dir), symbols=symbols)
    all_candles = _load_candles(args.data_dir, symbols)
    total_bars = sum(len(tf_c) for s in all_candles.values() for tf_c in s.values())

    if total_bars == 0:
        print(f"No candle data found in {args.data_dir}. Run run_candle_engine.py first.")
        sys.exit(1)

    log.info("wf.start", total_bars=total_bars)

    bt_cfg = BacktestConfig(
        initial_equity=args.equity,
        risk=RiskConfig(max_risk_per_trade=args.risk),
        execution=ExecutionConfig(),
    )

    # --- Walk-forward ---
    wf_cfg = WFConfig(
        train_bars=args.train,
        test_bars=args.test,
        step_bars=args.step,
        mode=args.mode,
    )
    wf_engine = WalkForwardEngine(
        strategy_factory=MomentumStrategy,
        bt_cfg=bt_cfg,
        wf_cfg=wf_cfg,
    )
    wf_result = wf_engine.run(all_candles)
    _print_wf_results(wf_result)

    # --- Monte Carlo on full backtest trades ---
    strategy = MomentumStrategy(MomentumConfig())
    runner = BacktestRunner(strategy, bt_cfg)
    full_metrics = runner.run(all_candles)

    all_trades = []
    for sym_candles in all_candles.values():
        pass  # trades come from portfolio; re-run collects them

    # Use the portfolio's closed trades collected during the full run above
    # Runner doesn't expose trades directly from metrics; use pnl_distribution
    trade_pnls = full_metrics.pnl_distribution

    if trade_pnls:
        from nexflow.services.strategy.signal_models import ClosedTrade, Direction, ExitReason
        synthetic_trades = [
            ClosedTrade(
                symbol="COMBINED", direction=Direction.LONG,
                entry_price=1.0, exit_price=1.0 + pnl, total_size=1.0,
                entry_time=i, exit_time=i + 1,
                pnl=pnl, fees=0.0, hold_bars=1,
                equity_at_entry=args.equity,
                exit_reason=ExitReason.TP1,
            )
            for i, pnl in enumerate(trade_pnls)
        ]

        mc_engine = MonteCarloEngine(MCConfig(n_simulations=args.mc_sims, seed=args.mc_seed))
        mc_result = mc_engine.run(synthetic_trades, args.equity)
        _print_mc_results(mc_result, args.equity)
    else:
        print("\nNo trades from full backtest — skipping Monte Carlo.")

    # --- Parameter sweep (optional) ---
    if args.sweep:
        print("\n" + "=" * 65)
        print("PARAMETER STABILITY SWEEP")
        print("=" * 65)
        ranges = [
            ParamRange("rel_vol_threshold", [1.1, 1.3, 1.5, 1.8, 2.0]),
            ParamRange("atr_period", [10, 14, 20]),
        ]

        def _make_strategy(rel_vol_threshold: float = 1.5, atr_period: int = 14) -> MomentumStrategy:
            return MomentumStrategy(MomentumConfig(
                rel_vol_threshold=rel_vol_threshold,
                atr_period=atr_period,
            ))

        sweeper = ParameterSweeper(
            strategy_factory=_make_strategy,  # type: ignore[arg-type]
            bt_cfg=bt_cfg,
            param_ranges=ranges,
        )
        summary = sweeper.run(all_candles)

        if summary.best_by_composite:
            b = summary.best_by_composite
            print(f"  Best composite : {b.params}  score={b.composite_score:.3f}  sharpe={b.metrics.sharpe:.2f}")
        if summary.best_by_sharpe:
            b = summary.best_by_sharpe
            print(f"  Best Sharpe    : {b.params}  sharpe={b.metrics.sharpe:.2f}")
        if summary.best_by_drawdown:
            b = summary.best_by_drawdown
            print(f"  Min drawdown   : {b.params}  dd={b.metrics.max_drawdown*100:.2f}%")
        print(f"\n  Stable region ({len(summary.stable_region)} combos):")
        for r in summary.stable_region[:5]:
            print(f"    {r.params}  stability={r.stability_score:.3f}  composite={r.composite_score:.3f}")
        print("=" * 65)


if __name__ == "__main__":
    main()
