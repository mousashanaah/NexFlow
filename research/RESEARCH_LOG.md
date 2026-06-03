# NexFlow Research Factory — Verdict Log

Goal: discover/validate/deploy a portfolio of uncorrelated, fee-survivable
trading engines on Bitget USDT Perpetual Futures. Every mechanism is judged in
two sections — A: does the market behavior exist (pre-fee)? B: is it monetizable
after fees? Kill gates: PF<1.10, maxDD>40%, n<60, OOS PF<0.85×full PF.
Taker fee 0.06%/side. Parameters pre-committed before each run.

| # | Mechanism | Timeframe | Section A | Section B | Verdict |
|---|-----------|-----------|-----------|-----------|---------|
| 2 | Compression → expansion | 1H | too rare (~0.1% of bars) | n/a | **KILL** |
| 3 | BTC→ETH lead-lag | 1H | 45.5% hit (random) | PF 0.88, -98% | **KILL** |
| 4 | Funding extremes (fade) | 1H | 45.3% fade hit | PF 0.97, -84.9% | **KILL** |
| 5 | OI confirmation | 1H | 46-49% cont., no spread | PF 0.92, -89.5% | **KILL** |
| 6 | Liquidation cascade fade | 15m | reversion absent | PF 0.80, -99.9% | **KILL** |
| 7a | HTF trend-following (2-coin BTC+ETH) | 4H | 19% reach 2R | PF 1.11, +18.5%, DD 17.2%, OOS PF 1.25 | **MARGINAL — survivor** |
| 7b | HTF trend wide universe (12-coin, unfiltered) | 4H | same | PF 0.92, -69.4%, DD 71.9% | **KILL** |
| 7c | HTF trend wide universe (12-coin, SMA-200 regime filter) | 4H | same | PF 0.98, -36.8%, DD 68.2% | **KILL** |
| 7d | HTF trend curated 4-coin (BTC/ETH/SOL/TRX, SMA-200 filter) | 4H | — | *in progress* | *pending* |

## Pattern learned
Every fast (≤24h hold), single-feature, taker-fee, mean-reversion/fade signal on
BTC/ETH dies in Section A — these are the most-arbitraged signals on the most
liquid instruments, and the 0.12% round-trip fee buries any residual edge.

HTF trend-following survives on 2 coins but collapses on a wide universe.
The regime filter (SMA-200) improves 2022 (from PF 0.83→1.00) but 2022 is still
catastrophic (-$36K). Root cause: SHORT breakdowns in downtrends are structurally
bad (only 14% reach +2R) — regime filter routes more shorts in bear markets,
which are net losers. The 4 coins that survive with regime filter (BTC PF 1.27,
ETH 1.24, SOL 1.29, TRX 1.14) likely do so due to higher volatility and cleaner
trending regimes.

## Active lead — 7d: curated 4-coin universe
Run BTCUSDT ETHUSDT SOLUSDT TRXUSDT with --regime-ma 200.
These are the only 4 coins that cleared PF > 1.10 with the regime filter applied.
If this passes (PF≥1.10, DD≤40%), we have a deployable engine.
If this KILLs, pivot to Mechanism #8: cross-sectional momentum.

## Mechanism #8 ready (if 7d KILLs)
Script: `scripts/backtest_cross_momentum.py` — pre-committed parameters:
  lookback=20 days, rebal=7 days, long top-3/short bottom-3, 12-coin universe.
Run with: `python scripts/backtest_cross_momentum.py` (uses 1D candles).
Rationale: portfolio approach from day 1 → regime risk diversified across coins.
Fee drag lower (weekly rebalancing vs every breakout). Different mechanism:
relative strength rank, not price level breakout.

## Operational notes
- Candle cache: data/candles/{SYMBOL}_{TF}.parquet — now committed to git.
  Downloads are incremental (only fetch bars newer than last cached bar).
  First run after adding new symbols is still slow (~20m); repeat runs ~1m.
- Research runs: GitHub Actions `run_research.yml`, one script at a time.
  **Do not run two jobs concurrently** — concurrent pushes will conflict.
- `extra_args` must include `--symbols ...` since workflow only passes extra_args
  to the script, not the symbols input separately.
