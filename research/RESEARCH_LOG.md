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
| 7b | HTF trend wide 12-coin (unfiltered) | 4H | same | PF 0.92, -69.4%, DD 71.9% | **KILL** |
| 7c | HTF trend wide 12-coin + SMA-200 filter | 4H | same | PF 0.98, -36.8%, DD 68.2% | **KILL** |
| 7d | HTF trend 4-coin (BTC/ETH/SOL/TRX) + SMA-200 | 4H | same | PF 1.16, +45.5%, DD 34.4%, OOS PF 1.26 | **MARGINAL** |
| 8 | Cross-sectional momentum (rank-based) | 1D | — | *in progress* | *pending* |

## Pattern learned
Every fast (≤24h hold), single-feature, taker-fee, mean-reversion/fade signal on
BTC/ETH dies in Section A — these are the most-arbitraged signals on the most
liquid instruments, and the 0.12% round-trip fee buries any residual edge.

HTF trend (7a-7d): the mechanism is real but narrow. Only 4 of 12 coins survive
with a regime filter. IS PF is 0.98 (barely profitable in 2021-2024); the
apparent MARGINAL result is driven by 2025-2026 recency (PF 1.49/2.11). Two
massive short trades in May 2026 (+$10K ETH, +$9.5K BTC) distort the picture.
CAGR 7.3% is insufficient for the "rich by year-end" target.

**Structural insight:** Channel breakouts work best when trends are long and
smooth. 2025-2026 provided that. 2021-2024 was choppier — many fake breakouts
before the real move, then trailing stop hit on consolidation. This is the
fundamental ceiling of single-entry breakout logic.

## Active lead — Mechanism #8: Cross-sectional momentum
Script: `scripts/backtest_cross_momentum.py`  
Pre-committed parameters: lookback=20 days, rebal=7 days, long top-3/short bottom-3, 12-coin universe  
Rationale: portfolio approach eliminates single-coin concentration risk; weekly
rebalancing means low fee drag; relative-strength ranking captures cross-coin
momentum which is distinct from absolute-price breakout.

If this passes, it can run alongside 7d as two uncorrelated engines.

## Candle data cache (as of 2026-06-03)
Committed to git: BTC, ETH, SOL, TRX (1H + 1D). Remaining 8 coins needed for #8.
The first run of #8 will download BNB/XRP/ADA/DOGE/AVAX/LINK/LTC/DOT and cache them.

## Operational notes
- Candle cache: data/candles/{SYMBOL}_{TF}.parquet — committed to git.
  Downloads are incremental. First run after new symbols: slow (~20m per 12 symbols).
  Repeat runs with cached data: ~1-3 min.
- Research runs: GitHub Actions `run_research.yml`, one script at a time.
  **Do not run two jobs concurrently** — concurrent pushes will conflict.
- `extra_args` must include `--symbols ...` (workflow only passes extra_args to script).
