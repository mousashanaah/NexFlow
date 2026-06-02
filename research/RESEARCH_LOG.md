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
| 7 | HTF trend-following (trailing) | 4H | per-breakout weak (19% reach 2R) | PF 1.11, +18.5%, DD 17.2%, **OOS PF 1.25 > IS 0.83** | **MARGINAL — survivor** |

## Pattern learned
Every fast (≤24h hold), single-feature, taker-fee, mean-reversion/fade signal on
BTC/ETH dies in Section A — these are the most-arbitraged signals on the most
liquid instruments, and the 0.12% round-trip fee buries any residual edge.

The first non-KILL (#7) broke that pattern on three axes at once: trend (not
fade), slow (4H, multi-day holds → low fee drag), and let-winners-run (trailing
exit → positive skew). Its edge is the *exit structure*, not the entry: a few big
winners pay for many small losers, and it held up out-of-sample.

## Active lead
HTF trend-following is the backbone of managed-futures funds **as a diversified
portfolio across many uncorrelated markets** — single-instrument it is mediocre
(CAGR 3.2% on 2 symbols), but a wide basket smooths equity and raises aggregate
return. Next test: the EXACT pre-committed rule (30-bar 4H channel, 3×ATR
trailing) across a wide universe of Bitget alt-perps. No parameter changes —
this is out-of-sample validation across instruments, not optimization.

## Operational notes
- Funding cache: data/funding/{SYMBOL}_funding.parquet (Binance, committed once).
- OI cache: data/oi/{SYMBOL}_OI_1H.parquet (Bybit, committed once).
- Research runs: GitHub Actions `run_research.yml`, one script at a time.
  **Do not run two jobs concurrently** — both push to the same branch and the
  second commit is rejected (lost the #6 result file this way; recovered from the
  job log).
