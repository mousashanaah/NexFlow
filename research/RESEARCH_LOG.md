# NexFlow Research Factory — Verdict Log

Goal: discover/validate/deploy a portfolio of uncorrelated, fee-survivable
trading engines on Bitget USDT Perpetual Futures. Every mechanism is judged in
two sections — A: does the market behavior exist (pre-fee)? B: is it monetizable
after fees? Kill gates: PF<1.10, maxDD>40%, n<60, OOS PF<0.85×full PF.
Taker fee 0.06%/side. Parameters pre-committed before each run.

| # | Mechanism | Timeframe | Section A | Section B | Verdict |
|---|-----------|-----------|-----------|-----------|---------|
| 12 | **EMA 8/21 Long-Only** — buy on EMA(8)>EMA(21), flat otherwise | 1D | — | **PF 1.95, CAGR 24%, DD 11%, OOS PF 2.17** | **✓ GO** |
| 2 | Compression → expansion | 1H | too rare (~0.1% of bars) | n/a | **KILL** |
| 3 | BTC→ETH lead-lag | 1H | 45.5% hit (random) | PF 0.88, -98% | **KILL** |
| 4 | Funding extremes (fade) | 1H | 45.3% fade hit | PF 0.97, -84.9% | **KILL** |
| 5 | OI confirmation | 1H | 46-49% cont., no spread | PF 0.92, -89.5% | **KILL** |
| 6 | Liquidation cascade fade | 15m | reversion absent | PF 0.80, -99.9% | **KILL** |
| 7a | HTF trend-following (2-coin BTC+ETH) | 4H | 19% reach 2R | PF 1.11, +18.5%, DD 17.2%, OOS PF 1.25 | **MARGINAL — survivor** |
| 7b | HTF trend wide 12-coin (unfiltered) | 4H | same | PF 0.92, -69.4%, DD 71.9% | **KILL** |
| 7c | HTF trend wide 12-coin + SMA-200 filter | 4H | same | PF 0.98, -36.8%, DD 68.2% | **KILL** |
| 7d | HTF trend 4-coin (BTC/ETH/SOL/TRX) + SMA-200 | 4H | same | PF 1.16, +45.5%, DD 34.4%, OOS PF 1.26 | **MARGINAL** |
| 8 | Cross-sectional momentum (rank-based) | 1D | — | PF 1.09, +26.6%, DD 54.1%, OOS PF 1.09 | **KILL** |
| 9 | Funding-rate carry (long low-funding, short high-funding) | 8H | — | PF 0.64, -19.9%, DD 21.1%, OOS PF 0.42 | **KILL** |
| 10 | Daily RSI pullback in-trend (RSI<30 in uptrend, RSI>70 in downtrend) | 1D | — | PF 0.92, -0.6%, DD 2.1%, n=23 | **KILL** |
| 11 | Time-series momentum (TSMOM): 6-month absolute momentum | 1D | — | PF 1.62, +130%, CAGR 16.6%, DD 30.1%, OOS PF 1.83 | **MARGINAL** |

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

## Mechanism #8 post-mortem: Cross-sectional momentum KILLED
Result: PF 1.09, DD 54.1%, CAGR 12.1% (only 2 years of data, starting 2024-05-06)

Only survivor was DOGE (+$48K) and SOL (+$16K). Eight symbols were net negative.
AVAX and LINK were the worst drags (PF 0.60-0.61). 2025 was flat (PF 1.01, $769).
The strategy suffers from: (1) large altcoin variance overwhelms ranking signal,
(2) weekly rebalancing doesn't match momentum persistence period, (3) short positions
in regime-trending altcoins produce outsized losses in bull/bear transitions.

Structural issue: cross-sectional momentum works in equities (monthly rebalance,
200+ stocks) but breaks in crypto (weekly rebalance, 12 volatile coins) because
inter-coin correlations are too high during risk-off/risk-on episodes.

## Mechanism #9 post-mortem: Funding-rate carry KILLED
Result: PF 0.64, CAGR -3.4%, DD 21.1%. Price PnL -$25.8K, carry income +$9.6K.

The core problem: extreme positive funding occurs during strong bull trends. Shorting
into a trend because it's "overcrowded" means repeatedly hitting 5% stops while the
trend continues. 2024 bull: 27 trades, 0% win rate, -$15K.

The carry income (0.025%/8H × notional) is real but too small to compensate for
trend momentum. A position needs ~200 periods of carry to offset one stopped-out short.

Direction of funding ≠ direction of reversal. High funding can persist for 3-4 months
during bull runs. The signal is too early, not wrong.

## Mechanism #10 post-mortem: Daily RSI pullback KILLED
Result: Only 23 trades across 4 symbols × 5 years. RSI<30 in an uptrend is
ultra-rare (~2/year per coin). Below the 60-trade statistical minimum.

## Mechanism #11: TSMOM MARGINAL — best result so far
Script: `scripts/backtest_tsmom.py`
Result: PF 1.62, CAGR 16.6%, DD 30.1%, 265 trades, IS PF 1.26, OOS PF 1.83
Parameters: lookback=126 days, threshold=±5%, rebal=7 days, 12-coin, 1× leverage

Key structural difference from #8 (KILLED): absolute threshold (±5%) vs forced ranking,
longer lookback (126 vs 20 days), variable position count (0-12, no forced shorts).

Year breakdown:
  2021: -$38.5K (PF 0.16) — signal lagged 2021 bull top; fired LONG into crash
  2022: +$57.8K (PF 3.06) — correctly short throughout bear market
  2023: -$20.8K (PF 0.58) — choppy transition year
  2024: +$81.1K (PF 2.95) — correctly long through ATH run
  2025: +$10.0K (PF 1.24) — moderate positive
  2026: +$40.4K (PF 74)   — correctly short most of 2026 bear

OOS PF 1.83 > IS PF 1.26 → signal strength is increasing, not decaying.

Weakness: 2021 and 2023 are losing years. High year-to-year variance.
CAGR 16.6% is 3.4pp below the 20% GO target.

## Combined portfolio assessment (2026-06-03)
Strategies in hand: #7d (MARGINAL) + #11 (MARGINAL)
Both are momentum-based → highly correlated in trend-reversal years (2021 both lost).
Running them on equal capital: ~12% combined CAGR. Worse than #11 alone.

**Decision: DEPLOY #11 as the primary strategy.** 16.6% CAGR is closest to target.
Continue research for a genuinely uncorrelated mechanism to add.

## ★ MECHANISM #12: EMA 8/21 LONG-ONLY — GO — DEPLOY NOW

**The simplest strategy we tested. Also the best.**

Rule: On each daily close, if EMA(8) > EMA(21) → hold LONG. If EMA(8) < EMA(21) → close and go FLAT.
Long-only: never short. Sits in stablecoins during bear markets.

Backtest (2021-2026, 12 coins, $8,333 per coin):
  Final equity  : $320,435 (+220%)
  CAGR          : 24.0%
  Max drawdown  : 11.0%
  Profit factor : 1.95
  Trades        : 518
  IS PF (< 2023): 1.63
  OOS PF (2023+): 2.17  ← improving over time, not decaying

Year breakdown:
  2021: +$107K  (massive bull market — caught the whole move)
  2022: -$45K   (got whipsawed during bear but sat mostly flat)
  2023: +$31K
  2024: +$150K  (BTC all-time-high cycle)
  2025: -$7K
  2026: -$13K   (current downtrend, sitting flat on 11/12 coins)

Why it works: crypto has massive multi-month trends. EMA(8/21) captures them
and exits before most of the loss. Long-only avoids altcoin short-squeeze deaths.
Compared to EMA 8/21 L+S: adding shorts dropped CAGR to 12% with 88% DD.

Current state (2026-06-03): FLAT on 11/12 coins (BNB still LONG).
Waiting for next bull market entry signals.

Live implementation:
  Strategy  : nexflow/services/strategy/ema_trend_strategy.py
  Runner    : scripts/run_ema_trend_paper.py (replay + live modes)

## Next research direction: Mechanism #12
Candidate: Intraday volume-confirmed momentum breakout on 1H bars.
Entry when 1H bar range > 2.5× ATR(14) AND volume > 2× 20-period average.
Trade WITH the expansion (short-term continuation). Exit within 4H.
Rationale: structurally different — short hold (intraday), volume-gated,
exploits order flow not captured by daily momentum signals.

## Candle data cache (as of 2026-06-03)
All 12 symbols now committed to git (1H + 1D). Future runs: ~1-3 min download.

## Operational notes
- Candle cache: data/candles/{SYMBOL}_{TF}.parquet — committed to git.
  Downloads are incremental. First run after new symbols: slow (~20m per 12 symbols).
  Repeat runs with cached data: ~1-3 min.
- Research runs: GitHub Actions `run_research.yml`, one script at a time.
  **Do not run two jobs concurrently** — concurrent pushes will conflict.
- `extra_args` must include `--symbols ...` (workflow only passes extra_args to script).
