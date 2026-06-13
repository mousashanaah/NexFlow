# NexFlow V8.63 — Stock Book Specification (strict, no-lookahead)

This is the production spec for adding a **stock-futures book** to V8.63, trading
Bitget TradFi stock perpetuals alongside the existing crypto book. Every number
here comes from a **lookahead-free** backtest (signals detected at the close that
prints them, executed at the *next* bar's open; stops modelled intraday).

Sources of truth:
- `scripts/test_stock_deep_research.py` — universe characterization, combo finder,
  scalp/pairs/insider tests, crypto↔stock pairing.
- `scripts/test_stock_risk_mgmt.py` — strict exit / sizing / check-frequency /
  circuit-breaker sweeps. **This file's `RECOMMENDED` dict is the live config.**
- `scripts/test_multi_asset.py` — combined crypto+stock allocation (delegates to
  the strict engine above).

---

## 1. The honesty correction (why earlier numbers were inflated)

The first-pass research entered at the **same close** that generated the signal
(lookahead) and only checked stops on the **close**. Rebuilding the engine
lookahead-free dropped the naive equal-weight combo from **+58% → +13.8% CAGR**.

The high returns are still real, but they come from **full capital deployment**,
not signal magic. Equal-weight `/6` left 4/6 of the account idle in cash whenever
fewer than six names had signals. Deploying full equity across the *active* names
recovers the return legitimately.

---

## 2. Universe (de-biased)

Combo-finder ran **all C(11, 3..6) = 1,419 combinations** of the tradeable names
(ETFs excluded as they're regime anchors, not trades). Winner:

> **GOOGL + AMD + NFLX + MSFT + AMZN + COIN**

- Strict standalone: **CAGR +51.9% · DD 31.2% · Sharpe 1.43 · PF 3.77 · 6/6 walk-fwd**
- Notably **excludes** NVDA and MSTR despite their huge CAGRs — their drawdowns
  drag portfolio risk-adjusted return. COIN earns its slot via diversification,
  not raw return.

**Caveat that still stands:** the 16-ticker pool was assembled with hindsight
(COIN/MSTR/NVDA are known winners). Yahoo Finance is blocked in the cloud env, so
the 69-ticker de-biased downloader (`scripts/download_debiased_universe.py`) must
be run **locally** and the combo finder re-run to confirm the combo isn't an
artifact of a small pool. Treat the magnitude as optimistic until then; the
*risk-adjusted shape* (Sharpe ~1.4, 6/6 WF) is what to trust.

---

## 3. Strategy (adapted from crypto V8.63)

| Element | Stock book | vs crypto V8.63 |
|---|---|---|
| Trend filter | 8/21 EMA bull | same |
| Confirmation | MACD > signal | same |
| Regime gate | **per-asset SMA200** (price > own 200d SMA) | crypto uses BTC-anchored AND-regime |
| Momentum gate | 90d momentum > 0 | crypto uses 20d |
| Direction | long-only | crypto adds TSMOM shorts |
| Entry exec | next bar open (no lookahead) | — |

Per-asset SMA200 (not an SPY anchor) is the key parameter that made the stock book
work — an SPY anchor produced near-zero trades.

---

## 4. Exits / take-profit (sweep A)

Tested 15 exit methods. Ranked by risk-adjusted score (Sharpe − 1.5·DD + WF bonus):

| Exit | CAGR | DD | Sharpe | WF |
|---|---|---|---|---|
| **MACD-cross + 10% hard stop** ✅ | +13.8%* | **14.2%** | **1.11** | 6/6 |
| TP +30% + 10% stop | +17.2% | 16.5% | 1.13 | 6/6 |
| Trailing 15% from peak | +15.1% | 20.4% | 1.00 | 6/6 |
| EMA-cross + 10% stop (baseline) | +15.5% | 19.3% | 0.92 | 6/6 |
| Chandelier 3×ATR | +13.1% | 19.1% | 0.87 | 6/6 |

\* equal-weight base; full-deployment sizing lifts this to ~52% (§5).

**Decision: MACD-cross exit + 10% hard stop.**
- Lowest drawdown (14.2%) and best Sharpe of the family.
- MACD crosses down *before* the slower EMA cross → exits losers earlier.
- More trades (222) = more responsive, but win rate holds at ~40% with a 3.0:1
  win/loss ratio (PF ~2.0 at base, ~3.8 deployed).
- A fixed take-profit (+30%) was competitive but **caps winners** — bad for a
  trend system whose edge is letting winners run. We let MACD decide the exit.

**Take-profit verdict:** no fixed TP. The MACD-cross *is* the profit-taking rule —
it exits when momentum rolls over, which is when a trend trade should close.

---

## 5. Position sizing & leverage (sweep B)

| Sizing | CAGR | DD | Sharpe | WF |
|---|---|---|---|---|
| Equal 1/6 slot (cash idle) | +13.8% | 14.2% | 1.11 | 6/6 |
| **Equal across active, gross 1.0x** ✅ | **+51.2%** | 30.7% | 1.41 | **6/6** |
| Inverse-vol full-deploy, 1.0x | +55.4% | 30.2% | 1.40 | 5/6 |
| Equal across active, gross 1.3x | +56.6% | 30.9% | 1.46 | 5/6 |
| Risk-stop 2.5%/trade, gross 1.3x | +21.9% | 18.8% | 1.22 | 6/6 |
| Risk-stop 1.5%/trade, gross 1.0x | +12.3% | 13.5% | 1.09 | 6/6 |

**Decision: equal-across-active, gross 1.0x (no leverage).**
- Picks the highest return that **stays 6/6 on walk-forward**. Inverse-vol and
  1.3x leverage squeeze out a few more % of CAGR but drop a WF window — not worth
  trading robustness for ~5% CAGR.
- **No leverage**: gross notional never exceeds equity. This is the primary
  account-safety control.
- For a "lose as little as possible" mandate, `RECOMMENDED_SAFE` (risk-stop
  2.5%/trade) gives **+21.9% CAGR / 18.8% DD / 6/6** — roughly half the return for
  two-thirds the drawdown. Switch by changing one import in the bot.

---

## 6. How often should the bot check? (sweep C) — **your question, answered**

Intraday stop monitoring vs once-per-day close-only:

| Mode | CAGR | DD | Sharpe |
|---|---|---|---|
| Intraday stop check | +55.4% | 30.2% | 1.40 |
| Close-only stop check | +57.0% | 30.4% | 1.43 |

Difference: **DD +0.2%, CAGR −1.5%** — negligible, and close-only is *slightly
better* (avoids intraday whipsaw).

**Decision: this is a daily-bar strategy. Check ONCE per day, near the US market
close (21:00 UTC / 16:00 ET), when the daily bar completes.**
- The current crypto cadence (every ~6h) is **wrong for the stock book** — it
  risks acting on incomplete bars and adds whipsaw with zero DD benefit.
- The crypto book still runs its own 24/7 loop; only the stock decision logic is
  gated to one post-close evaluation per day.
- Optional cheap insurance: 1–2 *stop-only* glances during US hours to catch a
  true gap crash. The data says it won't improve returns, but it bounds tail risk
  for free. Not required.

---

## 7. Account safety: circuit breaker (sweep D) — **counterintuitive, important**

| Portfolio circuit breaker | CAGR | DD | Sharpe |
|---|---|---|---|
| **OFF** ✅ | +55.4% | 30.2% | 1.40 |
| close-all at 30% DD | +33.7% | 34.3% | 1.15 |
| close-all at 25% DD | +10.6% | 33.4% | 0.52 |
| close-all at 20% DD | +16.7% | 26.6% | 0.78 |

**Decision: NO portfolio circuit breaker on the stock book.**
- Every threshold *hurt*: a 25% kill-switch cut CAGR to 10.6% while DD stayed
  33% — it sells the bottom and misses the recovery. Classic trend-following trap.
- This **differs from crypto V8.63**, which does use a 20% DD breaker. For the
  long-only stock trend book, per-position 10% stops + no leverage are the correct
  and sufficient risk controls.

---

## 8. Crypto ↔ stock pairing

- Monthly return correlation crypto vs stock book = **0.21** (genuinely low →
  real diversification, not just two long-beta books).
- H1-2025: crypto −31.3%, stock book +21.6% → 50/50 only −6.7%. The books offset
  because they run on different regime calendars.
- Combined allocations (strict engine, full period, $5K):

| Mix | Equity | CAGR | maxDD* | H1-2025 | Walk-fwd |
|---|---|---|---|---|---|
| 100% crypto | $58.2K | 56.9% | 25% | −31.3% | 5/6 |
| 100% stock | $48.8K | 51.9% | 31% | +21.6% | 6/6 |
| 70/30 | $55.4K | 55.5% | 31%* | −17.0% | 5/6 |
| **50/50** | $53.5K | 54.5% | 31%* | **−6.7%** | 5/6 |

\* combined DD shown as worst-single-book (conservative); true combined DD is
lower because of the 0.21 correlation.

**Dynamic allocation** (regime-confidence): weight the book whose regime is
"on" (crypto book ≈70% when BTC > SMA200, else flip to stock-heavy). Beat fixed
50/50 in the worst years (2022, 2025). Logic is specced in
`test_stock_deep_research.pairing_logic_test()`; production wiring pending.

**Recommendation:** run **50/50** as the base allocation for capital preservation,
or 70/30 crypto-tilt if maximizing absolute return and you can stomach crypto's
H1-style drawdowns. The diversification benefit is the whole point — don't go
100% of either.

---

## 9. Side strategies tested

- **Stock scalping (burst continuation):** PASS at the daily level — Bitget stock
  perp fees (~0.02% maker) are low enough that 3-day continuation after a ≥4%
  day is profitable (NFLX +1.18%/trade avg). Viable as an *overlay*, not HFT.
  Not yet integrated; lower priority than the core trend book.
- **Stock pairs (mean-reversion):** FAIL — all 8 pairs negative, same as crypto.
  Stocks trend too hard for spread reversion. Rejected.
- **Insider signal (SEC EDGAR Form 4):** methodology validated; EDGAR is blocked
  in the cloud env. Form-4 is lagging and diluted by sell filings. Build locally
  if desired, but low expected edge. Deprioritized.

---

## 10. Open items before going live

1. Run `download_debiased_universe.py` locally → re-run combo finder on 60+
   tickers to kill residual survivorship bias.
2. Bitget stock perps are **live-mode only** (no paper). Either dry-run the stock
   decision logic against live prices without sending orders, or allocate a small
   real slice.
3. Wire the once-daily post-close cadence + dynamic allocation into the bot
   (`run_trio_paper.py` extension).
