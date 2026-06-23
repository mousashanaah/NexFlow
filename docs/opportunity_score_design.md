# Opportunity Score — Design Intent

## What Is It Optimizing For?

**Current answer: earliest viable discovery.**

Not highest expected return. Not highest survival probability. Not best risk-adjusted outcome.

The score is a prior — a hypothesis about which tokens deserve attention *before* we have enough outcome data to know what actually works.

---

## The Three Possible Objectives (and why we chose one)

### Option A: Earliest Discovery
Optimize for finding tokens in the first 1–6 hours of their existence.

**Current formula is biased toward this.** Freshness contributes 0–25 pts (the largest single tier). A 30-minute-old token with $5K liquidity can outscore a 24-hour-old token with $500K liquidity.

**When this is correct:** The system's purpose is finding the next PEPE/WIF/BONK before they move 100x. Those moves happen in hours, not days. If you're not in by hour 6, you've often missed the asymmetric part.

**When this is wrong:** Most 30-minute-old tokens rug or die by hour 48. High freshness score ≠ high survival probability.

### Option B: Highest Expected Return
Optimize for tokens likely to multiply the most over 7–30 days.

**Not achievable yet.** We have zero classified outcomes. We don't know which signals correlate with 7d returns. Building a return-optimized ranking before we have outcome data would be pure fiction.

**This is what wallet score and narrative score will eventually provide** — once the attribution database has enough classified observations.

### Option C: Highest Survival Probability
Optimize for tokens that are still alive and liquid in 7 days.

**Partially captured by risk gate and liquidity tier.** A token that passes RugCheck with score 18+ and has $500K+ liquidity is more likely to survive than a fresh $3K pool. But survival without price appreciation is not alpha.

---

## Current Formula: Known Biases

| Signal | Weight | Known bias |
|---|---|---|
| Freshness | 0–25 | Over-weights new tokens; most will rug |
| Volume/Liq ratio | 0–25 | Good momentum proxy; can be gamed with wash trading |
| Liquidity tier | 0–20 | Under-weights large launches; favors micro-caps |
| Risk quality | 0–15 | Accurate when RugCheck returns data; UNVERIFIED = 0 bonus |
| Wallet score | 0–15 | Currently always 0 (no outcomes) |
| Narrative bonus | 0–10 | Currently always 0 (no outcome-backed win rates) |
| FOMO available | +5 | Correct: actionable > non-actionable, all else equal |
| SCAM_SIGNAL | −15 | Correct but conservative |

---

## What Changes When Outcomes Arrive

The formula is intentionally front-loaded with hypothesis signals (freshness, volume ratio) that will be *measured* against actual outcomes. After ~50 classified tokens per category, the attribution report will answer:

- Does freshness predict 7d winners? (If not, reduce its weight)
- Does volume/liq ratio predict survival? (If yes, increase its weight)
- Which narrative categories produce winners? (Then the narrative bonus activates)
- Which wallet score buckets correlate with returns? (Then wallet bonus activates)

**The current formula should be treated as a hypothesis, not a proven ranking.**

---

## What Rank #1 Means Today

A token ranks #1 on the current board when it has:

1. Passed the risk gate (non-negotiable)
2. Is very new (< 2 hours old)
3. Has volume > 3× its liquidity (high momentum)
4. Has enough liquidity to matter (>$10K)
5. Is available on FOMO (actionable now)
6. Has a decent risk quality score

**It does NOT mean:** highest probability of being a winner. It means highest priority for immediate human review — the combination of signals suggests something unusual is happening early.

---

## What Rank #1 Should Mean (Post-Outcome Data)

Once we have 100+ classified outcomes, rank #1 should mean:

1. Passed risk gate
2. Wallet score in the top quartile (proven early-buyer quality)
3. Narrative category with documented >50% win rate
4. Volume/liq ratio in historically predictive range
5. FOMO available

Freshness will still matter but should not dominate once we can measure what actually predicts returns.

---

## Recommendation

Do not modify the formula until the attribution database has at least 30 classified outcomes. The current biases are known and acceptable — they favor early discovery, which matches the stated objective of finding PEPE/WIF/BONK equivalents before they move.

The formula should be reviewed at:
- 30 classified outcomes (first meaningful signal test)
- 100 classified outcomes (statistically significant per-signal correlation)
- 200 classified outcomes (ready for weight optimization)

**The most important action right now is collecting outcomes, not refining the formula.**
