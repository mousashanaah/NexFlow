# V9 Confidence — Session Handoff

_Last updated: 2026-06-16_

This file exists so a fresh Claude Code session can pick up exactly where the
previous one left off. Read this first, then read the key files below.

## What V9 Confidence is
Fully autonomous multi-asset quant system. Two books run side by side with a
confidence allocator splitting capital each day:
- **Crypto book** — full V8.63 engine (`scripts/crypto_book_v863.py`): BTC-regime
  AND-entry, EMA/MACD/4H confluence longs, TSMOM shorts in bear, ATR sizing,
  15% stops, 20% circuit breaker, 12 coins.
- **Stock book** — MSTR + AMD + GOOGL + META, no-lookahead trend system, runs on
  Bitget STOCK perps (`SUSDT-FUTURES`).
- **Allocator** — crypto_score (0-4) + stock_score (0-3) → 5 regimes.

Backtested 2021–2026: ~76.6% CAGR, zero losing years on the full system.

## Live status (as of handoff)
- Running LIVE on real money via Bitget. Balance ~$97.44.
- Command: `python scripts/run_v9_confidence.py --mode live --capital 100 --stock-live`
- Credentials via env vars: `BITGET_API_KEY`, `BITGET_API_SECRET`, `BITGET_PASSPHRASE`.
  Real money = `BITGET_PAPER` unset. Demo = `BITGET_PAPER=1`.
- Current regime: **BEAR** (shorts mode). Crypto slice ~$43.85, stock ~$53.59.
- Positions: flat (0 crypto + 0 stock at last check).
- Development branch: `claude/gallant-mendel-vJRA6`.

## Recent work completed this/last sessions
- **Shorts upgrades shipped**: 14-day rebalance (was 7) + bearish-confluence
  position sizing (`_short_confluence_notional`). Trailing stops were TESTED and
  REJECTED (created losing years).
- **Stale-data bug fixed**: startup gap-backfill in `seed()` via
  `_fetch_daily_candles()` (Bitget `history-candles` caps at limit=200).
- **Double-ingest bug fixed**: `last_daily_ts` timestamp dedup in `daily_check()`.
- **Regime-lag research**: tested 0/1/2/3 day lag — non-monotonic noise, daily
  00:05 UTC check is optimal. Continuous uptime is the real robustness lever.

## Next scheduled events
- Stock daily check: 21:05 UTC.
- Crypto daily check: 00:05 UTC + stops every 6h.
- Short rebalance timer: re-arms ~14 days from last bot restart (restarted
  2026-06-16, so ~June 30). Timer resets on every restart.

## Key files
- `scripts/run_v9_confidence.py` — production launcher / scheduler.
- `scripts/crypto_book_v863.py` — crypto engine (V8.63).
- `scripts/backtest_full_regime_system.py` — parameterized backtester.
- `nexflow/exchange/bitget_client.py` — authenticated Bitget REST client.

## Open threads / ideas (not started)
- **Co-Invest connector**: user enabled it; explore what market data it exposes
  and how it can feed V9. (Connectors only load at session startup.)
- **Second project ideas** (from llm-council): SaaS signal wrapper on V9 +
  funding-rate farming were the top picks. Dropshipping rejected.
- **Outreach**: messaging IG trader "dovy.fx" (building "Hybrid trading AI") to
  get his take and potentially collaborate. Keep details high-level.

## House rules
- Develop on `claude/gallant-mendel-vJRA6`. Never push elsewhere without
  permission. Don't open a PR unless explicitly asked.
- Don't restart the live bot casually — it resets the rebalance timer.
