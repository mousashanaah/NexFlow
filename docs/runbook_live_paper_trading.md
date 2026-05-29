# NexFlow — Live Paper Trading Runbook

**Audience:** Non-technical operator  
**Purpose:** Run a live paper trading session against Bitget market data with simulated execution (no real money at risk)  
**Prerequisite:** The host machine must be outside a cloud datacenter. Bitget blocks connections from AWS, GCP, Azure, and similar providers. A home server, dedicated host, or residential VPS works fine.

---

## 1. Exact command to start live paper mode

```bash
python scripts/run_paper_trader.py \
  --mode live \
  --symbols BTCUSDT,ETHUSDT \
  --equity 100000 \
  --journal-dir logs/paper \
  --log-level INFO
```

**What each flag means:**

| Flag | What it does | Change it when… |
|---|---|---|
| `--mode live` | Connects to Bitget WebSocket for real market data | Never (for this runbook) |
| `--symbols` | Which trading pairs to watch | You want to add/remove pairs |
| `--equity` | Starting paper account balance in USDT | You want a different simulated account size |
| `--journal-dir` | Folder where all session data is saved | You want a different save location |
| `--log-level INFO` | How much detail to print to the log file | Set to `DEBUG` if something looks wrong |

To hide the live dashboard (e.g., when running in the background):

```bash
python scripts/run_paper_trader.py --mode live --symbols BTCUSDT,ETHUSDT --no-dashboard
```

---

## 2. Required environment variables

NexFlow paper trading does **not** require any API keys or environment variables. The system connects only to Bitget's **public** WebSocket endpoint:

```
wss://ws.bitget.com/v2/ws/public
```

No authentication is needed to receive live market data. No orders are sent to the exchange.

If you are behind a proxy, set the standard system variables before running:

```bash
export HTTPS_PROXY=http://your-proxy:port
export HTTP_PROXY=http://your-proxy:port
```

---

## 3. Expected console output

The terminal clears and refreshes every 5 seconds. A healthy session looks like this:

```
╔══ NexFlow Paper Trader ══════════════════════════════════╗
  2026-05-29 14:22:10 UTC    ● LIVE
  ────────────────────────────────────────────────────────
  Equity              :  100,000.00 USDT
  Realized PnL        :      +0.00
  Unrealized PnL      :      +0.00
  Drawdown            :    0.000%
  Rolling Sharpe      :    0.000
  ────────────────────────────────────────────────────────
  Open Positions      : 0
  Total Trades        : 0
  Win Rate            : 0.0%
  Profit Factor       : 0.0
  Expectancy (R)      : 0.0000
  ────────────────────────────────────────────────────────
  Long  trades        :    0  WR 0.0%
  Short trades        :    0  WR 0.0%
  ────────────────────────────────────────────────────────
  Risk Monitor
  Reconnects/hr       : 0
  Consec losses       : 0
  Latency spikes      : 0
╚══════════════════════════════════════════════════════════╝
```

**Status indicator meanings:**

| What you see | Meaning |
|---|---|
| `● LIVE` (green) | Connected, receiving data, kill-switch is off |
| `⚠ KILL ACTIVE: drawdown_exceeded` | Trading paused — see Section 7 |
| `⚠ KILL ACTIVE: consecutive_losses` | Trading paused — see Section 7 |
| `⚠ KILL ACTIVE: stale_feed` | No market data for > 2 minutes |
| `⚠ KILL ACTIVE: ws_unhealthy` | WebSocket connection dropped |

Shortly after the dashboard, structured log lines appear in the terminal:

```
2026-05-29 14:22:10 [info] paper_trader.live_start  symbols=['BTCUSDT', 'ETHUSDT']
2026-05-29 14:23:01 [info] router.fill  symbol=BTCUSDT direction=long price=67450.20 size=0.285
2026-05-29 14:31:17 [info] router.trade_closed  symbol=BTCUSDT exit_reason=tp1 pnl=+112.40
```

No log lines for several minutes after startup is normal — the strategy needs at least 22 one-minute candles per symbol before it can evaluate any signals.

---

## 4. Where journals are stored

Each session creates one file:

```
logs/paper/journal_YYYYMMDD_HHMMSS_<session-id>.jsonl
```

Example:

```
logs/paper/journal_20260529_142210_a3f1c9e2.jsonl
```

**Every signal, fill, stop exit, kill-switch event, and equity snapshot is recorded here in real time.** The file is written line-by-line so it survives a crash — nothing is lost if the process is killed unexpectedly.

Each line is a JSON object. You can inspect it at any time while the session is running:

```bash
tail -f logs/paper/journal_*.jsonl
```

---

## 5. Where reports are generated

Journals are not automatically converted to a report while the session runs. After stopping, generate the HTML report manually:

```bash
python scripts/generate_paper_report.py \
  --journal-dir logs/paper \
  --output reports/paper_session.html \
  --summary
```

The `--summary` flag also prints a compact text digest to the terminal. The HTML report opens automatically in your browser unless you add `--no-open`.

The report is a single self-contained HTML file — no internet connection needed to view it.

---

## 6. How to stop gracefully

Press **Ctrl+C** once in the terminal where the trader is running.

The system will:
1. Force-close any open simulated positions at the last known mid-price
2. Write a `SESSION_END` event to the journal
3. Print a final state summary to the terminal
4. Exit cleanly

**Do not kill the terminal window or press Ctrl+C twice** — the first press triggers an orderly shutdown that takes 1–3 seconds. A hard kill may leave an open position in the journal without a close record (the analyzer handles this gracefully, but it is messier to review).

If running in the background (e.g., via `nohup` or `screen`):

```bash
# Find the process
pgrep -f run_paper_trader.py

# Send graceful stop signal
kill -SIGINT <pid>
```

---

## 7. What metrics to monitor during runtime

Check the dashboard every time it refreshes (every 5 seconds). The following are the critical numbers:

### Normal operation — these should be stable

| Metric | Healthy range | Concern |
|---|---|---|
| **Drawdown** | < 1.0% (green) | Yellow > 0.5%, Red > 2% |
| **Reconnects/hr** | 0–1 | > 3 suggests network instability |
| **Consec losses** | < 4 | Kill activates at 6 |
| **Latency spikes** | 0 | > 2 consecutive spikes risks kill |

### Kill-switch thresholds (trading stops automatically)

| Condition | Trigger | What to do |
|---|---|---|
| `drawdown_exceeded` | Account drops 5% from start | Let it sit; review journal after session |
| `consecutive_losses` | 6 losses in a row | Let it sit; review journal after session |
| `stale_feed` | No candle for 2 minutes | Check network; consider restarting |
| `ws_unhealthy` | WebSocket reconnect fails | Check network; restart if sustained |
| `latency_spike` | 3 consecutive latency events > 3s | Check host load; restart if sustained |

**Kill-switch does not stop the process.** It only blocks new entries. Existing positions are still managed (stops and take-profits still fire). The process must be stopped manually with Ctrl+C.

### After a few hours — look for

- **Total Trades > 0** — confirms signals are reaching execution
- **Win Rate** — informational only in early sessions; sample size is too small to be meaningful
- **Profit Factor** — meaningful only after 30+ trades
- **Equity** trending away from 100,000 in either direction

---

## 8. How to verify candles are being produced

Run this in a second terminal while the session is live:

```bash
tail -f logs/paper/journal_*.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    try:
        e = json.loads(line)
        if e.get('event') in ('SIGNAL', 'FILL', 'EQUITY_SNAPSHOT', 'FEED_STALE'):
            print(e.get('ts'), e.get('event'), e.get('symbol',''))
    except: pass
"
```

**What to expect:**

- `EQUITY_SNAPSHOT` events appear every 60 seconds. If you see them, candles are flowing.
- `FEED_STALE` events mean a symbol stopped sending candles. This is a problem.
- No events at all for the first 2–3 minutes is normal (warmup period).

Alternatively, check the structured log for candle activity:

```bash
grep "candle\|fill\|signal" /path/to/app.log
```

The candle engine emits log lines at `DEBUG` level. To see them:

```bash
python scripts/run_paper_trader.py --mode live --symbols BTCUSDT,ETHUSDT --log-level DEBUG 2>&1 | grep "candle"
```

---

## 9. How to verify signals are being evaluated

Signals are evaluated on every finalized 1-minute candle, but most will not generate a trade. There are two checks:

**Check 1 — Watch for SIGNAL events in the journal:**

```bash
tail -f logs/paper/journal_*.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    try:
        e = json.loads(line)
        if e.get('event') in ('SIGNAL', 'REJECTED'):
            print(e.get('ts'), e.get('event'), e.get('symbol'), e.get('direction',''), e.get('reason',''))
    except: pass
"
```

A `SIGNAL` event means the strategy produced a trade candidate. A `REJECTED` event immediately after means it was blocked (already in a position, kill active, size too small, etc.). Both are expected and healthy. If you run for 30+ minutes and see **zero** SIGNAL events, something is wrong — see below.

**Check 2 — Verify the strategy has enough bars:**

The MomentumStrategy requires:
- At least **22 one-minute candles** per symbol before it can generate any signal
- At least **7 five-minute candles** per symbol

At one candle per minute, expect the first signal evaluation at approximately **T+22 minutes**. Before that, no signals will appear.

**If zero SIGNAL events appear after 30 minutes:**

1. Confirm `EQUITY_SNAPSHOT` events are appearing (proves candles are arriving)
2. Check for `FEED_STALE` events
3. Restart with `--log-level DEBUG` and look for `strategy.insufficient_data` log lines

---

## 10. How to verify orders are being simulated

A simulated "order" is a `FILL` event in the journal. Check for fills:

```bash
tail -f logs/paper/journal_*.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    try:
        e = json.loads(line)
        if e.get('event') in ('FILL', 'PARTIAL_TP', 'STOP_HIT', 'FORCE_CLOSE'):
            print(e.get('ts'), e.get('event'), e.get('symbol'), 
                  'size=' + str(e.get('size','')), 
                  'pnl=' + str(round(e.get('pnl',0),2)))
    except: pass
"
```

**A healthy trade lifecycle looks like this in order:**

```
SIGNAL   BTCUSDT  long
FILL     BTCUSDT  size=0.285              ← entry simulated
PARTIAL_TP BTCUSDT  size=0.143  pnl=+112  ← first take-profit hit
PARTIAL_TP BTCUSDT  size=0.071  pnl=+87   ← second take-profit hit
STOP_HIT BTCUSDT  size=0.071  pnl=+22    ← remainder closed at stop (in profit here)
```

If you see `FILL` but never `STOP_HIT` or `PARTIAL_TP`, the position is still open — that is correct and expected. It will close when the price hits a stop or take-profit level on a subsequent candle.

**After stopping the session**, generate the full report to review all fills:

```bash
python scripts/generate_paper_report.py --journal-dir logs/paper --summary
```

This prints totals for fills, rejections, win rate, fees, slippage, and drawdown.

---

## Quick reference card

```
START      python scripts/run_paper_trader.py --mode live --symbols BTCUSDT,ETHUSDT --equity 100000 --journal-dir logs/paper
STOP       Ctrl+C  (wait 1–3 seconds for clean shutdown)
REPORT     python scripts/generate_paper_report.py --journal-dir logs/paper --summary
JOURNAL    logs/paper/journal_YYYYMMDD_*.jsonl
REPORT     reports/paper_session.html
WARMUP     ~22 minutes before first signal is possible
KILL THRESHOLDS   drawdown > 5%  |  6 consecutive losses  |  feed stale > 2 min
LIVE REQUIREMENT  Non-datacenter IP (Bitget blocks cloud hosts)
```
