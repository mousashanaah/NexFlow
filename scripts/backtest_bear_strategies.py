"""
Bear Market Short Strategies Backtest
Tests 4 mechanisms designed to profit in bear markets without blowing up in bull markets.

Mechanisms:
  A: Weekly-confirmed EMA short (BTC+ETH only)
  B: TSMOM short-only (12 coins)
  C: SMA200 + trend strength short (BTC+ETH only)
  D: Weekly MACD short (BTC+ETH only)
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/candles")
SYMBOLS_ALL = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT", "DOTUSDT", "TRXUSDT"
]
SYMBOLS_BTCETH = ["BTCUSDT", "ETHUSDT"]
FEE = 0.0006  # taker each side


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_daily(symbol: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / f"{symbol}_1D.parquet")
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    df = df[["ts", "open", "high", "low", "close", "volume"]].copy()
    df["symbol"] = symbol
    return df


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLCV to weekly using ISO week (last bar of each week = Friday close)."""
    df = df.copy()
    df["week"] = df["ts"].dt.isocalendar().week.astype(int)
    df["year"] = df["ts"].dt.isocalendar().year.astype(int)
    df["yearweek"] = df["year"] * 100 + df["week"]

    weekly = (
        df.groupby("yearweek")
        .agg(
            ts=("ts", "last"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
    )
    weekly = weekly.sort_values("ts").reset_index(drop=True)
    return weekly


# ---------------------------------------------------------------------------
# PnL helpers
# ---------------------------------------------------------------------------

def compute_yearly_pnl(trades: list) -> dict:
    """Aggregate trade PnL by year."""
    yearly = {}
    for t in trades:
        y = t["exit_ts"].year
        yearly[y] = yearly.get(y, 0.0) + t["pnl"]
    return yearly


def compute_stats(trades: list) -> dict:
    if not trades:
        return {"PF": 0, "CAGR": 0, "MaxDD": 0, "n_trades": 0}

    df = pd.DataFrame(trades)
    df = df.sort_values("exit_ts").reset_index(drop=True)

    wins = df[df["pnl"] > 0]["pnl"].sum()
    losses = abs(df[df["pnl"] < 0]["pnl"].sum())
    pf = wins / losses if losses > 0 else (float("inf") if wins > 0 else 0)

    # Equity curve
    df["cum_pnl"] = df["pnl"].cumsum()
    peak = df["cum_pnl"].cummax()
    dd = (df["cum_pnl"] - peak)
    max_dd_abs = dd.min()

    # CAGR
    start = df["entry_ts"].min()
    end = df["exit_ts"].max()
    years = (end - start).days / 365.25
    total = df["pnl"].sum()
    # CAGR relative to initial capital deployed (use starting notional)
    capital = df["notional"].iloc[0]
    cagr = (((capital + total) / capital) ** (1 / years) - 1) * 100 if years > 0 else 0

    return {
        "PF": round(pf, 3),
        "CAGR": round(cagr, 2),
        "MaxDD_abs": round(max_dd_abs, 0),
        "n_trades": len(df),
    }


def is_oos_pf(trades: list) -> tuple:
    """Return (IS_PF, OOS_PF) split at 2023-01-01."""
    cutoff = pd.Timestamp("2023-01-01", tz="UTC")
    is_trades = [t for t in trades if t["exit_ts"] < cutoff]
    oos_trades = [t for t in trades if t["exit_ts"] >= cutoff]
    wins_is = sum(t["pnl"] for t in is_trades if t["pnl"] > 0)
    loss_is = abs(sum(t["pnl"] for t in is_trades if t["pnl"] < 0))
    wins_oos = sum(t["pnl"] for t in oos_trades if t["pnl"] > 0)
    loss_oos = abs(sum(t["pnl"] for t in oos_trades if t["pnl"] < 0))
    is_pf = wins_is / loss_is if loss_is > 0 else (float("inf") if wins_is > 0 else 0)
    oos_pf = wins_oos / loss_oos if loss_oos > 0 else (float("inf") if wins_oos > 0 else 0)
    return round(is_pf, 3), round(oos_pf, 3)


def print_results(name: str, trades: list, capital_label: str):
    stats = compute_stats(trades)
    is_pf, oos_pf = is_oos_pf(trades)
    yearly = compute_yearly_pnl(trades)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Capital: {capital_label}")
    print(f"{'='*60}")
    print(f"  Overall PF:  {stats['PF']:.3f}")
    print(f"  CAGR:        {stats['CAGR']:.1f}%")
    print(f"  MaxDD:       ${stats.get('MaxDD_abs', 0):,.0f}")
    print(f"  n_trades:    {stats['n_trades']}")
    print(f"  IS PF (<2023):   {is_pf:.3f}")
    print(f"  OOS PF (>=2023): {oos_pf:.3f}")
    print()
    print("  Year-by-Year PnL:")
    for yr in sorted(yearly.keys()):
        tag = ""
        if yr in (2022, 2025, 2026):
            tag = " << BEAR (want +)"
        elif yr in (2021, 2024):
            tag = " << BULL (want 0 or small -)"
        print(f"    {yr}: ${yearly[yr]:>10,.0f}{tag}")
    total_pnl = sum(yearly.values())
    print(f"    TOTAL: ${total_pnl:>10,.0f}")

    # Success criteria check
    bear_years = [y for y in [2022, 2025, 2026] if y in yearly]
    bull_years = [y for y in [2021, 2024] if y in yearly]
    bear_pnl = sum(yearly.get(y, 0) for y in bear_years)
    bull_loss = sum(yearly.get(y, 0) for y in bull_years if yearly.get(y, 0) < 0)

    print()
    print(f"  CRITERIA CHECK:")
    print(f"    Bear years profit: ${bear_pnl:,.0f} {'PASS' if bear_pnl > 0 else 'FAIL'}")
    print(f"    Bull year losses:  ${bull_loss:,.0f} ({'PASS' if abs(bull_loss) < bear_pnl else 'FAIL'} vs bear)")
    print(f"    PF >= 1.10:        {'PASS' if stats['PF'] >= 1.10 else 'FAIL'} ({stats['PF']:.3f})")
    print(f"    MaxDD <= 50%:      (check manually against deployed capital)")


# ---------------------------------------------------------------------------
# Mechanism A: Weekly-confirmed EMA short
# ---------------------------------------------------------------------------

def run_mechanism_a():
    """Weekly EMA(8/21) + daily EMA(8/21) must both be bearish to enter short."""
    notional = 50_000
    all_trades = []

    for symbol in SYMBOLS_BTCETH:
        daily = load_daily(symbol)
        weekly = resample_weekly(daily)

        # Weekly EMAs
        weekly["ema8"] = ema(weekly["close"], 8)
        weekly["ema21"] = ema(weekly["close"], 21)
        weekly["w_bear"] = weekly["ema8"] < weekly["ema21"]

        # Daily EMAs
        daily["ema8"] = ema(daily["close"], 8)
        daily["ema21"] = ema(daily["close"], 21)
        daily["d_bear"] = daily["ema8"] < daily["ema21"]

        # Map weekly signal to daily rows by week label
        weekly["yearweek"] = weekly["ts"].dt.isocalendar().year.astype(int) * 100 + \
                              weekly["ts"].dt.isocalendar().week.astype(int)
        daily["yearweek"] = daily["ts"].dt.isocalendar().year.astype(int) * 100 + \
                             daily["ts"].dt.isocalendar().week.astype(int)

        # Use PREVIOUS week's signal to avoid lookahead
        yw_signal = weekly[["yearweek", "w_bear"]].rename(columns={"w_bear": "prev_w_bear"})
        yw_signal["yearweek_next"] = yw_signal["yearweek"].shift(-1)
        # For each daily row, find the most recent completed weekly bar
        # -> map daily yearweek to the weekly signal of THAT week (which was known at week end)
        # Use lag: daily rows in week W get the signal from week W-1 end
        week_signal_map = dict(zip(yw_signal["yearweek"], yw_signal["prev_w_bear"]))

        # Shift: daily bar in week W uses week W-1's weekly signal
        weekly_shifted = weekly[["yearweek", "w_bear"]].copy()
        weekly_shifted["w_bear_lag"] = weekly_shifted["w_bear"].shift(1)
        lag_map = dict(zip(weekly_shifted["yearweek"], weekly_shifted["w_bear_lag"]))

        daily["w_bear_lag"] = daily["yearweek"].map(lag_map)
        daily["w_bear_lag"] = daily["w_bear_lag"].fillna(False)

        # Entry signal: both weekly (lagged) and daily (previous day) bear
        daily["signal"] = daily["w_bear_lag"] & daily["d_bear"].shift(1).fillna(False)

        in_short = False
        entry_price = 0.0
        entry_ts = None

        for i, row in daily.iterrows():
            if pd.isna(row["ema8"]) or pd.isna(row["ema21"]):
                continue

            if not in_short:
                if row["signal"]:
                    in_short = True
                    entry_price = row["close"]
                    entry_ts = row["ts"]
            else:
                # Exit if weekly no longer bearish OR daily no longer bearish
                weekly_exit = not bool(daily.at[i, "w_bear_lag"])
                daily_exit = not bool(daily.at[i, "d_bear"])
                if weekly_exit or daily_exit:
                    exit_price = row["close"]
                    pnl = (entry_price - exit_price) / entry_price * notional
                    fees = 2 * FEE * notional
                    all_trades.append({
                        "symbol": symbol,
                        "entry_ts": entry_ts,
                        "exit_ts": row["ts"],
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl": pnl - fees,
                        "notional": notional,
                    })
                    in_short = False

        # Close any open trade at last bar
        if in_short:
            last = daily.iloc[-1]
            pnl = (entry_price - last["close"]) / entry_price * notional
            fees = 2 * FEE * notional
            all_trades.append({
                "symbol": symbol,
                "entry_ts": entry_ts,
                "exit_ts": last["ts"],
                "entry_price": entry_price,
                "exit_price": last["close"],
                "pnl": pnl - fees,
                "notional": notional,
            })

    print_results(
        "MECHANISM A: Weekly-confirmed EMA Short (BTC+ETH)",
        all_trades,
        "$50,000 per coin"
    )
    return all_trades


# ---------------------------------------------------------------------------
# Mechanism B: TSMOM short-only (12 coins)
# ---------------------------------------------------------------------------

def run_mechanism_b():
    """126-day return < -5% → short. Rebalance weekly."""
    notional_per_coin = 100_000 / 12
    all_trades = []

    for symbol in SYMBOLS_ALL:
        daily = load_daily(symbol)
        daily["ret126"] = daily["close"].pct_change(126)

        in_short = False
        entry_price = 0.0
        entry_ts = None
        last_rebal = None

        for i, row in daily.iterrows():
            if pd.isna(row["ret126"]):
                continue

            should_rebal = (
                last_rebal is None or
                (row["ts"] - last_rebal).days >= 7
            )

            if should_rebal:
                last_rebal = row["ts"]
                want_short = row["ret126"] < -0.05

                if not in_short and want_short:
                    in_short = True
                    entry_price = row["close"]
                    entry_ts = row["ts"]

                elif in_short and not want_short:
                    exit_price = row["close"]
                    pnl = (entry_price - exit_price) / entry_price * notional_per_coin
                    fees = 2 * FEE * notional_per_coin
                    all_trades.append({
                        "symbol": symbol,
                        "entry_ts": entry_ts,
                        "exit_ts": row["ts"],
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl": pnl - fees,
                        "notional": notional_per_coin,
                    })
                    in_short = False

        # Close open
        if in_short:
            last = daily.iloc[-1]
            pnl = (entry_price - last["close"]) / entry_price * notional_per_coin
            fees = 2 * FEE * notional_per_coin
            all_trades.append({
                "symbol": symbol,
                "entry_ts": entry_ts,
                "exit_ts": last["ts"],
                "entry_price": entry_price,
                "exit_price": last["close"],
                "pnl": pnl - fees,
                "notional": notional_per_coin,
            })

    print_results(
        "MECHANISM B: TSMOM Short-Only (12 coins, $8,333/coin)",
        all_trades,
        "$8,333 per coin (~$100K total)"
    )
    return all_trades


# ---------------------------------------------------------------------------
# Mechanism C: SMA200 + trend strength short
# ---------------------------------------------------------------------------

def run_mechanism_c():
    """SHORT: price < SMA200 AND EMA8 < EMA21 AND gap widening."""
    notional = 50_000
    all_trades = []

    for symbol in SYMBOLS_BTCETH:
        daily = load_daily(symbol)
        daily["ema8"] = ema(daily["close"], 8)
        daily["ema21"] = ema(daily["close"], 21)
        daily["sma200"] = sma(daily["close"], 200)
        daily["gap"] = daily["ema8"] - daily["ema21"]  # negative when bearish
        daily["gap_prev"] = daily["gap"].shift(1)
        # "widening" means gap is becoming more negative
        daily["widening"] = daily["gap"] < daily["gap_prev"]

        in_short = False
        entry_price = 0.0
        entry_ts = None

        for i, row in daily.iterrows():
            if pd.isna(row["sma200"]) or pd.isna(row["ema8"]):
                continue

            below_sma200 = row["close"] < row["sma200"]
            ema_bear = row["ema8"] < row["ema21"]
            widening = bool(row["widening"])

            if not in_short:
                if below_sma200 and ema_bear and widening:
                    in_short = True
                    entry_price = row["close"]
                    entry_ts = row["ts"]
            else:
                # Exit: price crosses above SMA200 OR EMA8 crosses above EMA21
                above_sma200 = row["close"] > row["sma200"]
                ema_bull = row["ema8"] > row["ema21"]
                if above_sma200 or ema_bull:
                    exit_price = row["close"]
                    pnl = (entry_price - exit_price) / entry_price * notional
                    fees = 2 * FEE * notional
                    all_trades.append({
                        "symbol": symbol,
                        "entry_ts": entry_ts,
                        "exit_ts": row["ts"],
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl": pnl - fees,
                        "notional": notional,
                    })
                    in_short = False

        if in_short:
            last = daily.iloc[-1]
            pnl = (entry_price - last["close"]) / entry_price * notional
            fees = 2 * FEE * notional
            all_trades.append({
                "symbol": symbol,
                "entry_ts": entry_ts,
                "exit_ts": last["ts"],
                "entry_price": entry_price,
                "exit_price": last["close"],
                "pnl": pnl - fees,
                "notional": notional,
            })

    print_results(
        "MECHANISM C: SMA200 + Trend Strength Short (BTC+ETH)",
        all_trades,
        "$50,000 per coin"
    )
    return all_trades


# ---------------------------------------------------------------------------
# Mechanism D: Weekly MACD short
# ---------------------------------------------------------------------------

def run_mechanism_d():
    """Weekly MACD(12,26,9): SHORT when histogram crosses below 0, EXIT when crosses above 0."""
    notional = 50_000
    all_trades = []

    for symbol in SYMBOLS_BTCETH:
        daily = load_daily(symbol)
        weekly = resample_weekly(daily)

        # MACD on weekly closes
        weekly["ema12"] = ema(weekly["close"], 12)
        weekly["ema26"] = ema(weekly["close"], 26)
        weekly["macd"] = weekly["ema12"] - weekly["ema26"]
        weekly["signal_line"] = ema(weekly["macd"], 9)
        weekly["hist"] = weekly["macd"] - weekly["signal_line"]
        weekly["hist_prev"] = weekly["hist"].shift(1)

        # Cross below 0: hist < 0 and prev >= 0
        weekly["cross_below"] = (weekly["hist"] < 0) & (weekly["hist_prev"] >= 0)
        # Cross above 0: hist > 0 and prev <= 0
        weekly["cross_above"] = (weekly["hist"] > 0) & (weekly["hist_prev"] <= 0)

        # Map weekly signal to daily rows: use the CLOSE of the week bar as entry next day
        # We'll simulate: on the daily row that is the last day of the week, we act
        daily["yearweek"] = daily["ts"].dt.isocalendar().year.astype(int) * 100 + \
                             daily["ts"].dt.isocalendar().week.astype(int)
        weekly["yearweek"] = weekly["ts"].dt.isocalendar().year.astype(int) * 100 + \
                              weekly["ts"].dt.isocalendar().week.astype(int)

        # Build signal maps
        cross_below_map = dict(zip(weekly["yearweek"], weekly["cross_below"]))
        cross_above_map = dict(zip(weekly["yearweek"], weekly["cross_above"]))

        daily["cross_below"] = daily["yearweek"].map(cross_below_map).fillna(False)
        daily["cross_above"] = daily["yearweek"].map(cross_above_map).fillna(False)

        # Simulate on daily: enter/exit on close of day when weekly signal fires
        # To avoid lookahead within week, only act on the last daily bar of each week
        # Find last bar of each week
        last_bar_of_week = set(
            daily.groupby("yearweek")["ts"].idxmax().values
        )

        in_short = False
        entry_price = 0.0
        entry_ts = None

        for i, row in daily.iterrows():
            if i not in last_bar_of_week:
                continue
            if pd.isna(row.get("cross_below")):
                continue

            if not in_short:
                if row["cross_below"]:
                    in_short = True
                    entry_price = row["close"]
                    entry_ts = row["ts"]
            else:
                if row["cross_above"]:
                    exit_price = row["close"]
                    pnl = (entry_price - exit_price) / entry_price * notional
                    fees = 2 * FEE * notional
                    all_trades.append({
                        "symbol": symbol,
                        "entry_ts": entry_ts,
                        "exit_ts": row["ts"],
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl": pnl - fees,
                        "notional": notional,
                    })
                    in_short = False

        if in_short:
            last = daily.iloc[-1]
            pnl = (entry_price - last["close"]) / entry_price * notional
            fees = 2 * FEE * notional
            all_trades.append({
                "symbol": symbol,
                "entry_ts": entry_ts,
                "exit_ts": last["ts"],
                "entry_price": entry_price,
                "exit_price": last["close"],
                "pnl": pnl - fees,
                "notional": notional,
            })

    print_results(
        "MECHANISM D: Weekly MACD Short (BTC+ETH)",
        all_trades,
        "$50,000 per coin"
    )
    return all_trades


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  BEAR MARKET SHORT STRATEGIES BACKTEST")
    print("  Testing 4 mechanisms across 2021-2026")
    print("  Bear years: 2022, 2025, 2026  |  Bull years: 2021, 2023, 2024")
    print("="*60)

    trades_a = run_mechanism_a()
    trades_b = run_mechanism_b()
    trades_c = run_mechanism_c()
    trades_d = run_mechanism_d()

    # Summary comparison
    print("\n" + "="*60)
    print("  SUMMARY COMPARISON")
    print("="*60)
    print(f"{'Mechanism':<45} {'PF':>6} {'CAGR':>7} {'Trades':>7} {'IS PF':>7} {'OOS PF':>7}")
    print("-"*85)
    for name, trades in [
        ("A: Weekly-confirmed EMA (BTC+ETH)", trades_a),
        ("B: TSMOM Short-Only (12 coins)", trades_b),
        ("C: SMA200+Trend Strength (BTC+ETH)", trades_c),
        ("D: Weekly MACD (BTC+ETH)", trades_d),
    ]:
        s = compute_stats(trades)
        is_pf, oos_pf = is_oos_pf(trades)
        print(f"  {name:<43} {s['PF']:>6.3f} {s['CAGR']:>6.1f}% {s['n_trades']:>7} {is_pf:>7.3f} {oos_pf:>7.3f}")

    print("\n  SUCCESS CRITERIA: PF>=1.10, Bear years profitable, Bull losses < Bear gains")
    print("="*60)
