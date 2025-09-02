# replayer.py
# Replay-mode intraday breakout (robust column handling + VWAP + morning-range breakout)
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import time

# ---------- CONFIG ----------
STOCKS = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
INTERVAL = "5m"
PERIOD = "5d"             # short period for replay
MORNING_RANGE_START = "09:15"
MORNING_RANGE_END = "09:45"
TRADE_WINDOW_START = "09:45"
TRADE_WINDOW_END = "15:20"
VWAP_COL = "VWAP"
TARGET_MULT = 1.5
SAVE_CSV = "replay_trades.csv"

# ---------- HELPERS ----------
EXPECTED = ["Open", "High", "Low", "Close", "Volume"]

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns if present and normalize column strings."""
    if isinstance(df.columns, pd.MultiIndex):
        # Try to find a level that contains expected names
        for lvl in range(df.columns.nlevels):
            lvl_vals = df.columns.get_level_values(lvl)
            if set([c.lower() for c in EXPECTED]).issubset(set([str(x).lower() for x in lvl_vals])):
                df.columns = lvl_vals
                break
        else:
            # fallback: join tuples into single strings
            df.columns = ['_'.join([str(x) for x in col]).strip() for col in df.columns.values]

    # ensure all column names are plain strings
    df.columns = [str(c) for c in df.columns]
    return df

def pick_expected_columns(df: pd.DataFrame):
    """Return df with columns in EXPECTED order if they exist (case-insensitive)."""
    cols_map = {c.lower(): c for c in df.columns}
    missing = [c for c in EXPECTED if c.lower() not in cols_map]
    if missing:
        return None  # caller will handle
    picked = [cols_map[c.lower()] for c in EXPECTED]
    return df.loc[:, picked]

def fetch_and_clean(ticker: str):
    try:
        df = yf.download(ticker, period=PERIOD, interval=INTERVAL, auto_adjust=True, progress=False)
    except Exception as e:
        print(f"Download failed for {ticker}: {e}")
        return None
    if df is None or df.empty:
        print(f"No intraday data for {ticker} (period={PERIOD}, interval={INTERVAL})")
        return None

    df = flatten_columns(df)

    df_expected = pick_expected_columns(df)
    if df_expected is None:
        print(f"{ticker}: could not find required OHLCV columns after flattening. Columns returned: {list(df.columns)}")
        return None

    # force numeric and drop rows with NA in essential fields
    df_expected = df_expected.apply(pd.to_numeric, errors="coerce")
    df_expected = df_expected.dropna(subset=["Close", "High", "Low"])  # keep only rows with core values

    # timezone handling: yfinance often returns UTC-naive
    try:
        if df_expected.index.tz is None:
            df_expected = df_expected.tz_localize("UTC").tz_convert("Asia/Kolkata")
        else:
            df_expected = df_expected.tz_convert("Asia/Kolkata")
    except Exception:
        # if timezone ops fail, continue with naive index (still ok for replay)
        pass

    return df_expected

def compute_vwap(df: pd.DataFrame):
    parts = []
    for day, g in df.groupby(df.index.date):
        g_day = g.between_time(MORNING_RANGE_START, "15:30")
        if g_day.empty:
            continue
        vwap = (g_day["Close"] * g_day["Volume"]).cumsum() / g_day["Volume"].cumsum()
        g_day = g_day.copy()
        g_day[VWAP_COL] = vwap
        parts.append(g_day)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts).sort_index()

# ---------- STRATEGY ----------
def backtest_one_symbol(df: pd.DataFrame, ticker: str):
    trades = []
    df_vwap = compute_vwap(df)
    if df_vwap.empty:
        return trades

    for day, day_df in df_vwap.groupby(df_vwap.index.date):
        day_df = day_df.between_time(MORNING_RANGE_START, "15:30")
        if day_df.empty:
            continue

        morning = day_df.between_time(MORNING_RANGE_START, MORNING_RANGE_END)
        if morning.empty:
            continue
        rng_high = morning["High"].max()
        rng_low = morning["Low"].min()

        in_trade = False
        trade = None

        post_morning = day_df.between_time(TRADE_WINDOW_START, "15:30")
        for idx, row in post_morning.iterrows():
            # skip if VWAP missing
            if pd.isna(row.get(VWAP_COL, np.nan)):
                continue

            # ENTRY
            if not in_trade:
                # long breakout
                if (row["Close"] > rng_high) and (row["Close"] > row[VWAP_COL]):
                    entry = float(row["Close"])
                    sl = float(rng_low)
                    target = entry + TARGET_MULT * (entry - sl)
                    trade = {"stock": ticker, "type": "long", "entry_time": str(idx),
                             "entry": entry, "sl": sl, "target": float(target),
                             "exit_time": None, "exit": None, "pnl": None}
                    in_trade = True
                    continue

                # short breakout
                if (row["Close"] < rng_low) and (row["Close"] < row[VWAP_COL]):
                    entry = float(row["Close"])
                    sl = float(rng_high)
                    target = entry - TARGET_MULT * (sl - entry)
                    trade = {"stock": ticker, "type": "short", "entry_time": str(idx),
                             "entry": entry, "sl": sl, "target": float(target),
                             "exit_time": None, "exit": None, "pnl": None}
                    in_trade = True
                    continue

            # EXIT checks
            if in_trade and trade:
                if trade["type"] == "long":
                    # stop hit
                    if row["Low"] <= trade["sl"]:
                        exit_price = trade["sl"]
                        trade.update({"exit": exit_price, "exit_time": str(idx),
                                      "pnl": exit_price - trade["entry"]})
                        trades.append(trade)
                        in_trade, trade = False, None
                        break
                    # target hit
                    if row["High"] >= trade["target"]:
                        exit_price = trade["target"]
                        trade.update({"exit": exit_price, "exit_time": str(idx),
                                      "pnl": exit_price - trade["entry"]})
                        trades.append(trade)
                        in_trade, trade = False, None
                        break
                else:  # short
                    if row["High"] >= trade["sl"]:
                        exit_price = trade["sl"]
                        trade.update({"exit": exit_price, "exit_time": str(idx),
                                      "pnl": trade["entry"] - exit_price})
                        trades.append(trade)
                        in_trade, trade = False, None
                        break
                    if row["Low"] <= trade["target"]:
                        exit_price = trade["target"]
                        trade.update({"exit": exit_price, "exit_time": str(idx),
                                      "pnl": trade["entry"] - exit_price})
                        trades.append(trade)
                        in_trade, trade = False, None
                        break

        # day end square-off
        if in_trade and trade:
            last_price = float(day_df.iloc[-1]["Close"])
            trade.update({"exit": last_price, "exit_time": str(day_df.index[-1])})
            trade["pnl"] = (last_price - trade["entry"]) if trade["type"] == "long" else (trade["entry"] - last_price)
            trades.append(trade)
            in_trade, trade = False, None

    return trades

# ---------- RUN ----------
all_trades = []
for stk in STOCKS:
    df = fetch_and_clean(stk)
    if df is None:
        continue
    t_trades = backtest_one_symbol(df, stk)
    print(f"{stk}: {len(t_trades)} trades")
    all_trades.extend(t_trades)

if not all_trades:
    print("No trades generated. Either data missing or filters too tight.")
else:
    results = pd.DataFrame(all_trades)
    results["pnl"] = pd.to_numeric(results["pnl"], errors="coerce").fillna(0.0)
    total = results["pnl"].sum()
    wins = (results["pnl"] > 0).sum()
    trades = len(results)
    win_rate = wins / trades * 100 if trades>0 else 0.0
    equity = np.cumsum(results["pnl"].values)
    peak = np.maximum.accumulate(equity) if len(equity)>0 else np.array([0.0])
    drawdowns = equity - peak
    max_dd = drawdowns.min() if len(drawdowns)>0 else 0.0
    avg_per_day = trades / max(1, len(results["entry_time"].apply(lambda x: x.split(" ")[0]).unique()))

    print("\n=== Replay Breakout Summary ===")
    print(f"Total PnL: {round(total,2)}")
    print(f"Win Rate: {round(win_rate,2)}")
    print(f"Max Drawdown: {round(max_dd,2)}")
    print(f"Trades: {trades}")
    print(f"Avg Trades/Day: {round(avg_per_day,2)}")

    results.to_csv(SAVE_CSV, index=False)
    print(f"\nSaved -> {SAVE_CSV}")
