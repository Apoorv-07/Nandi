# intra_2.py
# Range-breakout intraday backtest (5m). Uses last 60 days ONLY (Yahoo limitation).
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import time

# ---------- CONFIG ----------
STOCKS = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
INTERVAL = "5m"
PERIOD = "60d"            # Yahoo: intraday (5m) only for ~last 60 days
MORNING_RANGE_START = "09:15"
MORNING_RANGE_END = "09:45"
TRADE_WINDOW_START = "09:45"   # start looking for breakouts after morning range
TRADE_WINDOW_END = "15:20"     # final exit enforced at/after this
VWAP_COL = "VWAP"
TARGET_MULT = 1.5
SAVE_CSV = "breakout_trades.csv"

# ---------- HELPERS ----------
def fetch_intraday(ticker):
    try:
        df = yf.download(ticker, period=PERIOD, interval=INTERVAL, group_by="column",
                         auto_adjust=True, progress=False)
    except Exception as e:
        print(f"Download failed for {ticker}: {e}")
        return None
    if df is None or df.empty:
        print(f"No intraday data for {ticker} (period={PERIOD}, interval={INTERVAL})")
        return None

    # flatten multiindex if present
    if isinstance(df.columns, pd.MultiIndex):
        if df.columns.nlevels == 2:
            lev0 = df.columns.get_level_values(0)
            lev1 = df.columns.get_level_values(1)
            if len(set(lev0)) == 1:
                df.columns = lev1
            elif len(set(lev1)) == 1:
                df.columns = lev0
            else:
                df.columns = ["_".join(map(str, t)) for t in df.columns]
        else:
            df.columns = ["_".join(map(str, t)) for t in df.columns]

    # ensure necessary columns
    expected = ["Open", "High", "Low", "Close", "Volume"]
    for c in expected:
        if c not in df.columns:
            print(f"{ticker}: missing column {c} after download. Skipping.")
            return None

    # numeric and timezone handling
    df = df[expected].apply(pd.to_numeric, errors="coerce").dropna()
    # set tz -> yfinance often returns UTC-naive; localize to UTC then convert to IST
    try:
        if df.index.tz is None:
            df = df.tz_localize("UTC").tz_convert("Asia/Kolkata")
        else:
            df = df.tz_convert("Asia/Kolkata")
    except Exception:
        # fallback: leave as-is (assume already localised)
        pass

    return df

def compute_daily_vwap(df):
    # df must be tz-aware or naive index but consistent
    parts = []
    for day, g in df.groupby(df.index.date):
        # consider market hours only
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

# ---------- STRATEGY: morning-range breakout ----------
def backtest_one_symbol(df, ticker):
    trades = []
    # build per-day df with VWAP column
    df_vwap = compute_daily_vwap(df)
    if df_vwap.empty:
        return trades

    for day, day_df in df_vwap.groupby(df_vwap.index.date):
        # only market hours inside day_df
        day_df = day_df.between_time(MORNING_RANGE_START, "15:30")
        if day_df.empty:
            continue

        # morning range (first 30 minutes)
        try:
            morning = day_df.between_time(MORNING_RANGE_START, MORNING_RANGE_END)
        except Exception:
            morning = day_df.loc[(day_df.index.time >= time(9,15)) & (day_df.index.time <= time(9,45))]
        if morning.empty:
            continue
        rng_high = morning["High"].max()
        rng_low = morning["Low"].min()

        in_trade = False
        trade = None

        # iterate after morning range
        post_morning = day_df.between_time(TRADE_WINDOW_START, "15:30")
        for idx, row in post_morning.iterrows():
            # skip if VWAP missing
            if pd.isna(row.get(VWAP_COL, np.nan)):
                continue

            # if not in trade, check breakout
            if not in_trade:
                # long breakout
                if (row["Close"] > rng_high) and (row[VWAP_COL] is not None) and (row["Close"] > row[VWAP_COL]):
                    entry = float(row["Close"])
                    sl = float(rng_low)
                    target = entry + TARGET_MULT * (entry - sl)
                    trade = {
                        "stock": ticker,
                        "type": "long",
                        "entry_time": str(idx),
                        "entry": entry,
                        "sl": sl,
                        "target": float(target),
                        "exit_time": None,
                        "exit": None,
                        "pnl": None
                    }
                    in_trade = True
                    continue

                # short breakout
                if (row["Close"] < rng_low) and (row[VWAP_COL] is not None) and (row["Close"] < row[VWAP_COL]):
                    entry = float(row["Close"])
                    sl = float(rng_high)
                    target = entry - TARGET_MULT * (sl - entry)
                    trade = {
                        "stock": ticker,
                        "type": "short",
                        "entry_time": str(idx),
                        "entry": entry,
                        "sl": sl,
                        "target": float(target),
                        "exit_time": None,
                        "exit": None,
                        "pnl": None
                    }
                    in_trade = True
                    continue

            # if in trade, check exits
            if in_trade and trade is not None:
                # long: stop or target
                if trade["type"] == "long":
                    # stop hit
                    if row["Low"] <= trade["sl"]:
                        exit_price = float(trade["sl"])
                        trade["exit"] = exit_price
                        trade["exit_time"] = str(idx)
                        trade["pnl"] = (exit_price - trade["entry"])
                        trades.append(trade)
                        in_trade = False
                        trade = None
                        break  # only one trade per day (change if you want multiple)
                    # target hit
                    if row["High"] >= trade["target"]:
                        exit_price = float(trade["target"])
                        trade["exit"] = exit_price
                        trade["exit_time"] = str(idx)
                        trade["pnl"] = (exit_price - trade["entry"])
                        trades.append(trade)
                        in_trade = False
                        trade = None
                        break
                else:  # short
                    if row["High"] >= trade["sl"]:
                        exit_price = float(trade["sl"])
                        trade["exit"] = exit_price
                        trade["exit_time"] = str(idx)
                        trade["pnl"] = (trade["entry"] - exit_price)
                        trades.append(trade)
                        in_trade = False
                        trade = None
                        break
                    if row["Low"] <= trade["target"]:
                        exit_price = float(trade["target"])
                        trade["exit"] = exit_price
                        trade["exit_time"] = str(idx)
                        trade["pnl"] = (trade["entry"] - exit_price)
                        trades.append(trade)
                        in_trade = False
                        trade = None
                        break

        # day end: if still open, square off at last available price in day_df (or at TRADE_WINDOW_END)
        if in_trade and trade is not None:
            last_price = float(day_df.iloc[-1]["Close"])
            trade["exit"] = last_price
            trade["exit_time"] = str(day_df.index[-1])
            if trade["type"] == "long":
                trade["pnl"] = (last_price - trade["entry"])
            else:
                trade["pnl"] = (trade["entry"] - last_price)
            trades.append(trade)
            in_trade = False
            trade = None

    return trades

# ---------- RUN BACKTEST ----------
all_trades = []
for stk in STOCKS:
    df = fetch_intraday(stk)
    if df is None:
        continue
    t_trades = backtest_one_symbol(df, stk)
    print(f"{stk}: {len(t_trades)} trades")
    all_trades.extend(t_trades)

# ---------- AGGREGATE + METRICS ----------
if not all_trades:
    print("No trades generated. Either data missing or filters too tight.")
else:
    results = pd.DataFrame(all_trades)
    # ensure pnl column exists and numeric
    if "pnl" not in results.columns:
        # compute from entry/exit
        def compute_row_pnl(r):
            try:
                if r["type"] == "long":
                    return r["exit"] - r["entry"]
                return r["entry"] - r["exit"]
            except Exception:
                return 0.0
        results["pnl"] = results.apply(compute_row_pnl, axis=1)
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

    print("\n=== Intraday Breakout Summary ===")
    print(f"Total PnL: {round(total,2)}")
    print(f"Win Rate: {round(win_rate,2)}")
    print(f"Max Drawdown: {round(max_dd,2)}")
    print(f"Trades: {trades}")
    print(f"Avg Trades/Day: {round(avg_per_day,2)}")

    # save
    results.to_csv(SAVE_CSV, index=False)
    print(f"\nSaved -> {SAVE_CSV}")
