# intraday_backtest.py
# Intraday backtester: EMA9/EMA21 crossover + RSI filter + ATR stop
# Saves trades to intraday_trades.csv and prints metrics summary.
import pandas as pd
import numpy as np
import yfinance as yf
from math import floor

# ---------------- CONFIG ----------------
TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]    # change as needed
PORTFOLIO = 5000.0
RISK_PCT = 0.01           # 1% per trade (use 0.02 for 2%)
MAX_PER_TRADE_ABS = None  # if set, overrides RISK_PCT calc (e.g., 100)
INTERVAL = "5m"
PERIOD = "60d"            # history window for backtest
TRADE_WINDOW_START = "09:30"
TRADE_WINDOW_END = "11:30"
RSI_THRESHOLD = 55        # require RSI > this for long entries
TARGET_MULT = 1.5         # reward = TARGET_MULT * risk (used for target exit)
ATR_MULT = 1.0            # stop = entry - ATR*ATR_MULT
SAVE_TRADES_CSV = "intraday_trades.csv"

# ---------------- HELPERS / INDICATORS ----------------
def normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    # flatten multiindex if any and ensure Open/High/Low/Close/Volume present
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        if df.columns.nlevels == 2:
            lev0 = df.columns.get_level_values(0)
            lev1 = df.columns.get_level_values(1)
            if len(set(lev0)) == 1:
                df.columns = lev1
            elif len(set(lev1)) == 1:
                df.columns = lev0
            else:
                df.columns = ["_".join(map(str,t)) for t in df.columns]
        else:
            df.columns = ["_".join(map(str,t)) for t in df.columns]
    if "Adj Close" in df.columns and "Close" not in df.columns:
        df["Close"] = df["Adj Close"]
    required = ["Open","High","Low","Close","Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns after normalization: {missing}. Got: {list(df.columns)}")
    df = df[required].apply(pd.to_numeric, errors="coerce").dropna()
    return df.sort_index()

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(period, min_periods=period).mean()
    ma_down = down.rolling(period, min_periods=period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

# ---------------- FETCH ----------------
def fetch_intraday(ticker):
    df = yf.download(ticker, period=PERIOD, interval=INTERVAL, group_by="column", auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data for {ticker}")
    df = normalize_ohlc(df)
    # timezone: make sure timestamps are in IST for window filtering
    try:
        if df.index.tz is None:
            df = df.tz_localize("UTC").tz_convert("Asia/Kolkata")
        else:
            df = df.tz_convert("Asia/Kolkata")
    except Exception:
        # fallback: assume naive index already in IST
        pass
    return df

# ---------------- STRATEGY (per-symbol) ----------------
def prepare_df(df):
    df = df.copy()
    df["EMA9"] = ema(df["Close"], 9)
    df["EMA21"] = ema(df["Close"], 21)
    df["RSI14"] = rsi(df["Close"], 14)
    df["ATR14"] = atr(df, 14)
    return df

def within_window(ts, start_str, end_str):
    t = ts.time()
    start_h, start_m = map(int, start_str.split(":"))
    end_h, end_m = map(int, end_str.split(":"))
    from datetime import time as dtime
    return dtime(start_h, start_m) <= t <= dtime(end_h, end_m)

def intraday_backtest_for_symbol(ticker):
    df = fetch_intraday(ticker)
    df = prepare_df(df)
    trades = []
    # group by date (calendar day)
    for date, group in df.groupby(df.index.date):
        open_position = None  # dict with entry details
        rows = group.sort_index()
        for idx, row in rows.iterrows():
            if not within_window(idx, TRADE_WINDOW_START, TRADE_WINDOW_END):
                # If window ended and position open -> square off at prev close inside window end
                if open_position is not None:
                    # square-off at last available price within day/window
                    exit_price = row["Close"]
                    trades.append({**open_position, "ExitDate": idx, "ExitPrice": float(exit_price),
                                   "Profit": (exit_price - open_position["EntryPrice"]) * open_position["Qty"], "Ticker": ticker})
                    open_position = None
                continue

            # skip until indicators filled
            if np.isnan(row["EMA9"]) or np.isnan(row["EMA21"]) or np.isnan(row["RSI14"]) or np.isnan(row["ATR14"]):
                continue

            # check entry: EMA9 crosses above EMA21 (use previous row)
            prev = rows.loc[:idx].iloc[-2] if len(rows.loc[:idx])>=2 else None
            crossed_up = False
            if prev is not None and not np.isnan(prev["EMA9"]) and not np.isnan(prev["EMA21"]):
                crossed_up = (prev["EMA9"] <= prev["EMA21"]) and (row["EMA9"] > row["EMA21"])
            # ENTRY
            if open_position is None and crossed_up and row["RSI14"] > RSI_THRESHOLD:
                entry_price = row["Close"]
                stop_price = entry_price - ATR_MULT * row["ATR14"]
                risk_per_trade = MAX_PER_TRADE_ABS if MAX_PER_TRADE_ABS is not None else (RISK_PCT * PORTFOLIO)
                # ensure non-zero stop distance
                dist = entry_price - stop_price
                if dist <= 0:
                    continue
                qty = floor(risk_per_trade / dist)
                if qty <= 0:
                    # position too small for given risk -> skip
                    continue
                open_position = {
                    "Ticker": ticker,
                    "EntryDate": idx,
                    "EntryPrice": float(entry_price),
                    "StopPrice": float(stop_price),
                    "Qty": int(qty),
                    "RiskPerTrade": float(risk_per_trade)
                }
                # continue to next candle (we only take one entry at a time)
                continue

            # if position open, check exit conditions each candle
            if open_position is not None:
                # stop hit
                if row["Low"] <= open_position["StopPrice"]:
                    exit_price = open_position["StopPrice"]  # assume executed at stop
                    trades.append({**open_position, "ExitDate": idx, "ExitPrice": float(exit_price),
                                   "Profit": (exit_price - open_position["EntryPrice"]) * open_position["Qty"], "Ticker": ticker})
                    open_position = None
                    continue
                # target hit
                target = open_position["EntryPrice"] + TARGET_MULT * (open_position["EntryPrice"] - open_position["StopPrice"])
                if row["High"] >= target:
                    exit_price = target
                    trades.append({**open_position, "ExitDate": idx, "ExitPrice": float(exit_price),
                                   "Profit": (exit_price - open_position["EntryPrice"]) * open_position["Qty"], "Ticker": ticker})
                    open_position = None
                    continue
                # reverse crossover -> exit (EMA9 crosses below EMA21)
                if prev is not None:
                    crossed_down = (prev["EMA9"] >= prev["EMA21"]) and (row["EMA9"] < row["EMA21"])
                    if crossed_down:
                        exit_price = row["Close"]
                        trades.append({**open_position, "ExitDate": idx, "ExitPrice": float(exit_price),
                                       "Profit": (exit_price - open_position["EntryPrice"]) * open_position["Qty"], "Ticker": ticker})
                        open_position = None
                        continue
        # day end: if open_position remains, square off at last available price for that day
        if open_position is not None:
            last_price = rows.iloc[-1]["Close"]
            trades.append({**open_position, "ExitDate": rows.index[-1], "ExitPrice": float(last_price),
                           "Profit": (last_price - open_position["EntryPrice"]) * open_position["Qty"], "Ticker": ticker})
            open_position = None
    return trades

# ---------------- AGGREGATE + METRICS ----------------
def compute_metrics(trades):
    if not trades:
        return {"Total PnL":0.0,"Win Rate":0.0,"Max Drawdown":0.0,"Trades":0}
    profits = [t["Profit"] for t in trades]
    total = sum(profits)
    wins = sum(1 for p in profits if p>0)
    win_rate = wins / len(profits) * 100
    equity = np.cumsum(profits)
    peak = np.maximum.accumulate(equity)
    drawdowns = equity - peak
    max_dd = drawdowns.min() if len(drawdowns)>0 else 0.0
    avg_trades_per_day = len(trades) / max(1, len(set([pd.to_datetime(t["EntryDate"]).date() for t in trades])))
    return {"Total PnL": round(total,2), "Win Rate": round(win_rate,2), "Max Drawdown": round(max_dd,2), "Trades": len(trades), "Avg Trades/Day": round(avg_trades_per_day,2)}

# ---------------- MAIN ----------------
if __name__ == "__main__":
    all_trades = []
    for t in TICKERS:
        try:
            t_trades = intraday_backtest_for_symbol(t)
            all_trades.extend(t_trades)
            print(f"{t}: {len(t_trades)} trades")
        except Exception as e:
            print(f"Skipped {t}: {e}")

    if all_trades:
        df_trades = pd.DataFrame(all_trades)
        # normalize datetime columns to strings
        df_trades["EntryDate"] = df_trades["EntryDate"].astype(str)
        df_trades["ExitDate"] = df_trades["ExitDate"].astype(str)
        df_trades.to_csv(SAVE_TRADES_CSV, index=False)
        metrics = compute_metrics(all_trades)
        print("\n=== Intraday Backtest Summary ===")
        for k,v in metrics.items():
            print(f"{k}: {v}")
        print(f"\nSaved trades -> {SAVE_TRADES_CSV}")
    else:
        print("No trades generated. Tighten filters or increase history/window.")
