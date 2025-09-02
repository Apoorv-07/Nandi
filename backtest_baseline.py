# backtest_baseline.py
import pandas as pd
import numpy as np
import yfinance as yf

# ---------------- CONFIG ----------------
TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
MAX_PER_TRADE = 3000
CAPITAL = 30000

# ---------------- UTIL: normalize yfinance columns ----------------
def normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Flatten MultiIndex to single level
    if isinstance(df.columns, pd.MultiIndex):
        # If any level has only one unique value, drop that level
        if df.columns.nlevels == 2:
            lev0 = df.columns.get_level_values(0)
            lev1 = df.columns.get_level_values(1)
            if len(set(lev0)) == 1:
                df.columns = lev1
            elif len(set(lev1)) == 1:
                df.columns = lev0
            else:
                df.columns = ["_".join([str(a) for a in tup]) for tup in df.columns]
        else:
            df.columns = ["_".join([str(x) for x in tup]) for tup in df.columns]

    # Standardize common names
    # yfinance gives: Open, High, Low, Close, Adj Close, Volume
    cols = {c: c.strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)

    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    if "Close" not in df.columns and "Price" in df.columns:
        df["Close"] = df["Price"]

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns after normalization: {missing}. Got: {list(df.columns)}")

    # Keep only required and ensure sorted by date
    out = df[required].sort_index()
    # ensure numeric
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.dropna()
    return out

# ---------------- INDICATORS ----------------
def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA20"] = df["Close"].rolling(20, min_periods=20).mean()
    df["SMA50"] = df["Close"].rolling(50, min_periods=50).mean()
    df["ATR14"] = atr(df, 14)

    # Long-only regime: 1 when SMA20>SMA50, else 0
    df["Signal"] = 0
    cond_long = (df["SMA20"] > df["SMA50"]).astype(int)
    df.loc[cond_long == 1, "Signal"] = 1

    # Entry price and fixed stop at entry time
    df["Entry"] = np.where(df["Signal"] == 1, df["Close"], np.nan)
    df["Stop"] = np.where(df["Signal"] == 1, df["Entry"] - 2 * df["ATR14"], np.nan)
    return df

# ---------------- BACKTEST ----------------
def backtest(df: pd.DataFrame):
    trades = []
    position = 0
    entry_price = np.nan
    stop_price = np.nan
    qty = 0

    for i in range(len(df)):
        row = df.iloc[i]

        # Entry
        if position == 0 and row["Signal"] == 1 and not np.isnan(row["Stop"]):
            entry_price = row["Close"]
            stop_price = row["Stop"]
            qty = int(MAX_PER_TRADE // entry_price)
            if qty > 0:
                position = qty
                trades.append({"Date": df.index[i], "Type": "BUY", "Price": float(entry_price), "Qty": int(qty)})

        # Exit: stop-loss or regime off (Signal goes 0)
        elif position > 0 and (row["Close"] < stop_price or row["Signal"] == 0):
            exit_price = row["Close"]
            trades.append({"Date": df.index[i], "Type": "SELL", "Price": float(exit_price), "Qty": int(position)})
            position = 0
            entry_price = np.nan
            stop_price = np.nan
            qty = 0

    return trades

# ---------------- MULTI STOCK ----------------
def fetch_data(ticker: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period="3y",
        interval="1d",
        group_by="column",    # <- force single-level columns
        auto_adjust=True,
        progress=False,
    )
    if df is None or len(df) == 0:
        raise ValueError(f"No data for {ticker}")
    return normalize_ohlc(df)

def backtest_multiple():
    all_trades = []
    for ticker in TICKERS:
        df = fetch_data(ticker)
        df = generate_signals(df)
        trades = backtest(df)
        for t in trades:
            t["Ticker"] = ticker.replace(".NS", "")
        all_trades.extend(trades)
    return all_trades

# ---------------- METRICS ----------------
def compute_metrics(trades):
    pnl = []
    last_buy = None
    for t in trades:
        if t["Type"] == "BUY":
            last_buy = t
        elif t["Type"] == "SELL" and last_buy:
            profit = (t["Price"] - last_buy["Price"]) * last_buy["Qty"]
            pnl.append(profit)
            last_buy = None

    total_pnl = float(np.sum(pnl)) if pnl else 0.0
    wins = sum(1 for p in pnl if p > 0)
    win_rate = (wins / len(pnl)) * 100 if pnl else 0.0

    # proper max drawdown on cumulative PnL
    if pnl:
        equity = np.cumsum(pnl)
        peak = np.maximum.accumulate(equity)
        drawdowns = equity - peak
        max_dd = float(drawdowns.min())
    else:
        max_dd = 0.0

    return {
        "Total PnL": round(total_pnl, 2),
        "Win Rate": round(win_rate, 2),
        "Max Drawdown": round(max_dd, 2),
        "Trades": len(pnl),
    }

# ---------------- MAIN ----------------
if __name__ == "__main__":
    print("Fetching data & running backtest...")
    trades = backtest_multiple()
    metrics = compute_metrics(trades)
    print("\n=== Baseline Backtest Results ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
