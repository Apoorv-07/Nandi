import yfinance as yf
import pandas as pd
import os

tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]

# ensure folders exist
os.makedirs("data/daily", exist_ok=True)
os.makedirs("data/4h", exist_ok=True)

for ticker in tickers:
    # Daily data
    df_daily = yf.download(ticker, period="3y", interval="1d")
    df_daily.to_csv(f"data/daily/{ticker.replace('.NS','')}_daily.csv")
    print(f"Saved daily {ticker}")

    # 4H data
    df_4h = yf.download(ticker, period="60d", interval="4h")
    df_4h.to_csv(f"data/4h/{ticker.replace('.NS','')}_4h.csv")
    print(f"Saved 4h {ticker}")
