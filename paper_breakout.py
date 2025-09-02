import yfinance as yf
import pandas as pd
import datetime as dt
import time

symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
capital = 100000
positions = []
trades = []

def breakout_signal(df):
    # simple rule: close > prev high
    if df["Close"].iloc[-1] > df["High"].iloc[-2]:
        return "BUY"
    elif df["Close"].iloc[-1] < df["Low"].iloc[-2]:
        return "SELL"
    return None

while True:
    now = dt.datetime.now().time()
    if now >= dt.time(9,15) and now <= dt.time(15,30):  # NSE market hours
        for sym in symbols:
            df = yf.download(sym, period="2d", interval="5m")
            signal = breakout_signal(df)

            if signal:
                price = df["Close"].iloc[-1]
                qty = 1  # paper: 1 share
                trades.append({
                    "time": dt.datetime.now(),
                    "symbol": sym,
                    "signal": signal,
                    "price": price,
                    "qty": qty
                })
                print(trades[-1])

        # log every loop
        pd.DataFrame(trades).to_csv("paper_trades.csv", index=False)

        time.sleep(300)  # wait 5m
    else:
        print("Market closed")
        break
