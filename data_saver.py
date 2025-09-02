import yfinance as yf
import pandas as pd

sym = "RELIANCE.NS"
df = yf.download(sym, period="5d", interval="5m")
df.to_csv("historical.csv")
print(df.head())
