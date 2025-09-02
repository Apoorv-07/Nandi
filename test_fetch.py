import yfinance as yf
import pandas as pd

df = yf.download("RELIANCE.NS", period="3y", interval="1d")
df.to_csv("RELIANCE_daily.csv")
print(df.head())
