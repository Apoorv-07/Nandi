# live_intraday.py
# Live-paper intraday bot: morning-range breakout + VWAP confirm
# WARNING: paper-trading only. Uses yfinance intraday (may be slightly delayed).
import time
import math
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, time as dtime, timedelta

# ---------- CONFIG (edit if needed) ----------
STOCKS = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
INTERVAL = "5m"
PERIOD = "3d"                # short recent history to build morning range
MORNING_RANGE_START = "09:15"
MORNING_RANGE_END = "09:45"
TRADE_WINDOW_START = "09:45"
TRADE_WINDOW_END = "15:20"   # force square-off at/after this
VWAP_COL = "VWAP"
TARGET_MULT = 1.5
PORTFOLIO = 5000.0           # your requested portfolio size
RISK_PCT = 0.01              # 1% risk per trade
MIN_QTY = 1
SLEEP_SECS = 30              # loop sleep (seconds)
SAVE_CSV = "live_paper_trades.csv"

# ---------- HELPERS ----------
EXPECTED = ["Open", "High", "Low", "Close", "Volume"]

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        # try pick a level that matches expected names
        for lvl in range(df.columns.nlevels):
            lvl_vals = df.columns.get_level_values(lvl)
            if set([c.lower() for c in EXPECTED]).issubset(set([str(x).lower() for x in lvl_vals])):
                df.columns = lvl_vals
                break
        else:
            df.columns = ['_'.join([str(x) for x in col]).strip() for col in df.columns.values]
    df.columns = [str(c) for c in df.columns]
    return df

def fetch_recent(ticker: str):
    # fetch recent intraday candles
    try:
        df = yf.download(ticker, period=PERIOD, interval=INTERVAL, group_by="column",
                         auto_adjust=True, progress=False)
    except Exception as e:
        print(f"[{ticker}] download error: {e}")
        return None
    if df is None or df.empty:
        print(f"[{ticker}] no data returned.")
        return None
    df = flatten_columns(df)
    # try to pick expected columns case-insensitively
    cols_map = {c.lower(): c for c in df.columns}
    if not set([c.lower() for c in EXPECTED]).issubset(set(cols_map.keys())):
        print(f"[{ticker}] missing OHLCV columns after flattening. Columns: {list(df.columns)}")
        return None
    picked = [cols_map[c.lower()] for c in EXPECTED]
    df = df.loc[:, picked].apply(pd.to_numeric, errors="coerce").dropna()
    # timezone handling -> convert to IST
    try:
        if df.index.tz is None:
            df = df.tz_localize("UTC").tz_convert("Asia/Kolkata")
        else:
            df = df.tz_convert("Asia/Kolkata")
    except Exception:
        pass
    return df

def compute_vwap_per_day(df: pd.DataFrame):
    frames = []
    for day, g in df.groupby(df.index.date):
        g_day = g.between_time(MORNING_RANGE_START, "15:30")
        if g_day.empty:
            continue
        vwap = (g_day["Close"] * g_day["Volume"]).cumsum() / g_day["Volume"].cumsum()
        g_day = g_day.copy()
        g_day[VWAP_COL] = vwap
        frames.append(g_day)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_index()

def in_market_time(now=None):
    if now is None:
        now = datetime.now().time()
    return dtime(9,15) <= now <= dtime(15,30)

def time_in_str(ts):
    return ts.strftime("%Y-%m-%d %H:%M:%S")

# ---------- PAPER TRADE RECORDING ----------
def append_trade_row(row: dict):
    df = pd.DataFrame([row])
    header = not pd.io.common.file_exists(SAVE_CSV)
    df.to_csv(SAVE_CSV, mode="a", index=False, header=header)

# ---------- STRATEGY STATE ----------
# track per-stock whether we've taken today's trade (single trade/day)
state = {s: {"last_trade_date": None, "open_trade": None} for s in STOCKS}

# ---------- MAIN LOOP ----------
print("LIVE PAPER BOT STARTED — (paper-trading only). Press Ctrl+C to stop.")
try:
    while True:
        now = datetime.now().astimezone()  # local tz
        if not in_market_time(now.time()):
            print(f"[{now.strftime('%H:%M:%S')}] Market closed or outside hours. Sleeping...")
            time.sleep(60)
            continue

        for stk in STOCKS:
            df = fetch_recent(stk)
            if df is None or df.empty:
                continue

            df_vwap = compute_vwap_per_day(df)
            if df_vwap.empty:
                # no usable days in fetched data
                continue

            # process todays data
            today_date = now.date()
            # group by days; pick today's group if present else the most recent day
            if today_date in df_vwap.index.date:
                day_df = df_vwap[df_vwap.index.date == today_date]
            else:
                # market might be open but no today rows due to yfinance delay; pick latest day in df
                latest_day = sorted(set(df_vwap.index.date))[-1]
                day_df = df_vwap[df_vwap.index.date == latest_day]

            # ensure day_df within market hours
            day_df = day_df.between_time(MORNING_RANGE_START, "15:30")
            if day_df.empty:
                continue

            # skip if we've already taken a trade today for this stock
            last_trade_date = state[stk]["last_trade_date"]
            if last_trade_date == day_df.index[0].date():
                # check whether we need to monitor open trade (if any)
                open_trade = state[stk]["open_trade"]
                if open_trade:
                    # check exit conditions on latest candle
                    latest = day_df.iloc[-1]
                    idx = day_df.index[-1]
                    exited = False

                    # stop hit
                    if open_trade["type"] == "long":
                        if latest["Low"] <= open_trade["sl"]:
                            exit_price = open_trade["sl"]
                            pnl = (exit_price - open_trade["entry"]) * open_trade["qty"]
                            open_trade.update({
                                "exit_time": time_in_str(idx), "exit_price": exit_price, "pnl": pnl
                            })
                            append_trade_row(open_trade)
                            print(f"[{stk}] STOP hit -> closed long at {exit_price} pnl {pnl:.2f}")
                            state[stk]["open_trade"] = None
                            exited = True
                        elif latest["High"] >= open_trade["target"]:
                            exit_price = open_trade["target"]
                            pnl = (exit_price - open_trade["entry"]) * open_trade["qty"]
                            open_trade.update({
                                "exit_time": time_in_str(idx), "exit_price": exit_price, "pnl": pnl
                            })
                            append_trade_row(open_trade)
                            print(f"[{stk}] TARGET hit -> closed long at {exit_price} pnl {pnl:.2f}")
                            state[stk]["open_trade"] = None
                            exited = True
                    else:  # short
                        if latest["High"] >= open_trade["sl"]:
                            exit_price = open_trade["sl"]
                            pnl = (open_trade["entry"] - exit_price) * open_trade["qty"]
                            open_trade.update({
                                "exit_time": time_in_str(idx), "exit_price": exit_price, "pnl": pnl
                            })
                            append_trade_row(open_trade)
                            print(f"[{stk}] STOP hit -> closed short at {exit_price} pnl {pnl:.2f}")
                            state[stk]["open_trade"] = None
                            exited = True
                        elif latest["Low"] <= open_trade["target"]:
                            exit_price = open_trade["target"]
                            pnl = (open_trade["entry"] - exit_price) * open_trade["qty"]
                            open_trade.update({
                                "exit_time": time_in_str(idx), "exit_price": exit_price, "pnl": pnl
                            })
                            append_trade_row(open_trade)
                            print(f"[{stk}] TARGET hit -> closed short at {exit_price} pnl {pnl:.2f}")
                            state[stk]["open_trade"] = None
                            exited = True

                    # force square-off if trade still open and we've passed TRADE_WINDOW_END
                    if (not exited) and (now.time() >= dtime(int(TRADE_WINDOW_END.split(":")[0]), int(TRADE_WINDOW_END.split(":")[1]))):
                        exit_price = float(latest["Close"])
                        if open_trade["type"] == "long":
                            pnl = (exit_price - open_trade["entry"]) * open_trade["qty"]
                        else:
                            pnl = (open_trade["entry"] - exit_price) * open_trade["qty"]
                        open_trade.update({
                            "exit_time": time_in_str(idx), "exit_price": exit_price, "pnl": pnl
                        })
                        append_trade_row(open_trade)
                        print(f"[{stk}] Square-off -> closed at {exit_price} pnl {pnl:.2f}")
                        state[stk]["open_trade"] = None

                continue  # already traded today, skip entry logic

            # no trade today yet — check morning range and possible breakout
            # build morning range from day's candles
            try:
                morning = day_df.between_time(MORNING_RANGE_START, MORNING_RANGE_END)
            except Exception:
                morning = day_df.loc[(day_df.index.time >= dtime(9,15)) & (day_df.index.time <= dtime(9,45))]
            if morning.empty:
                # morning range not yet available (too early) — skip
                continue

            rng_high = morning["High"].max()
            rng_low = morning["Low"].min()

            # check latest candle after morning range
            latest = day_df.iloc[-1]
            idx = day_df.index[-1]
            # only consider signals after TRADE_WINDOW_START
            if idx.time() < dtime(int(TRADE_WINDOW_START.split(":")[0]), int(TRADE_WINDOW_START.split(":")[1])):
                continue

            # require VWAP present and confirm price vs VWAP
            if pd.isna(latest.get(VWAP_COL, np.nan)):
                continue

            # LONG breakout
            if (latest["Close"] > rng_high) and (latest["Close"] > latest[VWAP_COL]):
                entry_price = float(latest["Close"])
                sl_price = float(rng_low)
                if entry_price <= sl_price:
                    continue
                risk_amt = PORTFOLIO * RISK_PCT
                qty = math.floor(risk_amt / (entry_price - sl_price)) if (entry_price - sl_price) > 0 else 0
                qty = max(qty, 0)
                if qty < MIN_QTY:
                    print(f"[{stk}] computed qty < {MIN_QTY} (qty={qty}) — skipping trade")
                else:
                    target_price = entry_price + TARGET_MULT * (entry_price - sl_price)
                    trade = {
                        "timestamp": time_in_str(now),
                        "ticker": stk,
                        "type": "long",
                        "entry_time": time_in_str(idx),
                        "entry_price": entry_price,
                        "sl": sl_price,
                        "target": target_price,
                        "qty": int(qty),
                        "exit_time": None,
                        "exit_price": None,
                        "pnl": None,
                        "mode": "paper"
                    }
                    state[stk]["last_trade_date"] = day_df.index[0].date()
                    state[stk]["open_trade"] = trade
                    append_trade_row({**trade, **{"note":"ENTRY"}})
                    print(f"[{stk}] PAPER ENTRY long @ {entry_price} qty {qty} sl {sl_price} tgt {round(target_price,2)}")

            # SHORT breakout
            elif (latest["Close"] < rng_low) and (latest["Close"] < latest[VWAP_COL]):
                entry_price = float(latest["Close"])
                sl_price = float(rng_high)
                if sl_price <= entry_price:
                    continue
                risk_amt = PORTFOLIO * RISK_PCT
                qty = math.floor(risk_amt / (sl_price - entry_price)) if (sl_price - entry_price) > 0 else 0
                qty = max(qty, 0)
                if qty < MIN_QTY:
                    print(f"[{stk}] computed qty < {MIN_QTY} (qty={qty}) — skipping trade")
                else:
                    target_price = entry_price - TARGET_MULT * (sl_price - entry_price)
                    trade = {
                        "timestamp": time_in_str(now),
                        "ticker": stk,
                        "type": "short",
                        "entry_time": time_in_str(idx),
                        "entry_price": entry_price,
                        "sl": sl_price,
                        "target": target_price,
                        "qty": int(qty),
                        "exit_time": None,
                        "exit_price": None,
                        "pnl": None,
                        "mode": "paper"
                    }
                    state[stk]["last_trade_date"] = day_df.index[0].date()
                    state[stk]["open_trade"] = trade
                    append_trade_row({**trade, **{"note":"ENTRY"}})
                    print(f"[{stk}] PAPER ENTRY short @ {entry_price} qty {qty} sl {sl_price} tgt {round(target_price,2)}")

        # end for each stock
        time.sleep(SLEEP_SECS)

except KeyboardInterrupt:
    print("Stopped by user. Exiting.")
