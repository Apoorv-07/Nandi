# live_intraday.py (Action-friendly, single-iteration mode + state persistence)
# Live-paper intraday bot: morning-range breakout + VWAP confirm
# NOTE: This script is modified to run once (one loop) and persist state/trades to files
import os
import json
import time
import math
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, time as dtime, timedelta, timezone
from zoneinfo import ZoneInfo
import argparse

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
SLEEP_SECS = 30              # only used for continuous mode
SAVE_CSV = "live_paper_trades.csv"
STATE_FILE = "live_state.json"
TIMEZONE = ZoneInfo("Asia/Kolkata")

# ---------- HELPERS ----------
EXPECTED = ["Open", "High", "Low", "Close", "Volume"]

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
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
    cols_map = {c.lower(): c for c in df.columns}
    if not set([c.lower() for c in EXPECTED]).issubset(set(cols_map.keys())):
        print(f"[{ticker}] missing OHLCV columns after flattening. Columns: {list(df.columns)}")
        return None
    picked = [cols_map[c.lower()] for c in EXPECTED]
    df = df.loc[:, picked].apply(pd.to_numeric, errors="coerce").dropna()
    try:
        if df.index.tz is None:
            df = df.tz_localize("UTC").tz_convert(TIMEZONE)
        else:
            df = df.tz_convert(TIMEZONE)
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
        now = datetime.now(TIMEZONE).time()
    return dtime(9,15) <= now <= dtime(15,30)

def time_in_str(ts):
    if isinstance(ts, datetime):
        return ts.astimezone(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")
    return str(ts)

# ---------- STATE HANDLING ----------
def load_state():
    if not os.path.exists(STATE_FILE):
        return {s: {"last_trade_date": None, "open_trade": None} for s in STOCKS}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            st = json.load(f)
        # ensure stocks present
        for s in STOCKS:
            if s not in st:
                st[s] = {"last_trade_date": None, "open_trade": None}
        return st
    except Exception as e:
        print("Failed to load state file:", e)
        return {s: {"last_trade_date": None, "open_trade": None} for s in STOCKS}

def save_state(state):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, default=str, indent=2)
    except Exception as e:
        print("Failed to save state file:", e)

# ---------- PAPER TRADE RECORDING ----------
def append_trade_row(row: dict):
    df = pd.DataFrame([row])
    header = not os.path.exists(SAVE_CSV)
    df.to_csv(SAVE_CSV, mode="a", index=False, header=header)

# ---------- CORE SINGLE-ITERATION PROCESS ----------
def single_iteration(state):
    now = datetime.now(TIMEZONE)
    if not in_market_time(now.time()):
        print(f"[{now.strftime('%H:%M:%S')}] Market closed or outside hours. Exiting iteration.")
        return state

    for stk in STOCKS:
        df = fetch_recent(stk)
        if df is None or df.empty:
            continue

        df_vwap = compute_vwap_per_day(df)
        if df_vwap.empty:
            continue

        today_date = now.date()
        if today_date in df_vwap.index.date:
            day_df = df_vwap[df_vwap.index.date == today_date]
        else:
            latest_day = sorted(set(df_vwap.index.date))[-1]
            day_df = df_vwap[df_vwap.index.date == latest_day]

        day_df = day_df.between_time(MORNING_RANGE_START, "15:30")
        if day_df.empty:
            continue

        last_trade_date = state.get(stk, {}).get("last_trade_date")
        if last_trade_date is not None:
            try:
                last_trade_date = datetime.fromisoformat(last_trade_date).date()
            except Exception:
                last_trade_date = None

        # Monitor open trade if present for today's date
        if last_trade_date == day_df.index[0].date():
            open_trade = state[stk].get("open_trade")
            if open_trade:
                latest = day_df.iloc[-1]
                idx = day_df.index[-1]
                exited = False

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
                else:
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

            continue

        # ENTRY logic
        try:
            morning = day_df.between_time(MORNING_RANGE_START, MORNING_RANGE_END)
        except Exception:
            morning = day_df.loc[(day_df.index.time >= dtime(9,15)) & (day_df.index.time <= dtime(9,45))]
        if morning.empty:
            continue

        rng_high = morning["High"].max()
        rng_low = morning["Low"].min()

        latest = day_df.iloc[-1]
        idx = day_df.index[-1]
        if idx.time() < dtime(int(TRADE_WINDOW_START.split(":")[0]), int(TRADE_WINDOW_START.split(":")[1])):
            continue

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
                state[stk]["last_trade_date"] = day_df.index[0].date().isoformat()
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
                state[stk]["last_trade_date"] = day_df.index[0].date().isoformat()
                state[stk]["open_trade"] = trade
                append_trade_row({**trade, **{"note":"ENTRY"}})
                print(f"[{stk}] PAPER ENTRY short @ {entry_price} qty {qty} sl {sl_price} tgt {round(target_price,2)}")

    return state

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run a single iteration and exit (for CI scheduling).")
    parser.add_argument("--continuous", action="store_true", help="Run in continuous loop (original behavior).")
    args = parser.parse_args()

    state = load_state()

    if args.continuous:
        print("LIVE PAPER BOT STARTED (continuous). Ctrl+C to stop.")
        try:
            while True:
                state = single_iteration(state)
                save_state(state)
                time.sleep(SLEEP_SECS)
        except KeyboardInterrupt:
            print("Stopped by user. Exiting.")
            save_state(state)
    else:
        # default: single iteration
        state = single_iteration(state)
        save_state(state)
        print("Single iteration finished.")

if __name__ == "__main__":
    main()
