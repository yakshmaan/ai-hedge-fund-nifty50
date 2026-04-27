"""
fetch_data.py
-------------
Downloads OHLCV data for all Nifty 50 stocks from Yahoo Finance.
Saves one CSV per stock into data/raw/.

OHLCV = Open, High, Low, Close, Volume — the raw daily price data.
Run this once to build your dataset. Re-run anytime to refresh.

Usage:
    python pipeline/fetch_data.py
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────────

# Yahoo Finance uses ".NS" suffix for NSE-listed stocks
# These are the current Nifty 50 constituents
NIFTY50_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "BAJFINANCE.NS", "WIPRO.NS", "ONGC.NS", "NTPC.NS",
    "POWERGRID.NS", "ULTRACEMCO.NS", "BAJAJFINSV.NS", "HCLTECH.NS", "JSWSTEEL.NS",
    "TATASTEEL.NS", "ADANIENT.NS", "ADANIPORTS.NS", "COALINDIA.NS", "DIVISLAB.NS",
    "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS", "HEROMOTOCO.NS", "HINDALCO.NS",
    "INDUSINDBK.NS", "M&M.NS", "NESTLEIND.NS", "SBILIFE.NS", "SHREECEM.NS",
    "TATACONSUM.NS", "TATAMOTORS.NS", "TECHM.NS", "CIPLA.NS", "APOLLOHOSP.NS",
    "BAJAJ-AUTO.NS", "BPCL.NS", "BRITANNIA.NS", "HDFCLIFE.NS", "UPL.NS",
]

START_DATE = "2018-01-01"
END_DATE   = datetime.today().strftime("%Y-%m-%d")  # today's date automatically
RAW_DIR    = "data/raw"

# ── MAIN ──────────────────────────────────────────────────────────────────────

def fetch_stock(symbol: str) -> pd.DataFrame | None:
    """
    Download daily OHLCV data for one stock.
    Returns a cleaned DataFrame or None if download fails.
    """
    try:
        df = yf.download(symbol, start=START_DATE, end=END_DATE, progress=False)

        if df.empty:
            print(f"  [WARN] No data returned for {symbol}")
            return None

        # yfinance returns a MultiIndex column when auto_adjust=True sometimes
        # Flatten if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Keep only the columns we need
        df = df[["Open", "High", "Low", "Close", "Volume"]]

        # Add the ticker as a column so we know which stock this is
        df["Symbol"] = symbol

        # Make sure the index (Date) is clean
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"

        return df

    except Exception as e:
        print(f"  [ERROR] Failed to fetch {symbol}: {e}")
        return None


def fetch_all():
    """
    Loop through all Nifty 50 symbols, download each,
    and save to data/raw/<SYMBOL>.csv
    """
    os.makedirs(RAW_DIR, exist_ok=True)

    success = 0
    failed  = []

    print(f"Fetching {len(NIFTY50_SYMBOLS)} stocks from {START_DATE} to {END_DATE}...\n")

    for i, symbol in enumerate(NIFTY50_SYMBOLS, 1):
        print(f"[{i:02d}/{len(NIFTY50_SYMBOLS)}] {symbol}", end=" ... ")

        df = fetch_stock(symbol)

        if df is not None:
            # Save: strip the ".NS" suffix from filename for cleanliness
            filename = symbol.replace(".NS", "").replace("&", "_") + ".csv"
            filepath = os.path.join(RAW_DIR, filename)
            df.to_csv(filepath)
            print(f"saved ({len(df)} rows)")
            success += 1
        else:
            print("FAILED")
            failed.append(symbol)

    # Summary
    print(f"\n{'─'*50}")
    print(f"Done. {success}/{len(NIFTY50_SYMBOLS)} stocks fetched successfully.")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print(f"Raw data saved to: {RAW_DIR}/")


if __name__ == "__main__":
    fetch_all()