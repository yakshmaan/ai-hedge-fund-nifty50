"""
store_data.py
-------------
Loads all cleaned CSVs into a single SQLite database: nifty50.db

Why SQLite instead of just CSVs?
  - You can query across all 50 stocks in one line
  - Much faster than opening 50 CSV files
  - All future agents just do: query the db, get data
  - No external server needed — it's just one file

Database schema:
    Table: prices
        Date        TEXT    (YYYY-MM-DD)
        Symbol      TEXT    (e.g. RELIANCE)
        Open        REAL
        High        REAL
        Low         REAL
        Close       REAL
        Volume      REAL
        Daily_Return REAL
        Is_Outlier  INTEGER (0 or 1)

Run AFTER clean_data.py.

Usage:
    python pipeline/store_data.py
"""

import pandas as pd
import sqlite3
import os
import glob

# ── CONFIG ────────────────────────────────────────────────────────────────────

CLEAN_DIR = "data/clean"
DB_PATH   = "data/nifty50.db"


# ── STORE LOGIC ───────────────────────────────────────────────────────────────

def build_database():
    """
    Read all clean CSVs and write them into nifty50.db.
    If the database already exists, it replaces the prices table.
    """
    clean_files = glob.glob(os.path.join(CLEAN_DIR, "*.csv"))

    if not clean_files:
        print(f"No files found in {CLEAN_DIR}/. Run clean_data.py first.")
        return

    print(f"Loading {len(clean_files)} stocks into database...\n")

    all_dfs = []

    for filepath in sorted(clean_files):
        symbol = os.path.basename(filepath).replace(".csv", "")
        df = pd.read_csv(filepath, index_col="Date", parse_dates=True)
        df["Symbol"] = symbol  # make sure symbol column is set
        all_dfs.append(df)

    # Combine all stocks into one big DataFrame
    combined = pd.concat(all_dfs)
    combined = combined.reset_index()  # make Date a regular column
    combined["Date"] = combined["Date"].astype(str)  # SQLite stores dates as text

    # Write to SQLite
    conn = sqlite3.connect(DB_PATH)

    combined.to_sql(
        name="prices",
        con=conn,
        if_exists="replace",   # replace table if it already exists
        index=False,
    )

    # Create an index on (Symbol, Date) so queries are fast
    conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_date ON prices (Symbol, Date)")
    conn.commit()

    # Verify
    row_count = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    symbol_count = conn.execute("SELECT COUNT(DISTINCT Symbol) FROM prices").fetchone()[0]
    date_range = conn.execute("SELECT MIN(Date), MAX(Date) FROM prices").fetchone()

    conn.close()

    print(f"{'─'*50}")
    print(f"Database built: {DB_PATH}")
    print(f"  Stocks  : {symbol_count}")
    print(f"  Rows    : {row_count:,}")
    print(f"  From    : {date_range[0]}")
    print(f"  To      : {date_range[1]}")


def test_query():
    """
    Run a quick test query to confirm the DB works.
    Gets RELIANCE closing prices for the last 5 trading days.
    """
    if not os.path.exists(DB_PATH):
        print("Database not found. Run build_database() first.")
        return

    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql("""
        SELECT Date, Close, Daily_Return
        FROM prices
        WHERE Symbol = 'RELIANCE'
        ORDER BY Date DESC
        LIMIT 5
    """, conn)

    conn.close()

    print("\nTest query — RELIANCE last 5 days:")
    print(df.to_string(index=False))
    print("\nPhase 1 complete. Your database is ready.")


if __name__ == "__main__":
    build_database()
    test_query()