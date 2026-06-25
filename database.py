"""
database.py
SQLite-backed persistence layer for the AI Mental Health Sentiment Analyzer.
"""

import sqlite3
from datetime import datetime, date, timedelta

import pandas as pd

DB_PATH = "mental_health.db"


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = _get_conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS journal_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_text TEXT NOT NULL,
            compound REAL NOT NULL,
            pos REAL NOT NULL,
            neu REAL NOT NULL,
            neg REAL NOT NULL,
            mood_label TEXT NOT NULL,
            crisis_flag INTEGER NOT NULL DEFAULT 0,
            timestamp TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT NOT NULL,
            response TEXT NOT NULL,
            crisis_flag INTEGER NOT NULL DEFAULT 0,
            timestamp TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def log_entry(entry_text: str, sentiment: dict, crisis_flag: bool):
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO journal_entries "
        "(entry_text, compound, pos, neu, neg, mood_label, crisis_flag, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            entry_text,
            sentiment["compound"],
            sentiment["pos"],
            sentiment["neu"],
            sentiment["neg"],
            sentiment["label"],
            int(crisis_flag),
            datetime.now().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def get_all_entries(limit: int = None) -> pd.DataFrame:
    conn = _get_conn()
    query = "SELECT * FROM journal_entries ORDER BY id DESC"
    if limit:
        query += f" LIMIT {int(limit)}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def log_chat(message: str, response: str, crisis_flag: bool):
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO chat_log (message, response, crisis_flag, timestamp) VALUES (?, ?, ?, ?)",
        (message, response, int(crisis_flag), datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def get_chat_history(limit: int = 50) -> pd.DataFrame:
    conn = _get_conn()
    df = pd.read_sql_query(
        "SELECT * FROM chat_log ORDER BY id DESC LIMIT ?", conn, params=(limit,)
    )
    conn.close()
    return df


def get_checkin_streak() -> int:
    """Number of consecutive days (including today) with at least one journal entry."""
    df = get_all_entries()
    if df.empty:
        return 0

    entry_dates = set(df["timestamp"].dt.date)
    streak = 0
    cursor_date = date.today()
    while cursor_date in entry_dates:
        streak += 1
        cursor_date -= timedelta(days=1)
    return streak


def get_weekly_average_sentiment() -> float:
    df = get_all_entries()
    if df.empty:
        return 0.0
    cutoff = datetime.now() - timedelta(days=7)
    recent = df[df["timestamp"] >= cutoff]
    if recent.empty:
        return 0.0
    return round(recent["compound"].mean(), 3)