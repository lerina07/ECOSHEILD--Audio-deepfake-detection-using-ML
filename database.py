"""
database.py — SQLite operations for AudioForensics AI
Handles users, sessions, analysis history, and report storage.
"""

import sqlite3
import os
from datetime import datetime
from contextlib import contextmanager

DB_PATH = "audioforensics.db"


# ===============================
# CONNECTION CONTEXT MANAGER
# ===============================
@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row          # rows behave like dicts
    conn.execute("PRAGMA journal_mode=WAL") # concurrent reads
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ===============================
# SCHEMA INIT
# ===============================
def init_db():
    """Create all tables if they don't exist. Call once at app startup."""
    with get_db() as conn:
        conn.executescript("""
            -- ── Users ──────────────────────────────────────────────────────
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT    NOT NULL UNIQUE,
                email         TEXT    NOT NULL UNIQUE,
                password_hash TEXT    NOT NULL,
                created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            -- ── Analysis records ────────────────────────────────────────────
            CREATE TABLE IF NOT EXISTS analyses (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                file_name     TEXT    NOT NULL,
                duration_sec  REAL,
                prediction    TEXT    NOT NULL,   -- 'REAL' | 'FAKE'
                real_conf     REAL    NOT NULL,
                fake_conf     REAL    NOT NULL,
                transcript    TEXT,
                created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            -- ── PDF reports (stored as BLOBs) ───────────────────────────────
            CREATE TABLE IF NOT EXISTS reports (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER NOT NULL UNIQUE REFERENCES analyses(id) ON DELETE CASCADE,
                pdf_data    BLOB    NOT NULL,
                created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            );
        """)


# ===============================
# USER OPERATIONS
# ===============================
def create_user(username: str, email: str, password_hash: str) -> int:
    """Insert a new user. Returns new user id."""
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username.strip(), email.strip().lower(), password_hash),
        )
        return cur.lastrowid


def get_user_by_username(username: str):
    """Return user row or None."""
    with get_db() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE username = ?", (username.strip(),)
        ).fetchone()


def get_user_by_email(email: str):
    """Return user row or None."""
    with get_db() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE email = ?", (email.strip().lower(),)
        ).fetchone()


def get_user_by_id(user_id: int):
    with get_db() as conn:
        return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def username_exists(username: str) -> bool:
    return get_user_by_username(username) is not None


def email_exists(email: str) -> bool:
    return get_user_by_email(email) is not None


# ===============================
# ANALYSIS OPERATIONS
# ===============================
def save_analysis(
    user_id: int,
    file_name: str,
    duration_sec: float,
    prediction: str,
    real_conf: float,
    fake_conf: float,
    transcript: str = "",
) -> int:
    """Insert an analysis record and return its id."""
    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO analyses
               (user_id, file_name, duration_sec, prediction, real_conf, fake_conf, transcript)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (user_id, file_name, round(duration_sec, 3),
             prediction.upper(), round(real_conf, 4), round(fake_conf, 4), transcript or ""),
        )
        return cur.lastrowid


def get_user_analyses(user_id: int) -> list:
    """Return all analyses for a user, newest first."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT a.*, (SELECT 1 FROM reports r WHERE r.analysis_id = a.id) AS has_report
               FROM analyses a
               WHERE a.user_id = ?
               ORDER BY a.created_at DESC""",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_analysis_by_id(analysis_id: int, user_id: int):
    """Fetch a single analysis (ownership-checked)."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM analyses WHERE id = ? AND user_id = ?",
            (analysis_id, user_id),
        ).fetchone()
        return dict(row) if row else None


def delete_analysis(analysis_id: int, user_id: int) -> bool:
    """Delete an analysis (and its report via CASCADE). Returns True if deleted."""
    with get_db() as conn:
        cur = conn.execute(
            "DELETE FROM analyses WHERE id = ? AND user_id = ?",
            (analysis_id, user_id),
        )
        return cur.rowcount > 0


# ===============================
# REPORT OPERATIONS
# ===============================
def save_report(analysis_id: int, pdf_bytes: bytes) -> int:
    """Store PDF bytes. Returns report id."""
    with get_db() as conn:
        cur = conn.execute(
            "INSERT OR REPLACE INTO reports (analysis_id, pdf_data) VALUES (?, ?)",
            (analysis_id, pdf_bytes),
        )
        return cur.lastrowid


def get_report(analysis_id: int, user_id: int) -> bytes | None:
    """Return PDF bytes for a report (ownership-checked via JOIN)."""
    with get_db() as conn:
        row = conn.execute(
            """SELECT r.pdf_data
               FROM reports r
               JOIN analyses a ON a.id = r.analysis_id
               WHERE r.analysis_id = ? AND a.user_id = ?""",
            (analysis_id, user_id),
        ).fetchone()
        return row["pdf_data"] if row else None


def has_report(analysis_id: int) -> bool:
    with get_db() as conn:
        return conn.execute(
            "SELECT 1 FROM reports WHERE analysis_id = ?", (analysis_id,)
        ).fetchone() is not None


# ===============================
# STATS (for dashboard)
# ===============================
def get_user_stats(user_id: int) -> dict:
    with get_db() as conn:
        row = conn.execute(
            """SELECT
                COUNT(*)                                                    AS total,
                SUM(CASE WHEN prediction='REAL' THEN 1 ELSE 0 END)         AS real_count,
                SUM(CASE WHEN prediction='FAKE' THEN 1 ELSE 0 END)         AS fake_count,
                ROUND(COALESCE(AVG(real_conf), 0.0), 1)                    AS avg_real_conf,
                ROUND(COALESCE(AVG(fake_conf), 0.0), 1)                    AS avg_fake_conf
               FROM analyses WHERE user_id = ?""",
            (user_id,),
        ).fetchone()
        if not row:
            return {"total": 0, "real_count": 0, "fake_count": 0,
                    "avg_real_conf": 0.0, "avg_fake_conf": 0.0}
        d = dict(row)
        # Ensure floats are never None regardless of SQLite version behaviour
        d["avg_real_conf"] = d["avg_real_conf"] or 0.0
        d["avg_fake_conf"] = d["avg_fake_conf"] or 0.0
        d["total"]       = d["total"]       or 0
        d["real_count"]  = d["real_count"]  or 0
        d["fake_count"]  = d["fake_count"]  or 0
        return d