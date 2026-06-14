import sqlite3
from datetime import datetime, timedelta, timezone
from hashlib import sha256


def make_key(tool_name: str, *parts: str) -> str:
    return sha256(":".join((tool_name, *parts)).encode()).hexdigest()


class CacheStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._ensure_table()

    def _ensure_table(self) -> None:
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            "  key        TEXT PRIMARY KEY,"
            "  data       TEXT NOT NULL,"
            "  expires_at TEXT NOT NULL"
            ")"
        )
        self._conn.commit()

    def get(self, key: str) -> str | None:
        now = datetime.now(timezone.utc).isoformat()
        # Check for a live (non-expired) entry first — no DML on cache hits,
        # so no implicit transaction is opened on the happy path.
        row = self._conn.execute(
            "SELECT data FROM cache WHERE key = ? AND expires_at > ?", (key, now)
        ).fetchone()
        if row is not None:
            return str(row[0])
        # Cache miss: evict any expired entry and commit so the lock is released
        # before the caller fetches fresh data and calls set().
        self._conn.execute("DELETE FROM cache WHERE key = ? AND expires_at <= ?", (key, now))
        self._conn.commit()
        return None

    def set(self, key: str, data: str, ttl_hours: float) -> None:
        expires_at = (datetime.now(timezone.utc) + timedelta(hours=ttl_hours)).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (key, data, expires_at) VALUES (?, ?, ?)",
            (key, data, expires_at),
        )
        self._conn.commit()
