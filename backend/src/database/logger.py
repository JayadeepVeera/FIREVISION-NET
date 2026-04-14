from datetime import datetime
import sqlite3


class EventLogger:
    def __init__(self, db_path="firevision.db", enabled=True):
        self.enabled = enabled
        self.db_path = db_path
        self.conn = None

        if not self.enabled:
            return

        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self._create_tables()
        except Exception as e:
            print(f"[WARN] Logger init failed: {e}")
            self.conn = None
            self.enabled = False

    def _create_tables(self):
        if self.conn is None:
            return

        query = """
        CREATE TABLE IF NOT EXISTS firevision_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL,
            source TEXT DEFAULT 'web_camera',
            fps REAL,
            extra_text TEXT
        );
        """
        cur = self.conn.cursor()
        cur.execute(query)
        self.conn.commit()

    def log_event(self, status: str, fps: float = None, source: str = "web_camera", extra_text: str = ""):
        if not self.enabled or self.conn is None:
            return

        query = """
        INSERT INTO firevision_events (created_at, status, source, fps, extra_text)
        VALUES (?, ?, ?, ?, ?);
        """
        try:
            cur = self.conn.cursor()
            cur.execute(
                query,
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status, source, fps, extra_text)
            )
            self.conn.commit()
        except Exception as e:
            print(f"[WARN] log_event failed: {e}")

    def get_recent_events(self, limit=50):
        if not self.enabled or self.conn is None:
            return []

        query = """
        SELECT id, created_at, status, source, fps, extra_text
        FROM firevision_events
        ORDER BY datetime(created_at) DESC
        LIMIT ?;
        """
        try:
            cur = self.conn.cursor()
            cur.execute(query, (limit,))
            rows = cur.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            print(f"[WARN] get_recent_events failed: {e}")
            return []

    def clear_events(self):
        if not self.enabled or self.conn is None:
            return

        try:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM firevision_events;")
            self.conn.commit()
        except Exception as e:
            print(f"[WARN] clear_events failed: {e}")

    def close(self):
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None