from datetime import datetime
import sqlite3


class EventLogger:
    def __init__(self, db_path="firevision.db", enabled=True):
        self.enabled = enabled
        self.db_path = db_path
        self.conn = None

        if not self.enabled:
            return

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
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
        cur = self.conn.cursor()
        cur.execute(query, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status, source, fps, extra_text))
        self.conn.commit()

    def get_recent_events(self, limit=50):
        if not self.enabled or self.conn is None:
            return []

        query = """
        SELECT id, created_at, status, source, fps, extra_text
        FROM firevision_events
        ORDER BY datetime(created_at) DESC
        LIMIT ?;
        """
        cur = self.conn.cursor()
        cur.execute(query, (limit,))
        rows = cur.fetchall()
        return [dict(row) for row in rows]

    def clear_events(self):
        if not self.enabled or self.conn is None:
            return

        cur = self.conn.cursor()
        cur.execute("DELETE FROM firevision_events;")
        self.conn.commit()