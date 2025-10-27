import hashlib
import sqlite3
import json
import threading
from typing import TypedDict, Optional


class CacheSettings(TypedDict):
    """Cache configuration settings."""
    db_loc: str


DEFAULT_CACHE_SETTINGS: CacheSettings = {
    "db_loc": "./openai_cache.db",
}


class Sqlite3CacheProvider:
    """SQLite-based cache provider for OpenAI API responses.

    This provider stores and retrieves cached API responses in a local SQLite database.
    It is thread-safe and compatible with the latest OpenAI Python SDK (>=1.0.0).

    Example:
        >>> cache = Sqlite3CacheProvider()
        >>> key = cache.hash_params({"model": "gpt-4o-mini"})
        >>> cache.insert(key, {"model": "gpt-4o-mini"}, {"result": "ok"})
        >>> print(cache.get(key))
    """

    CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS cache(
        key TEXT PRIMARY KEY NOT NULL,
        request_params JSON NOT NULL,
        response JSON NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    def __init__(self, settings: CacheSettings = DEFAULT_CACHE_SETTINGS):
        """Initialize the cache provider and ensure the table exists.

        Args:
            settings (CacheSettings): Cache configuration, including the SQLite file path.
        """
        # Allow the connection to be used across threads
        self.conn: sqlite3.Connection = sqlite3.connect(
            settings.get("db_loc"),
            check_same_thread=False,
            isolation_level=None,  # Enable autocommit mode for thread safety
        )
        self.lock = threading.Lock()
        self._create_table_if_not_exists()

    def _create_table_if_not_exists(self):
        """Create the cache table if it doesn't already exist."""
        with self.lock:
            self.conn.execute(self.CREATE_TABLE)
            self.conn.commit()

    def hash_params(self, params: dict) -> str:
        """Generate a deterministic MD5 hash from a dictionary of parameters.

        Args:
            params (dict): Request parameters to hash.

        Returns:
            str: MD5 hash string.
        """
        # Sort keys to ensure consistent hash for equivalent parameter sets
        stringified = json.dumps(params, sort_keys=True).encode("utf-8")
        return hashlib.md5(stringified).hexdigest()

    def get(self, key: str) -> Optional[str]:
        """Retrieve a cached response by its key.

        Args:
            key (str): Cache key (MD5 hash of request parameters).

        Returns:
            Optional[str]: Cached JSON string if found, otherwise None.
        """
        with self.lock:
            cursor = self.conn.execute("SELECT response FROM cache WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row[0] if row else None

    def insert(self, key: str, request: dict, response: dict):
        """Insert or replace a cached response.

        Args:
            key (str): Cache key (MD5 hash of request parameters).
            request (dict): Original request parameters.
            response (dict): API response to cache.
        """
        with self.lock:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO cache (key, request_params, response)
                VALUES (?, ?, ?)
                """,
                (
                    key,
                    json.dumps(request, sort_keys=True),
                    json.dumps(response),
                ),
            )
            self.conn.commit()

    def clear(self):
        """Delete all cached entries."""
        with self.lock:
            self.conn.execute("DELETE FROM cache")
            self.conn.commit()

    def close(self):
        """Close the database connection."""
        with self.lock:
            self.conn.close()
