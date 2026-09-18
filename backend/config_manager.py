import os
import json
import logging
from typing import Any, Optional, Dict, Tuple
import pymysql
from pymysql.cursors import DictCursor
from config import settings

logger = logging.getLogger(__name__)

PUBLISHED_BY_CONFIG_ID_SQL = """
SELECT
    c.id AS numeric_config_id,
    c.identifier AS config_id,
    c.name AS config_name,
    vs.version_number,
    vs.snapshot_data,
    vs.published_at
FROM configurations AS c
INNER JOIN cfg_version_snapshots AS vs
    ON vs.configuration_id = c.id
WHERE c.identifier = %s
  AND (
      c.status = 'active'
      OR (
          c.status = 'pending_changes'
          AND COALESCE(c.previous_status, 'active') = 'active'
      )
  )
  AND c.is_archived = 0
ORDER BY vs.version_number DESC
LIMIT 1;
"""
HISTORICAL_BY_CONFIG_ID_SQL = """
SELECT
    c.id AS numeric_config_id,
    c.identifier AS config_id,
    c.name AS config_name,
    vs.version_number,
    vs.snapshot_data,
    vs.published_at,
    c.status,
    c.is_archived
FROM configurations AS c
INNER JOIN cfg_version_snapshots AS vs
    ON vs.configuration_id = c.id
WHERE c.identifier = %s
ORDER BY vs.version_number DESC
LIMIT 1;
"""

PUBLISHED_BY_TRACKER_ID_SQL = """
SELECT
    c.id AS numeric_config_id,
    c.identifier AS config_id,
    c.name AS config_name,
    vs.version_number,
    vs.snapshot_data,
    vs.published_at
FROM configurations AS c
INNER JOIN tracker_mappings AS tm
    ON tm.configuration_id = c.id
INNER JOIN cfg_version_snapshots AS vs
    ON vs.configuration_id = c.id
WHERE tm.tracker_id = %s
  AND (
      c.status = 'active'
      OR (
          c.status = 'pending_changes'
          AND COALESCE(c.previous_status, 'active') = 'active'
      )
  )
  AND c.is_archived = 0
ORDER BY vs.version_number DESC
LIMIT 1;
"""

ALL_ACTIVE_CONFIGS_SQL = """
SELECT
    c.id AS numeric_config_id,
    c.identifier AS config_id,
    c.name AS config_name,
    vs.version_number,
    vs.published_at
FROM configurations AS c
INNER JOIN cfg_version_snapshots AS vs
    ON vs.configuration_id = c.id
WHERE c.is_archived = 0
ORDER BY c.name ASC, vs.version_number DESC;
"""

class ConfigManager:
    def __init__(self):
        self.cache_dir = os.path.join(settings.CACHE_DIR, "configs")
        os.makedirs(self.cache_dir, exist_ok=True)
        self._memory_cache: Dict[str, Dict[str, Any]] = {}

    def _get_connection(self) -> pymysql.Connection:
        ca_path = settings.MYSQL_SSL_CA
        if not os.path.isabs(ca_path):
            ca_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ca_path))
            
        ssl_config = {"ca": ca_path} if os.path.exists(ca_path) else None
        
        return pymysql.connect(
            host=settings.MYSQL_HOST,
            port=settings.MYSQL_PORT,
            database=settings.MYSQL_DATABASE,
            user=settings.MYSQL_USER,
            password=settings.MYSQL_PASSWORD,
            ssl=ssl_config,
            cursorclass=DictCursor,
            connect_timeout=10,
            read_timeout=30,
            autocommit=True
        )

    def get_cached_config(self, config_id: str) -> Optional[Dict[str, Any]]:
        if config_id in self._memory_cache:
            return self._memory_cache[config_id]
        
        cache_file = os.path.join(self.cache_dir, f"{config_id}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._memory_cache[config_id] = data
                    return data
            except Exception as e:
                logger.warning(f"Failed to read disk cache for {config_id}: {e}")
        return None

    def save_cached_config(self, config_id: str, data: Dict[str, Any]) -> None:
        self._memory_cache[config_id] = data
        cache_file = os.path.join(self.cache_dir, f"{config_id}.json")
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to write disk cache for {config_id}: {e}")

    def fetch_published_config(self, config_id: Optional[str] = None, tracker_id: Optional[str] = None) -> Dict[str, Any]:
        """Loads published snapshot for config_id or tracker_id, with disk/memory caching."""
        lookup_key = config_id or f"tracker_{tracker_id}"
        cached = self.get_cached_config(lookup_key)
        if cached:
            return cached

        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                if config_id:
                    cursor.execute(PUBLISHED_BY_CONFIG_ID_SQL, (config_id,))
                elif tracker_id:
                    cursor.execute(PUBLISHED_BY_TRACKER_ID_SQL, (tracker_id,))
                else:
                    raise ValueError("Must provide either config_id or tracker_id")
                
                row = cursor.fetchone()
                if not row:
                    raise LookupError(f"No active published config found for {config_id or tracker_id}")

                raw_snapshot = row["snapshot_data"]
                snapshot = json.loads(raw_snapshot) if isinstance(raw_snapshot, str) else raw_snapshot

                result = {
                    "metadata": {
                        "numeric_config_id": row["numeric_config_id"],
                        "config_id": row["config_id"],
                        "config_name": row["config_name"],
                        "version_number": row["version_number"],
                        "published_at": str(row["published_at"]) if row["published_at"] else None,
                    },
                    "snapshot": snapshot
                }

                self.save_cached_config(lookup_key, result)
                if row["config_id"] != lookup_key:
                    self.save_cached_config(row["config_id"], result)

                return result
        finally:
            conn.close()

    def fetch_historical_config(self, config_id: str) -> Dict[str, Any]:
        """Load the newest stored snapshot even when the config is inactive."""
        cache_key = f"audit_{config_id}"
        cached = self.get_cached_config(cache_key)
        if cached:
            return cached

        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(HISTORICAL_BY_CONFIG_ID_SQL, (config_id,))
                row = cursor.fetchone()
            if not row:
                raise LookupError(f"No stored config snapshot found for {config_id}")
            raw_snapshot = row["snapshot_data"]
            snapshot = json.loads(raw_snapshot) if isinstance(raw_snapshot, str) else raw_snapshot
            result = {
                "metadata": {
                    "numeric_config_id": row["numeric_config_id"],
                    "config_id": row["config_id"],
                    "config_name": row["config_name"],
                    "version_number": row["version_number"],
                    "published_at": str(row["published_at"]) if row["published_at"] else None,
                    "lookup_mode": "historical_snapshot",
                    "config_status": row["status"],
                    "is_archived": bool(row["is_archived"]),
                },
                "snapshot": snapshot,
            }
            self.save_cached_config(cache_key, result)
            return result
        finally:
            conn.close()

    def list_available_configs(self) -> list[Dict[str, Any]]:
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(ALL_ACTIVE_CONFIGS_SQL)
                rows = cursor.fetchall()
                seen = set()
                unique = []
                for r in rows:
                    if r["config_id"] not in seen:
                        seen.add(r["config_id"])
                        unique.append({
                            "numeric_config_id": r["numeric_config_id"],
                            "config_id": r["config_id"],
                            "config_name": r["config_name"],
                            "version_number": r["version_number"],
                            "published_at": str(r["published_at"]) if r["published_at"] else None
                        })
                return unique
        finally:
            conn.close()

config_manager = ConfigManager()
