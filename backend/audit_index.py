import logging
import os
from typing import Any, Dict, List, Optional

import pymysql
from pymysql.cursors import DictCursor

from config import settings

logger = logging.getLogger(__name__)


class AuditIndex:
    """Read-only index of processed articles from pipeline_audit_log.

    The table stores config_id as the configuration string identifier and may contain
    multiple rows for one correlation ID after reprocessing. Every public query keeps
    only the newest row per (config_id, correlation_id).
    """

    def _get_connection(self) -> pymysql.Connection:
        ca_path = settings.MYSQL_SSL_CA
        if not os.path.isabs(ca_path):
            ca_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ca_path))
        ssl_config = {"ca": ca_path} if os.path.exists(ca_path) else {"ssl": {}}
        return pymysql.connect(
            host=settings.MYSQL_HOST,
            port=settings.MYSQL_PORT,
            database=settings.MYSQL_DATABASE,
            user=settings.MYSQL_USER,
            password=settings.MYSQL_PASSWORD,
            ssl=ssl_config,
            cursorclass=DictCursor,
            autocommit=True,
            connect_timeout=10,
            read_timeout=30,
        )

    def list_articles(
        self,
        config_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Return recent audit rows deduplicated by correlation ID.

        The audit table has separate indexes on config_id and created_at, but the
        read-only rig cannot add a composite index. A full window-function query is
        too slow on the production table. Read a bounded recent slice using the
        created_at/config_id indexes, then deduplicate reprocessing rows in Python.
        """
        limit = max(1, min(int(limit), 1000))
        offset = max(0, int(offset))
        scan_limit = min(max((limit + offset) * 10, 500), 5000)
        where = "WHERE config_id = %s" if config_id else ""
        params: list[Any] = [config_id] if config_id else []
        params.append(scan_limit)

        sql = f"""
            SELECT
                config_id,
                config_name,
                correlation_id,
                tracker_id,
                status,
                processing_source,
                created_at AS last_processed_at
            FROM pipeline_audit_log
            {where}
            ORDER BY created_at DESC, id DESC
            LIMIT %s
        """

        connection = self._get_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(sql, params)
                rows = cursor.fetchall()
        finally:
            connection.close()

        unique: list[Dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for row in rows:
            key = (str(row["config_id"]), str(row["correlation_id"]))
            if key in seen:
                continue
            seen.add(key)
            unique.append(self._to_blob_entry(row))
            if len(unique) >= offset + limit:
                break
        return unique[offset:offset + limit]

    def list_config_counts(self) -> List[Dict[str, Any]]:
        """Return unique processed article counts grouped by config identifier."""
        sql = """
            SELECT
                config_id,
                MAX(config_name) AS config_name,
                COUNT(DISTINCT correlation_id) AS count,
                MAX(created_at) AS last_processed_at
            FROM pipeline_audit_log
            GROUP BY config_id
            ORDER BY last_processed_at DESC, config_id ASC
        """
        connection = self._get_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(sql)
                rows = cursor.fetchall()
            return [
                {
                    "config_id": row["config_id"],
                    "config_name": row["config_name"],
                    "count": int(row["count"]),
                    "last_processed_at": str(row["last_processed_at"]),
                }
                for row in rows
            ]
        finally:
            connection.close()

    def latest_config_id(self) -> Optional[str]:
        """Return the config identifier from the newest audit row."""
        connection = self._get_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT config_id FROM pipeline_audit_log "
                    "ORDER BY created_at DESC, id DESC LIMIT 1"
                )
                row = cursor.fetchone()
            return row["config_id"] if row else None
        finally:
            connection.close()

    @staticmethod
    def _to_blob_entry(row: Dict[str, Any]) -> Dict[str, Any]:
        correlation_id = str(row["correlation_id"])
        return {
            "name": f"{correlation_id}{settings.AUDIT_BLOB_SUFFIX}.json",
            "correlation_id": correlation_id,
            "article_id": None,
            "headline": "Audit article (open to load headline)",
            "config_id": row["config_id"],
            "config_name": row.get("config_name"),
            "headline": f"Processed article {correlation_id}",
            "status": row.get("status"),
            "processing_source": row.get("processing_source"),
            "processed_at": str(row["last_processed_at"]),
            "last_processed_at": str(row["last_processed_at"]),
            "cached": False,
            "source": "pipeline_audit_log",
            "media_type": None,
        }


audit_index = AuditIndex()
