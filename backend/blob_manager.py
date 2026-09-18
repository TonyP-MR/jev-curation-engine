import os
import json
import logging
from typing import Any, Optional, Dict, List
from azure.storage.blob import BlobServiceClient
from config import settings

logger = logging.getLogger(__name__)

class BlobManager:
    def __init__(self):
        self.cache_dir = os.path.join(settings.CACHE_DIR, "blobs")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.index_path = os.path.join(settings.CACHE_DIR, "blob_index.json")
        self._service: Optional[BlobServiceClient] = None
        self._container = None
        self._index = self._load_index()

    def _load_index(self) -> Dict[str, Any]:
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not read blob index: {e}")
        return {"blobs": {}}

    def _save_index(self) -> None:
        try:
            with open(self.index_path, "w", encoding="utf-8") as f:
                json.dump(self._index, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not write blob index: {e}")

    def _index_entry(self, name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        inbound = data.get("inbound_data") or {}
        return {
            "name": name,
            "correlation_id": data.get("correlation_id") or name.replace(".json", ""),
            "article_id": data.get("article_id"),
            "headline": inbound.get("headline") or data.get("headline") or "Untitled",
            "config_id": data.get("config_id"),
            "config_name": data.get("config_name"),
            "tracker_id": data.get("tracker_id"),
            "processed_at": data.get("processed_at"),
            "media_type": inbound.get("media_type") or data.get("media_type") or "article",
            "cached": True,
        }

    def _get_container(self):
        if self._container is None:
            if settings.AZURE_STORAGE_KEY:
                self._service = BlobServiceClient(
                    account_url=settings.AUDIT_BLOB_ACCOUNT_URL,
                    credential=settings.AZURE_STORAGE_KEY
                )
            else:
                from azure.identity import DefaultAzureCredential
                self._service = BlobServiceClient(
                    account_url=settings.AUDIT_BLOB_ACCOUNT_URL,
                    credential=DefaultAzureCredential()
                )
            self._container = self._service.get_container_client(settings.AUDIT_BLOB_CONTAINER)
        return self._container

    def refresh_index(self, max_blobs: int = 150) -> Dict[str, Any]:
        """Rebuild the local metadata index from cached JSON files."""
        entries: Dict[str, Any] = {}
        for fname in os.listdir(self.cache_dir):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(self.cache_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    entries[fname] = self._index_entry(fname, json.load(f))
            except Exception as e:
                logger.warning(f"Error indexing cached blob {fname}: {e}")
        self._index = {"blobs": entries}
        self._save_index()
        return self._index

    def sync_remote(self, max_blobs: int = 100) -> int:
        """Download a bounded page of remote audit blobs and update the local index.

        Blob names do not contain config IDs and the archive has no query index, so
        filtering by config requires downloading JSON. Refresh intentionally bounds
        this operation to keep the UI responsive; opening an uncached row downloads
        that exact blob on demand.
        """
        container = self._get_container()
        downloaded = 0
        seen = set(self._index.get("blobs", {}))
        for item in container.list_blobs(results_per_page=max_blobs):
            if downloaded >= max_blobs:
                break
            if (
                not item.name.endswith(".json")
                or item.name.startswith("validation-failure/")
                or item.name.startswith("delivery-audit/")
                or item.name in seen
            ):
                continue
            try:
                raw = container.get_blob_client(item.name).download_blob().readall()
                data = json.loads(raw.decode("utf-8"))
                cache_path = os.path.join(self.cache_dir, item.name)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, default=str)
                self._index.setdefault("blobs", {})[item.name] = self._index_entry(item.name, data)
                seen.add(item.name)
                downloaded += 1
            except Exception as e:
                logger.warning(f"Could not cache remote blob {item.name}: {e}")
        self._save_index()
        return downloaded

    def list_blobs(
        self,
        limit: int = 50,
        prefix: str = "",
        config_id: Optional[str] = None,
        include_azure: bool = False,
    ) -> List[Dict[str, Any]]:
        """List audit blobs from the local index, optionally filtered by config.

        The local index is the primary source so the picker is instant and every row
        carries its true `config_id`. `include_azure` additionally appends uncached
        remote names (metadata-poor).
        """
        blobs = self._index.get("blobs", {})
        if not blobs:
            self.refresh_index()
            blobs = self._index.get("blobs", {})

        results = list(blobs.values())
        if prefix:
            results = [b for b in results if b["name"].startswith(prefix)]
        if config_id:
            results = [b for b in results if b.get("config_id") == config_id]

        results.sort(key=lambda b: b.get("processed_at") or "", reverse=True)
        results = results[:limit]

        if include_azure:
            results = self._append_remote_names(results, limit, prefix)

        return results

    def _append_remote_names(
        self, results: List[Dict[str, Any]], limit: int, prefix: str
    ) -> List[Dict[str, Any]]:
        """Append uncached remote blob names to a listing (best-effort)."""
        seen = {b["name"] for b in results}
        try:
            container = self._get_container()
            count = 0
            for b in container.list_blobs(name_starts_with=prefix, results_per_page=100):
                if count >= limit:
                    break
                if (
                    not b.name.endswith(".json")
                    or b.name.startswith("validation-failure/")
                    or b.name.startswith("delivery-audit/")
                    or b.name in seen
                ):
                    continue
                corr_id = b.name.split("_")[0] if "_" in b.name else b.name.replace(".json", "")
                results.append(
                    {
                        "name": b.name,
                        "correlation_id": corr_id,
                        "size_bytes": b.size,
                        "last_modified": str(b.last_modified) if b.last_modified else None,
                        "cached": False,
                        "headline": f"Uncached blob: {corr_id[:12]}...",
                        "config_id": None,
                    }
                )
                count += 1
        except Exception as e:
            logger.warning(f"Could not list remote blobs from Azure container: {e}")
        return results

    def get_blob(self, name_or_correlation_id: str) -> Dict[str, Any]:
        """Gets a blob payload by exact blob name or correlation ID, checking disk cache first."""
        # Clean up identifier
        blob_name = name_or_correlation_id
        if not blob_name.endswith(".json"):
            blob_name = f"{name_or_correlation_id}{settings.AUDIT_BLOB_SUFFIX}.json"

        cache_path = os.path.join(self.cache_dir, blob_name)
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # Download from Azure
        container = self._get_container()
        blob_client = container.get_blob_client(blob_name)
        raw_bytes = blob_client.download_blob().readall()
        data = json.loads(raw_bytes.decode("utf-8"))

        # Save to local cache and index
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        self._index.setdefault("blobs", {})[blob_name] = self._index_entry(blob_name, data)
        self._save_index()

        return data

    def cache_blobs_batch(self, blob_names: List[str]) -> List[str]:
        """Pre-downloads and caches a batch of blobs."""
        cached = []
        for name in blob_names:
            try:
                self.get_blob(name)
                cached.append(name)
            except Exception as e:
                logger.error(f"Failed to cache blob {name}: {e}")
        return cached

blob_manager = BlobManager()
