"""
Data Caching Module for HAEM.

Provides efficient caching of fetched NWP data to reduce
network requests and improve performance.
"""

import hashlib
import json
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np

from haem.models.meteorological import ModelData, NWPModel

logger = logging.getLogger(__name__)


class DataCache:
    """
    File-based cache for NWP model data.

    Caches ModelData objects to disk with configurable TTL.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        ttl_hours: int = 24,
        max_size_gb: float = 10.0,
    ):
        """
        Initialize the data cache.

        Args:
            cache_dir: Directory for cache files
            ttl_hours: Time-to-live for cache entries
            max_size_gb: Maximum cache size in GB
        """
        self.cache_dir = cache_dir or Path.home() / ".haem" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_hours = ttl_hours
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)

        # Index file for tracking cache entries
        self.index_file = self.cache_dir / "cache_index.json"
        self._index = self._load_index()

    def _load_index(self) -> dict[str, dict[str, Any]]:
        """Load cache index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_index(self) -> None:
        """Save cache index to disk."""
        with open(self.index_file, "w") as f:
            json.dump(self._index, f, indent=2, default=str)

    def _generate_key(
        self,
        model: NWPModel,
        run_time: datetime,
        fields: Optional[list[str]] = None,
    ) -> str:
        """Generate a unique cache key."""
        fields_str = ",".join(sorted(fields or []))
        key_data = f"{model.value}_{run_time.isoformat()}_{fields_str}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]

    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{key}.pkl"

    def get(
        self,
        model: NWPModel,
        run_time: datetime,
        fields: Optional[list[str]] = None,
    ) -> Optional[ModelData]:
        """
        Retrieve cached model data.

        Args:
            model: NWP model
            run_time: Model run time
            fields: Fields that were requested

        Returns:
            ModelData if found and valid, None otherwise
        """
        key = self._generate_key(model, run_time, fields)

        if key not in self._index:
            return None

        entry = self._index[key]

        # Check TTL
        cached_time = datetime.fromisoformat(entry["cached_at"])
        if datetime.utcnow() - cached_time > timedelta(hours=self.ttl_hours):
            logger.debug(f"Cache entry expired for {key}")
            self.invalidate(model, run_time, fields)
            return None

        # Load from disk
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            logger.warning(f"Cache file missing for {key}")
            del self._index[key]
            self._save_index()
            return None

        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            logger.info(f"Cache hit for {model.value} run {run_time}")
            return data
        except (pickle.UnpicklingError, IOError) as e:
            logger.error(f"Failed to load cache entry {key}: {e}")
            return None

    def put(
        self,
        model_data: ModelData,
        fields: Optional[list[str]] = None,
    ) -> None:
        """
        Store model data in cache.

        Args:
            model_data: Model data to cache
            fields: Fields that were fetched
        """
        key = self._generate_key(
            model_data.run.model,
            model_data.run.init_time,
            fields,
        )

        cache_path = self._get_cache_path(key)

        # Serialize to disk
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Update index
            self._index[key] = {
                "model": model_data.run.model.value,
                "run_time": model_data.run.init_time.isoformat(),
                "fields": fields or [],
                "cached_at": datetime.utcnow().isoformat(),
                "size_bytes": cache_path.stat().st_size,
            }
            self._save_index()

            logger.info(f"Cached {model_data.run.model.value} run {model_data.run.init_time}")

            # Check cache size and cleanup if needed
            self._cleanup_if_needed()

        except IOError as e:
            logger.error(f"Failed to write cache entry: {e}")

    def invalidate(
        self,
        model: NWPModel,
        run_time: datetime,
        fields: Optional[list[str]] = None,
    ) -> None:
        """Remove a specific cache entry."""
        key = self._generate_key(model, run_time, fields)

        if key in self._index:
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                cache_path.unlink()
            del self._index[key]
            self._save_index()
            logger.info(f"Invalidated cache entry for {model.value} run {run_time}")

    def clear(self) -> None:
        """Clear all cache entries."""
        for key in list(self._index.keys()):
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                cache_path.unlink()

        self._index = {}
        self._save_index()
        logger.info("Cache cleared")

    def _cleanup_if_needed(self) -> None:
        """Remove old entries if cache size exceeds limit."""
        total_size = sum(
            entry.get("size_bytes", 0)
            for entry in self._index.values()
        )

        if total_size <= self.max_size_bytes:
            return

        logger.info(f"Cache size ({total_size / 1e9:.2f} GB) exceeds limit, cleaning up")

        # Sort by cached_at (oldest first)
        sorted_keys = sorted(
            self._index.keys(),
            key=lambda k: self._index[k].get("cached_at", ""),
        )

        # Remove oldest entries until under limit
        for key in sorted_keys:
            if total_size <= self.max_size_bytes * 0.8:  # Clean to 80%
                break

            entry = self._index[key]
            size = entry.get("size_bytes", 0)

            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                cache_path.unlink()

            del self._index[key]
            total_size -= size

        self._save_index()
        logger.info(f"Cache cleaned, new size: {total_size / 1e9:.2f} GB")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(
            entry.get("size_bytes", 0)
            for entry in self._index.values()
        )

        models = {}
        for entry in self._index.values():
            model = entry.get("model", "unknown")
            models[model] = models.get(model, 0) + 1

        return {
            "total_entries": len(self._index),
            "total_size_mb": total_size / 1e6,
            "entries_by_model": models,
            "cache_dir": str(self.cache_dir),
            "ttl_hours": self.ttl_hours,
        }
