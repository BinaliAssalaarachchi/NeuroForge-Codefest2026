import json
import hashlib
from pathlib import Path
from typing import Any, Optional
import config

class DiskCache:
    """
    Simple persistent JSON disk cache for caching LLM responses and embeddings.
    Prevents burning API rate limits / credits during repeated debugging runs.
    """
    def __init__(self, cache_file: Optional[Path] = None):
        if cache_file is None:
            cache_file = config.CACHE_DIR / "response_cache.json"
        self.cache_file = cache_file
        self.data = self._load()

    def _load(self) -> dict:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save(self):
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Cache Warning] Could not save cache: {e}")

    def _make_key(self, namespace: str, key_text: str) -> str:
        hashed = hashlib.sha256(key_text.encode("utf-8")).hexdigest()
        return f"{namespace}:{hashed}"

    def get(self, namespace: str, key_text: str) -> Optional[Any]:
        key = self._make_key(namespace, key_text)
        return self.data.get(key)

    def set(self, namespace: str, key_text: str, value: Any):
        key = self._make_key(namespace, key_text)
        self.data[key] = value
        self._save()

    def clear(self):
        self.data = {}
        self._save()

# Global default cache instance
default_cache = DiskCache()
