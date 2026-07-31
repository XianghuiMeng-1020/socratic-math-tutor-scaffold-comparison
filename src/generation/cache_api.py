"""Deterministic API caching for reproducible generation."""
import json
import os
from typing import Optional


def _cache_dir(project_root: str) -> str:
    return os.path.join(project_root, "outputs", "cache")


def get_cached_response(project_root: str, key: str) -> Optional[str]:
    """Get cached API response by key."""
    path = os.path.join(_cache_dir(project_root), "api_cache.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        cache = json.load(f)
    return cache.get(key)


def cache_response(project_root: str, key: str, value: str) -> None:
    """Cache API response."""
    d = _cache_dir(project_root)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "api_cache.json")
    cache = {}
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            cache = json.load(f)
    cache[key] = value
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
