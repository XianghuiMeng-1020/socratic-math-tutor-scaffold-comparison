"""I/O utilities for the experiment pipeline."""
import os
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _resolve_path(s: str) -> str:
    """Resolve ${project_root} in paths. Uses PROJECT_ROOT env if set."""
    root = os.environ.get("PROJECT_ROOT", "e:/m/xian math")
    return s.replace("${project_root}", root)


def _resolve_dict(d: Any) -> Any:
    """Recursively resolve ${project_root} in dict values."""
    if isinstance(d, dict):
        return {k: _resolve_dict(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_resolve_dict(x) for x in d]
    if isinstance(d, str) and "${" in d:
        return _resolve_path(d)
    return d


def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    if d is None:
        return {}
    return _resolve_dict(d)


def save_json(path: str, obj: Any, indent: int = 2) -> None:
    """Save object to JSON file."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def load_json(path: str) -> Any:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str, encoding: str = "utf-8", errors: str = "replace") -> List[Dict[str, Any]]:
    """Load JSONL file."""
    out = []
    with open(path, "r", encoding=encoding, errors=errors) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    """Save list of dicts to JSONL."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def ensure_dir(path: str) -> None:
    """Ensure directory exists."""
    if path:
        Path(path).mkdir(parents=True, exist_ok=True)


def compute_hash(content: str) -> str:
    """Compute SHA256 hash of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compute_file_hash(path: str) -> Optional[str]:
    """Compute SHA256 hash of file contents."""
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()
