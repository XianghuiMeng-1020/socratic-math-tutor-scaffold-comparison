"""Export all tables to outputs/tables."""
import os
from typing import List

from ..utils.io import ensure_dir


def export_all_tables(project_root: str) -> List[str]:
    """Ensure all expected tables exist. Returns list of paths."""
    tables_dir = os.path.join(project_root, "outputs", "tables")
    ensure_dir(tables_dir)
    return [os.path.join(tables_dir, f) for f in os.listdir(tables_dir) if f.endswith(".csv")]
