"""Bootstrap 95% CI for effect sizes."""
import numpy as np
from typing import Tuple


def bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 10000) -> Tuple[float, float]:
    """Bootstrap 95% CI for mean difference (a - b)."""
    diff = np.mean(a) - np.mean(b)
    na, nb = len(a), len(b)
    boot_diffs = []
    for _ in range(n_boot):
        sa = np.random.choice(a, size=na, replace=True)
        sb = np.random.choice(b, size=nb, replace=True)
        boot_diffs.append(np.mean(sa) - np.mean(sb))
    lo = np.percentile(boot_diffs, 2.5)
    hi = np.percentile(boot_diffs, 97.5)
    return float(lo), float(hi)
