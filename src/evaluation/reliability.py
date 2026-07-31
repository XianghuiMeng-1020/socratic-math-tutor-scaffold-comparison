"""Reliability: Pearson r, Cohen's kappa, stability (MAD)."""
from typing import Dict, List
import numpy as np


def pearson_r(a: List[float], b: List[float]) -> float:
    """Pearson correlation."""
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def cohens_kappa(a: List[int], b: List[int]) -> float:
    """Cohen's kappa for agreement."""
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(a, b))


def mad(scores: List[float]) -> float:
    """Median absolute deviation."""
    if not scores:
        return 0.0
    m = np.median(scores)
    return float(np.median([abs(x - m) for x in scores]))
