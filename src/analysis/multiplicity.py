"""Multiplicity correction audit."""
from typing import List


def holm_bonferroni(pvalues: List[float], alpha: float = 0.05) -> List[float]:
    """Return adjusted p-values."""
    n = len(pvalues)
    order = sorted(range(n), key=lambda i: pvalues[i])
    adjusted = [0.0] * n
    for i, idx in enumerate(order):
        adjusted[idx] = min(1.0, pvalues[idx] * (n - i))
    return adjusted
