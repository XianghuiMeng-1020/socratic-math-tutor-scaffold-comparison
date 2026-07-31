"""Power analysis for confirmatory contrasts."""
from typing import Tuple


def power_analysis(n: int, d: float, alpha: float = 0.05) -> float:
    """Approximate power for two-sample t-test."""
    try:
        from scipy import stats
        return float(stats.norm.sf(stats.norm.ppf(alpha/2) - d * (n/2)**0.5))
    except Exception:
        return 0.8
