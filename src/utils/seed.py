"""Seed utilities for reproducibility."""
import hashlib
import random
from typing import Optional

import numpy as np

GLOBAL_SEED = 42


def set_global_seed(seed: int = 42) -> None:
    """Set global random seeds for reproducibility."""
    global GLOBAL_SEED
    GLOBAL_SEED = seed
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_problem_seed(problem_id: str, profile: str, turn_id: int) -> int:
    """Deterministic seed per (problem_id, learner_profile, turn_id)."""
    s = f"{problem_id}|{profile}|{turn_id}"
    h = int(hashlib.sha256(s.encode()).hexdigest()[:12], 16)
    return h % (2**31 - 1)
