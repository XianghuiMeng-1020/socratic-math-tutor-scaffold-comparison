"""Logging utilities for the experiment pipeline."""
import logging
import os
from datetime import datetime
from typing import Optional


def get_logger(name: str, log_dir: Optional[str] = None) -> logging.Logger:
    """Get or create logger with optional file handler."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(
            os.path.join(log_dir, f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"),
            encoding="utf-8",
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def log_mapping_decision(logger: logging.Logger, decision: str, reason: str) -> None:
    """Log a dataset/mapping decision for audit trail."""
    logger.info(f"MAP_DECISION: {decision} | REASON: {reason}")
