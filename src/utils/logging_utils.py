"""
Small logging wrapper so every script prints in the same format and can
optionally tee output into a log file under ``experiments/``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def get_logger(name: str = "dr", log_file: str | Path | None = None,
               level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger. Safe to call repeatedly."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:  # already configured in this process
        return logger

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
    logger.addHandler(stream)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
        logger.addHandler(fh)

    return logger
