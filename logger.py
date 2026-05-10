from __future__ import annotations

import sys
from loguru import logger


def configure_logging(level: str = "INFO"):
    logger.remove()
    logger.add(sys.stderr, level=level, enqueue=True, backtrace=False, diagnose=False)
    return logger


log = configure_logging()
