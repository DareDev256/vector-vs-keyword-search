from __future__ import annotations

import logging
from typing import Optional


_CONFIGURED = False


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        _CONFIGURED = True
    return logging.getLogger(name)

