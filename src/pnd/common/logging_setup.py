"""Logging configuration (D6: progress prints become ``logger.info``)."""

from __future__ import annotations

import logging
import sys


def configure(level: str = "INFO") -> None:
    """Basic stdout handler with a compact format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )
