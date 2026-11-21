"""D5: one global seed for python / numpy (TensorFlow is seeded after graph reset in ``pnd.modeling``)."""

from __future__ import annotations

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_global_seed(seed: int | None) -> None:
    """Seed python and numpy. ``None`` keeps the original unseeded behaviour."""
    if seed is None:
        logger.info("seed=None (original unseeded behaviour)")
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    # drives Q29's np.random.permutation unchanged
    np.random.seed(seed)
    logger.info("seed=%s", seed)
