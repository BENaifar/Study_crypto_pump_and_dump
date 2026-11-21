"""D10: load the authors' session pickles, which reference ``__main__.Session``/``__main__.Message``."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

_LEGACY_CLASSES = {("__main__", "Session"), ("__main__", "Message")}


class LegacyUnpickler(pickle.Unpickler):
    """Map ``__main__.Session``/``__main__.Message`` to :mod:`pnd.data.telegram.models`."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in _LEGACY_CLASSES:
            from pnd.data.telegram import models

            return getattr(models, name)
        return super().find_class(module, name)


def load_legacy(path: Path) -> Any:
    """Load a pickle that may contain legacy ``__main__`` classes."""
    with open(path, "rb") as f:
        return LegacyUnpickler(f).load()
