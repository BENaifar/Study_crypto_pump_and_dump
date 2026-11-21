"""Small I/O helpers shared by all stages."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def ensure_parent(path: Path) -> Path:
    """D1: create the parent directory of an *output* path."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return Path(path)


def ensure_dir(path: Path) -> Path:
    """D1: create an *output* directory."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def load_pickle(path: Path) -> Any:
    """[orig: load_obj in every Telegram script] ``pickle.load`` of a binary file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj: Any, path: Path) -> None:
    """[orig: save_obj] ``pickle.dump`` with ``HIGHEST_PROTOCOL`` (output dir created, D1)."""
    ensure_parent(path)
    with open(path, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def read_text_first_line_csv(path: Path) -> list[str]:
    """[orig: keyword_filter.py:L43-L49] ``f.readlines()[0].split(",")`` (trailing newline kept on last token)."""
    with open(path, encoding="utf-8") as f:  # D13
        first = f.readlines()[0]
    return first.split(",")
