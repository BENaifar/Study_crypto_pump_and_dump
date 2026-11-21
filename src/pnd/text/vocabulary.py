"""Symbol vocabularies. [orig: keyword_filter.py:L43-L50, message_fg.py:L47-L54]"""

from __future__ import annotations

from pathlib import Path

from pnd.common.io import read_text_first_line_csv


def load_symbol_list(path: Path) -> list[str]:
    """Q01: first line split on ',' (crypto symbols or exchange names)."""
    return read_text_first_line_csv(path)
