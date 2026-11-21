"""Channel list helpers. [orig: get_channel_post.py:L26-L40]"""

from __future__ import annotations

from pathlib import Path


def read_channel_lines(path: Path) -> list[str]:
    """``f.readlines()`` of a channel list file (one t.me URL per line)."""
    with open(path, encoding="utf-8") as f:  # D13
        return f.readlines()


def entity_from_line(line: str, start: int = 13) -> str:
    """Q36: ``channel[13:-1]`` strips ``https://t.me/`` and the trailing newline."""
    return line[start:-1]
