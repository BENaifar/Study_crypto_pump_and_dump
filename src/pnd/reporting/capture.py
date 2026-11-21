"""Capture everything a phase logs or prints into its own log file."""

from __future__ import annotations

import contextlib
import io
import logging
import sys
from collections.abc import Iterator
from pathlib import Path


class _Tee(io.TextIOBase):
    """Text stream that records everything written to it (the console stays quiet during a phase)."""

    def __init__(self, copy: io.StringIO) -> None:
        self._copy = copy

    def write(self, s: str) -> int:
        self._copy.write(s)
        return len(s)

    def flush(self) -> None:
        pass


@contextlib.contextmanager
def capture_phase(log_file: Path) -> Iterator[io.StringIO]:
    """Route logging records and stdout/stderr of the phase into ``log_file`` (and a buffer)."""
    buffer = io.StringIO()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    stdout, stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(buffer)
    sys.stderr = _Tee(buffer)
    try:
        yield buffer
    finally:
        sys.stdout, sys.stderr = stdout, stderr
        root.removeHandler(handler)
        log_file.write_text(_strip_progress(buffer.getvalue()), encoding="utf-8")


def _strip_progress(text: str) -> str:
    """Drop the carriage-return progress bars written by the ported Binance downloader (Q15)."""
    lines = []
    for line in text.splitlines():
        if "\r" in line:
            line = line.split("\r")[-1]
        if line.startswith("[") and set(line) <= set("[]#."):
            continue
        lines.append(line)
    return "\n".join(lines) + "\n"
