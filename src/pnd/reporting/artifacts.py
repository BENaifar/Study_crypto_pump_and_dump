"""Artifact specifications and probes (what a phase reads / writes, and what is found there)."""

from __future__ import annotations

import json
import os
import pickle
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from pnd.config import Config
from pnd.reporting.models import ArtifactState

Locator = Callable[[Config], Path]

PREVIEW_ROWS = 3
PREVIEW_COLS = 6
PREVIEW_CELL_CHARS = 40
MAX_PICKLE_BYTES = 60 * 1024 * 1024


def key(dotted: str) -> Locator:
    """Locate an artifact through a ``paths.*`` config key."""
    def locate(cfg: Config) -> Path:
        node = cfg.paths
        for part in dotted.split("."):
            node = node[part]
        return Path(node)
    locate.__doc__ = f"paths.{dotted}"
    return locate


def child(dotted: str, name: str) -> Locator:
    """A file inside a directory given by a config key."""
    parent = key(dotted)
    return lambda cfg: parent(cfg) / name


def fixed(relative: str) -> Locator:
    """A path relative to the project root (used for the authors' original folders)."""
    return lambda cfg: cfg.root / relative


@dataclass(frozen=True)
class ArtifactSpec:
    """Describes one input or output of a phase."""

    name: str
    locate: Locator
    description: str
    kind: str = "file"            # file | tsv | csv | csv_noheader | text | json | pickle_list | dir
    pattern: str = "*"            # for kind == "dir"
    recursive: bool = False       # search sub-folders too (archive trees)
    sep: str = ","
    columns: tuple[str, ...] = field(default_factory=tuple)


def _iter(path: Path, spec: ArtifactSpec):
    return path.rglob(spec.pattern) if spec.recursive else path.glob(spec.pattern)


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def _count_lines(path: Path) -> int:
    count = 0
    last = b""
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            count += chunk.count(b"\n")
            last = chunk[-1:]
    return count + (1 if last and last != b"\n" else 0)


def _cell(value: object) -> str:
    text = str(value).replace("\n", " ").replace("\t", " ")
    return text if len(text) <= PREVIEW_CELL_CHARS else text[: PREVIEW_CELL_CHARS - 1] + "…"


def _preview_table(path: Path, spec: ArtifactSpec, state: ArtifactState) -> None:
    header_mode = None if spec.kind in ("csv_noheader",) or spec.columns else "infer"
    names = list(spec.columns) or None
    df = pd.read_csv(path, sep=spec.sep, nrows=PREVIEW_ROWS, header=None if header_mode is None else 0,
                     names=names, on_bad_lines="skip", engine="python", dtype=str)
    state.columns = [str(c) for c in df.columns]
    cols = list(df.columns)[:PREVIEW_COLS]
    state.preview_header = [str(c) for c in cols]
    state.preview_rows = [[_cell(v) for v in row] for row in df[cols].itertuples(index=False)]


def probe(spec: ArtifactSpec, role: str, cfg: Config) -> ArtifactState:
    """Inspect an artifact without modifying it."""
    path = spec.locate(cfg)
    state = ArtifactState(name=spec.name, role=role, path=_rel(path, cfg.root), description=spec.description,
                          kind=spec.kind)
    try:
        if spec.kind == "dir":
            if not path.is_dir():
                return state
            files = [p for p in _iter(path, spec) if p.is_file()]
            state.exists = True
            state.records = len(files)
            state.records_label = f"files matching {spec.pattern}"
            state.size_bytes = sum(p.stat().st_size for p in files)
            state.preview_header = ["first files"]
            state.preview_rows = [[_rel(p, path)] for p in sorted(files)[:PREVIEW_ROWS]]
            return state

        if not path.is_file():
            return state
        state.exists = True
        state.size_bytes = path.stat().st_size

        if spec.kind in ("tsv", "csv", "csv_noheader"):
            lines = _count_lines(path)
            has_header = spec.kind != "csv_noheader" and not spec.columns
            state.records = max(lines - (1 if has_header else 0), 0)
            state.records_label = "rows"
            if state.records:
                _preview_table(path, spec, state)
        elif spec.kind == "text":
            state.records = _count_lines(path)
            state.records_label = "lines"
            with open(path, encoding="utf-8", errors="replace") as f:
                state.preview_header = ["first lines"]
                lines = [next(f, "").rstrip("\n") for _ in range(PREVIEW_ROWS)]
                state.preview_rows = [[_cell(line)] for line in lines if line]
        elif spec.kind == "json":
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            state.records = len(data)
            state.records_label = "top-level keys"
            state.preview_header = ["first keys"]
            state.preview_rows = [[_cell(k)] for k in list(data)[:PREVIEW_ROWS]]
        elif spec.kind == "pickle_list":
            if state.size_bytes <= MAX_PICKLE_BYTES:
                with open(path, "rb") as f:
                    state.records = len(pickle.load(f))
                state.records_label = "items"
    except Exception as exc:  # probing must never break a run
        state.note = f"probe error: {exc!r}"
    return state


def fingerprint(spec: ArtifactSpec, cfg: Config) -> tuple:
    """(exists, size, mtime) summary used to decide whether a phase changed an artifact."""
    path = spec.locate(cfg)
    if spec.kind == "dir":
        if not path.is_dir():
            return (False,)
        stats = [p.stat() for p in _iter(path, spec) if p.is_file()]
        return (True, len(stats), sum(s.st_size for s in stats), max((s.st_mtime for s in stats), default=0))
    if not path.is_file():
        return (False,)
    s = os.stat(path)
    return (True, s.st_size, s.st_mtime)
