"""Serializable records of a pipeline run (written to ``run.json`` and read by the PDF builder)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ArtifactState:
    """What was found at an artifact's location at one point in time."""

    name: str
    role: str                      # "input" | "output"
    path: str                      # relative to the project root when possible
    description: str
    exists: bool = False
    kind: str = "file"
    size_bytes: int = 0
    records: int | None = None     # rows / lines / files / keys, depending on the kind
    records_label: str = ""
    columns: list[str] = field(default_factory=list)
    preview_header: list[str] = field(default_factory=list)
    preview_rows: list[list[str]] = field(default_factory=list)
    changed_in_phase: bool = False
    note: str = ""


@dataclass
class TableBlock:
    """A small result table shown in a phase section."""

    title: str
    header: list[str]
    rows: list[list[str]]
    note: str = ""


@dataclass
class PhaseResult:
    """Outcome of one phase."""

    number: int
    stage: str
    title: str
    original: str
    purpose: str
    status: str                    # "ok" | "failed" | "skipped"
    expected: bool                 # outcome matches the documented behaviour of the original code
    explanation: str
    quirks: list[str]
    duration_s: float = 0.0
    error_type: str = ""
    error_message: str = ""
    traceback_tail: list[str] = field(default_factory=list)
    inputs: list[ArtifactState] = field(default_factory=list)
    outputs: list[ArtifactState] = field(default_factory=list)
    highlights: list[str] = field(default_factory=list)
    tables: list[TableBlock] = field(default_factory=list)
    log_tail: list[str] = field(default_factory=list)
    log_file: str = ""


@dataclass
class RunRecord:
    """A whole pipeline run."""

    run_id: str
    mode: str
    started_utc: str
    finished_utc: str
    duration_s: float
    seed: int | None
    project_root: str
    run_dir: str
    environment: dict[str, str]
    overrides: list[str]
    limits: dict[str, int | None]
    phases: list[PhaseResult] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> RunRecord:
        data = json.loads(path.read_text(encoding="utf-8"))
        phases = []
        for p in data.pop("phases"):
            p["inputs"] = [ArtifactState(**a) for a in p["inputs"]]
            p["outputs"] = [ArtifactState(**a) for a in p["outputs"]]
            p["tables"] = [TableBlock(**t) for t in p["tables"]]
            phases.append(PhaseResult(**p))
        return RunRecord(**data, phases=phases)
