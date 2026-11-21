"""Run the whole pipeline phase by phase and record what each phase read, wrote and reported.

Each run lives in ``runs/<run_id>/``:

* ``workspace/``: a private copy of every ``datasets/...`` path, so a run never alters the shared copies or the
  authors' originals. Two folders stay shared because they are pure inputs or download caches: the raw Telegram
  pickles (read-only, 845 MB) and the Binance archive caches (``datasets/binance``, ``datasets/verification``).
* ``logs/<nn>-<stage>.log``: everything a phase logged or printed.
* ``reports/``: CSV/MD outputs of the stages, ``run.json`` (machine-readable record) and the PDF report.

Modes:

* ``literal`` (default): the configuration exactly as shipped; every documented error happens.
* ``bridged``: additionally bridges the pure *path* mismatches (Q03, Q04, Q17, Q18, Q22, Q29) with config
  overrides, so later phases can be observed. Logic errors (Q14, Q23f, ...) are never bridged.
"""

from __future__ import annotations

import logging
import os
import platform
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from pnd.config import Config, find_project_root, load_config
from pnd.pipeline.registry import get_stage
from pnd.reporting.artifacts import fingerprint, probe
from pnd.reporting.capture import capture_phase
from pnd.reporting.models import PhaseResult, RunRecord
from pnd.reporting.phases import PhaseSpec, build_catalogue

logger = logging.getLogger(__name__)

SHARED_PREFIXES = ("datasets/telegram/raw", "datasets/binance", "datasets/verification")
MODES = ("literal", "bridged")


def _workspace_overrides(raw_paths: dict, workspace_rel: str, prefix: str = "paths") -> list[str]:
    """Redirect every ``datasets/...`` path into the run workspace, except the shared ones."""
    overrides: list[str] = []
    for name, value in raw_paths.items():
        dotted = f"{prefix}.{name}"
        if isinstance(value, dict):
            overrides += _workspace_overrides(value, workspace_rel, dotted)
        elif isinstance(value, str) and value.startswith("datasets/") and not value.startswith(SHARED_PREFIXES):
            overrides.append(f"{dotted}={workspace_rel}/{value}")
        elif isinstance(value, str) and value == "reports":
            overrides.append(f"{dotted}={workspace_rel.rsplit('/', 1)[0]}/reports")
    return overrides


def _bridge_overrides(ws: str) -> list[str]:
    """Path-only bridges (documented in QUIRKS.md). Values are workspace paths."""
    return [
        f"paths.message_features.load_dir={ws}/datasets/telegram/PreSelection",          # Q03
        f"paths.classifier.preselect_input={ws}/datasets/telegram/preselect_sample",      # Q04
        "paths.process_klines.walk_dir=datasets/binance/data",                           # Q17 (shared cache)
        f"paths.positive_samples.concat_dir={ws}/datasets/data/concat",                    # Q18
        f"paths.dataset.pos_input={ws}/datasets/featgen/pump_sample_raw.csv",             # Q22
        f"paths.model.train_input={ws}/datasets/featgen/train_sample.csv",                # Q29
        f"paths.model.test_input={ws}/datasets/featgen/test_sample.csv",                  # Q29
    ]


def _environment() -> dict[str, str]:
    import numpy
    import pandas
    import sklearn

    env = {"python": platform.python_version(), "platform": platform.platform(), "numpy": numpy.__version__,
           "pandas": pandas.__version__, "scikit-learn": sklearn.__version__}
    try:
        import tensorflow as tf

        env["tensorflow"] = tf.__version__
    except Exception:  # pragma: no cover
        env["tensorflow"] = "unavailable"
    return env


def _relativize(text: str, root: Path) -> str:
    """Show project paths relative to the project root (shorter, machine independent)."""
    for form in (str(root) + "\\", str(root) + "/", str(root).replace("\\", "/") + "/"):
        text = text.replace(form, "")
    return text


def _tail(text: str, n: int = 14) -> list[str]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return lines[-n:]


def _highlights(text: str, patterns: tuple[str, ...]) -> list[str]:
    import re

    found: list[str] = []
    for line in text.splitlines():
        if any(re.search(p, line) for p in patterns) and line not in found:
            found.append(line.split(": ", 1)[-1] if " INFO " in line else line)
    return found[:10]


class PipelineRunner:
    """Executes the phase catalogue for one run and returns a :class:`RunRecord`."""

    def __init__(self, mode: str = "literal", run_id: str | None = None, verify_limit: int | None = 100,
                 klines_limit: int | None = None, extra_overrides: list[str] | None = None,
                 only: list[str] | None = None, root: Path | None = None) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.root = root or find_project_root()
        self.run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{mode}"
        self.run_dir = self.root / "runs" / self.run_id
        self.verify_limit = verify_limit
        self.klines_limit = klines_limit
        self.extra_overrides = list(extra_overrides or [])
        self.only = set(only or [])

    # ------------------------------------------------------------------ config
    def build_config(self) -> tuple[Config, list[str]]:
        base = load_config(root=self.root)
        ws = f"runs/{self.run_id}/workspace"
        overrides = _workspace_overrides(base.raw["paths"], ws)
        if self.mode == "bridged":
            overrides += _bridge_overrides(ws)
        overrides += self.extra_overrides
        return load_config(overrides=overrides, root=self.root), overrides

    # ------------------------------------------------------------------ run
    def run(self) -> RunRecord:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        logging.getLogger("tensorflow").setLevel(logging.ERROR)  # deprecation chatter, not phase output
        cfg, overrides = self.build_config()
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "reports").mkdir(parents=True, exist_ok=True)
        started = time.time()
        record = RunRecord(
            run_id=self.run_id, mode=self.mode, started_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            finished_utc="", duration_s=0.0, seed=cfg.seed, project_root=str(self.root), run_dir=str(self.run_dir),
            environment=_environment(), overrides=overrides,
            limits={"verify_limit": self.verify_limit, "klines_limit": self.klines_limit})

        catalogue = build_catalogue(self.verify_limit, self.klines_limit)
        for number, spec in enumerate(catalogue):
            if self.only and spec.stage not in self.only and spec.stage != "migrate-data":
                continue
            print(f"[{number:02d}] {spec.stage:<22}", end="", flush=True)
            result = self.run_phase(number, spec, cfg)
            record.phases.append(result)
            flag = "" if result.expected else "  (UNEXPECTED)"
            print(f"{result.status:<8} {result.duration_s:7.1f}s{flag}")
            record.duration_s = time.time() - started
            record.save(self.run_dir / "reports" / "run.json")

        record.finished_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        record.duration_s = time.time() - started
        record.save(self.run_dir / "reports" / "run.json")
        return record

    def run_phase(self, number: int, spec: PhaseSpec, cfg: Config) -> PhaseResult:
        expectation = spec.expectations[self.mode]
        result = PhaseResult(number=number, stage=spec.stage, title=spec.title, original=spec.original,
                             purpose=spec.purpose, status="skipped", expected=True,
                             explanation=expectation.explanation, quirks=list(spec.quirks))
        result.inputs = [probe(a, "input", cfg) for a in spec.inputs]
        before = [fingerprint(a, cfg) for a in spec.outputs]

        if not spec.run:
            result.outputs = [probe(a, "output", cfg) for a in spec.outputs]
            for state in result.outputs:
                if state.exists:
                    state.note = "authors' shipped file; stands in for this skipped phase"
            result.expected = expectation.status == "skipped"
            return result

        log_file = self.run_dir / "logs" / f"{number:02d}-{spec.stage}.log"
        result.log_file = str(log_file.relative_to(self.run_dir)).replace("\\", "/")
        console_handlers = [h for h in logging.getLogger().handlers]
        saved_levels = [h.level for h in console_handlers]
        for h in console_handlers:
            h.setLevel(logging.WARNING)
        logging.getLogger().setLevel(logging.INFO)
        start = time.time()
        with capture_phase(log_file) as buffer:
            try:
                get_stage(spec.stage)(**spec.stage_kwargs).run(cfg)
                result.status = "ok"
            except Exception as exc:
                result.status = "failed"
                result.error_type = type(exc).__name__
                result.error_message = _relativize(str(exc), self.root)[:600]
                result.traceback_tail = _relativize(traceback.format_exc(), self.root).strip().splitlines()[-8:]
                traceback.print_exc()
        result.duration_s = time.time() - start
        for h, level in zip(console_handlers, saved_levels):
            h.setLevel(level)

        text = _relativize(buffer.getvalue(), self.root)
        result.log_tail = _tail(_relativize(log_file.read_text(encoding="utf-8"), self.root))
        result.highlights = _highlights(text, spec.highlight_patterns)
        result.outputs = [probe(a, "output", cfg) for a in spec.outputs]
        for state, a, fp in zip(result.outputs, spec.outputs, before):
            state.changed_in_phase = fingerprint(a, cfg) != fp
        if spec.extract is not None:
            try:
                result.tables = spec.extract(cfg)
            except Exception as exc:  # extraction must never break a run
                result.highlights.append(f"(result extraction failed: {exc!r})")

        result.expected = (result.status == expectation.status
                           and (not expectation.error_type or result.error_type == expectation.error_type))
        if not result.expected:
            result.explanation = (f"Documented behaviour: {expectation.explanation} Observed: {result.status}"
                                  + (f" ({result.error_type}: {result.error_message[:200]})" if result.error_type else "")
                                  + ".")
        return result
