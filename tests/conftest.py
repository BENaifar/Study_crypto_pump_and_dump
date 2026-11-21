"""Shared fixtures."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

from pnd.config import find_project_root, load_config

REPO_ROOT = find_project_root(Path(__file__).parent)
ORIG_TELEGRAM = REPO_ROOT / "Data" / "Telegram"
ORIG_FEATGEN = REPO_ROOT / "TargetCoinPrediction" / "FeatGeneration"
ORIG_SEQMODEL = REPO_ROOT / "TargetCoinPrediction" / "SeqModel"
DATASETS = REPO_ROOT / "datasets"


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def make_cfg():
    """``make_cfg(*overrides)`` -> Config loaded from configs/default.yaml with overrides."""
    def _make(*overrides: str):
        return load_config(overrides=list(overrides), root=REPO_ROOT)
    return _make


@pytest.fixture
def cfg(make_cfg):
    return make_cfg()


def original_source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_original_function(path: Path, name: str, namespace: dict | None = None):
    """Extract one top-level function from an original script (without running its side effects)."""
    tree = ast.parse(original_source(path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            src = textwrap.dedent(ast.get_source_segment(original_source(path), node))
            ns = dict(namespace or {})
            exec(compile(src, str(path), "exec"), ns)
            return ns[name]
    raise LookupError(f"{name} not found in {path}")


def original_assignment(path: Path, name: str):
    """Evaluate a top-level literal assignment of an original script."""
    tree = ast.parse(original_source(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise LookupError(f"{name} not found in {path}")


requires_datasets = pytest.mark.skipif(not (DATASETS / "telegram" / "raw").is_dir(),
                                       reason="run `pnd migrate-data` first")
