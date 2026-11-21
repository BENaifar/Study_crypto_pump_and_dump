"""Build the .ipynb notebooks from their percent-format sources in notebooks/_src/.

Source format (jupytext "percent"): a line ``# %% [markdown]`` starts a Markdown cell whose lines carry a ``# ``
comment prefix (stripped on build), and a line ``# %%`` starts a code cell. Usage: ``uv run python notebooks/_src/build.py [name ...]``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat

SRC = Path(__file__).resolve().parent
OUT = SRC.parent


def parse(text: str) -> list:
    cells, kind, buf = [], None, []

    def flush():
        if kind is not None:
            body = "\n".join(buf).strip("\n")
            if body:
                cells.append(nbformat.v4.new_markdown_cell(body) if kind == "md" else nbformat.v4.new_code_cell(body))

    for line in text.splitlines():
        if line.startswith("# %% [markdown]"):
            flush()
            kind, buf = "md", []
        elif line.startswith("# %%"):
            flush()
            kind, buf = "code", []
        elif kind == "md":
            buf.append(line[2:] if line.startswith("# ") else line[1:] if line.startswith("#") else line)
        else:
            buf.append(line)
    flush()
    return cells


def build(name: str, keep_outputs: bool = False) -> Path:
    """Rebuild the notebook from its source. With ``keep_outputs``, code cells whose source is unchanged keep the
    outputs of the existing executed notebook (used to edit Markdown without re-running long notebooks)."""
    src = SRC / f"{name}.py"
    nb = nbformat.v4.new_notebook()
    nb.cells = parse(src.read_text(encoding="utf-8"))
    out_path = OUT / f"{name}.ipynb"
    if keep_outputs and out_path.is_file():
        old = [c for c in nbformat.read(out_path, as_version=4).cells if c.cell_type == "code"]
        new = [c for c in nb.cells if c.cell_type == "code"]
        if len(old) != len(new) or any(o.source != n.source for o, n in zip(old, new)):
            raise ValueError(f"{name}: code changed since the last execution; re-run the notebook instead")
        for o, n in zip(old, new):
            n.outputs, n.execution_count = o.outputs, o.execution_count
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    out = OUT / f"{name}.ipynb"
    nbformat.write(nb, out)
    return out


if __name__ == "__main__":
    keep = "--keep-outputs" in sys.argv
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or sorted(p.stem for p in SRC.glob("I*.py"))
    for n in names:
        print("built", build(n, keep_outputs=keep))
