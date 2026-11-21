"""Build, execute and export every brief notebook, then assemble the complete report.

    uv run python notebooks/run_all.py                 # everything (QUICK sizes unless PND_QUICK=0)
    uv run python notebooks/run_all.py I2 I3           # only notebooks whose name starts with these prefixes
    uv run python notebooks/run_all.py --no-pdf        # skip the PDF export
    uv run python notebooks/run_all.py --report-only   # only rebuild and export the complete report

Each step notebook is executed in a fresh kernel (so it is independent of the others); outputs are written back
into the .ipynb, and PDFs go to notebooks/pdf/ (step PDFs keep the code, the complete report hides it).
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import re

import nbformat
from nbclient import NotebookClient

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ANSI = re.compile(chr(27) + r"\[[0-9;]*m")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "_src"))

import build  # noqa: E402

STEPS = ["I0_data_acquisition", "I1_literature_synthesis", "I2_descriptive_statistics", "I3_event_study",
         "I4_intra_channel_homogeneity", "I5_coordination_network", "IIA1_arma_garch", "IIA2_cusum_isolation_forest",
         "IIB_machine_learning", "IIC_deep_learning", "IID_bonus_btc_forecast"]
PDF_DIR = HERE / "pdf"


def execute(name: str) -> float:
    path = build.build(name)
    nb = nbformat.read(path, as_version=4)
    start = time.time()
    client = NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(HERE)}})
    try:
        client.execute()
    finally:
        nbformat.write(nb, path)
    return time.time() - start


def export_pdf(name: str, hide_code: bool = False) -> Path:
    PDF_DIR.mkdir(exist_ok=True)
    cmd = [sys.executable, str(HERE / "_src" / "topdf.py"), str(HERE / f"{name}.ipynb"), str(PDF_DIR)]
    if hide_code:
        cmd.append("--no-input")
    subprocess.run(cmd, check=True, capture_output=True)
    return PDF_DIR / f"{name}.pdf"


def main(argv: list[str]) -> int:
    pdf = "--no-pdf" not in argv
    report_only = "--report-only" in argv
    prefixes = [a for a in argv if not a.startswith("--")]
    steps = [s for s in STEPS if not prefixes or any(s.startswith(p) for p in prefixes)]
    failed = []
    if not report_only:
        for name in steps:
            print(f"[run] {name} ...", flush=True)
            try:
                dt = execute(name)
                print(f"      done in {dt / 60:.1f} min", flush=True)
            except Exception as exc:  # keep going so one failure does not hide the others
                failed.append(name)
                msg = re.sub(ANSI, "", str(exc)).strip().splitlines()
                print(f"      FAILED: {type(exc).__name__}: {msg[-1] if msg else ''}"[:600], flush=True)
                continue
            if pdf:
                print(f"      pdf -> {export_pdf(name)}", flush=True)
    if not prefixes or report_only:
        import build_complete_report

        out = build_complete_report.build()
        print(f"[report] {out}", flush=True)
        if pdf:
            print(f"      pdf -> {export_pdf(out.stem, hide_code=True)}", flush=True)
    if failed:
        print("FAILED notebooks:", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
