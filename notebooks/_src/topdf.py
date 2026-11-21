"""Export an executed notebook to PDF: nbconvert HTML (MathJax) printed by headless Chromium (Playwright).

    uv run python notebooks/_src/topdf.py NOTEBOOK.ipynb OUT_DIR [--no-input]

nbconvert's own ``webpdf`` exporter fails on Windows because Jupyter installs the selector event loop policy, which
cannot start subprocesses; this script runs in a clean process with the default Proactor loop.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter

CSS = """
<style>
  body { font-family: "Segoe UI", Helvetica, Arial, sans-serif; font-size: 10.5pt; }
  .jp-RenderedHTMLCommon table { font-size: 8pt; }
  .jp-RenderedHTMLCommon td, .jp-RenderedHTMLCommon th { padding: 2px 5px; }
  .jp-OutputArea-output pre { font-size: 7.5pt; white-space: pre-wrap; }
  .jp-RenderedImage img { max-width: 100%; height: auto; }
  .jp-Cell { break-inside: auto; }
  .jp-OutputArea-output { overflow-x: hidden; }
  div.jp-InputArea-editor { font-size: 7.5pt; }
  .jp-InputArea-editor pre, .highlight pre, .jp-CodeMirrorEditor pre { white-space: pre-wrap; word-break: break-word; }
  @page { size: A4; margin: 14mm 12mm 14mm 12mm; }
</style>
"""


def to_html(nb_path: Path, no_input: bool) -> str:
    nb = nbformat.read(nb_path, as_version=4)
    exp = HTMLExporter(template_name="lab")
    exp.exclude_input = no_input
    exp.exclude_input_prompt = True
    exp.exclude_output_prompt = True
    html, _ = exp.from_notebook_node(nb)
    return html.replace("</head>", CSS + "</head>", 1)


def render(html: str, out_pdf: Path) -> None:
    from playwright.sync_api import sync_playwright

    with tempfile.TemporaryDirectory() as tmp:
        page_path = Path(tmp) / "page.html"
        page_path.write_text(html, encoding="utf-8")
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(page_path.as_uri(), wait_until="networkidle", timeout=180_000)
            try:  # wait until MathJax is loaded, then until every formula is typeset
                page.wait_for_function("() => window.MathJax && (window.MathJax.Hub || window.MathJax.typesetPromise)",
                                       timeout=60_000)
                page.evaluate("() => window.MathJax.Hub ? new Promise(r => window.MathJax.Hub.Queue(r))"
                              " : window.MathJax.typesetPromise()")
            except Exception:
                pass
            page.wait_for_timeout(1500)
            page.emulate_media(media="print")
            page.pdf(path=str(out_pdf), format="A4", print_background=True,
                     margin={"top": "14mm", "bottom": "14mm", "left": "12mm", "right": "12mm"})
            browser.close()


def main(argv: list[str]) -> int:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    nb_path, out_dir = Path(argv[0]), Path(argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (nb_path.stem + ".pdf")
    render(to_html(nb_path, "--no-input" in argv), out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
