"""Render a :class:`RunRecord` as a PDF report with ReportLab.

One section per phase: what it does, every input and output (with what was actually found on disk before and
after the phase), the outcome compared with the documented behaviour of the original code, result tables,
previews and a log excerpt.
"""

from __future__ import annotations

import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from pnd.reporting.models import ArtifactState, PhaseResult, RunRecord, TableBlock

PAGE_W, PAGE_H = A4
MARGIN = 16 * mm
CONTENT_W = PAGE_W - 2 * MARGIN

INK = colors.HexColor("#1f2933")
MUTED = colors.HexColor("#616e7c")
RULE = colors.HexColor("#d9dee4")
HEAD_BG = colors.HexColor("#eef1f4")
STATUS_COLORS = {"ok": colors.HexColor("#1b7f4c"), "failed": colors.HexColor("#b4372f"),
                 "skipped": colors.HexColor("#7b8794")}

_REPLACEMENTS = {"≥": ">=", "≤": "<=", "→": "->", "×": "x", "​": "", "✓": "v", "✗": "x"}


def safe(text: object) -> str:
    """Escape for ReportLab mini-markup and keep to characters the built-in Helvetica can draw."""
    s = str(text)
    for a, b in _REPLACEMENTS.items():
        s = s.replace(a, b)
    s = s.encode("cp1252", errors="replace").decode("cp1252")
    return escape(s)


def human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{n} B"


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=base["BodyText"], fontName="Helvetica", fontSize=9, leading=12.5,
                          textColor=INK, spaceAfter=4)
    return {
        "title": ParagraphStyle("title", parent=body, fontName="Helvetica-Bold", fontSize=20, leading=25,
                                spaceAfter=6),
        "subtitle": ParagraphStyle("subtitle", parent=body, fontSize=11, leading=15, textColor=MUTED, spaceAfter=14),
        "h1": ParagraphStyle("h1", parent=body, fontName="Helvetica-Bold", fontSize=14, leading=18, spaceBefore=4,
                             spaceAfter=4),
        "h2": ParagraphStyle("h2", parent=body, fontName="Helvetica-Bold", fontSize=10.5, leading=14, spaceBefore=8,
                             spaceAfter=3),
        "body": body,
        "meta": ParagraphStyle("meta", parent=body, fontSize=8.5, textColor=MUTED),
        "cell": ParagraphStyle("cell", parent=body, fontSize=7.6, leading=9.6, spaceAfter=0),
        "cellb": ParagraphStyle("cellb", parent=body, fontName="Helvetica-Bold", fontSize=7.6, leading=9.6,
                                spaceAfter=0),
        "mono": ParagraphStyle("mono", parent=body, fontName="Courier", fontSize=6.6, leading=8.2,
                               textColor=colors.HexColor("#323f4b")),
        "center": ParagraphStyle("center", parent=body, alignment=TA_CENTER),
    }


S = _styles()


def _p(text: str, style: str = "cell") -> Paragraph:
    return Paragraph(text, S[style])


def _table(header: list[str], rows: list[list], widths: list[float], header_bg=HEAD_BG) -> Table:
    data = [[_p(safe(h), "cellb") for h in header]]
    for row in rows:
        data.append([c if isinstance(c, Paragraph) else _p(safe(c)) for c in row])
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, RULE),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def _status_badge(status: str, expected: bool) -> str:
    color = STATUS_COLORS.get(status, MUTED).hexval()[2:]
    label = status.upper()
    tail = "as documented" if expected else "NOT as documented"
    tail_color = "1b7f4c" if expected else "b4372f"
    return f'<font color="#{color}"><b>{label}</b></font> <font color="#{tail_color}">({tail})</font>'


def _artifact_state(a: ArtifactState) -> str:
    if not a.exists:
        return '<font color="#b4372f"><b>missing</b></font>'
    parts = ["present"]
    if a.records is not None:
        parts.append(f"{a.records:,} {safe(a.records_label)}")
    if a.size_bytes:
        parts.append(human_size(a.size_bytes))
    text = ", ".join(parts)
    if a.role == "output" and a.changed_in_phase:
        text += '<br/><font color="#1b7f4c"><b>written in this phase</b></font>'
    elif a.role == "output" and not a.note:
        text += '<br/><font color="#616e7c">not changed by this phase</font>'
    if a.note:
        text += f"<br/>{safe(a.note)}"
    return text


def _artifact_table(items: list[ArtifactState], role: str) -> Table:
    rows = []
    for a in items:
        rows.append([_p(f"<b>{safe(a.name)}</b>"), _p(safe(a.description)), _p(safe(a.path)),
                     _p(_artifact_state(a))])
    header = [role, "What it is", "Location", "State before phase" if role == "Input" else "State after phase"]
    return _table(header, rows, [0.19 * CONTENT_W, 0.37 * CONTENT_W, 0.25 * CONTENT_W, 0.19 * CONTENT_W])


def _previews(items: list[ArtifactState], seen: set) -> list:
    flow = []
    for a in items:
        if not a.exists or not a.preview_rows or a.path.endswith(".md"):
            continue
        fingerprint = (a.path, a.size_bytes)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        n = max(len(a.preview_header), 1)
        widths = [CONTENT_W / n] * n
        title = f"Preview: {safe(a.name)}"
        if a.columns and len(a.columns) > len(a.preview_header):
            title += f" (first {len(a.preview_header)} of {len(a.columns)} columns)"
        flow.append(KeepTogether([_p(title, "meta"), _table(a.preview_header, a.preview_rows, widths),
                                  Spacer(1, 4)]))
    return flow


def _result_block(t: TableBlock) -> list:
    n = len(t.header)
    first = 0.5 if n == 2 else 0.28
    widths = [CONTENT_W * first] + [CONTENT_W * (1 - first) / (n - 1)] * (n - 1)
    flow = [_p(f"<b>{safe(t.title)}</b>", "body"), _table(t.header, t.rows, widths)]
    if t.note:
        flow.append(_p(safe(t.note), "meta"))
    flow.append(Spacer(1, 4))
    return [KeepTogether(flow)]


def load_quirk_glossary(quirks_md: Path) -> dict[str, str]:
    """``Qnn``/``Dnn`` -> its text in QUIRKS.md (table rows, and full bullet text for the findings)."""
    glossary: dict[str, str] = {}
    if not quirks_md.is_file():
        return glossary
    current: str | None = None
    for line in quirks_md.read_text(encoding="utf-8").splitlines():
        clean = re.sub(r"\*\*|`", "", line).strip()
        m = re.match(r"\|\s*([QD]\d+)\s*\|\s*(.+?)\s*\|", line)
        if m:
            glossary[m.group(1)] = re.sub(r"\*\*|`", "", m.group(2))
            current = None
            continue
        m2 = re.match(r"-\s*\*\*(Q\d+)\s+(.+?)\*\*(.*)", line)
        if m2:
            current = m2.group(1)
            glossary[current] = (m2.group(2) + " " + re.sub(r"\*\*|`", "", m2.group(3))).strip()
            continue
        if current and line.startswith("  ") and clean:
            glossary[current] += " " + clean.lstrip("- ").strip()
        elif current and (not line.startswith("  ") or not clean):
            current = None if line.startswith("- ") or line.startswith("#") else current
    return glossary


class PdfReportBuilder:
    """Builds the PDF for one run."""

    def __init__(self, record: RunRecord, glossary: dict[str, str]) -> None:
        self.r = record
        self.glossary = glossary
        self._seen_previews: set = set()

    # -------------------------------------------------------------- page decoration
    def _decorate(self, canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN, PAGE_H - 10 * mm, f"P&D target-coin pipeline: run {self.r.run_id} ({self.r.mode})")
        canvas.drawRightString(PAGE_W - MARGIN, 10 * mm, f"page {doc.page}")
        canvas.setStrokeColor(RULE)
        canvas.line(MARGIN, PAGE_H - 11.5 * mm, PAGE_W - MARGIN, PAGE_H - 11.5 * mm)
        canvas.restoreState()

    # -------------------------------------------------------------- sections
    def _cover(self) -> list:
        r = self.r
        ran = [p for p in r.phases if p.status != "skipped"]
        counts = {s: sum(p.status == s for p in r.phases) for s in ("ok", "failed", "skipped")}
        unexpected = [p for p in r.phases if not p.expected]
        flow = [Spacer(1, 30 * mm),
                _p("Pipeline run report", "title"),
                _p("Sequence-based Target Coin Prediction for Cryptocurrency Pump-and-Dump (Hu et al., SIGMOD 2023): "
                   "bug-for-bug modular re-implementation", "subtitle")]
        rows = [["Run id", r.run_id], ["Mode", r.mode], ["Started", r.started_utc], ["Finished", r.finished_utc or "(in progress)"],
                ["Duration", f"{r.duration_s / 60:.1f} min"], ["Seed", str(r.seed)],
                ["Phases", f"{len(r.phases)} ({counts['ok']} ok, {counts['failed']} failed, {counts['skipped']} skipped; "
                           f"{len(ran)} executed)"],
                ["Outcomes not matching the documented behaviour", str(len(unexpected)) + (
                    ": " + ", ".join(p.stage for p in unexpected) if unexpected else "")],
                ["Limits", ", ".join(f"{k}={v}" for k, v in r.limits.items())]]
        rows += [[k, v] for k, v in r.environment.items()]
        flow.append(_table(["Item", "Value"], rows, [0.32 * CONTENT_W, 0.68 * CONTENT_W]))
        flow.append(PageBreak())
        return flow

    def _how_to_read(self) -> list:
        mode_text = {
            "literal": "<b>literal</b>: the configuration exactly as shipped. Every documented error of the original "
                       "code happens, so several phases fail or produce nothing. That is the expected result.",
            "bridged": "<b>bridged</b>: as literal, plus config overrides that bridge the pure path mismatches "
                       "(Q03, Q04, Q17, Q18, Q22, Q29) so later phases can be observed. Logic errors such as Q14 "
                       "(pump month never downloaded) or Q23f (negatives never saved) are never bridged.",
        }[self.r.mode]
        flow = [_p("How to read this report", "h1"),
                _p("Each phase below corresponds to one script of the authors' repository and one "
                   "<font face='Courier'>pnd</font> command. For every phase the report shows what it does, each input "
                   "(and what was on disk before the phase ran), each output (and what was on disk afterwards, "
                   "including whether the phase wrote it), the outcome, result tables, data previews and the end of "
                   "the phase log.", "body"),
                _p("<b>Status</b> is what happened (OK, FAILED, SKIPPED). <b>As documented</b> means the outcome is "
                   "exactly what the original code does, including its errors, as catalogued in QUIRKS.md. A failure "
                   "can therefore be the correct, faithful result. Interactive labeling steps and the opt-in Telegram "
                   "and CoinGecko fetchers are skipped; the authors' shipped outputs stand in for them.", "body"),
                _p("Run mode: " + mode_text, "body"),
                _p("Each run works in its own folder (<font face='Courier'>runs/&lt;run id&gt;/workspace</font>) that "
                   "starts as a copy of the authors' shipped files, so runs never alter the originals. The raw Telegram "
                   "pickles and the Binance download caches are shared, read-only or append-only.", "body"),
                Spacer(1, 6), _p("Overview", "h1")]
        rows = []
        for p in self.r.phases:
            key = p.highlights[0] if p.highlights else (p.tables[0].title if p.tables else "")
            if p.status == "failed":
                key = f"{p.error_type}"
            rows.append([str(p.number), _p(f"<b>{safe(p.title)}</b><br/><font color='#616e7c'>pnd {safe(p.stage)}</font>"),
                         _p(safe(p.original)), _p(_status_badge(p.status, p.expected)),
                         f"{p.duration_s:.1f} s" if p.status != "skipped" else "-", _p(safe(key))])
        flow.append(_table(["#", "Phase", "Original script", "Status", "Time", "Key result"], rows,
                           [0.05 * CONTENT_W, 0.24 * CONTENT_W, 0.24 * CONTENT_W, 0.17 * CONTENT_W,
                            0.08 * CONTENT_W, 0.22 * CONTENT_W]))
        flow.append(PageBreak())
        return flow

    def _phase(self, p: PhaseResult) -> list:
        flow: list = [_p(f"Phase {p.number:02d}: {safe(p.title)}", "h1"),
                      _p(f"Original: <font face='Courier'>{safe(p.original)}</font> &nbsp;|&nbsp; Command: "
                         f"<font face='Courier'>pnd {safe(p.stage)}</font> &nbsp;|&nbsp; "
                         + (f"Duration: {p.duration_s:.1f} s" if p.status != "skipped" else "not executed"), "meta"),
                      _p("<b>What this phase does.</b> " + safe(p.purpose), "body")]

        flow.append(_p("Inputs", "h2"))
        flow.append(_artifact_table(p.inputs, "Input") if p.inputs else _p("None.", "body"))
        flow.append(_p("Outputs", "h2"))
        flow.append(_artifact_table(p.outputs, "Output") if p.outputs else
                    _p("No file output (results are printed / logged only).", "body"))

        flow.append(_p("Outcome", "h2"))
        outcome = [_p(_status_badge(p.status, p.expected), "body"), _p(safe(p.explanation), "body")]
        if p.error_type:
            outcome.append(_p(f"<b>Error:</b> {safe(p.error_type)}: {safe(p.error_message)}", "body"))
        if p.quirks:
            lines = [f"<b>{q}</b>: {safe(self.glossary.get(q, ''))}" for q in p.quirks]
            outcome.append(_p("<b>Documented errors involved</b> (see QUIRKS.md):<br/>" + "<br/>".join(lines), "cell"))
        box = Table([[outcome]], colWidths=[CONTENT_W])
        box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, RULE), ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fafbfc")),
                                 ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                                 ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
        flow.append(box)

        if p.tables or p.highlights:
            flow.append(_p("Results", "h2"))
            for t in p.tables:
                flow += _result_block(t)
            if p.highlights:
                flow.append(_p("<b>Key log lines</b>", "body"))
                for h in p.highlights:
                    flow.append(_p("&bull; " + safe(h[:300]), "cell"))

        previews = _previews(p.inputs + p.outputs, self._seen_previews)
        if previews:
            flow.append(_p("Data previews", "h2"))
            flow += previews

        if p.log_tail or p.traceback_tail:
            flow.append(_p(f"Log excerpt <font color='#616e7c' size='7.5'>(full log: runs/{safe(self.r.run_id)}/"
                           f"{safe(p.log_file)})</font>", "h2"))
            lines = p.traceback_tail if p.status == "failed" and p.traceback_tail else p.log_tail
            text = "\n".join(_wrap(safe_plain(ln), 125) for ln in lines)
            flow.append(Preformatted(text, S["mono"]))
        flow.append(PageBreak())
        return flow

    def _findings(self) -> list:
        flow = [_p("Findings about the authors' data", "h1"),
                _p("Discovered while reproducing the pipeline; each is pinned by a test.", "meta")]
        for q in ("Q37", "Q38"):
            if q in self.glossary:
                flow.append(_p(f"<b>{q}</b>: {safe(self.glossary[q])}", "body"))
        verify = next((p for p in self.r.phases if p.stage == "verify-binance" and p.tables), None)
        if verify:
            flow += _result_block(verify.tables[0])
        flow += [Spacer(1, 6), _p("Deviations from the original code", "h1"),
                 _p("The only changes allowed by the re-implementation; none changes a methodological result.", "meta")]
        rows = [[d, self.glossary[d]] for d in sorted((k for k in self.glossary if k.startswith("D")),
                                                     key=lambda k: int(k[1:]))]
        if rows:
            flow.append(_table(["ID", "Deviation"], rows, [0.08 * CONTENT_W, 0.92 * CONTENT_W]))
        if self.r.overrides:
            flow += [Spacer(1, 6), _p("Configuration overrides used by this run", "h2")]
            flow.append(Preformatted("\n".join(_wrap(safe_plain(o), 125) for o in self.r.overrides), S["mono"]))
        return flow

    def build(self, out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        doc = SimpleDocTemplate(str(out_path), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                                topMargin=MARGIN + 2 * mm, bottomMargin=MARGIN, title=f"Pipeline run {self.r.run_id}",
                                author="pnd pipeline runner")
        story = self._cover() + self._how_to_read()
        for p in self.r.phases:
            story += self._phase(p)
        story += self._findings()
        doc.build(story, onFirstPage=self._decorate, onLaterPages=self._decorate)
        return out_path


def safe_plain(text: str) -> str:
    """Like :func:`safe` but without XML escaping (for Preformatted blocks)."""
    s = str(text)
    for a, b in _REPLACEMENTS.items():
        s = s.replace(a, b)
    return s.encode("cp1252", errors="replace").decode("cp1252")


def _wrap(line: str, width: int) -> str:
    return "\n".join(line[i:i + width] for i in range(0, max(len(line), 1), width))


def build_pdf(run_dir: Path, project_root: Path) -> Path:
    """Render ``<run_dir>/reports/run.json`` to ``<run_dir>/reports/pipeline_report.pdf``."""
    record = RunRecord.load(run_dir / "reports" / "run.json")
    glossary = load_quirk_glossary(project_root / "QUIRKS.md")
    return PdfReportBuilder(record, glossary).build(run_dir / "reports" / "pipeline_report.pdf")
