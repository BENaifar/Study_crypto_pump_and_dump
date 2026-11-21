import pickle
from datetime import datetime, timezone

from pnd.classification.features import message_feature_row
from pnd.pipeline.runner import SHARED_PREFIXES, _relativize, _workspace_overrides
from pnd.reporting.artifacts import ArtifactSpec, probe
from pnd.reporting.models import ArtifactState, PhaseResult, RunRecord, TableBlock
from pnd.reporting.pdf import PdfReportBuilder, load_quirk_glossary
from pnd.reporting.phases import build_catalogue
from pnd.text.vocabulary import load_symbol_list
from tests.conftest import ORIG_TELEGRAM, REPO_ROOT


def _msg(i, text):
    return {"id": i, "peer_id": {"channel_id": 7}, "message": text,
            "date": datetime(2021, 3, 1, 12, 30, tzinfo=timezone.utc)}


def test_d15_sets_give_identical_feature_rows():
    crypto = load_symbol_list(ORIG_TELEGRAM / "raw" / "crypto_symbol")
    exchange = load_symbol_list(ORIG_TELEGRAM / "raw" / "exchange")
    kw = ["target", "buy", "pump", "left", "announcement", "hold"]
    skip = ["", "the", "we", "pump", "buy"]
    minutes = [58, 59, 0, 1, 2, 28, 29, 30, 31, 32]
    for i, text in enumerate(["BTC pump on Binance, target 50%!", "Hold ETH... announcement soon 🚀",
                              "nothing to see here", "xrp bittrex kucoin left"]):
        m = _msg(i, text)
        assert (message_feature_row(m, crypto, exchange, kw, skip, minutes)
                == message_feature_row(m, set(crypto), set(exchange), set(kw), set(skip), set(minutes)))


def test_workspace_overrides_redirect_everything_but_shared_caches(cfg):
    overrides = _workspace_overrides(cfg.raw["paths"], "runs/x/workspace")
    targets = dict(o.split("=", 1) for o in overrides)
    assert targets["paths.pd_log.cleaned"] == "runs/x/workspace/datasets/telegram/Labeled/PD_logs_cleaned.txt"
    assert targets["paths.verification.report_dir"] == "runs/x/reports"
    assert "paths.telegram.raw_dir" not in targets
    assert "paths.klines.store_directory" not in targets
    assert all(not v.split("workspace/", 1)[-1].startswith(SHARED_PREFIXES) for v in targets.values())


def test_relativize_strips_project_root():
    assert _relativize(f"saved to {REPO_ROOT}\\runs\\a.csv", REPO_ROOT) == "saved to runs\\a.csv"


def test_catalogue_covers_every_registered_stage():
    from pnd.pipeline.registry import all_stages

    stages = [p.stage for p in build_catalogue(None, None)]
    assert len(stages) == len(set(stages))
    assert set(stages) >= set(all_stages()) - {"setup-nltk"}
    assert all(p.purpose and set(p.expectations) == {"literal", "bridged"} for p in build_catalogue(None, None))


def test_probe_counts_rows_and_previews(tmp_path, cfg):
    f = tmp_path / "t.tsv"
    f.write_text("a\tb\n1\tx\n2\ty\n", encoding="utf-8")
    state = probe(ArtifactSpec("t", lambda c: f, "d", "tsv", sep="\t"), "input", cfg)
    assert state.exists and state.records == 2 and state.columns == ["a", "b"]
    assert state.preview_rows == [["1", "x"], ["2", "y"]]
    missing = probe(ArtifactSpec("m", lambda c: tmp_path / "nope.csv", "d", "csv"), "input", cfg)
    assert not missing.exists


def test_glossary_reads_quirks_and_deviations():
    g = load_quirk_glossary(REPO_ROOT / "QUIRKS.md")
    assert {"Q01", "Q33", "Q37", "Q38", "D1", "D15"} <= set(g)


def test_pdf_builds_from_a_record(tmp_path):
    art = ArtifactState(name="In", role="input", path="x/in.csv", description="input file", exists=True, records=3,
                        records_label="rows", preview_header=["a"], preview_rows=[["é ≥ 🚀 <tag>"]])
    phase = PhaseResult(number=0, stage="clean-pd-log", title="Clean", original="PDlog_clean.py", purpose="p",
                        status="failed", expected=True, explanation="as documented", quirks=["Q13"],
                        error_type="KeyError", error_message="-1", inputs=[art],
                        tables=[TableBlock("t", ["k", "v"], [["a", "1"]])], log_tail=["line & <b>"])
    record = RunRecord(run_id="t", mode="literal", started_utc="s", finished_utc="f", duration_s=1.0, seed=1,
                       project_root=".", run_dir=".", environment={"python": "3.11"}, overrides=["a=b"],
                       limits={"verify_limit": 1}, phases=[phase])
    record.save(tmp_path / "run.json")
    reloaded = RunRecord.load(tmp_path / "run.json")
    out = PdfReportBuilder(reloaded, {"Q13": "fill"}).build(tmp_path / "r.pdf")
    assert out.read_bytes()[:4] == b"%PDF"
    pickle.dumps(reloaded)
