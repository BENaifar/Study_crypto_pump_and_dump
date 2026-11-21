"""Command-line interface: ``pnd <stage> [--config] [--set key=value] [--seed N]``."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from pnd.common.logging_setup import configure
from pnd.config import load_config
from pnd.pipeline.registry import get_stage

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Bug-for-bug modular re-implementation of Hu et al. (SIGMOD 2023).")

_state: dict = {"config": None, "overrides": []}


@app.callback()
def main(
    config: Optional[Path] = typer.Option(None, "--config", help="YAML config (default: configs/default.yaml)."),
    set_: list[str] = typer.Option([], "--set", help="Override: dotted.key=value (repeatable)."),
    seed: Optional[str] = typer.Option(None, "--seed", help="Shortcut for --set seed=<int|null>."),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    configure(log_level)
    overrides = list(set_)
    if seed is not None:
        overrides.append(f"seed={seed}")
    _state["config"] = config
    _state["overrides"] = overrides


SetOpt = typer.Option([], "--set", help="Override: dotted.key=value (repeatable).")
SeedOpt = typer.Option(None, "--seed", help="Shortcut for --set seed=<int|null>.")


def _local(set_: list[str], seed: Optional[str]) -> list[str]:
    return [*set_, *([f"seed={seed}"] if seed is not None else [])]


def _run(stage_name: str, extra_overrides: list[str] | None = None, **stage_kwargs) -> None:
    cfg = load_config(_state["config"], [*_state["overrides"], *(extra_overrides or [])])
    stage = get_stage(stage_name)(**stage_kwargs)
    stage.run(cfg)


def _simple(stage_name: str, doc: str) -> None:
    def command(set_: list[str] = SetOpt, seed: Optional[str] = SeedOpt) -> None:
        _run(stage_name, _local(set_, seed))

    command.__doc__ = doc
    app.command(stage_name)(command)


for _name, _doc in [
    ("migrate-data", "Copy the authors' shipped inputs into data/ (never moves or edits originals)."),
    ("setup-nltk", "Download NLTK stopwords / punkt / punkt_tab."),
    ("fetch-telegram", "[opt-in] Telegram history (get_channel_post.py). Needs TELEGRAM_API_ID/HASH."),
    ("filter-keywords", "keyword_filter.py: raw/*.pkl -> PreSelection/*.pkl."),
    ("label-messages", "pump_message_label.py: interactive message labeling."),
    ("message-features", "message_fg.py: PreSelection -> preselect_sample."),
    ("benchmark-classifiers", "train_classifier.py: LR / GBDT / RF / NB comparison."),
    ("predict-pump", "predic_classifier.py: GBDT pump-message prediction."),
    ("aggregate-sessions", "sess_aggregate.py: 12h sessions with >1 predicted pump."),
    ("label-sessions", "PD_label.py: interactive session labeling."),
    ("clean-pd-log", "PDlog_clean.py: PD_logs_raw.txt -> PD_logs_cleaned.txt."),
    ("process-klines", "data_process.py: unzip + concat klines."),
    ("fetch-coingecko", "[opt-in] tmp.py: CoinGecko pre-3d history (needs the coingecko extra)."),
    ("positive-samples", "pos_sample_process.py: pump_sample_raw.csv."),
    ("negative-samples", "neg_sample_process.py: negative frame (never saved, Q23f)."),
    ("build-dataset", "gen_feature.py: train_sample.csv / test_sample.csv."),
]:
    _simple(_name, _doc)


@app.command("fetch-klines")
def fetch_klines(limit: Optional[int] = typer.Option(None, "--limit", help="Only the first N P&D rows."),
                 set_: list[str] = SetOpt, seed: Optional[str] = SeedOpt) -> None:
    """kline_collect_Binance.py: download monthly 1m klines from data.binance.vision."""
    _run("fetch-klines", _local(set_, seed), limit=limit)


@app.command("verify-binance")
def verify_binance(limit: Optional[int] = typer.Option(None, "--limit", help="Only the first N binance rows."),
                   set_: list[str] = SetOpt, seed: Optional[str] = SeedOpt) -> None:
    """[new] Audit the P&D log and CoinGecko prices against Binance klines."""
    _run("verify-binance", _local(set_, seed), limit=limit)


def _model_overrides(model, max_seq_length, batch_size, epoch, learning_rate, dropout_rate, checkpointDir) -> list[str]:
    values = {"model": model, "max_seq_length": max_seq_length, "batch_size": batch_size, "epoch": epoch,
              "learning_rate": learning_rate, "dropout_rate": dropout_rate, "checkpointDir": checkpointDir}
    return [f"model.{k}={v}" for k, v in values.items() if v is not None]


@app.command("train")
def train(
    model: Optional[str] = typer.Option(None, "--model"),
    max_seq_length: Optional[int] = typer.Option(None, "--max_seq_length"),
    batch_size: Optional[int] = typer.Option(None, "--batch_size"),
    epoch: Optional[int] = typer.Option(None, "--epoch"),
    learning_rate: Optional[float] = typer.Option(None, "--learning_rate"),
    dropout_rate: Optional[float] = typer.Option(None, "--dropout_rate"),
    checkpointDir: Optional[str] = typer.Option(None, "--checkpointDir"),
    set_: list[str] = SetOpt,
    seed: Optional[str] = SeedOpt,
) -> None:
    """run_train.py: train dnn / snn / snnta."""
    _run("train", _local(set_, seed) + _model_overrides(model, max_seq_length, batch_size, epoch, learning_rate, dropout_rate,
                                   checkpointDir))


@app.command("evaluate")
def evaluate(
    model: Optional[str] = typer.Option(None, "--model"),
    max_seq_length: Optional[int] = typer.Option(None, "--max_seq_length"),
    batch_size: Optional[int] = typer.Option(None, "--batch_size"),
    epoch: Optional[int] = typer.Option(None, "--epoch"),
    learning_rate: Optional[float] = typer.Option(None, "--learning_rate"),
    dropout_rate: Optional[float] = typer.Option(None, "--dropout_rate"),
    checkpointDir: Optional[str] = typer.Option(None, "--checkpointDir"),
    set_: list[str] = SetOpt,
    seed: Optional[str] = SeedOpt,
) -> None:
    """run_eval.py: HitRatio@k on checkpoints 15..30."""
    _run("evaluate", _local(set_, seed) + _model_overrides(model, max_seq_length, batch_size, epoch, learning_rate, dropout_rate,
                                      checkpointDir))


@app.command("run-pipeline")
def run_pipeline(
    mode: str = typer.Option("literal", "--mode", help="literal (as shipped) or bridged (path mismatches bridged)."),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Folder name under runs/ (default: timestamp-mode)."),
    verify_limit: Optional[int] = typer.Option(100, "--verify-limit", help="Rows audited by verify-binance (0 = all)."),
    klines_limit: Optional[int] = typer.Option(None, "--klines-limit", help="P&D rows for fetch-klines (default all)."),
    only: list[str] = typer.Option([], "--only", help="Run only these stages (repeatable; workspace prep always runs)."),
    pdf: bool = typer.Option(True, "--pdf/--no-pdf", help="Render the PDF report at the end."),
    set_: list[str] = SetOpt,
) -> None:
    """Run every phase, record inputs/outputs/outcomes in runs/<id>/reports/run.json and render the PDF report."""
    from pnd.pipeline.runner import PipelineRunner
    from pnd.reporting.pdf import build_pdf

    runner = PipelineRunner(mode=mode, run_id=run_id, verify_limit=verify_limit or None,
                            klines_limit=klines_limit, extra_overrides=[*_state["overrides"], *set_], only=only)
    typer.echo(f"run {runner.run_id} -> {runner.run_dir}")
    record = runner.run()
    unexpected = [p.stage for p in record.phases if not p.expected]
    typer.echo(f"done in {record.duration_s / 60:.1f} min; outcomes not as documented: {unexpected or 'none'}")
    if pdf:
        typer.echo(f"report: {build_pdf(runner.run_dir, runner.root)}")


@app.command("report")
def report(run_dir: Path = typer.Argument(..., help="A runs/<id> folder containing reports/run.json.")) -> None:
    """(Re)render the PDF report of an existing run."""
    from pnd.config import find_project_root
    from pnd.reporting.pdf import build_pdf

    typer.echo(f"report: {build_pdf(run_dir.resolve(), find_project_root())}")


if __name__ == "__main__":  # pragma: no cover
    app()
