"""Catalogue of pipeline phases: what each one does, what it reads and writes, and what is expected of it.

This module is data only (plus small result extractors); the runner executes it and the PDF builder renders it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from pnd.config import Config
from pnd.reporting.artifacts import ArtifactSpec, child, fixed, key
from pnd.reporting.models import TableBlock

PRESELECT_COLUMNS = ("channel_id", "sample_id", "message", "message_wo_stop", "date", "weekday", "on_time",
                     "crypto_signal", "exchange_signal", "keyword_signal", "length")

Extractor = Callable[[Config], list[TableBlock]]


@dataclass(frozen=True)
class Expectation:
    """What the original code is documented to do in a given mode."""

    status: str                    # "ok" | "failed" | "skipped"
    explanation: str
    error_type: str = ""


@dataclass(frozen=True)
class PhaseSpec:
    stage: str
    title: str
    original: str
    purpose: str
    inputs: tuple[ArtifactSpec, ...]
    outputs: tuple[ArtifactSpec, ...]
    quirks: tuple[str, ...]
    expectations: dict[str, Expectation]          # mode -> expectation
    run: bool = True                              # False for interactive / opt-in phases
    highlight_patterns: tuple[str, ...] = ()
    extract: Extractor | None = None
    stage_kwargs: dict = field(default_factory=dict)


# ---------------------------------------------------------------- result extractors

def _benchmark_table(cfg: Config) -> list[TableBlock]:
    path = Path(cfg.paths.verification.report_dir) / "classifier_benchmark.csv"
    if not path.is_file():
        return []
    df = pd.read_csv(path, index_col=0)
    rows = [[str(i)] + [f"{v:.2f}" for v in r] for i, r in zip(df.index, df.values)]
    return [TableBlock("Classifier comparison on the 30% test split (weighted, %)", ["model", *df.columns], rows,
                       "Paper Table 1 (2 classes, threshold 0.2): LR F1 0.902, RF F1 0.920. The code compares 3 "
                       "string classes with argmax predictions, so the numbers are not directly comparable (Q08).")]


def _pred_table(cfg: Config) -> list[TableBlock]:
    path = Path(cfg.paths.classifier.predict_output)
    if not path.is_file():
        return []
    df = pd.read_csv(path, sep="\t", usecols=["channel_id"])
    top = df.channel_id.value_counts().head(5)
    return [TableBlock("Predicted pump messages", ["metric", "value"],
                       [["predicted pump messages", f"{len(df):,}"], ["channels", f"{df.channel_id.nunique():,}"]]
                       + [[f"top channel {c}", f"{n:,}"] for c, n in top.items()])]


def _sessions_table(cfg: Config) -> list[TableBlock]:
    from pnd.common.legacy_pickle import load_legacy

    out = Path(cfg.paths.sessions.output_dir)
    files = sorted(out.glob("*_session.pkl"))
    if not files:
        return []
    n_sessions = n_msgs = 0
    for f in files:
        sessions = load_legacy(f)
        n_sessions += len(sessions)
        n_msgs += sum(len(s.message_list) for s in sessions)
    return [TableBlock("Predicted pump sessions", ["metric", "value"],
                       [["session files (channels)", f"{len(files):,}"], ["sessions kept (>1 pump msg)", f"{n_sessions:,}"],
                        ["messages inside kept sessions", f"{n_msgs:,}"]])]


def _pd_log_table(cfg: Config) -> list[TableBlock]:
    path = Path(cfg.paths.pd_log.cleaned)
    if not path.is_file():
        return []
    df = pd.read_csv(path, sep="\t", dtype=str)
    rows = [["P&D samples (rows)", f"{len(df):,}"], ["channels", f"{df.channel_id.nunique():,}"],
            ["coins", f"{df.coin.nunique():,}"], ["exchanges (case-sensitive)", f"{df.exchange.nunique():,}"]]
    rows += [[f"exchange = {k}", f"{v:,}"] for k, v in df.exchange.value_counts().head(5).items()]
    rows += [[f"pair = {k}", f"{v:,}"] for k, v in df.pair.value_counts().head(3).items()]
    return [TableBlock("Cleaned P&D log", ["metric", "value"], rows,
                       "Paper Table 2: 1,335 samples, 709 events, 108 channels, 278 coins, 18 exchanges.")]


def _klines_table(cfg: Config) -> list[TableBlock]:
    store = Path(cfg.paths.klines.store_directory)
    zips = sorted(store.rglob("*.zip"))
    if not zips:
        return []
    months = pd.Series([z.stem.rsplit("-", 2)[-2] + "-" + z.stem.rsplit("-", 1)[-1] for z in zips]).value_counts()
    rows = [["monthly 1m archives on disk", f"{len(zips):,}"], ["distinct symbols", f"{len({z.parent.parent.name for z in zips}):,}"]]
    rows += [[f"month {m}", f"{n:,}"] for m, n in months.sort_index().head(6).items()]
    return [TableBlock("Downloaded kline archives (shared download cache)", ["metric", "value"], rows,
                       "Every archive is the month BEFORE a pump; the pump month itself is never requested (Q14).")]


def _positive_table(cfg: Config) -> list[TableBlock]:
    path = Path(cfg.paths.positive_samples.output)
    if not path.is_file():
        return []
    df = pd.read_csv(path)
    binance = df[df.exchange == "binance"]
    has = lambda c: int(df[c].notna().sum()) if c in df.columns else 0  # noqa: E731
    return [TableBlock("Positive samples", ["metric", "value"], [
        ["rows (all P&D samples)", f"{len(df):,}"], ["binance rows", f"{len(binance):,}"],
        ["rows with pre-pump kline features (pre_1h_price)", f"{has('pre_1h_price'):,}"],
        ["rows with after-pump returns (insider_return)", f"{has('insider_return'):,}"],
        ["rows with CoinGecko price (pre_3d_price_btc)", f"{has('pre_3d_price_btc'):,}"],
        ["columns", f"{len(df.columns):,}"]])]


def _dataset_table(cfg: Config) -> list[TableBlock]:
    rows = []
    for name, key_ in (("train", "train_output"), ("test", "test_output")):
        path = Path(cfg.paths.dataset[key_])
        if path.is_file():
            labels = pd.read_csv(path, header=None, usecols=[0])[0]
            rows.append([name, f"{len(labels):,}", f"{int(labels.sum()):,}", f"{len(labels) - int(labels.sum()):,}"])
    if not rows:
        return []
    return [TableBlock("Train / test samples", ["split", "rows", "positives", "negatives"], rows,
                       "Paper Table 4: 948 positives / 195,665 negatives in total.")]


def _verify_table(cfg: Config) -> list[TableBlock]:
    path = Path(cfg.paths.verification.report_dir) / "binance_verification.csv"
    if not path.is_file():
        return []
    df = pd.read_csv(path)

    def pct(col: str) -> str:
        s = df[col].dropna().astype(bool)
        return f"{int(s.sum())}/{len(s)} ({100 * s.mean():.1f}%)" if len(s) else "n/a"

    def med(col: str) -> str:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        return f"{s.median():.4f} (IQR {s.quantile(.25):.4f}-{s.quantile(.75):.4f}, n={len(s)})" if len(s) else "n/a"

    rows = [["rows audited", f"{len(df):,}"], ["previous-month archive exists", pct("archive_prev_found")],
            ["pump-month archive exists", pct("archive_current_found")], ["pump visible (>=5% within 60 min)", pct("pump_visible")],
            ["symbol still TRADING today", pct("symbol_trading_now")], ["median peak / pre-pump price", med("pump_ratio")],
            ["median minutes to peak", med("minutes_to_peak")],
            ["CoinGecko price_btc vs Binance VWAP, rel. error", med("cg_vs_binance_rel_err")]]
    defaulted = df[df.pair_or_exchange_defaulted == True]  # noqa: E712
    if len(defaulted):
        rows.append(["pump visible, rows defaulted to BTC/binance by Q13",
                     f"{int(defaulted.pump_visible.fillna(False).astype(bool).sum())}/{len(defaulted)}"])
    return [TableBlock("Binance audit of the authors' P&D log", ["check", "result"], rows)]


# ---------------------------------------------------------------- artifacts (shared specs)

TELEGRAM_RAW = ArtifactSpec("Raw Telegram messages", key("telegram.raw_dir"),
                            "One pickle per channel (authors' shipped Telethon dumps): a list of message dicts with "
                            "id, date, message text, from_id and peer_id.", "dir", "*.pkl")
CRYPTO = ArtifactSpec("Crypto symbol vocabulary", key("telegram.crypto_symbol"),
                      "Comma-separated coin tickers / names (first line only is used).", "text")
EXCHANGE = ArtifactSpec("Exchange vocabulary", key("telegram.exchange"),
                        "Comma-separated exchange names (first line only is used).", "text")
PRESELECTION = ArtifactSpec("Pre-selected messages", key("keyword_filter.output_dir"),
                            "Per-channel pickles holding only the channel posts that mention a coin, an exchange or "
                            "the keyword 'target'.", "dir", "*.pkl")
LABELS = ArtifactSpec("Manual message labels", key("classifier.label_file"),
                      "Space-separated 'label channel_id message_id'. Labels: 0 non-pump, 1 pump, 2 pump-related "
                      "(merged into 0 for prediction), ? unsure (dropped). Authors' 5,146 lines.", "csv",
                      sep=" ", columns=("label", "channel_id", "sample_id"))
PRESELECT_WRITTEN = ArtifactSpec("Message feature table (as written)", key("message_features.output"),
                                 "TSV, one row per pre-selected message: ids, cleaned text, text without stop-words, date, "
                                 "weekday, on-time flag, crypto/exchange/keyword counts, length.", "tsv", sep="\t",
                                 columns=PRESELECT_COLUMNS)
PRESELECT_READ = ArtifactSpec("Message feature table (where the classifiers look)", key("classifier.preselect_input"),
                              "Same TSV, at the path the classifier scripts read (PreSelection/preselect_sample).",
                              "tsv", sep="\t", columns=PRESELECT_COLUMNS)
PRED = ArtifactSpec("Predicted pump messages", key("classifier.predict_output"),
                    "TSV of unlabeled messages that GBDT scores >= 0.25 as pump: channel_id, sample_id, message, date, "
                    "pred_label.", "tsv", sep="\t")
SESSIONS = ArtifactSpec("Predicted pump sessions", key("sessions.output_dir"),
                        "One pickle per channel: a list of Session objects (messages split at >=12h gaps) that contain "
                        "more than one predicted pump message.", "dir", "*_session.pkl")
PD_RAW = ArtifactSpec("Raw P&D log", key("pd_log.raw"),
                      "Authors' manual session labels: channel_id, session_id, coin, exchange, pair, timestamp; "
                      "unknown exchange/pair written as '?'.", "csv_noheader", sep="\t")
PD_CLEAN = ArtifactSpec("Cleaned P&D log", key("pd_log.cleaned"),
                        "The P&D log with '?' exchange/pair filled (same coin+day, previous row of the same channel, "
                        "then BTC / binance). Main input of everything downstream.", "tsv", sep="\t")
KLINE_ZIPS = ArtifactSpec("Binance monthly kline archives", key("klines.store_directory"),
                          "data.binance.vision monthly 1-minute OHLCV zips: open time, OHLC, volume, quote volume, "
                          "trades, taker-buy volumes.", "dir", "*.zip", recursive=True)
CG_JSON = ArtifactSpec("CoinGecko pre-pump statistics", key("positive_samples.coingecko_json"),
                       "Authors' shipped CoinGecko /coins/{id}/history responses keyed COIN_YYYYMMDD (pump date), "
                       "fetched 3 days before the pump: market cap, price, volume, Twitter/Reddit/Alexa.", "json")
POS = ArtifactSpec("Positive samples", key("positive_samples.output"),
                   "One row per P&D sample: the log columns + after-pump prices/returns + 9 windows x 15 pre-pump "
                   "price/volume features + 9 CoinGecko features.", "csv")


def _expect(literal: Expectation, bridged: Expectation | None = None) -> dict[str, Expectation]:
    return {"literal": literal, "bridged": bridged or literal}


OK = Expectation("ok", "Runs as in the original.")
SKIP_INTERACTIVE = Expectation("skipped", "Interactive (a human answers prompts per message/session); not run "
                                          "unattended. Downstream phases use the authors' shipped output instead.")
SKIP_OPTIN = Expectation("skipped", "Network fetcher outside the data decision (only Binance is re-fetched); "
                                    "the authors' shipped files are used instead.")


def build_catalogue(verify_limit: int | None, klines_limit: int | None) -> list[PhaseSpec]:
    """The phases in pipeline order."""
    return [
        PhaseSpec(
            "migrate-data", "Prepare the run workspace", "(new)",
            "Copies the authors' shipped inputs (labels, P&D logs, session pickles, CoinGecko files, coin embedding) "
            "from the untouched original folders into this run's workspace, so every run starts from the same state "
            "and never alters the originals. Raw Telegram pickles and Binance downloads are shared read-only caches.",
            (ArtifactSpec("Original Labeled folder", fixed("Data/Telegram/Labeled"),
                          "Authors' labels, P&D logs, predictions and session pickles.", "dir"),
             ArtifactSpec("Original FeatGeneration/raw folder", fixed("TargetCoinPrediction/FeatGeneration/raw"),
                          "Authors' CoinGecko JSON/TXT, symbol map and required-file list.", "dir")),
            (ArtifactSpec("Workspace Labeled folder", lambda cfg: Path(cfg.paths.pd_log.raw).parent,
                          "Working copy used by this run.", "dir"),
             ArtifactSpec("Workspace featgen/raw folder",
                          lambda cfg: Path(cfg.paths.negative_samples.required_files).parent,
                          "Working copy used by this run.", "dir"),
             ArtifactSpec("Coin embedding", key("model.wv_embedding"),
                          "Pre-trained word2vec coin vectors (378 x 8) used by the models.", "file")),
            (), _expect(OK)),

        PhaseSpec(
            "fetch-telegram", "Collect Telegram channel history", "Data/Telegram/get_channel_post.py",
            "Downloads every message of every broadcast channel listed in Pump_channel2 with Telethon and stores one "
            "pickle per channel.",
            (ArtifactSpec("Channel list", key("telegram.channel_list"), "t.me URLs, one per line (608 channels).", "text"),),
            (TELEGRAM_RAW,), ("Q36",), _expect(SKIP_OPTIN), run=False),

        PhaseSpec(
            "filter-keywords", "Keyword pre-selection", "Data/Telegram/keyword_filter.py",
            "Keeps channel posts (from_id is None) whose words include a known coin, an exchange or a keyword. The "
            "keyword list is [target, buy, pump], but 'buy' and 'pump' are on the skip-list, so only 'target' can "
            "match; the first match decides.",
            (TELEGRAM_RAW, CRYPTO, EXCHANGE), (PRESELECTION,), ("Q01", "Q02"), _expect(OK),
            highlight_patterns=()),

        PhaseSpec(
            "label-messages", "Manual message labeling", "Data/Telegram/pump_message_label.py",
            "Shows each unlabeled pre-selected message posted after 2021-01-01 and appends the human's label to "
            "label.txt.", (PRESELECTION,), (LABELS,), ("Q12",), _expect(SKIP_INTERACTIVE), run=False),

        PhaseSpec(
            "message-features", "Message features and text cleaning", "Data/Telegram/message_fg.py",
            "For every pre-selected message: lower-case, strip emoji, URLs and punctuation, remove NLTK stop-words, "
            "and count coin / exchange / keyword mentions; also weekday and whether it was posted near a full or half "
            "hour. Writes one TSV row per message.",
            (PRESELECTION,
             ArtifactSpec("Folder the script actually opens", key("message_features.load_dir"),
                          "load_obj(file) has no directory prefix, so pickles are opened from the working directory, "
                          "not from PreSelection/ (Q03).", "dir", "*.pkl"),
             CRYPTO, EXCHANGE),
            (PRESELECT_WRITTEN,), ("Q03", "Q04", "Q05", "Q06", "Q07"),
            _expect(Expectation("failed", "FileNotFoundError: the pickle names come from PreSelection/ but are opened "
                                          "from the working directory (Q03).", "FileNotFoundError"), OK)),

        PhaseSpec(
            "benchmark-classifiers", "Compare message classifiers", "Data/Telegram/train_classifier.py",
            "Joins the labels with the message features, fits TF-IDF (1-3 grams) on all labeled rows, splits 70/30 "
            "and compares Logistic Regression, GBDT, Random Forest and Naive Bayes.",
            (LABELS, PRESELECT_READ),
            (ArtifactSpec("Benchmark table", child("verification.report_dir", "classifier_benchmark.csv"),
                          "Accuracy / precision / recall / F1 per model (also printed).", "csv"),),
            ("Q04", "Q08"),
            _expect(Expectation("failed", "FileNotFoundError: the feature table was written to ./preselect_sample but "
                                          "is read from ./PreSelection/preselect_sample (Q04), and in literal mode it was "
                                          "not written at all (Q03).", "FileNotFoundError"), OK),
            extract=_benchmark_table),

        PhaseSpec(
            "predict-pump", "Predict pump messages", "Data/Telegram/predic_classifier.py",
            "Trains GBDT on all labeled messages (class 2 merged into 0) and labels every unlabeled message; "
            "probability >= 0.25 counts as pump. TF-IDF is fitted on labeled and unlabeled messages together.",
            (LABELS, PRESELECT_READ), (PRED,), ("Q04", "Q09"),
            _expect(Expectation("failed", "FileNotFoundError: same missing feature table as the benchmark (Q03/Q04). "
                                          "Downstream phases fall back to the authors' shipped prediction file.",
                                "FileNotFoundError"), OK),
            extract=_pred_table),

        PhaseSpec(
            "aggregate-sessions", "Aggregate messages into sessions", "Data/Telegram/sess_aggregate.py",
            "For each channel with predicted pump messages: order its posts in time, split them where two posts are "
            ">= 12 hours apart, and keep the sessions containing more than one predicted pump message.",
            (PRED, TELEGRAM_RAW), (SESSIONS,), ("Q10", "Q38"),
            _expect(Expectation("failed", "FileNotFoundError: the authors' prediction file names 15 channels whose raw "
                                          "pickles were never shipped (finding Q38).", "FileNotFoundError"), OK),
            extract=_sessions_table),

        PhaseSpec(
            "label-sessions", "Manual session labeling (P&D log)", "Data/Telegram/PD_label.py",
            "Shows each predicted pump session around its pump messages; the human records coin, exchange, pair and "
            "pump time, producing the raw P&D log.",
            (SESSIONS, ArtifactSpec("Browsed-session list", key("session_labeler.browsed_input"),
                                    "Read at start-up but never shipped or written (Q11).", "text")),
            (ArtifactSpec("Pump attack records", key("session_labeler.attack_output"),
                          "Tab-separated channel, session, coin, exchange, pair, time.", "text"),),
            ("Q11",), _expect(SKIP_INTERACTIVE), run=False),

        PhaseSpec(
            "clean-pd-log", "Clean the P&D log", "Data/Telegram/PDlog_clean.py",
            "Fills unknown ('?') exchange and pair values: first from another row with the same coin on the same day, "
            "then from the previous row of the same channel, and finally defaults to BTC / binance.",
            (PD_RAW,), (PD_CLEAN,), ("Q13", "Q37"),
            _expect(Expectation("ok", "Runs as in the original. Its output differs from the authors' published "
                                      "PD_logs_cleaned.txt by 22 hand edits (Q37) and replaces it for the rest of the run.")),
            highlight_patterns=(r"fail_cnt", r"saved \d+ rows"), extract=_pd_log_table),

        PhaseSpec(
            "fetch-klines", "Download Binance klines", "Data/Binance/kline_collect_Binance.py",
            "For each P&D row, downloads the 1-minute kline archive of coin+pair for the month BEFORE the pump "
            "(pump date - 31 days, or - 15 days in the first 3 days of a month).",
            (PD_CLEAN,), (KLINE_ZIPS,), ("Q14", "Q15", "Q16"),
            _expect(Expectation("ok", "Runs; only previous-month archives are fetched, never the pump month (Q14).")),
            stage_kwargs={"limit": klines_limit}, extract=_klines_table),

        PhaseSpec(
            "process-klines", "Unzip and join klines", "Data/Binance/data_process.py",
            "Unzips the archives and, per P&D row, writes the previous month + pump month as one CSV.",
            (PD_CLEAN, ArtifactSpec("Folder searched for zips", key("process_klines.walk_dir"),
                                    "Data/data relative to the script (the archives are in Data/Binance/data) (Q17).",
                                    "dir", "*.zip", recursive=True)),
            (ArtifactSpec("Unzipped klines", key("process_klines.unzip_dir"), "Monthly CSVs.", "dir", "*.csv"),
             ArtifactSpec("Joined klines", key("process_klines.concat_dir"), "Previous + pump month per P&D row.",
                          "dir", "*.csv")),
            ("Q17",),
            _expect(Expectation("ok", "Runs but produces nothing: it searches the wrong folder (Q17), and the pump-month "
                                      "archive it needs is never downloaded (Q14)."),
                    Expectation("ok", "Path bridged (Q17), but still nothing joined because the pump month is never "
                                      "downloaded (Q14).")),
            highlight_patterns=(r"concat files written",)),

        PhaseSpec(
            "fetch-coingecko", "Collect CoinGecko statistics", "TargetCoinPrediction/FeatGeneration/tmp.py",
            "Fetches CoinGecko history 3 days before each pump for 16 hard-coded symbols; the result is never saved.",
            (PD_CLEAN,), (CG_JSON,), ("Q35",), _expect(SKIP_OPTIN), run=False),

        PhaseSpec(
            "positive-samples", "Features of pumped coins", "TargetCoinPrediction/FeatGeneration/pos_sample_process.py",
            "For each Binance P&D row: after-pump prices/returns (insider = price 60 min before, outsider = price at "
            "the pump), 73 hourly volume-weighted price/volume buckets ending 30 min before the pump summarised over "
            "9 windows, and 9 CoinGecko statistics (looked up under the pump date).",
            (PD_CLEAN, ArtifactSpec("Joined klines (where this script looks)", key("positive_samples.concat_dir"),
                                    "../../Binance/data/concat relative to the script: a third location (Q18).",
                                    "dir", "*.csv"), CG_JSON),
            (POS,), ("Q18", "Q19", "Q20", "Q21"),
            _expect(Expectation("ok", "Runs; no kline file is found, so only the CoinGecko features are filled (Q14, Q17, "
                                      "Q18).")),
            highlight_patterns=(r"kline features", r"coingecko features", r"saved \d+ rows"), extract=_positive_table),

        PhaseSpec(
            "negative-samples", "Features of non-pumped coins", "TargetCoinPrediction/FeatGeneration/neg_sample_process.py",
            "For each pump, every other listed Binance coin with kline data for that month becomes a negative sample "
            "with the same pre-pump features and CoinGecko statistics. The merged table is never saved.",
            (PD_CLEAN,
             ArtifactSpec("Required kline file list", key("negative_samples.required_files"),
                          "Authors' list of symbol+pair-month files needed for negatives.", "text"),
             ArtifactSpec("Joined klines (listing location)", key("negative_samples.concat_list_dir"),
                          "Where candidate negatives are listed from.", "dir", "*.csv"),
             ArtifactSpec("Negative CoinGecko statistics", key("negative_samples.gecko_txt"),
                          "Authors' shipped TSV: key, market cap, price, volume, social indices.", "csv_noheader",
                          sep="\t")),
            (ArtifactSpec("Negative samples (expected by the next phase)", key("dataset.neg_input"),
                          "feature/neg_sample_raw.csv, which no code ever writes (Q23f / Q24).", "csv"),),
            ("Q23",),
            _expect(Expectation("ok", "Runs; builds the negative table in memory and does not save it (Q23f).")),
            highlight_patterns=(r"negative candidates", r"not saved")),

        PhaseSpec(
            "build-dataset", "Build train / test samples", "TargetCoinPrediction/FeatGeneration/gen_feature.py",
            "Imputes missing CoinGecko values per class, z-scores features over all rows, maps coins/channels to ids, "
            "builds each channel's history of earlier pumps as a sequence (values glued with no delimiter), splits "
            "train/test at 2021-05-09 17:00:21 UTC and shuffles.",
            (ArtifactSpec("Positive samples (where this script looks)", key("dataset.pos_input"),
                          "feature/pump_sample_raw.csv; the positive phase writes pump_sample_raw.csv (Q22).", "csv"),
             ArtifactSpec("Negative samples", key("dataset.neg_input"), "Never produced (Q24).", "csv")),
            (ArtifactSpec("Training samples", key("dataset.train_output"), "Header-less CSV in COLUMN_LIST order.",
                          "csv_noheader"),
             ArtifactSpec("Test samples", key("dataset.test_output"), "Header-less CSV in COLUMN_LIST order.",
                          "csv_noheader")),
            ("Q22", "Q24", "Q25", "Q26", "Q27", "Q28", "Q29"),
            _expect(Expectation("failed", "FileNotFoundError: the positive table is looked for in feature/ (Q22) and the "
                                          "negative table is never written (Q24).", "FileNotFoundError"),
                    Expectation("failed", "FileNotFoundError: neg_sample_raw.csv is never written by any code (Q23f/Q24); "
                                          "this cannot be bridged by a path.", "FileNotFoundError")),
            extract=_dataset_table),

        PhaseSpec(
            "train", "Train the target-coin model (SNN)", "TargetCoinPrediction/SeqModel/run_train.py",
            "Builds the TensorFlow 1 graph of the selected model (default SNN: positional attention over the channel's "
            "pump sequence) and trains with Adam; any exception ends training.",
            (ArtifactSpec("Training samples (where training looks)", key("model.train_input"),
                          "feature/train_sample.csv; build-dataset writes train_sample.csv one folder up (Q29).",
                          "csv_noheader"),
             ArtifactSpec("Coin embedding", key("model.wv_embedding"), "Frozen 378 x 8 coin vectors.", "file")),
            (ArtifactSpec("Checkpoints", lambda cfg: Path(cfg.paths.model.checkpoint_root) / cfg.model.checkpointDir,
                          "TF checkpoints saved every int(132314 / batch_size) iterations.", "dir", "*.index"),),
            ("Q29", "Q30", "Q31", "Q32"),
            _expect(Expectation("ok", "Returns normally after 0 iterations: the missing input raises inside the training "
                                      "loop, which prints the error and stops (Q32).")),
            highlight_patterns=(r"training stopped after \d+ iterations", r"NotFound|No such file")),

        PhaseSpec(
            "evaluate", "Evaluate (AUC, HitRatio@k)", "TargetCoinPrediction/SeqModel/run_eval.py",
            "Restores checkpoints 15..30 and reports AUC and HitRatio@1/3/5/10/20/50 per (channel, pump time) list.",
            (ArtifactSpec("Test samples (where evaluation looks)", key("model.test_input"),
                          "feature/test_sample.csv (Q29).", "csv_noheader"),
             ArtifactSpec("Checkpoints", lambda cfg: Path(cfg.paths.model.checkpoint_root) / cfg.model.checkpointDir,
                          "Expected snn15 ... snn30.", "dir", "*.index")),
            (), ("Q33",),
            _expect(Expectation("ok", "Returns normally: the first checkpoint (epoch 15) is missing, the restore error "
                                      "is printed and the loop stops (Q33).")),
            highlight_patterns=(r"not a valid checkpoint|epoch=|HitRate",)),

        PhaseSpec(
            "verify-binance", "Audit the authors' data against Binance", "(new)",
            "Downloads the pump month AND the previous month for each Binance P&D row into its own cache and checks: "
            "archive availability, whether a >=5% spike follows the logged time within 60 minutes, current trading "
            "status, and CoinGecko price_btc vs the Binance VWAP 3 days earlier.",
            (ArtifactSpec("Cleaned P&D log (verified)", key("verification.pd_log"), "Rows to audit.", "tsv", sep="\t"),
             ArtifactSpec("Raw P&D log", key("verification.pd_log_raw"), "To flag rows defaulted by Q13.",
                          "csv_noheader", sep="\t"),
             ArtifactSpec("CoinGecko statistics", key("verification.coingecko_json"), "Prices to cross-check.", "json")),
            (ArtifactSpec("Audit table", child("verification.report_dir", "binance_verification.csv"),
                          "One row per audited P&D sample.", "csv"),
             ArtifactSpec("Audit summary", child("verification.report_dir", "binance_verification.md"),
                          "Markdown summary.", "text")),
            ("Q13", "Q14", "Q37"), _expect(OK), stage_kwargs={"limit": verify_limit}, extract=_verify_table),
    ]
