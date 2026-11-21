# pnd: a bug-for-bug, modular re-implementation of *Sequence-based Target Coin Prediction for Cryptocurrency Pump-and-Dump* (Hu et al., SIGMOD 2023)

This repository re-builds the authors' research scripts (still present, untouched, under `Data/` and
`TargetCoinPrediction/`; their original README is in [docs/ORIGINAL_README.md](docs/ORIGINAL_README.md))
as an installable `src/`-layout package with one class per responsibility.

**The methodology is reproduced exactly, including every error.** Each preserved error, every finding about the
authors' data, and the few allowed deviations are listed in [QUIRKS.md](QUIRKS.md).

## Install

```bash
uv python install 3.11
uv sync --extra dev          # TensorFlow 2.15 (tf.compat.v1), pandas 2.2, scikit-learn 1.5
uv run pnd setup-nltk        # stopwords / punkt
uv run pnd migrate-data      # copies the authors' shipped inputs into datasets/ (never moves or edits originals)
```

Optional extras: `--extra telegram` (Telethon) and `--extra coingecko` (pycoingecko). Neither is used by default.

## Data provenance

| Source | How it is obtained |
|---|---|
| Binance 1-minute klines | **Downloaded** from data.binance.vision (`fetch-klines`, `verify-binance`) |
| Telegram messages, labels, P&D logs, session pickles | Authors' shipped files (fetcher ported, opt-in: `fetch-telegram`) |
| CoinGecko pre-3-day statistics | Authors' shipped JSON/TXT (fetcher ported, opt-in: `fetch-coingecko`) |
| Coin embedding `wv_embedding.pkl` | Authors' shipped file |

`datasets/` mirrors the folders the original scripts ran in, so the original relative-path errors reproduce
literally. The folder is not called `data/` because on Windows that is the same folder as the original `Data/`.

| New | Original working directory |
|---|---|
| `datasets/telegram/` | `Data/Telegram/` |
| `datasets/binance/` | `Data/Binance/` |
| `datasets/` | `Data/` |
| `datasets/featgen/` | `TargetCoinPrediction/FeatGeneration/` |
| `datasets/seqmodel/` | `TargetCoinPrediction/SeqModel/` |
| `datasets/legacy_root/` | repository root |

## Pipeline

| Command | Original script | Default input → output |
|---|---|---|
| `pnd fetch-telegram` *(opt-in)* | get_channel_post.py | `raw/Pump_channel2` → `raw/<id>.pkl` |
| `pnd filter-keywords` | keyword_filter.py | `raw/*.pkl` → `PreSelection/*.pkl` |
| `pnd label-messages` *(interactive)* | pump_message_label.py | → `Labeled/label.txt` |
| `pnd message-features` | message_fg.py | `PreSelection/` → `preselect_sample` |
| `pnd benchmark-classifiers` | train_classifier.py | `PreSelection/preselect_sample` → `reports/classifier_benchmark.csv` |
| `pnd predict-pump` | predic_classifier.py | → `Labeled/pred_pump_message.csv` |
| `pnd aggregate-sessions` | sess_aggregate.py | → `Labeled/<id>_session.pkl` |
| `pnd label-sessions` *(interactive)* | PD_label.py | → `Labeled/pump_attack_test.txt` |
| `pnd clean-pd-log` | PDlog_clean.py | `PD_logs_raw.txt` → `PD_logs_cleaned.txt` (overwrites, as the original) |
| `pnd fetch-klines [--limit N]` | kline_collect_Binance.py | → `datasets/binance/data/spot/...zip` |
| `pnd process-klines` | data_process.py | → `datasets/data/{unzip,concat}` |
| `pnd fetch-coingecko` *(opt-in)* | tmp.py | in memory only (never saved, as the original) |
| `pnd positive-samples` | pos_sample_process.py | → `datasets/featgen/pump_sample_raw.csv` |
| `pnd negative-samples` | neg_sample_process.py | builds the frame, saves nothing (as the original) |
| `pnd build-dataset` | gen_feature.py | `featgen/feature/*_raw.csv` → `featgen/{train,test}_sample.csv` |
| `pnd train` / `pnd evaluate` | run_train.py / run_eval.py | checkpoints → AUC + HitRatio@k |
| `pnd verify-binance [--limit N]` | *(new)* | → `reports/binance_verification.{md,csv}` |

Because the errors are kept, the literal chain does **not** produce a finished dataset. For example:
- `fetch-klines` never downloads the pump month (Q14).
- `process-klines` looks in the wrong folder (Q17).
- `negative-samples` never saves (Q23f).

That is intended.

### Run the whole pipeline and get a PDF report

```bash
uv run pnd run-pipeline                      # literal mode: the code exactly as shipped
uv run pnd run-pipeline --mode bridged       # also bridges the pure path mismatches (Q03, Q04, Q17, Q18, Q22, Q29)
uv run pnd report runs/<run id>              # re-render the PDF of an existing run
```

Each run gets its own folder, `runs/<run id>/`:
- `workspace/`: a fresh copy of the authors' shipped files. Runs never alter the originals or `datasets/`. The raw Telegram pickles and the Binance download caches are shared.
- `logs/NN-<stage>.log`: everything each phase logged or printed.
- `reports/run.json`: a machine-readable record of every phase.
- `reports/pipeline_report.pdf`: the report, generated with ReportLab.

For every phase, the report shows:
- what the phase does;
- each input, with what was on disk before the phase ran;
- each output, with what was on disk afterwards and whether this phase wrote it;
- the outcome, compared with the documented behaviour of the original code;
- result tables, data previews and a log excerpt.

Interactive labeling and the opt-in Telegram and CoinGecko fetchers are listed as skipped, and the authors' shipped outputs stand in for them.

Options:
- `--verify-limit N` (default 100; 0 = all rows) sets how many rows the Binance audit covers.
- `--klines-limit N` (default: all rows) limits `fetch-klines`.
- `--only <stage>` (repeatable) runs only the named stages.
- `--set key=value` applies a config override.

Expected runtime of a full literal run is about 15–25 minutes. Keyword filtering of the 845 MB of raw messages takes about 7 minutes; the Binance downloads take most of the rest.

**Bridged mode needs a lot of memory.** It runs `predict-pump` on about 1 M messages: TF-IDF with 1–3-grams over the whole corpus, as in the original. That needs well over 12 GB of RAM.

### Bridging a path error without touching code

Every path is a config key, so you can bridge a gap explicitly. For example:

```bash
uv run pnd benchmark-classifiers --set paths.classifier.preselect_input=datasets/telegram/preselect_sample
uv run pnd build-dataset --set paths.dataset.pos_input=datasets/featgen/pump_sample_raw.csv
```

`--set` takes any `dotted.key=value` from [configs/default.yaml](configs/default.yaml) and can be repeated.

### Seed

`seed: 1234` in the config (or `--seed N`) seeds python, numpy, TensorFlow and `tf.data.shuffle`. `--seed null`
restores the original unseeded behaviour. The original hard-coded sklearn `random_state`s are unchanged.

## Notebooks (project brief)

The `notebooks/` folder answers the project brief (*Détection et prédiction des schémas « Pump-and-Dump »*). One
notebook per step, each independent: it loads the raw data itself and never reads another notebook's outputs. These
notebooks follow the **paper**, not the bug-for-bug port: the methods are written from scratch, and only the
authors' data is reused. Shared data access and statistics live in `src/pnd_paper/`.

| Notebook | Brief step |
|---|---|
| `I0_data_acquisition` | data download (Binance archive), audit against the brief and [A1] |
| `I1_literature_synthesis` | I.1 four questions, MAR and MiCA |
| `I2_descriptive_statistics` | I.2 distributions, χ², Gini, Kolmogorov–Smirnov |
| `I3_event_study` | I.3 market model, CAR tests (Student, Wilcoxon, BMP, Holm), abnormal volume, minute anatomy |
| `I4_intra_channel_homogeneity` | I.4 cosine homogeneity, permutation test, Benjamini–Hochberg |
| `I5_coordination_network` | I.5 TF-IDF bipartite network, Louvain, date-permutation null model |
| `IIA1_arma_garch` | II.A.1 ADF, KPSS, Ljung–Box, ARMA–GARCH–t (MLE written from scratch) |
| `IIA2_cusum_isolation_forest` | II.A.2–3 CUSUM with ARL₀ calibration, Isolation Forest, McNemar |
| `IIB_machine_learning` | II.B ranking dataset, LR, conditional logit, RF, LightGBM (binary and per-event softmax), Optuna, SHAP |
| `IIC_deep_learning` | II.C DNN, LSTM, GRU, TCN, SNN, SNN_S in PyTorch, ablations, attention weights |
| `IID_bonus_btc_forecast` | II.D ARIMA, ARIMAX, LSTM on VADER sentiment, Diebold–Mariano |
| `00_complete_report` | all steps assembled with a validation chapter against [A1] (generated) |

```bash
uv sync --extra dev --extra paper
uv run playwright install chromium            # PDF export (nbconvert webpdf)
make all                                      # = uv run python notebooks/run_all.py (QUICK sizes)
make all FULL=1                               # sizes required by the brief (PND_QUICK=0)
uv run python notebooks/run_all.py IIB        # one notebook (prefix match)
```

Outputs:
- executed notebooks in `notebooks/`;
- PDFs in `notebooks/pdf/`: the step PDFs show code, the complete report hides it;
- tables and figures in `results/<notebook>/`;
- the Binance download cache in `datasets/paper_cache/`.

The notebook sources are kept as percent-format scripts in `notebooks/_src/`; `run_all.py` rebuilds the `.ipynb` files from them.

## Layout

```
src/pnd/
  config.py, cli.py           configuration (frozen) and Typer CLI
  pipeline/                   Stage protocol + registry (stages register themselves)
  common/                     I/O, legacy unpickler, seeding, time helpers, interactive InputPort
  data/telegram|binance|coingecko   acquisition (one client per source) and domain models
  text/, filtering/           message cleaning and keyword pre-selection
  labeling/                   interactive message and session labelers
  classification/             message features, TF-IDF, model zoo, benchmark, predictor
  sessions/, pd_log/          session aggregation and P&D log cleaning
  features/                   kline statistics, positive/negative samples, dataset builder
  modeling/                   TF1 data loader, layers, DNN/SNN/SNNTA, metrics, trainer, evaluator
  verification/               Binance audit of the authors' data
tests/  unit/ golden/ quirks/ network/
```

## Tests

```bash
uv run ruff check src tests
uv run pytest               # unit, golden (vs. the authors' shipped files) and quirk tests
uv run pytest -m network    # downloads from Binance
```

Key golden results:
- Session aggregation reproduces the authors' shipped session pickles exactly (160 reproducible channels).
- P&D-log cleaning reproduces the shipped log up to 22 hand edits by the authors, pinned in the test (QUIRKS.md Q37).

## Citation

```
@article{hu2022sequence,
  title={Sequence-Based Target Coin Prediction for Cryptocurrency Pump-and-Dump},
  author={Hu, Sihao and Zhang, Zhen and Lu, Shengliang and He, Bingsheng and Li, Zhao},
  journal={arXiv preprint arXiv:2204.12929},
  year={2022}
}
```
