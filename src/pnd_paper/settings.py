"""Paths, protocol constants and QUICK/FULL sizes shared by the brief notebooks.

``PND_QUICK=1`` (default) selects reduced sizes for a fast validation run; ``PND_QUICK=0`` gives the sizes the
brief asks for. The authors' data folders are only ever read.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def _find_root() -> Path:
    here = Path(__file__).resolve()
    for p in [Path.cwd(), *Path.cwd().parents, *here.parents]:
        if (p / "pyproject.toml").is_file() and (p / "Data").is_dir():
            return p
    raise FileNotFoundError("project root (pyproject.toml + Data/) not found")


ROOT = _find_root()

# --- authors' data (read-only) ---------------------------------------------------------------------------------
PD_LOG = ROOT / "Data" / "Telegram" / "Labeled" / "PD_logs_cleaned.txt"
PD_LOG_RAW = ROOT / "Data" / "Telegram" / "Labeled" / "PD_logs_raw.txt"
TELEGRAM_RAW = ROOT / "Data" / "Telegram" / "raw"
GECKO_TARGETS = ROOT / "TargetCoinPrediction" / "FeatGeneration" / "raw" / "coinDate2Statistics_pred3d.json"
GECKO_CANDIDATES = ROOT / "TargetCoinPrediction" / "FeatGeneration" / "raw" / "neg_coin_gecko_statistics.txt"
NEG_REQUIRED_FILES = ROOT / "TargetCoinPrediction" / "FeatGeneration" / "raw" / "neg_binance_required_file_name.txt"
SYMBOL_MAP = ROOT / "TargetCoinPrediction" / "FeatGeneration" / "raw" / "binanceSymbol2CoinId.json"
WV_EMBEDDING = ROOT / "TargetCoinPrediction" / "FeatGeneration" / "feature" / "wv_embedding.pkl"

# --- caches and outputs ----------------------------------------------------------------------------------------
CACHE = ROOT / "datasets" / "paper_cache"
RESULTS = ROOT / "results"

# --- protocol (brief, Part II) ---------------------------------------------------------------------------------
TRAIN_END = pd.Timestamp("2021-01-19", tz="UTC")     # train: tau < TRAIN_END
VALID_END = pd.Timestamp("2021-05-10", tz="UTC")     # valid: TRAIN_END <= tau < VALID_END; test: tau >= VALID_END
HR_KS = (1, 3, 5, 10, 20, 30)
WINDOWS_H = (1, 3, 6, 12, 24, 48, 60, 72)           # paper §5.1 windows x (hours)
SEED = 1234

PAPER_HR = {  # [A1] reference values quoted by the brief (HR@1 / HR@10 / HR@30)
    "LR": (0.156, 0.449, 0.662), "RF": (0.189, 0.537, 0.731), "TCN": (0.256, 0.573, 0.770),
    "SNN": (0.260, 0.596, 0.797), "SNNS": (0.277, 0.623, 0.823),
}


@dataclass(frozen=True)
class Sizes:
    seeds: int
    perm_draws: int
    null_draws: int
    optuna_trials: int
    dl_epochs: int
    garch_targets: int
    cusum_events: int | None
    arl_sims: int
    rolling_origins: int


QUICK = os.environ.get("PND_QUICK", "1") != "0"
SIZES = (Sizes(seeds=2, perm_draws=1_000, null_draws=200, optuna_trials=25, dl_epochs=15, garch_targets=10,
               cusum_events=60, arl_sims=200, rolling_origins=40)
         if QUICK else
         Sizes(seeds=5, perm_draws=10_000, null_draws=1_000, optuna_trials=150, dl_epochs=60, garch_targets=10,
               cusum_events=None, arl_sims=2_000, rolling_origins=200))


def seed_everything(seed: int = SEED) -> None:
    """Seed python, numpy and (if installed) torch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass


def split_of(ts: pd.Series) -> pd.Series:
    """'train' / 'valid' / 'test' for each pump time (chronological split identical to [A1])."""
    return pd.Series(np.where(ts < TRAIN_END, "train", np.where(ts < VALID_END, "valid", "test")), index=ts.index)


# --- facts reported by [A1] (arXiv 2204.12929 v2), used only for comparison tables ----------------------------
PAPER_FACTS = {
    "samples": 1335, "events": 709, "channels": 108, "coins": 278, "exchanges": 18,
    "binance_events": 445, "binance_share_events": 0.628, "binance_btc_positive_samples": 948,
    "train_pos": 648, "valid_pos": 100, "test_pos": 200, "negatives_total": 195_665, "positive_ratio": 0.0048,
    "pre_pump_return_x60": 0.095, "frequent_trading_start_h": 57, "repump_probability": 0.601,
    "channels_per_binance_pump": 2.25, "cos_same_channel": 0.92, "cos_pumped": 0.80, "cos_random": 0.72,
}
PAPER_TABLE5 = pd.DataFrame(  # test-set HR@k, Table 5 of [A1]
    {"LR": [0.156, 0.269, 0.322, 0.449, 0.608, 0.662], "RF": [0.189, 0.348, 0.417, 0.537, 0.687, 0.731],
     "DNN": [0.225, 0.278, 0.383, 0.498, 0.626, 0.727], "LSTM": [0.207, 0.339, 0.423, 0.551, 0.648, 0.696],
     "GRU": [0.229, 0.339, 0.414, 0.529, 0.626, 0.714], "TCN": [0.256, 0.348, 0.427, 0.573, 0.692, 0.770],
     "SNN": [0.260, 0.383, 0.465, 0.596, 0.727, 0.797], "SNNS": [0.277, 0.414, 0.513, 0.623, 0.739, 0.823]},
    index=[f"HR@{k}" for k in HR_KS])
