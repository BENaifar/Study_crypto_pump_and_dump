"""Stage registry (open/closed: new stages register themselves, the CLI never changes)."""

from __future__ import annotations

import importlib

from pnd.pipeline.stage import Stage

_REGISTRY: dict[str, type[Stage]] = {}

# Modules that define stages; imported lazily so that e.g. TensorFlow loads only when needed.
_STAGE_MODULES: dict[str, str] = {
    "migrate-data": "pnd.data.migration",
    "setup-nltk": "pnd.text.cleaning",
    "fetch-telegram": "pnd.data.telegram.fetcher",
    "filter-keywords": "pnd.filtering.keyword_filter",
    "label-messages": "pnd.labeling.message_labeler",
    "message-features": "pnd.classification.features",
    "benchmark-classifiers": "pnd.classification.benchmark",
    "predict-pump": "pnd.classification.predictor",
    "aggregate-sessions": "pnd.sessions.aggregation",
    "label-sessions": "pnd.labeling.session_labeler",
    "clean-pd-log": "pnd.pd_log.cleaning",
    "fetch-klines": "pnd.data.binance.positive_klines",
    "process-klines": "pnd.data.binance.processing",
    "fetch-coingecko": "pnd.data.coingecko.client",
    "positive-samples": "pnd.features.positive_samples",
    "negative-samples": "pnd.features.negative_samples",
    "build-dataset": "pnd.features.dataset_builder",
    "train": "pnd.modeling.trainer",
    "evaluate": "pnd.modeling.evaluator",
    "verify-binance": "pnd.verification.binance_checks",
}


def register(cls: type[Stage]) -> type[Stage]:
    """Class decorator adding a stage to the registry under ``cls.name``."""
    _REGISTRY[cls.name] = cls
    return cls


def get_stage(name: str) -> type[Stage]:
    """Return the stage class registered under ``name`` (importing its module on demand)."""
    if name not in _REGISTRY:
        if name not in _STAGE_MODULES:
            raise KeyError(f"Unknown stage: {name}")
        importlib.import_module(_STAGE_MODULES[name])
    return _REGISTRY[name]


def all_stages() -> list[str]:
    """Names of all known stages, in pipeline order."""
    return list(_STAGE_MODULES)
