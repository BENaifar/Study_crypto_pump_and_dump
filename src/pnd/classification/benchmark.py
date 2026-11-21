"""Compare the four message classifiers. [orig: Data/Telegram/train_classifier.py]"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from pnd.classification.model_zoo import MODEL_NAMES, make_all
from pnd.classification.vectorizer import make_tfidf
from pnd.common.io import ensure_parent
from pnd.common.seeding import set_global_seed
from pnd.config import Config
from pnd.pipeline.registry import register

logger = logging.getLogger(__name__)

PRESELECT_COLUMNS = ["channel_id", "sample_id", "message", "message_wo_stop", "date", "weekday", "on_time",
                     "crypto_signal", "exchange_signal", "keyword_signal", "length"]


class color:  # Text style
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def model_train(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


class _Metrics:
    """Holds the four metric lists that were module globals in the original."""

    def __init__(self) -> None:
        self.acc_list: list[float] = []
        self.pr_list: list[float] = []
        self.re_list: list[float] = []
        self.f1_list: list[float] = []

    def model_accuracy(self, model, X_test, y_test, model_name):
        """[orig: train_classifier.py:L47-L56]"""
        logger.info("Classifier: %s", model_name)
        pred_y = model.predict(X_test)
        logger.info(color.BOLD + 'Confusion Matrix:\n' + color.END + "%s", confusion_matrix(y_test, pred_y))
        logger.info(color.BOLD + 'Report : ' + color.END)
        logger.info("\n%s", classification_report(y_test, pred_y))
        self.acc_list.append((round(accuracy_score(y_test, pred_y), 4) * 100))
        self.pr_list.append((round(precision_score(y_test, pred_y, average='weighted'), 4) * 100))
        self.re_list.append((round(recall_score(y_test, pred_y, average='weighted'), 4) * 100))
        self.f1_list.append((round(f1_score(y_test, pred_y, average='weighted'), 4) * 100))


def load_labeled_preselect(label_file, preselect_input) -> pd.DataFrame:
    """Q08: inner merge of labels with features, '?' dropped, labels kept as strings. [orig: L24-L28]"""
    labeled_sample = pd.read_csv(label_file, sep=" ", names=["label", "channel_id", "sample_id"])
    labeled_sample = labeled_sample[labeled_sample.label != "?"]
    pre_select_sample = pd.read_csv(preselect_input, sep="\t", names=PRESELECT_COLUMNS)
    pre_select_sample = pd.merge(labeled_sample, pre_select_sample, how="inner", on=["channel_id", "sample_id"])
    pre_select_sample = pre_select_sample.dropna(axis=0, how="any")
    return pre_select_sample


def benchmark(pre_select_sample: pd.DataFrame, classifier_cfg) -> pd.DataFrame:
    """[orig: train_classifier.py:L61-L94]"""
    tf_idf = make_tfidf(classifier_cfg.tfidf)

    # QUIRK(Q08): TF-IDF fitted on ALL rows before the train/test split (leakage) [orig: L63-L65]
    X = tf_idf.fit_transform(pre_select_sample.message_wo_stop)
    y = pre_select_sample.label.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=classifier_cfg.test_size,
                                                        random_state=classifier_cfg.split_random_state)

    ml_models = make_all(classifier_cfg)
    model_names = MODEL_NAMES

    trained_ml_models = []
    for i in ml_models:
        tt = model_train(i, X_train, y_train)
        trained_ml_models.append(tt)

    metrics = _Metrics()
    for i in range(len(trained_ml_models)):
        md = trained_ml_models[i]
        name = model_names[i]
        metrics.model_accuracy(md, X_test, y_test, name)

    performance_matrix = pd.DataFrame({'Accuracy': metrics.acc_list, 'Precision': metrics.pr_list,
                                       'Recall': metrics.re_list, 'F1 Score': metrics.f1_list},
                                      index=model_names)
    return performance_matrix


@register
class BenchmarkClassifiersStage:
    """Print / save the performance matrix of LR, GBDT, RF and NB."""

    name = "benchmark-classifiers"

    def run(self, cfg: Config) -> None:
        set_global_seed(cfg.seed)
        p = cfg.paths.classifier
        sample = load_labeled_preselect(p.label_file, p.preselect_input)
        performance_matrix = benchmark(sample, cfg.classifier)
        logger.info("\n%s", performance_matrix)
        out = ensure_parent(cfg.paths.verification.report_dir / "classifier_benchmark.csv")
        performance_matrix.to_csv(out)
        logger.info("saved %s", out)
