"""Label the unlabeled pre-selected messages with GBDT. [orig: Data/Telegram/predic_classifier.py]"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pnd.classification.benchmark import PRESELECT_COLUMNS
from pnd.classification.model_zoo import make_gbdt
from pnd.classification.vectorizer import make_tfidf
from pnd.common.io import ensure_parent
from pnd.common.seeding import set_global_seed
from pnd.config import Config
from pnd.pipeline.registry import register

logger = logging.getLogger(__name__)


def predict_pump(pre_select_sample: pd.DataFrame, labeled_sample: pd.DataFrame, classifier_cfg) -> pd.DataFrame:
    """Q09. [orig: predic_classifier.py:L22-L50]"""
    labeled_sample = labeled_sample[labeled_sample.label != "?"]
    # QUIRK(Q09): class "2" merged into "0" [orig: L23]
    labeled_sample.loc[labeled_sample.label == "2", "label"] = "0"
    pre_select_sample = pre_select_sample.dropna(axis=0, how="any")

    pre_select_sample = pd.merge(pre_select_sample, labeled_sample, how="left", on=["channel_id", "sample_id"])
    pre_select_sample.label.fillna("-1", inplace=True)

    # QUIRK(Q04/Q09): TF-IDF fitted on train + predict rows together [orig: L29-L30]
    tf_idf = make_tfidf(classifier_cfg.tfidf)
    X_all = tf_idf.fit_transform(pre_select_sample.message_wo_stop)

    X_predict = X_all[np.where(pre_select_sample.label.values == "-1")[0]]
    X_train = X_all[np.where(pre_select_sample.label.values != "-1")[0]]
    y_train = pre_select_sample.label.values[np.where(pre_select_sample.label.values != "-1")[0]]

    # use GBDT model
    gbdt_model = make_gbdt(classifier_cfg.gbdt)
    gbdt_model.fit(X_train, y_train)
    y_pred_proba = gbdt_model.predict_proba(X_predict)

    y_pred = np.zeros_like(y_pred_proba[:, 1])

    # QUIRK(Q09): `threshold` is declared but the literal 0.25 is applied [orig: L43-L44]
    threshold = classifier_cfg.threshold_variable  # noqa: F841
    y_pred[np.where(y_pred_proba[:, 1] >= classifier_cfg.threshold_literal)[0]] = 1

    pred_sample = pre_select_sample[pre_select_sample.label.values == "-1"]
    # QUIRK(Q09): reset_index concat keeps an extra "index" column [orig: L47]
    pred_sample_final = pd.concat([pred_sample.reset_index(), pd.DataFrame(y_pred, columns=["pred_label"])], axis=1)

    # keep only the samples predicted as pump
    pred_pump_sample = pred_sample_final[pred_sample_final.pred_label == 1].reset_index()
    return pred_pump_sample


@register
class PredictPumpStage:
    """``PreSelection/preselect_sample`` + label.txt -> ``Labeled/pred_pump_message.csv``."""

    name = "predict-pump"

    def run(self, cfg: Config) -> None:
        set_global_seed(cfg.seed)
        p = cfg.paths.classifier
        pre_select_sample = pd.read_csv(p.preselect_input, sep="\t", names=PRESELECT_COLUMNS)
        labeled_sample = pd.read_csv(p.label_file, sep=" ", names=["label", "channel_id", "sample_id"])
        pred_pump_sample = predict_pump(pre_select_sample, labeled_sample, cfg.classifier)
        out = ensure_parent(p.predict_output)
        pred_pump_sample[["channel_id", "sample_id", "message", "date", "pred_label"]].to_csv(out, index=False, sep="\t")
        logger.info("saved %d predicted pump messages to %s", len(pred_pump_sample), out)
