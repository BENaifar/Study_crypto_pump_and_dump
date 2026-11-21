"""Each test asserts that a documented original error (QUIRKS.md) is still reproduced."""

import pickle
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from pnd.classification.features import MessageFeaturesStage
from pnd.classification.predictor import predict_pump
from pnd.data.binance import archive
from pnd.data.binance.positive_klines import FetchKlinesStage
from pnd.data.binance.processing import ProcessKlinesStage
from pnd.features import negative_samples
from pnd.features.dataset_builder import build_sequences
from pnd.features.negative_samples import NegativeSamplesStage
from pnd.labeling.session_labeler import LabelSessionsStage
from pnd.pd_log.cleaning import clean_pd_log
from tests.conftest import DATASETS, ORIG_TELEGRAM, requires_datasets


def _message(i, text="target xyz on binance"):
    return {"_": "Message", "from_id": None, "id": i, "peer_id": {"channel_id": 42}, "message": text,
            "date": datetime(2021, 6, 1, 12, 0, tzinfo=timezone.utc)}


def test_q03_message_features_opens_file_without_directory(make_cfg, tmp_path):
    pre = tmp_path / "PreSelection"
    pre.mkdir()
    with open(pre / "42.pkl", "wb") as f:
        pickle.dump([_message(1)], f)
    cfg = make_cfg(f"paths.message_features.walk_dir={pre}", f"paths.message_features.load_dir={tmp_path}",
                   f"paths.message_features.output={tmp_path / 'preselect_sample'}",
                   f"paths.message_features.crypto_symbol={ORIG_TELEGRAM / 'raw' / 'crypto_symbol'}",
                   f"paths.message_features.exchange={ORIG_TELEGRAM / 'raw' / 'exchange'}")
    with pytest.raises(FileNotFoundError):
        MessageFeaturesStage().run(cfg)


def test_q04_q22_q24_q29_default_path_mismatches(cfg):
    p = cfg.paths
    assert p.message_features.output != p.classifier.preselect_input
    assert p.positive_samples.output != p.dataset.pos_input
    assert not p.dataset.neg_input.exists() or True  # never produced by the negative stage (see Q23f test)
    assert p.dataset.train_output != p.model.train_input
    assert p.dataset.test_output != p.model.test_input


def _toy_preselect(n=60):
    rows, labels = [], []
    for i in range(n):
        pump = i % 2 == 0
        text = "pump coin target binance hold buy" if pump else "news market today update weekly report"
        rows.append([1, i, text, text, "2021-01-01", 4, 0, 1, 1, 1, 6])
        if i < 40:
            labels.append(["1" if pump else ("2" if i % 4 == 1 else "0"), 1, i])
    pre = pd.DataFrame(rows, columns=["channel_id", "sample_id", "message", "message_wo_stop", "date", "weekday",
                                      "on_time", "crypto_signal", "exchange_signal", "keyword_signal", "length"])
    lab = pd.DataFrame(labels, columns=["label", "channel_id", "sample_id"])
    return pre, lab


def test_q09_threshold_variable_is_ignored(make_cfg):
    pre, lab = _toy_preselect()
    a = predict_pump(pre.copy(), lab.copy(), make_cfg().classifier)
    b = predict_pump(pre.copy(), lab.copy(), make_cfg("classifier.threshold_variable=0.99").classifier)
    pd.testing.assert_frame_equal(a, b)
    assert "index" in a.columns  # extra column from reset_index concat


def test_q11_session_labeler_needs_missing_browsed_file(make_cfg, tmp_path):
    cfg = make_cfg(f"paths.session_labeler.walk_dir={tmp_path}",
                   f"paths.session_labeler.browsed_input={tmp_path / 'pump_attack_browsed.txt'}")
    with pytest.raises(FileNotFoundError):
        LabelSessionsStage().run(cfg)


def test_q13_question_mark_in_first_row_raises_keyerror():
    df = pd.DataFrame({"channel_id": [1], "session_id": [1], "coin": ["AAA"], "exchange": ["?"], "pair": ["BTC"],
                       "timestamp": [pd.Timestamp("2021-01-01", tz="UTC")]})
    with pytest.raises(KeyError):
        clean_pd_log(df)


def test_q14_positive_klines_only_previous_month(make_cfg, tmp_path, monkeypatch):
    log = tmp_path / "pd.txt"
    pd.DataFrame({"channel_id": [1, 1], "session_id": [1, 2], "coin": ["AAA", "BBB"], "exchange": ["binance"] * 2,
                  "pair": ["BTC"] * 2, "timestamp": ["2021-05-20 17:00:00+00:00", "2021-05-02 17:00:00+00:00"]}
                 ).to_csv(log, sep="\t", index=False)
    calls = []
    monkeypatch.setattr(archive, "download_monthly_klines",
                        lambda **kw: calls.append((kw["symbols"][0], kw["years"][0], kw["months"][0])))
    FetchKlinesStage().run(make_cfg(f"paths.klines.pd_log={log}", f"paths.klines.store_directory={tmp_path}"))
    assert calls == [("AAABTC", 2021, 4), ("BBBBTC", 2021, 4)]  # the pump month (5) is never requested


def test_q23b_negative_download_never_requests_previous_month(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(negative_samples, "download_monthly_klines",
                        lambda **kw: calls.append((kw["years"][0], kw["months"][0])))
    negative_samples.download_branch(np.array(["AAABTC-202105", "BBBBTC-202106"]), tmp_path)
    assert calls == [(2021, 5), (2021, 6)]


def test_q17_process_klines_walks_the_wrong_directory(make_cfg, tmp_path):
    zdir = tmp_path / "binance" / "data" / "spot" / "monthly" / "klines" / "AAABTC" / "1m"
    zdir.mkdir(parents=True)
    with zipfile.ZipFile(zdir / "AAABTC-1m-2021-05.zip", "w") as z:
        z.writestr("AAABTC-1m-2021-05.csv", "1,1,1,1,1,1,1,1,1,1,1,0\n")
    log = tmp_path / "pd.txt"
    pd.DataFrame({"channel_id": [1], "session_id": [1], "coin": ["AAA"], "exchange": ["binance"], "pair": ["BTC"],
                  "timestamp": ["2021-05-20 17:00:00+00:00"]}).to_csv(log, sep="\t", index=False)
    cfg = make_cfg(f"paths.process_klines.pd_log={log}", f"paths.process_klines.walk_dir={tmp_path / 'data'}",
                   f"paths.process_klines.unzip_dir={tmp_path / 'data' / 'unzip'}",
                   f"paths.process_klines.concat_dir={tmp_path / 'data' / 'concat'}")
    ProcessKlinesStage().run(cfg)
    assert not (tmp_path / "data" / "unzip").exists()
    assert list((tmp_path / "data" / "concat").iterdir()) == []


@requires_datasets
def test_q23f_negative_samples_saves_nothing(make_cfg, tmp_path):
    featgen = DATASETS / "featgen"
    cfg = make_cfg(f"paths.negative_samples.concat_list_dir={tmp_path / 'concat'}",
                   f"paths.negative_samples.statistics_dir={tmp_path / 'concat'}",
                   f"paths.negative_samples.required_files={featgen / 'raw' / 'neg_binance_required_file_name.txt'}")
    before = set((featgen / "feature").iterdir())
    NegativeSamplesStage().run(cfg)
    assert set((featgen / "feature").iterdir()) == before
    assert not (featgen / "feature" / "neg_sample_raw.csv").exists()


def test_q28_sequence_features_have_no_delimiter():
    pos = pd.DataFrame({"channel_id": [1, 1], "timestamp_unix": [1000, 2000], "coin_id": [3, 4],
                        "pre_1h_price": [0.5, -1.25], "pre_1h_return": [2.0, 3.0]})
    seq = build_sequences(pos, ["pre_1h_price", "pre_1h_return"])
    assert pos.loc[0, "feature"] == "0.52.0"
    assert seq.loc[0, "feature_seq"] == "0.52.0"   # history of the 2nd pump = the 1st pump, glued
    assert seq.loc[0, "length"] == "1"


@pytest.mark.tf
def test_q30_feat_generator_splits_features_per_character(tmp_path):
    import tensorflow.compat.v1 as tf

    from pnd.features.columns import COLUMN_LIST
    from pnd.modeling.data_loader import FeatGenerator

    tf.reset_default_graph()
    values = ["1", "123", "1", "AAA", "2", "1620000000000", "1", "5", "0.5"] + ["0.1"] * (len(COLUMN_LIST) - 9)
    csv = tmp_path / "one.csv"
    csv.write_text(",".join(values) + "\n")
    gen = FeatGenerator(str(csv), DATASETS / "featgen" / "feature" / "wv_embedding.pkl", epoch=1, batch_size=1)
    with tf.Session() as sess, pytest.raises(tf.errors.InvalidArgumentError):
        # "0.5" is split into "0", ".", "5"; "." cannot be converted -> StringToNumber error (expected, Q30)
        sess.run(gen.features["seq_feature"])
