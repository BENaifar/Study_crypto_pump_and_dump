"""Golden tests against the authors' shipped artifacts (read from the pristine originals under Data/)."""

import pandas as pd
import pytest

from pnd.common.legacy_pickle import load_legacy
from pnd.data.binance.symbols import BINANCE_SYMBOLS
from pnd.data.telegram.models import Session
from pnd.pd_log.cleaning import clean_pd_log, read_raw_pd_log
from pnd.sessions.aggregation import AggregateSessionsStage
from tests.conftest import ORIG_FEATGEN, ORIG_TELEGRAM, original_assignment, requires_datasets

# Q37 (finding): the shipped PD_logs_cleaned.txt was hand-edited after PDlog_clean.py ran.
# These are the ONLY differences between the shipped file and the code's output on the shipped raw log.
EXPECTED_EXCHANGE_CASE_EDITS = 20     # e.g. "Yobit" -> "yobit", "Binance" -> "binance" (5 others left as-is)
EXPECTED_COIN_EDITS = {357: ("Ong", "ONG")}
EXPECTED_DROPPED_LAST_ROW = ("1249251103", "219", "LOOM", "yobit", "BTC", "2020-05-19 17:00:03+00:00")


def test_g1_pd_log_cleaning_reproduces_shipped_file_up_to_documented_hand_edits(tmp_path):
    df, _, _ = clean_pd_log(read_raw_pd_log(ORIG_TELEGRAM / "Labeled" / "PD_logs_raw.txt"))
    out = tmp_path / "cleaned.txt"
    df.to_csv(out, index=False, sep="\t")
    ours = pd.read_csv(out, sep="\t", dtype=str)
    shipped = pd.read_csv(ORIG_TELEGRAM / "Labeled" / "PD_logs_cleaned.txt", sep="\t", dtype=str)

    assert len(ours) == len(shipped) + 1
    assert tuple(ours.iloc[-1]) == EXPECTED_DROPPED_LAST_ROW
    ours = ours.iloc[: len(shipped)].reset_index(drop=True)

    for col in ("channel_id", "session_id", "pair", "timestamp"):
        assert (ours[col] == shipped[col]).all(), col

    coin_diff = ours.index[ours.coin != shipped.coin].tolist()
    assert {i: (ours.coin[i], shipped.coin[i]) for i in coin_diff} == EXPECTED_COIN_EDITS

    exch_diff = ours.exchange != shipped.exchange
    assert exch_diff.sum() == EXPECTED_EXCHANGE_CASE_EDITS
    assert (ours.exchange[exch_diff].str.lower() == shipped.exchange[exch_diff]).all()


def _session_signature(s):
    return (s.id, [m.message_id for m in s.message_list], s.first_pred_pump_offset, s.last_pred_pump_offset,
            s.pred_pump_cnt)


def _channels_with_raw() -> set[int]:
    return {int(f.stem) for f in (ORIG_TELEGRAM / "raw").glob("*.pkl")}


# Q38 (finding): the shipped raw pickles do not cover every channel the authors predicted on.
EXPECTED_PRED_CHANNELS_WITHOUT_RAW = 15
EXPECTED_SESSION_FILES_WITHOUT_RAW = 12


def test_q38_shipped_raw_messages_are_incomplete():
    pred = pd.read_csv(ORIG_TELEGRAM / "Labeled" / "pred_pump_message.csv", sep="\t")
    raw = _channels_with_raw()
    assert len(set(pred.channel_id) - raw) == EXPECTED_PRED_CHANNELS_WITHOUT_RAW
    session_channels = {int(f.name.split("_")[0]) for f in (ORIG_TELEGRAM / "Labeled").glob("*_session.pkl")}
    assert len(session_channels - raw) == EXPECTED_SESSION_FILES_WITHOUT_RAW


@requires_datasets
def test_q38_default_session_run_hits_missing_raw_file(make_cfg, tmp_path):
    cfg = make_cfg(f"paths.sessions.pred_input={ORIG_TELEGRAM / 'Labeled' / 'pred_pump_message.csv'}",
                   f"paths.sessions.output_dir={tmp_path}")
    with pytest.raises(FileNotFoundError):
        AggregateSessionsStage().run(cfg)


@requires_datasets
def test_g2_session_aggregation_reproduces_shipped_sessions(make_cfg, tmp_path):
    raw = _channels_with_raw()
    pred = pd.read_csv(ORIG_TELEGRAM / "Labeled" / "pred_pump_message.csv", sep="\t")
    pred_input = tmp_path / "pred_pump_message.csv"
    pred[pred.channel_id.isin(raw)].to_csv(pred_input, sep="\t", index=False)
    out_dir = tmp_path / "sessions"
    cfg = make_cfg(f"paths.sessions.pred_input={pred_input}", f"paths.sessions.output_dir={out_dir}")
    AggregateSessionsStage().run(cfg)

    shipped_files = sorted(p.name for p in (ORIG_TELEGRAM / "Labeled").glob("*_session.pkl")
                           if int(p.name.split("_")[0]) in raw)
    ours_files = sorted(p.name for p in out_dir.glob("*_session.pkl"))
    assert len(shipped_files) == 160
    assert ours_files == shipped_files

    for name in shipped_files:
        expected = [_session_signature(s) for s in load_legacy(ORIG_TELEGRAM / "Labeled" / name)]
        actual = [_session_signature(s) for s in load_legacy(out_dir / name)]
        assert actual == expected, name


def test_g3_legacy_unpickler_loads_all_shipped_sessions():
    files = list((ORIG_TELEGRAM / "Labeled").glob("*_session.pkl"))
    assert len(files) == 172
    for f in files:
        sessions = load_legacy(f)
        assert sessions and all(isinstance(s, Session) for s in sessions)


def test_g4_binance_symbol_list_is_character_exact():
    original = original_assignment(ORIG_FEATGEN / "neg_sample_process.py", "binance_symbol").split(",")
    assert list(BINANCE_SYMBOLS) == original
