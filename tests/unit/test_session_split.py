from datetime import datetime, timedelta, timezone

from pnd.data.telegram.models import Message
from pnd.sessions.aggregation import select_pred_pump_sessions, session_split

T0 = datetime(2021, 1, 1, tzinfo=timezone.utc)


def msg(i, minutes, pump=False):
    m = Message({"peer_id": {"channel_id": 1}, "id": i, "message": str(i), "date": T0 + timedelta(minutes=minutes)})
    m.pred_pump = pump
    return m


def test_exact_12h_gap_splits():
    sessions = session_split([msg(1, 0), msg(2, 12 * 60)])
    assert [len(s.message_list) for s in sessions] == [1, 1]


def test_shorter_gap_does_not_split():
    sessions = session_split([msg(1, 0), msg(2, 12 * 60 - 1)])
    assert [len(s.message_list) for s in sessions] == [2]


def test_last_message_closes_session_and_flag_propagates():
    sessions = session_split([msg(1, 0), msg(2, 1, pump=True), msg(3, 2000)])
    assert [s.pred_pump_flag for s in sessions] == [True, False]
    assert [s.id for s in sessions] == [0, 1]


def test_only_sessions_with_more_than_one_pump_q10():
    sessions = session_split([msg(1, 0, True), msg(2, 1), msg(3, 2, True), msg(4, 5000, True)])
    kept = select_pred_pump_sessions(sessions)
    assert len(kept) == 1
    s = kept[0]
    assert (s.first_pred_pump_offset, s.last_pred_pump_offset, s.pred_pump_cnt) == (0, 2, 2)
