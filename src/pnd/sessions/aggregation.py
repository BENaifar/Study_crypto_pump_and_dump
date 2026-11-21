"""Group channel messages into sessions and keep predicted pump sessions. [orig: Data/Telegram/sess_aggregate.py]"""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

import pandas as pd

from pnd.common.io import load_pickle, save_pickle
from pnd.common.seeding import set_global_seed
from pnd.config import Config
from pnd.data.telegram.models import Message, Session
from pnd.pipeline.registry import register

logger = logging.getLogger(__name__)


def session_split(message_list, threshold=12):
    """Q10: split when the gap to the next message is >= ``threshold`` hours. [orig: sess_aggregate.py:L48-L73]"""

    session_list = []
    threshold = timedelta(hours=threshold)
    session_message_list = []
    session_id = 0
    session_pred_pump = False

    for i in range(len(message_list)):

        session_message_list.append(message_list[i])
        if message_list[i].pred_pump:
            session_pred_pump = True

        if i == len(message_list) - 1:
            session_list.append(Session(session_id, session_message_list, session_pred_pump))

        else:
            delta = message_list[i + 1].date - message_list[i].date
            # QUIRK(Q10): ">=" 12h gap (the paper says 24h) [orig: L67]
            if delta >= threshold:
                session_list.append(Session(session_id, session_message_list, session_pred_pump))
                session_message_list = []
                session_pred_pump = False
                session_id += 1

    return session_list


def build_message_list(all_messages, pred_pump_sample_ids):
    """Reverse to chronological order, keep channel posts, flag predicted pumps. [orig: L87-L98]"""
    all_messages.reverse()

    message_list = []
    for m in all_messages:
        if m["_"] != "Message" or m["from_id"] != None:  # noqa: E711
            continue
        m_obj = Message(m)
        if m_obj.message_id in pred_pump_sample_ids:
            m_obj.pred_pump = True

        message_list.append(m_obj)
    return message_list


def select_pred_pump_sessions(session_list, min_cnt_exclusive: int = 1):
    """Compute pump offsets and keep sessions with ``pred_pump_cnt > 1``. [orig: L101-L114]"""
    pred_pump_session_list = []

    for session in session_list:
        if session.pred_pump_flag:
            for i in range(len(session.message_list)):
                m = session.message_list[i]
                if m.pred_pump:
                    if session.pred_pump_cnt == 0:
                        session.first_pred_pump_offset = i
                    session.pred_pump_cnt += 1
                    session.last_pred_pump_offset = i

            # QUIRK(Q10): at least two predicted pump messages [orig: L113]
            if session.pred_pump_cnt > min_cnt_exclusive:
                pred_pump_session_list.append(session)
    return pred_pump_session_list


@register
class AggregateSessionsStage:
    """``pred_pump_message.csv`` + raw pickles -> ``Labeled/<channel>_session.pkl``. [orig: L75-L117]"""

    name = "aggregate-sessions"

    def run(self, cfg: Config) -> None:
        set_global_seed(cfg.seed)
        p = cfg.paths.sessions
        pred_pump_sample = pd.read_csv(p.pred_input, sep="\t")
        channels = set(pred_pump_sample.channel_id.values)
        pred_pump_sample["id"] = pred_pump_sample.channel_id.astype(str) + "-" + pred_pump_sample.sample_id.astype(str)

        pred_pump_sample_ids = set(pred_pump_sample.id.values)

        for c in channels:
            all_messages = load_pickle(Path(p.raw_dir) / (str(c) + ".pkl"))
            message_list = build_message_list(all_messages, pred_pump_sample_ids)

            session_list = session_split(message_list, cfg.sessions.gap_hours)
            pred_pump_session_list = select_pred_pump_sessions(session_list,
                                                               cfg.sessions.min_pred_pump_cnt_exclusive)

            if len(pred_pump_session_list) > 0:
                save_pickle(pred_pump_session_list, Path(p.output_dir) / (str(c) + "_session.pkl"))
        logger.info("processed %d channels", len(channels))
