"""Telegram domain objects (single definition). [orig: sess_aggregate.py:L20-L45, PD_label.py:L17-L42]"""

from __future__ import annotations


class Message(object):
    """A channel message reduced to id / content / date / predicted-pump flag."""

    def __init__(self, m):
        self.message_id = str(m["peer_id"]["channel_id"]) + "-" + str(m["id"])
        self.content = m["message"]
        self.date = m["date"]
        self.pred_pump = False


class Session(object):
    """A run of messages separated from neighbours by >= the session gap."""

    def __init__(self, id, message_list, pred_pump_flag):
        self.id = id
        self.pred_pump_flag = pred_pump_flag
        self.message_list = message_list
        self.target_coin = None
        self.target_exchange = None
        self.target_timestamp = None
        self.session_id = None
        self.first_pred_pump_offset = None
        self.last_pred_pump_offset = None
        self.pred_pump_cnt = 0

    def labeling(self):
        """Unused in the original pipeline; ported as-is."""
        self.target_coin = input("coin")
        self.target_exchange = input("exchange")
        self.target_timestamp = input("timestamp")
