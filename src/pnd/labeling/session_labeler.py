"""Manual labeling of predicted pump sessions -> P&D log. [orig: Data/Telegram/PD_label.py]"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytz

from pnd.common.console import InputPort, StdInputPort
from pnd.common.legacy_pickle import load_legacy
from pnd.common.seeding import set_global_seed
from pnd.config import Config
from pnd.pipeline.registry import register


class SessionLabeler:
    """Q11: browse each predicted pump session around its pump messages and record pump attacks."""

    def __init__(self, input_port: InputPort, walk_dir: Path, browsed_input: Path,
                 browsed_output: Path, attack_output: Path, min_date: str = "2019-01-01") -> None:
        self.io = input_port
        self.walk_dir = Path(walk_dir)
        self.browsed_input = Path(browsed_input)
        self.browsed_output = Path(browsed_output)
        self.attack_output = Path(attack_output)
        y, m, d = (int(x) for x in min_date.split("-"))
        self.min_date = (y, m, d)
        self.browsed_sessions: list[str] = []

    def load_browsed(self) -> None:
        """QUIRK(Q11): reads pump_attack_browsed.txt (missing -> FileNotFoundError). [orig: L50-L52]"""
        with open(self.browsed_input, "r", encoding="utf-8") as f:  # D13
            browsed_sessions = f.readlines()
            browsed_sessions = list(map(lambda x: "-".join(x[:-1].split("\t")[:2]), browsed_sessions))
        self.browsed_sessions = browsed_sessions

    def session_list_print(self, sess_list, channel_id):
        """print the session information [orig: L54-L109]"""

        min_datetime = datetime(*self.min_date).replace(tzinfo=pytz.timezone('UTC'))

        for sess in sess_list:
            if sess.message_list[0].date <= min_datetime:
                continue
            if str(channel_id) + "-" + str(sess.id) in self.browsed_sessions:
                continue

            # QUIRK(Q11): window = [first-5, last+6) [orig: L67-L68]
            stop_offset = min(sess.last_pred_pump_offset + 6, len(sess.message_list))
            start_offset = max(sess.first_pred_pump_offset - 5, 0)

            delta = 0
            for j in range(start_offset, stop_offset):
                i = j + delta
                message = sess.message_list[i]
                date = message.date
                content = message.content

                self.io.show(content)
                self.io.show("------------| " + str(message.pred_pump) + " | " + str(i) + "/" + str(stop_offset - 1)
                             + " ｜ " + str(date) + " |---------------")

                op = self.io.ask("")
                if op == "1":
                    self.labeling(str(channel_id), str(sess.id))
                    self.io.show("------------------------------------")

                elif op == "?":

                    self.io.show("-------------")
                    self.io.show("channel:" + str(channel_id), " | " + "session_index:" + str(sess.id))
                    self.io.show("-------------")
                    delta -= 1

                elif op == "#":
                    break
                elif op == "##":
                    return

                elif op == "r":
                    if j > 0:
                        delta -= 2
                    else:
                        self.io.show("not illegal")
                        delta -= 1

                else:
                    pass

            # QUIRK(Q11): writes the *_test file, never the browsed file it reads [orig: L107-L109]
            with open(self.browsed_output, "a", encoding="utf-8") as f:  # D13
                f.write(str(channel_id) + "-" + str(sess.id) + "\n")

    def labeling(self, channel_id, session_id):
        """[orig: L112-L122]"""
        with open(self.attack_output, "a", encoding="utf-8") as f:  # D13
            coin = self.io.ask("Coin:")
            exchange = self.io.ask("Exchange:")
            pair = self.io.ask("Pair:")
            timestamp = self.io.ask("Pump_time:")

            write_list = [channel_id, session_id, coin, exchange, pair, timestamp]
            f.write("\t".join(write_list) + "\n")

    def run(self) -> None:
        """[orig: L124-L134]"""
        self.load_browsed()
        for root, dirs, files in os.walk(self.walk_dir):

            for file in files:
                if file.endswith(".pkl") == False:  # noqa: E712
                    continue
                channel_id = file.split("_")[0]
                pred_pump_session_list = load_legacy(self.walk_dir / (channel_id + "_session.pkl"))
                self.session_list_print(pred_pump_session_list, channel_id)


@register
class LabelSessionsStage:
    """Interactive CLI stage."""

    name = "label-sessions"

    def __init__(self, input_port: InputPort | None = None) -> None:
        self.input_port = input_port or StdInputPort()

    def run(self, cfg: Config) -> None:
        set_global_seed(cfg.seed)
        p = cfg.paths.session_labeler
        SessionLabeler(self.input_port, p.walk_dir, p.browsed_input, p.browsed_output, p.attack_output,
                       cfg.labeling.session_min_date).run()
