"""Manual labeling of pre-selected messages. [orig: Data/Telegram/pump_message_label.py]"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytz

from pnd.common.console import InputPort, StdInputPort
from pnd.common.io import load_pickle
from pnd.common.seeding import set_global_seed
from pnd.config import Config
from pnd.pipeline.registry import register


class MessageLabeler:
    """Q12: show each unlabeled message after the min date and append ``label channel id`` to label.txt."""

    def __init__(self, input_port: InputPort, walk_dir: Path, load_dir: Path, label_file: Path,
                 min_date: str = "2021-01-01") -> None:
        self.io = input_port
        self.walk_dir = Path(walk_dir)
        self.load_dir = Path(load_dir)
        self.label_file = Path(label_file)
        y, m, d = (int(x) for x in min_date.split("-"))
        self.min_date = (y, m, d)

    def run(self) -> None:
        with open(self.label_file, "r", encoding="utf-8") as f:  # D13
            labeled_samples = f.readlines()

        # QUIRK(Q12): id = "-".join(line[:-1].split(" ")[1:]) [orig: L20]
        labeled_samples = list(map(lambda x: "-".join(x[:-1].split(" ")[1:]), labeled_samples))

        with open(self.label_file, "a", encoding="utf-8") as f:  # D7, D13

            for root, dirs, files in os.walk(self.walk_dir):

                for file in files:

                    # QUIRK(Q12): min date with pytz UTC via .replace(tzinfo=...) [orig: L28]
                    min_datetime = datetime(*self.min_date).replace(tzinfo=pytz.timezone('UTC'))
                    if file.endswith(".pkl"):
                        all_messages = load_pickle(self.load_dir / file)
                        all_messages.reverse()
                        for i in range(len(all_messages)):
                            m = all_messages[i]
                            if m["date"] <= min_datetime:
                                continue
                            if str(m["peer_id"]["channel_id"]) + "-" + str(m["id"]) in labeled_samples:
                                continue

                            self.io.show(str(i + 1) + "/" + str(len(all_messages)) + " id:" + str(m["id"])
                                         + "-----" + str(m["date"]) + "--------------")
                            self.io.show("")
                            self.io.show(m["message"])
                            label = self.io.ask("")
                            if label == "##":
                                break  # jump to the next channel
                            elif label == "?" or label == "1" or label == "2":
                                pass  # nothing
                            elif label == "-1":
                                # back to the next message and relabel
                                pass
                            else:
                                label = "0"

                            f.write(str(label) + " " + str(m["peer_id"]["channel_id"]) + " " + str(m["id"]) + "\n")


@register
class LabelMessagesStage:
    """Interactive CLI stage."""

    name = "label-messages"

    def __init__(self, input_port: InputPort | None = None) -> None:
        self.input_port = input_port or StdInputPort()

    def run(self, cfg: Config) -> None:
        set_global_seed(cfg.seed)
        p = cfg.paths.label_messages
        MessageLabeler(self.input_port, p.walk_dir, p.load_dir, p.label_file, cfg.labeling.message_min_date).run()
