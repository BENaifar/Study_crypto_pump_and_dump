"""Hand-crafted message features + cleaned text -> ``preselect_sample`` TSV. [orig: Data/Telegram/message_fg.py]"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from pnd.common.io import ensure_parent, load_pickle
from pnd.common.seeding import set_global_seed
from pnd.config import Config
from pnd.pipeline.registry import register
from pnd.text.cleaning import remove_punctuation_url_emoji, remove_stopwords
from pnd.text.vocabulary import load_symbol_list

logger = logging.getLogger(__name__)


def message_feature_row(m, crypto_symbol_list, exchange_symbol_list, keyword_list, skip_tokens, on_time_minutes):
    """One TSV line for message ``m``. [orig: message_fg.py:L67-L103]"""
    sample_id = m["id"]
    channel_id = m["peer_id"]["channel_id"]
    date = m["date"]

    # weekday ?
    weekday = date.weekday()

    # near the full/half hour
    on_time = 0
    if date.minute in on_time_minutes:
        on_time = 1

    message = remove_punctuation_url_emoji(m["message"])
    message_wo_stop = remove_stopwords(message)

    word_list = message_wo_stop.split(" ")
    length = len(word_list)
    crypto_signal = 0
    exchange_signal = 0
    keyword_signal = 0
    for w in word_list:
        if w in skip_tokens:
            continue
        # QUIRK(Q05): three independent ifs, counted after stop-word removal [orig: L96-L101]
        if w in crypto_symbol_list:
            crypto_signal += 1
        if w in exchange_symbol_list:
            exchange_signal += 1
        if w in keyword_list:
            keyword_signal += 1

    # QUIRK(Q05): `message` may still contain tab characters, which corrupts the TSV [orig: L103]
    return "\t".join([str(channel_id), str(sample_id), message, message_wo_stop, str(date), str(weekday), str(on_time),
                      str(crypto_signal), str(exchange_signal), str(keyword_signal), str(length)]) + "\n"


@register
class MessageFeaturesStage:
    """``PreSelection/*.pkl`` -> ``./preselect_sample``. [orig: message_fg.py:L44-L107]"""

    name = "message-features"

    def run(self, cfg: Config) -> None:
        set_global_seed(cfg.seed)
        p = cfg.paths.message_features
        mf = cfg.message_features
        # D15: sets instead of lists for `in` tests (identical result, ~100x faster on 1M messages)
        crypto_symbol_list = set(load_symbol_list(p.crypto_symbol))
        exchange_symbol_list = set(load_symbol_list(p.exchange))
        keyword_list = set(mf.keywords)
        skip_tokens = set(cfg.keyword_filter.skip_tokens)
        on_time_minutes = set(mf.on_time_minutes)

        # QUIRK(Q04): written to ./preselect_sample, but the classifiers read PreSelection/preselect_sample [orig: L44]
        output = ensure_parent(Path(p.output))
        with open(output, "w", encoding="utf-8") as f_write:  # D7, D13

            for root, dirs, files in os.walk(p.walk_dir):

                cnt = 0
                for file in files:
                    cnt += 1
                    if file.endswith(".pkl") == False:  # noqa: E712
                        continue

                    # QUIRK(Q03): load_obj(file) had no directory prefix -> opened relative to the cwd [orig: L41, L64]
                    all_messages = load_pickle(Path(p.load_dir) / file)

                    for i in range(len(all_messages)):
                        m = all_messages[i]
                        arr = message_feature_row(m, crypto_symbol_list, exchange_symbol_list, keyword_list,
                                                  skip_tokens, on_time_minutes)
                        f_write.writelines(arr)

                        if i % 100 == 0:
                            logger.info(str(cnt) + "/" + str(len(files)) + " : " + str(i + 1) + "/" + str(len(all_messages)))
