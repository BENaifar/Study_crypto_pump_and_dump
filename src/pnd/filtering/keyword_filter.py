"""Keyword pre-selection of Telegram messages. [orig: Data/Telegram/keyword_filter.py]"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from pnd.common.io import load_pickle, save_pickle
from pnd.common.seeding import set_global_seed
from pnd.config import Config
from pnd.pipeline.registry import register
from pnd.text.vocabulary import load_symbol_list

logger = logging.getLogger(__name__)


class KeywordFilter:
    """Holds the vocabularies and the match logs that were module globals in the original."""

    def __init__(self, crypto_symbol_list: list[str], exchange_symbol_list: list[str],
                 keyword_list: list[str], skip_tokens: list[str]) -> None:
        # D15: sets for `in` tests (identical result, much faster)
        self.crypto_symbol_list = set(crypto_symbol_list)
        self.exchange_symbol_list = set(exchange_symbol_list)
        self.keyword_list = set(keyword_list)
        self.skip_tokens = set(skip_tokens)
        self.crypto_match: list[str] = []
        self.exchange_match: list[str] = []
        self.keyword_match: list[str] = []

    def pre_select(self, message) -> bool:
        """Q01. [orig: keyword_filter.py:L22-L39]"""
        # SYMBOL, EXCHANGE, OR KEYWORDS
        words = re.split('[ ,.?!/\n;#$]', message["message"].lower())
        for w in words:
            # QUIRK(Q01): "pump"/"buy" are in the skip-list, so only "target" can match as keyword [orig: L27]
            if w in self.skip_tokens:
                continue
            # QUIRK(Q01): returns at the FIRST match, crypto -> exchange -> keyword [orig: L29-L37]
            if w in self.crypto_symbol_list:
                self.crypto_match.append(w)
                return True
            if w in self.exchange_symbol_list:
                self.exchange_match.append(w)
                return True
            if w in self.keyword_list:
                self.keyword_match.append(w)
                return True

        return False


@register
class FilterKeywordsStage:
    """``raw/*.pkl`` -> ``PreSelection/*.pkl``. [orig: keyword_filter.py:L41-L69]"""

    name = "filter-keywords"

    def run(self, cfg: Config) -> None:
        set_global_seed(cfg.seed)
        p = cfg.paths.keyword_filter
        kw_filter = KeywordFilter(
            crypto_symbol_list=load_symbol_list(cfg.paths.telegram.crypto_symbol),
            exchange_symbol_list=load_symbol_list(cfg.paths.telegram.exchange),
            keyword_list=list(cfg.keyword_filter.keywords),
            skip_tokens=list(cfg.keyword_filter.skip_tokens),
        )
        self.filter_dir(kw_filter, Path(p.walk_dir), Path(p.load_dir), Path(p.output_dir))

    @staticmethod
    def filter_dir(kw_filter: KeywordFilter, walk_dir: Path, load_dir: Path, output_dir: Path) -> None:
        for root, dirs, files in os.walk(walk_dir):

            for file in files:
                message_list = []
                if file.endswith(".pkl"):
                    all_messgaes = load_pickle(load_dir / file)

                    for m in all_messgaes:
                        # QUIRK(Q02): only channel posts (from_id None) of type "Message" [orig: L60]
                        if m["from_id"] != None or m["_"] != "Message":  # noqa: E711
                            continue

                        if kw_filter.pre_select(m):
                            message_list.append(m)

                # QUIRK(Q02): the length check sits outside the ".pkl" branch [orig: L66-L67]
                if (len(message_list) > 0):
                    save_pickle(message_list, output_dir / (file[:-4] + ".pkl"))

        logger.info("pause")
