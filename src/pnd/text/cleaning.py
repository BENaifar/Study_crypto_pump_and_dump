"""Message text cleaning. [orig: Data/Telegram/message_fg.py:L1-L38]"""

from __future__ import annotations

import logging
import re

import emoji  # noqa: F401  (imported by the original module; kept per R1)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from urlextract import URLExtract  # noqa: F401  (imported by the original module; kept per R1)

from pnd.config import Config
from pnd.pipeline.registry import register

logger = logging.getLogger(__name__)

# Copied byte-exact from message_fg.py:L11
punctuations = '''’'!()-[]{};:'"\,<>./?@#$%^&*_~�=‘'''


def remove_punctuation_url_emoji(d):
    """Q06. [orig: message_fg.py:L13-L30]"""
    d = d.lower()

    def filter_emoji(desstr, restr=''):
        try:
            co = re.compile(u'[\U00010000-\U0010ffff]')
        except re.error:
            co = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
        return co.sub(restr, desstr)

    d = filter_emoji(d)
    d = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', d, flags=re.MULTILINE)  # This line is for removing url
    review = d.replace('\n', ' ')
    no_punct = ""
    for char in review:
        if char not in punctuations:
            no_punct = no_punct + char
    return no_punct


_ENGLISH_STOPWORDS: frozenset[str] | None = None


def _english_stopwords() -> frozenset[str]:
    """D11/D15: the NLTK English stop-word list, loaded once (membership result identical to the list)."""
    global _ENGLISH_STOPWORDS
    if _ENGLISH_STOPWORDS is None:
        _ENGLISH_STOPWORDS = frozenset(stopwords.words('english'))
    return _ENGLISH_STOPWORDS


def remove_stopwords(d):
    """Q07 (+D11/D15: stop-words loaded once and looked up in a set, same result). [orig: message_fg.py:L32-L38]"""
    text_tokens = word_tokenize(d.lower())
    english_stopwords = _english_stopwords()
    tokens_without_sw = [word for word in text_tokens if not word in english_stopwords]  # noqa: E713
    ls = " ".join(tokens_without_sw)
    return ls


@register
class SetupNltkStage:
    """Download the NLTK corpora used by :func:`remove_stopwords`."""

    name = "setup-nltk"

    def run(self, cfg: Config) -> None:
        import nltk

        for package in ("stopwords", "punkt", "punkt_tab"):
            ok = nltk.download(package, quiet=True)
            logger.info("nltk.download(%s) -> %s", package, ok)
