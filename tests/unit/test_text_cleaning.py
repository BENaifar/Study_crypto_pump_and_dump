import re

import pytest
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from pnd.text import cleaning
from tests.conftest import ORIG_TELEGRAM, load_original_function, original_assignment

MESSAGES = [
    "PUMP Signal!!! Buy #XYZ now 🚀🚀 https://t.me/joinchat/abc?x=1&y=2 on Binance.",
    "Target: 50%\nHold   it, don't sell… ‘quoted’ ’x’ �",
    "Coin name: (ABC) [BTC] {pair} <b>bold</b> a/b\\c ~=_*&^%$#@",
    "",
]


@pytest.fixture(scope="module")
def originals():
    ns = {"re": re, "word_tokenize": word_tokenize, "stopwords": stopwords,
          "punctuations": original_assignment(ORIG_TELEGRAM / "message_fg.py", "punctuations")}
    return (load_original_function(ORIG_TELEGRAM / "message_fg.py", "remove_punctuation_url_emoji", ns),
            load_original_function(ORIG_TELEGRAM / "message_fg.py", "remove_stopwords", ns))


def test_punctuation_constant_is_byte_exact():
    assert cleaning.punctuations == original_assignment(ORIG_TELEGRAM / "message_fg.py", "punctuations")


@pytest.mark.parametrize("text", MESSAGES)
def test_matches_original_functions(originals, text):
    orig_clean, orig_stop = originals
    assert cleaning.remove_punctuation_url_emoji(text) == orig_clean(text)
    cleaned = orig_clean(text)
    assert cleaning.remove_stopwords(cleaned) == orig_stop(cleaned)


def test_hand_computed_examples():
    assert cleaning.remove_punctuation_url_emoji("Buy https://x.com/a now 🚀\nHOLD!") == "buy  now  hold"
    assert cleaning.remove_stopwords("we will pump the coin now") == "pump coin"
