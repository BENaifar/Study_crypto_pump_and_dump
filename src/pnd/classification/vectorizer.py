"""TF-IDF factory. [orig: train_classifier.py:L61, predic_classifier.py:L29]"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer


def _split_tokenizer(x):
    """Same as the original ``lambda x: x.split()`` (named so it is picklable)."""
    return x.split()


def make_tfidf(tfidf_cfg) -> TfidfVectorizer:
    """``TfidfVectorizer(ngram_range=(1,3), min_df=5, tokenizer=..., max_features=20000, use_idf=True)``."""
    return TfidfVectorizer(ngram_range=tuple(tfidf_cfg["ngram_range"]), min_df=tfidf_cfg["min_df"],
                           tokenizer=_split_tokenizer, max_features=tfidf_cfg["max_features"],
                           use_idf=tfidf_cfg["use_idf"])
