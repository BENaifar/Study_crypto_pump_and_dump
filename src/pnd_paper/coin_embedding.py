r"""Coin-symbol word embeddings learned from the authors' Telegram corpus (cold-start variant SNN_S of [A1]).

[A1] pre-trains SkipGram on Telegram messages and replaces the learned coin-id embedding by the word vector of the
coin symbol. We use the closed-form equivalent of SkipGram with negative sampling (Levy & Goldberg, 2014): the
shifted positive PMI matrix of (coin, context word) co-occurrences factorised by a truncated SVD,

    PPMI(c, w) = max(0, log( #(c,w) * sum #(.,.) / (#(c,.) * #(.,w)^0.75 ...) )),   E = U_k Sigma_k^{1/2}.

Only messages posted **before** ``cutoff`` are used, so embeddings never contain information from the evaluation
periods. The result is cached in ``datasets/paper_cache/coin_embedding_<cutoff>.npz``.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import svds

from pnd_paper import settings
from pnd_paper.data.telegram import iter_messages

TOKEN = re.compile(r"[a-z][a-z0-9]{1,9}")


def build_coin_embedding(coins: list[str], cutoff: pd.Timestamp = settings.TRAIN_END, dim: int = 16, window: int = 5,
                         n_context: int = 20000, rebuild: bool = False) -> tuple[dict[str, np.ndarray], dict]:
    """Return ({COIN: vector}, info). Coins never mentioned before ``cutoff`` are absent from the dict."""
    path = settings.CACHE / f"coin_embedding_{cutoff:%Y%m%d}_{dim}.npz"
    if path.is_file() and not rebuild:
        z = np.load(path, allow_pickle=True)
        return dict(zip(z["coins"], z["vectors"])), z["info"].item()
    targets = {c.lower() for c in coins}
    cutoff_dt = cutoff.to_pydatetime()

    def messages():
        for _, date, text in iter_messages():
            if date is not None and date < cutoff_dt:
                yield TOKEN.findall(text.lower())

    word_freq: Counter = Counter()
    for toks in messages():                        # pass 1: context vocabulary
        word_freq.update(toks)
    context = [w for w, _ in word_freq.most_common(n_context)]
    cidx = {w: i for i, w in enumerate(context)}
    tlist = sorted(targets)
    tidx = {w: i for i, w in enumerate(tlist)}
    C = sparse.csr_matrix((len(tlist), len(context)))
    rows, cols = [], []
    n_msg = 0
    for toks in messages():                        # pass 2: (coin, context) co-occurrences
        hit = False
        for i, t in enumerate(toks):
            if t not in tidx:
                continue
            hit = True
            for j in range(max(0, i - window), min(len(toks), i + window + 1)):
                if j != i and toks[j] in cidx:
                    rows.append(tidx[t])
                    cols.append(cidx[toks[j]])
        n_msg += hit
        if len(rows) > 5_000_000:
            C = C + sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=C.shape)
            rows, cols = [], []
    C = C + sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=C.shape)
    C.sum_duplicates()
    total = C.sum()
    pc = np.asarray(C.sum(1)).ravel()
    pw = np.asarray(C.sum(0)).ravel() ** 0.75
    pw = pw / pw.sum()
    C = C.tocoo()
    pmi = np.log(C.data * 1.0 / total / (pc[C.row] / total) / pw[C.col])
    keep = pmi > 0
    P = sparse.csr_matrix((pmi[keep], (C.row[keep], C.col[keep])), shape=C.shape)
    mentioned = pc > 0
    k = min(dim, min(P.shape) - 1)
    U, s, _ = svds(P, k=k)
    E = U * np.sqrt(s)
    vecs = {tlist[i].upper(): E[i] for i in range(len(tlist)) if mentioned[i]}
    info = {"messages_with_coin": n_msg, "context_vocabulary": len(context), "coins_mentioned": int(mentioned.sum()),
            "coins_requested": len(tlist), "cutoff": str(cutoff), "dim": k}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, coins=np.array(list(vecs)), vectors=np.array(list(vecs.values())), info=np.array(info, dtype=object))
    return vecs, info
