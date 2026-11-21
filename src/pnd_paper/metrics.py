r"""Ranking metrics for target-coin prediction.

For event i with candidate set C_i, scores s(c) and target c*_i, the rank of the target is

    rho_i = 1 + #{c in C_i : s(c) > s(c*)} + 0.5 * #{c in C_i \ {c*} : s(c) = s(c*)}

(ties broken at random in expectation), HR@k = (1/N) sum_i 1{rho_i <= k} and MRR = (1/N) sum_i 1/rho_i.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from pnd_paper.settings import HR_KS


def target_ranks(df: pd.DataFrame, score: str = "score", group: str = "event_id", label: str = "label") -> pd.Series:
    """Rank of the target inside each event (events without a target are dropped)."""
    out = {}
    for eid, g in df.groupby(group, sort=False):
        pos = g[label].values == 1
        if not pos.any():
            continue
        s = g[score].values
        st = s[pos][0]
        others = s[~pos]
        out[eid] = 1.0 + (others > st).sum() + 0.5 * (others == st).sum()
    return pd.Series(out, name="rank")


def hit_ratios(ranks: pd.Series, ks=HR_KS) -> dict[str, float]:
    return {f"HR@{k}": float((ranks <= k).mean()) for k in ks}


def ranking_report(df: pd.DataFrame, score: str = "score", ks=HR_KS) -> dict[str, float]:
    """HR@k, MRR, AUC-ROC, AUC-PR (average precision) and the number of events."""
    ranks = target_ranks(df, score)
    rep = hit_ratios(ranks, ks)
    rep["MRR"] = float((1.0 / ranks).mean())
    rep["AUC"] = float(roc_auc_score(df["label"], df[score]))
    rep["AUC-PR"] = float(average_precision_score(df["label"], df[score]))
    rep["N_events"] = int(len(ranks))
    return rep


def random_baseline(df: pd.DataFrame, ks=HR_KS) -> dict[str, float]:
    """Expected HR@k of a uniformly random ranking: E[1{rho<=k}] = min(k, |C_i|)/|C_i|."""
    sizes = df.groupby("event_id").size()
    return {f"HR@{k}": float(np.mean(np.minimum(k, sizes) / sizes)) for k in ks}
