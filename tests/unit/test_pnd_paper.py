import numpy as np
import pandas as pd

from pnd_paper import metrics, stats
from pnd_paper.data.universe import last_bar_before


def _toy():
    return pd.DataFrame({
        "event_id": [0, 0, 0, 1, 1, 1, 1],
        "label":    [0, 1, 0, 0, 0, 0, 1],
        "score":    [0.9, 0.5, 0.1, 0.4, 0.4, 0.2, 0.4],
    })


def test_target_ranks_and_hr():
    ranks = metrics.target_ranks(_toy())
    assert ranks[0] == 2.0            # one candidate above the target
    assert ranks[1] == 2.0            # two ties -> 1 + 0 + 0.5 * 2
    hr = metrics.hit_ratios(ranks, ks=(1, 3))
    assert hr == {"HR@1": 0.0, "HR@3": 1.0}
    assert metrics.random_baseline(_toy(), ks=(1,))["HR@1"] == np.mean([1 / 3, 1 / 4])


def test_monte_carlo_p_never_zero():
    assert stats.monte_carlo_p(10.0, np.zeros(999)) == 1 / 1000
    assert stats.monte_carlo_p(0.0, np.zeros(9)) == 1.0


def test_mcnemar_and_holm_bh():
    r = stats.mcnemar([1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 1])
    assert (r["n10"], r["n01"]) == (4, 1) and 0 < r["p"] < 1
    assert np.allclose(stats.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert np.allclose(stats.benjamini_hochberg([0.01, 0.04, 0.03]), [0.03, 0.04, 0.04])


def test_diebold_mariano_sign():
    rng = np.random.default_rng(0)
    e1 = rng.normal(0, 1, 500)
    e2 = rng.normal(0, 2, 500)
    r = stats.diebold_mariano(e1, e2, h=1)
    assert r["DM_HLN"] < 0 and r["p"] < 1e-6


def test_gini_and_fmt():
    assert stats.gini([1, 1, 1, 1]) == 0.0
    assert abs(stats.gini([0, 0, 0, 1]) - 0.75) < 1e-12
    assert stats.fmt_p(0.00032) == r"p=3.20\times10^{-4}"


def test_information_horizon():
    tau = pd.Timestamp("2021-03-01 12:01", tz="UTC")
    assert last_bar_before(tau) == pd.Timestamp("2021-03-01 10:00", tz="UTC")   # bar 10:00-11:00 closes before 11:01
