"""Statistical tools used across the notebooks (tests, corrections, resampling, formatting)."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


# ------------------------------------------------------------------ formatting
def fmt_p(p: float) -> str:
    """LaTeX p-value: p = 3.2\\times10^{-4}; values below 1e-300 are shown as an inequality."""
    if p is None or not np.isfinite(p):
        return r"p=\text{n/a}"
    if p < 1e-300:
        return r"p<10^{-300}"
    if p >= 1e-3:
        return f"p={p:.3f}"
    m, e = f"{p:.2e}".split("e")
    return rf"p={float(m):.2f}\times10^{{{int(e)}}}"


def decision(p: float, alpha: float = 0.05) -> str:
    return f"reject $H_0$ at $\\alpha={alpha:g}$" if p < alpha else f"do not reject $H_0$ at $\\alpha={alpha:g}$"


def stars(p: float) -> str:
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else ""


# ------------------------------------------------------------------ multiple testing
def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (controls the family-wise error rate)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def benjamini_hochberg(pvals) -> np.ndarray:
    """BH adjusted p-values (q-values; controls the false discovery rate)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


# ------------------------------------------------------------------ resampling
def monte_carlo_p(observed: float, null: np.ndarray, greater: bool = True) -> float:
    """p_hat = (1 + #{T^(b) >= T}) / (B + 1)  (never zero; Davison & Hinkley 1997)."""
    null = np.asarray(null, float)
    hits = (null >= observed).sum() if greater else (null <= observed).sum()
    return float((1 + hits) / (len(null) + 1))


def bootstrap_ci(x, stat=np.mean, n_boot: int = 5000, level: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of ``stat``."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    boots = np.array([stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)])
    a = (1 - level) / 2
    return float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


def paired_bootstrap_diff(a, b, n_boot: int = 10000, seed: int = 0) -> dict[str, float]:
    """Paired bootstrap of mean(a - b) over events: CI and two-sided p-value of H0: E[a - b] = 0."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    centred = boots - d.mean()
    p = float((1 + (np.abs(centred) >= abs(d.mean())).sum()) / (n_boot + 1))
    return {"diff": float(d.mean()), "ci_low": float(np.quantile(boots, 0.025)),
            "ci_high": float(np.quantile(boots, 0.975)), "p": p}


# ------------------------------------------------------------------ paired classifiers
def mcnemar(hit_a, hit_b) -> dict[str, float]:
    """Exact McNemar test on paired binary outcomes: n01 = #(a=0,b=1), n10 = #(a=1,b=0), H0: P(n10) = P(n01)."""
    a = np.asarray(hit_a, bool)
    b = np.asarray(hit_b, bool)
    n10 = int((a & ~b).sum())
    n01 = int((~a & b).sum())
    n = n10 + n01
    p = float(stats.binomtest(n10, n, 0.5).pvalue) if n > 0 else 1.0
    return {"n10": n10, "n01": n01, "p": p}


# ------------------------------------------------------------------ forecasts
def diebold_mariano(e1, e2, h: int = 1, loss: str = "abs") -> dict[str, float]:
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample correction.

    d_t = L(e1_t) - L(e2_t); DM = dbar / sqrt(LRV/T) with a Newey-West long-run variance using h-1 lags;
    HLN statistic DM* = DM * sqrt((T + 1 - 2h + h(h-1)/T) / T) compared with Student t(T-1).
    Negative values mean model 1 has the smaller loss.
    """
    e1 = np.asarray(e1, float)
    e2 = np.asarray(e2, float)
    d = (np.abs(e1) - np.abs(e2)) if loss == "abs" else (e1**2 - e2**2)
    T = len(d)
    dbar = d.mean()
    dc = d - dbar
    lrv = dc @ dc / T
    for k in range(1, h):
        w = 1 - k / h
        lrv += 2 * w * (dc[k:] @ dc[:-k]) / T
    dm = dbar / math.sqrt(lrv / T) if lrv > 0 else np.nan
    corr = math.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_hln = dm * corr
    p = float(2 * stats.t.sf(abs(dm_hln), df=T - 1))
    return {"mean_loss_diff": float(dbar), "DM": float(dm), "DM_HLN": float(dm_hln), "p": p, "T": T}


# ------------------------------------------------------------------ distributions
def gini(x) -> float:
    """Gini coefficient G = sum_i (2i - n - 1) x_(i) / (n sum x)."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    return float(((2 * np.arange(1, n + 1) - n - 1) @ x) / (n * x.sum()))
