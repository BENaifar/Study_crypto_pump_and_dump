r"""ARMA(p,q)-GARCH(1,1) with standardised Student-t innovations, fitted by exact conditional maximum likelihood.

Model:
    r_t = mu + sum_{i=1}^p phi_i r_{t-i} + eps_t + sum_{j=1}^q theta_j eps_{t-j},
    eps_t = sigma_t eta_t,  sigma_t^2 = omega + alpha eps_{t-1}^2 + beta sigma_{t-1}^2,
    eta_t iid, standardised Student t with nu > 2 degrees of freedom (unit variance).

Log-likelihood (conditional on the first max(p,q) observations, sigma_0^2 = sample variance):
    l(psi) = sum_t [ log Gamma((nu+1)/2) - log Gamma(nu/2) - 0.5 log(pi (nu-2)) - log sigma_t
                     - (nu+1)/2 log(1 + eta_t^2/(nu-2)) ].
Standard errors come from the inverse of the numerical Hessian of -l at the optimum.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, special


@dataclass
class GarchFit:
    p: int
    q: int
    params: dict[str, float]
    se: dict[str, float]
    loglik: float
    n: int
    eta: np.ndarray = field(repr=False)
    sigma: np.ndarray = field(repr=False)
    converged: bool = True

    @property
    def k(self) -> int:
        return len(self.params)

    @property
    def aic(self) -> float:
        return -2 * self.loglik + 2 * self.k

    @property
    def bic(self) -> float:
        return -2 * self.loglik + self.k * np.log(self.n)

    @property
    def persistence(self) -> float:
        return self.params["alpha"] + self.params["beta"]


def _names(p: int, q: int) -> list[str]:
    return ["mu"] + [f"phi{i}" for i in range(1, p + 1)] + [f"theta{j}" for j in range(1, q + 1)] + ["omega", "alpha", "beta", "nu"]


def _filter(theta: np.ndarray, r: np.ndarray, p: int, q: int):
    mu = theta[0]
    phi = theta[1:1 + p]
    th = theta[1 + p:1 + p + q]
    omega, alpha, beta, nu = theta[1 + p + q:]
    n = len(r)
    m = max(p, q)
    eps = np.zeros(n)
    sig2 = np.empty(n)
    sig2[:] = np.var(r)
    for t in range(m, n):
        ar = 0.0
        for i in range(p):
            ar += phi[i] * r[t - 1 - i]
        ma = 0.0
        for j in range(q):
            ma += th[j] * eps[t - 1 - j]
        eps[t] = r[t] - mu - ar - ma
        if t > m:
            sig2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sig2[t - 1]
    return eps[m:], sig2[m:], nu


def negloglik(theta: np.ndarray, r: np.ndarray, p: int, q: int) -> float:
    eps, sig2, nu = _filter(theta, r, p, q)
    if np.any(sig2 <= 0) or not np.all(np.isfinite(sig2)):
        return 1e12
    eta2 = eps**2 / sig2
    ll = (special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
          - 0.5 * np.log(sig2) - (nu + 1) / 2 * np.log1p(eta2 / (nu - 2)))
    val = -ll.sum()
    return float(val) if np.isfinite(val) else 1e12


def _hessian(f, x, h=1e-4):
    k = len(x)
    H = np.empty((k, k))
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k)
            ej = np.zeros(k)
            ei[i] = h * max(1.0, abs(x[i]))
            ej[j] = h * max(1.0, abs(x[j]))
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * ei[i] * ej[j])
    return H


def fit_arma_garch(r, p: int = 1, q: int = 0) -> GarchFit:
    """Fit ARMA(p,q)-GARCH(1,1)-t by L-BFGS-B under stationarity-type bounds (|phi|,|theta|<1, alpha+beta<1 via bounds)."""
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    v = np.var(r)
    x0 = np.r_[np.mean(r), np.zeros(p), np.zeros(q), 0.1 * v, 0.08, 0.85, 6.0]
    bounds = ([(None, None)] + [(-0.99, 0.99)] * (p + q) + [(1e-8 * v, 10 * v), (1e-6, 0.5), (1e-6, 0.999), (2.05, 200.0)])
    cons_pen = lambda th: negloglik(th, r, p, q) + (1e6 * (th[-3] + th[-2] - 0.9999) ** 2 if th[-3] + th[-2] > 0.9999 else 0.0)  # noqa: E731
    res = optimize.minimize(cons_pen, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 2000})
    names = _names(p, q)
    try:
        cov = np.linalg.inv(_hessian(lambda th: negloglik(th, r, p, q), res.x))
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(len(names), np.nan)
    eps, sig2, _ = _filter(res.x, r, p, q)
    return GarchFit(p=p, q=q, params=dict(zip(names, res.x)), se=dict(zip(names, se)), loglik=-res.fun, n=len(eps),
                    eta=eps / np.sqrt(sig2), sigma=np.sqrt(sig2), converged=bool(res.success))


def select_arma_garch(r, max_p: int = 3, max_q: int = 3, criterion: str = "bic") -> tuple[GarchFit, list[dict]]:
    """Grid search over p, q <= max; returns the best fit and the table of criteria."""
    fits, table = [], []
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            f = fit_arma_garch(r, p, q)
            fits.append(f)
            table.append({"p": p, "q": q, "loglik": f.loglik, "AIC": f.aic, "BIC": f.bic, "converged": f.converged})
    best = min(fits, key=lambda f: getattr(f, criterion))
    return best, table


def simulate_t(nu: float, size, rng: np.random.Generator) -> np.ndarray:
    """Standardised Student-t draws (unit variance)."""
    return rng.standard_t(nu, size=size) * np.sqrt((nu - 2) / nu)


def filter_new(fit: GarchFit, r) -> tuple[np.ndarray, np.ndarray]:
    """Apply fitted parameters to a new series (e.g. the monitoring period): returns (eta_t, sigma_t)."""
    theta = np.array(list(fit.params.values()))
    eps, sig2, _ = _filter(theta, np.asarray(r, float), fit.p, fit.q)
    return eps / np.sqrt(sig2), np.sqrt(sig2)
