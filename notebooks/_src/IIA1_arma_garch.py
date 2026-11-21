# %% [markdown]
# # II.A.1 — Stationarity tests and ARMA–GARCH models of calm-period returns
#
# **Objective (brief II.A.1).** For 10 pump targets in a calm period, (i) test stationarity and serial dependence
# with the ADF, KPSS and Ljung–Box tests, and (ii) fit an ARMA$(p,q)$–GARCH(1,1) model with Student-$t$ innovations:
# $$r_t=\mu+\sum_{i=1}^{p}\phi_i r_{t-i}+\varepsilon_t+\sum_{j=1}^{q}\theta_j\varepsilon_{t-j},\qquad
# \varepsilon_t=\sigma_t\eta_t,\qquad \sigma_t^2=\omega+\alpha\varepsilon_{t-1}^2+\beta\sigma_{t-1}^2,$$
# with $\eta_t$ i.i.d. standardised Student-$t_\nu$ ($\mathbb E\eta_t=0$, $\operatorname{Var}\eta_t=1$). These models
# describe "normal" behaviour; II.A.2 uses their standardised residuals $\hat\eta_t$ to detect pumps.
#
# **Calm period.** For a pump at $\tau$ we use the hourly log-returns (in %) on $[\tau-30\,\text{d},\ \tau-72\,\text{h}]$,
# the estimation window of I.3, which ends before the pre-pump drift.

# %%
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as st
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from arch import arch_model

from pnd_paper import settings, tsmodels
from pnd_paper.data import pd_log, binance, universe
from pnd_paper.report import Reporter, md, show, close, SERIES

warnings.filterwarnings("ignore")
settings.seed_everything()
R = Reporter("IIA1_arma_garch")

ev_all = pd_log.binance_btc_events()
panels = binance.build_hourly_panel(universe.universe(ev_all) + ["BTCUSDT"], universe.PANEL_START, universe.PANEL_END)
close_ = panels["close"]


def calm_window(e):
    t = universe.last_bar_before(e.timestamp)
    s = close_[e.symbol].loc[t - pd.Timedelta(days=30):t - pd.Timedelta(hours=72)]
    return s


# choose 10 distinct coins with complete calm windows (seeded draw among eligible events)
rng = np.random.default_rng(settings.SEED)
cands = []
for e in ev_all.drop_duplicates("symbol").itertuples():
    if e.symbol in close_.columns:
        s = calm_window(e)
        if len(s) > 600 and s.notna().mean() > 0.99 and (s.diff() != 0).mean() > 0.9:
            cands.append(e)
chosen = [cands[i] for i in sorted(rng.choice(len(cands), settings.SIZES.garch_targets, replace=False))]
series = {e.coin: 100 * np.log(calm_window(e)).diff().dropna() for e in chosen}
info = pd.DataFrame([{"coin": e.coin, "pump time": f"{e.timestamp:%Y-%m-%d %H:%M}", "calm window": f"{s.index[0]:%Y-%m-%d} → {s.index[-1]:%Y-%m-%d}",
                      "n": len(s), "mean (%)": s.mean(), "sd (%)": s.std(), "skew": st.skew(s), "excess kurtosis": st.kurtosis(s)}
                     for e, s in zip(chosen, series.values())]).set_index("coin")
R.table(info, "targets")
show(info)
md(f"{len(cands)} target coins have a complete, actively traded calm window; 10 are drawn at random (seed "
   f"{settings.SEED}). Every series has a large excess kurtosis, which motivates Student-$t$ innovations.")

# %% [markdown]
# ## 1. Tests
#
# * **ADF** (augmented Dickey–Fuller). Regression $\Delta y_t=c+\gamma y_{t-1}+\sum_{k=1}^{K}\delta_k\Delta y_{t-k}+u_t$;
#   $H_0:\gamma=0$ (unit root) against $H_1:\gamma<0$ (stationary). The statistic $\hat\gamma/\mathrm{se}(\hat\gamma)$
#   follows the Dickey–Fuller distribution; $K$ is chosen by AIC.
# * **KPSS**. $y_t=\xi_t+u_t$ with a random walk $\xi_t$ of variance $\sigma^2_\xi$; $H_0:\sigma^2_\xi=0$
#   (level-stationary). Statistic $\eta=T^{-2}\sum_t S_t^2/\hat\lambda^2$ with $S_t$ the partial sums of residuals and
#   $\hat\lambda^2$ a Newey–West long-run variance. ADF and KPSS have opposite null hypotheses, so stationarity is
#   supported when ADF rejects **and** KPSS does not.
# * **Ljung–Box**. $Q(m)=T(T+2)\sum_{k=1}^{m}\hat\rho_k^2/(T-k)\overset{H_0}{\sim}\chi^2_m$, $H_0$: no autocorrelation
#   up to lag $m$. Applied to $r_t$ (linear dependence, i.e. the ARMA part) and to $r_t^2$ (volatility clustering,
#   i.e. the GARCH part), with $m=24$ hours.
# * **ARCH-LM** (Engle). Regression of $\hat u_t^2$ on $m$ of its lags; $TR^2\overset{H_0}{\sim}\chi^2_m$, $H_0$: no
#   ARCH effect.

# %%
rows = []
for coin, r in series.items():
    lp = np.log(close_[[e for e in chosen if e.coin == coin][0].symbol].loc[r.index])
    adf_p = adfuller(lp, autolag="AIC")
    adf_r = adfuller(r, autolag="AIC")
    kp_p = kpss(lp, regression="c", nlags="auto")
    kp_r = kpss(r, regression="c", nlags="auto")
    lb = acorr_ljungbox(r, lags=[24])
    lb2 = acorr_ljungbox(r**2, lags=[24])
    arch_lm = het_arch(r - r.mean(), nlags=12)
    rows.append({"coin": coin, "ADF log-price": adf_p[0], "p ADF log-price": adf_p[1], "ADF r": adf_r[0], "p ADF r": adf_r[1],
                 "KPSS log-price": kp_p[0], "p KPSS log-price": kp_p[1], "KPSS r": kp_r[0], "p KPSS r": kp_r[1],
                 "Q(24) r": lb.lb_stat.iloc[0], "p Q r": lb.lb_pvalue.iloc[0], "Q(24) r^2": lb2.lb_stat.iloc[0],
                 "p Q r^2": lb2.lb_pvalue.iloc[0], "ARCH-LM(12)": arch_lm[0], "p ARCH": arch_lm[1]})
tests = pd.DataFrame(rows).set_index("coin")
R.table(tests, "stationarity_tests")
show(tests, "{:.3g}")
md("**Reading the table.** statsmodels truncates KPSS p-values to [0.01, 0.10] (values printed as 0.01 or 0.1 mean "
   "'at most' / 'at least').\n\n"
   f"- Log-prices: ADF rejects the unit root for {(tests['p ADF log-price'] < 0.05).sum()}/10 coins and KPSS rejects "
   f"level-stationarity for {(tests['p KPSS log-price'] < 0.05).sum()}/10.\n"
   f"- Returns: ADF rejects the unit root for {(tests['p ADF r'] < 0.05).sum()}/10 coins, KPSS rejects stationarity for "
   f"{(tests['p KPSS r'] < 0.05).sum()}/10.\n"
   f"- Ljung–Box on $r_t$ rejects 'no autocorrelation' for {(tests['p Q r'] < 0.05).sum()}/10 coins; on $r_t^2$ for "
   f"{(tests['p Q r^2'] < 0.05).sum()}/10; ARCH-LM detects ARCH effects for {(tests['p ARCH'] < 0.05).sum()}/10.")

# %% [markdown]
# ## 2. ARMA$(p,q)$–GARCH(1,1)–$t$ estimation
#
# **Estimation.** Parameters $\psi=(\mu,\phi,\theta,\omega,\alpha,\beta,\nu)$ maximise the conditional log-likelihood
# $$\ell(\psi)=\sum_{t}\Big[\log\Gamma\big(\tfrac{\nu+1}{2}\big)-\log\Gamma\big(\tfrac{\nu}{2}\big)-\tfrac12\log\big(\pi(\nu-2)\big)
# -\log\sigma_t-\tfrac{\nu+1}{2}\log\Big(1+\frac{\eta_t^2}{\nu-2}\Big)\Big],\quad \eta_t=\varepsilon_t/\sigma_t,$$
# under $\omega>0$, $\alpha,\beta\ge0$, $\alpha+\beta<1$ (covariance stationarity) and $\nu>2$. Standard errors come
# from the inverse numerical Hessian, $\widehat{\operatorname{Var}}(\hat\psi)=[-\nabla^2\ell(\hat\psi)]^{-1}$. The
# estimator is written from scratch (`pnd_paper.tsmodels`) because the `arch` package only offers AR means; we
# validate it against `arch` on the AR(1) special case.
#
# **Order selection.** $p,q\in\{0,1,2,3\}$ (16 models per coin), chosen by
# $\mathrm{BIC}=-2\ell(\hat\psi)+k\log n$.

# %%
coin0 = next(iter(series))
f_ours = tsmodels.fit_arma_garch(series[coin0].values, 1, 0)
f_arch = arch_model(series[coin0].values, mean="AR", lags=1, vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
val = pd.DataFrame({"pnd_paper": [f_ours.params[k] for k in ("mu", "phi1", "omega", "alpha", "beta", "nu")] + [f_ours.loglik],
                    "arch package": list(f_arch.params.values) + [f_arch.loglikelihood]},
                   index=["mu", "phi1", "omega", "alpha", "beta", "nu", "log-likelihood"])
R.table(val, "validation_vs_arch")
show(val)
md(f"Validation on {coin0}, AR(1)–GARCH(1,1)–t: both estimators reach the same log-likelihood up to "
   f"{abs(f_ours.loglik - f_arch.loglikelihood):.2f} (flat likelihood in the $(\\omega,\\alpha,\\beta)$ direction explains small "
   "parameter differences).")

# %%
fits, grids = {}, {}
for coin, r in series.items():
    best, grid = tsmodels.select_arma_garch(r.values, 3, 3, "bic")
    fits[coin] = best
    grids[coin] = pd.DataFrame(grid)
param_rows = []
for coin, f in fits.items():
    row = {"coin": coin, "order": f"ARMA({f.p},{f.q})"}
    for k in ("mu", "omega", "alpha", "beta", "nu"):
        row[k] = f.params[k]
        row[f"se({k})"] = f.se[k]
    row["alpha+beta"] = f.persistence
    row["half-life (h)"] = np.log(0.5) / np.log(f.persistence) if f.persistence < 1 else np.inf
    row["BIC"] = f.bic
    param_rows.append(row)
params = pd.DataFrame(param_rows).set_index("coin")
R.table(params, "garch_parameters")
show(params, "{:.4g}")
at_bound = int((params["alpha+beta"] > 0.999).sum())
interior = params.loc[params["alpha+beta"] <= 0.999]
md(f"Selected orders: {', '.join(f'{c}: ARMA({f.p},{f.q})' for c, f in fits.items())}.\n\n"
   f"Persistence: for **{at_bound}/10** coins the estimate sits on the stationarity boundary "
   f"$\\hat\\alpha+\\hat\\beta\\approx1$ (integrated-GARCH behaviour: a volatility shock does not decay within the "
   f"30-day window); for the other {len(interior)} coins $\\hat\\alpha+\\hat\\beta$ ranges from "
   f"{interior['alpha+beta'].min():.2f} to {interior['alpha+beta'].max():.2f} (half-lives "
   f"{interior['half-life (h)'].min():.0f}–{interior['half-life (h)'].max():.0f} h). Median "
   f"$\\hat\\nu={params['nu'].median():.1f}$, i.e. very heavy tails (the kurtosis of $t_\\nu$ is infinite for $\\nu\\le4$).")

# %%
grid_all = pd.concat({c: g for c, g in grids.items()}, names=["coin"]).reset_index(level=0)
best_counts = grid_all.loc[grid_all.groupby("coin").BIC.idxmin(), ["p", "q"]].value_counts().rename("coins").reset_index()
show(best_counts.set_index(["p", "q"]))

# %% [markdown]
# ## 3. Residual diagnostics
#
# If the model is adequate, $\hat\eta_t$ is i.i.d. with the fitted $t_\nu$ law: no autocorrelation in $\hat\eta_t$
# (Ljung–Box), no remaining ARCH effect in $\hat\eta_t^2$, and a Kolmogorov–Smirnov test of $\hat\eta_t$ against the
# standardised $t_{\hat\nu}$ should not reject (the KS p-value is approximate because $\nu$ is estimated).

# %%
diag = []
for coin, f in fits.items():
    lb = acorr_ljungbox(f.eta, lags=[24], model_df=f.p + f.q)
    lb2 = acorr_ljungbox(f.eta**2, lags=[24], model_df=2)
    nu = f.params["nu"]
    ks = st.kstest(f.eta, lambda x, nu=nu: st.t.cdf(x * np.sqrt(nu / (nu - 2)), nu))
    diag.append({"coin": coin, "Q(24) eta": lb.lb_stat.iloc[0], "p": lb.lb_pvalue.iloc[0], "Q(24) eta^2": lb2.lb_stat.iloc[0],
                 "p ": lb2.lb_pvalue.iloc[0], "KS vs t_nu": ks.statistic, "p KS": ks.pvalue})
diag = pd.DataFrame(diag).set_index("coin")
R.table(diag, "residual_diagnostics")
show(diag, "{:.3g}")
md(f"Ljung–Box on $\\hat\\eta_t$ does not reject at 5 % for {(diag['p'] >= 0.05).sum()}/10 coins, on $\\hat\\eta_t^2$ for "
   f"{(diag['p '] >= 0.05).sum()}/10 coins, and the KS test does not reject the $t_\\nu$ law for "
   f"{(diag['p KS'] >= 0.05).sum()}/10 coins.")

# %%
two = list(series)[:2]
fig, axes = plt.subplots(2, 2, figsize=(10, 5.2), gridspec_kw={"width_ratios": [2.2, 1]})
for i, coin in enumerate(two):
    r = series[coin]
    f = fits[coin]
    idx = r.index[-len(f.sigma):]
    axes[i, 0].plot(r.index, r.values, color="#b9b8b2", lw=0.8, label="$r_t$ (%)")
    axes[i, 0].plot(idx, 2 * f.sigma, color=SERIES[0], lw=1.5, label=r"$\pm2\hat\sigma_t$")
    axes[i, 0].plot(idx, -2 * f.sigma, color=SERIES[0], lw=1.5)
    axes[i, 0].set_title(f"{coin}: hourly returns and conditional volatility, ARMA({f.p},{f.q})-GARCH(1,1)-t")
    axes[i, 0].legend(loc="upper left")
    nu = f.params["nu"]
    q_th = st.t.ppf((np.arange(1, len(f.eta) + 1) - 0.5) / len(f.eta), nu) * np.sqrt((nu - 2) / nu)
    axes[i, 1].scatter(q_th, np.sort(f.eta), s=5, color=SERIES[0])
    lim = [q_th.min(), q_th.max()]
    axes[i, 1].plot(lim, lim, color=SERIES[1], lw=1)
    axes[i, 1].set_title(f"QQ plot of $\\hat\\eta_t$ vs $t_{{{nu:.1f}}}$")
    axes[i, 1].set_xlabel("theoretical quantile")
R.fig(fig, "garch_fit")
close(fig)

# %% [markdown]
# ## 4. Interpretation (QUICK run; no QUICK/FULL difference in this notebook)
#
# * **Prices are integrated, returns are stationary.** On log-prices ADF never rejects the unit root (0/10) and KPSS
#   rejects level-stationarity for 8/10 coins. On returns ADF always rejects the unit root (10/10) and KPSS never
#   rejects stationarity (0/10). The two tests agree, so hourly log-returns, not prices, are the object to model.
# * **Linear and volatility dependence.** Ljung–Box on $r_t$ rejects for 9/10 coins. This dependence is short-lived:
#   BIC selects low orders, mostly an MA(1) with a negative coefficient, the signature of bid–ask bounce on illiquid
#   pairs. Ljung–Box on $r_t^2$ (9/10) and ARCH-LM (8/10) show strong volatility clustering, which justifies the GARCH
#   layer.
# * **GARCH parameters.** For most coins persistence reaches the boundary $\alpha+\beta\approx1$: on a 30-day window
#   volatility behaves like an integrated GARCH, and the half-life of a volatility shock is longer than the window.
#   For the others it lies between 0.70 and 0.99. $\hat\nu\approx3$ means extremely heavy tails, typical of small-cap
#   coins quoted in BTC. A Gaussian model would flag many ordinary hours as anomalies, which is why II.A.2 uses
#   Student innovations.
# * **Adequacy.** After filtering, residuals are close to i.i.d. $t_\nu$. Ljung–Box on $\hat\eta_t$ does not reject for
#   9/10 coins and on $\hat\eta_t^2$ for 10/10, and the KS test does not reject the fitted $t_\nu$ law for any coin.
#   $\hat\eta_t$ is therefore a sound input for the sequential detector of II.A.2.
#
# **Limitations.** Ten coins are a small sample. Calm periods of illiquid coins contain hours with no trade (zero
# returns), which violate the continuous-innovation assumption; these hours were kept so as not to bias the
# volatility estimates.
