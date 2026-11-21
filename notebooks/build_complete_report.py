"""Assemble notebooks/00_complete_report.ipynb from the executed step notebooks.

The step notebooks are copied with their outputs (no re-execution, so the report is consistent with the step PDFs);
headings are demoted by one level. Three chapters are added: an introduction, a "Validation against [A1]" chapter
computed from results/*/tables (executed here, it is the only part that reads other notebooks' outputs, by design),
and a conclusion with limitations, ethics and references.
"""

from __future__ import annotations

import re
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
OUT = HERE / "00_complete_report.ipynb"

PARTS = [
    ("Part I — Expertise of the phenomenon", ["I0_data_acquisition", "I1_literature_synthesis", "I2_descriptive_statistics",
                                              "I3_event_study", "I4_intra_channel_homogeneity", "I5_coordination_network"]),
    ("Part II — Modelling", ["IIA1_arma_garch", "IIA2_cusum_isolation_forest", "IIB_machine_learning", "IIC_deep_learning",
                             "IID_bonus_btc_forecast"]),
]

INTRO = r"""# Detection and prediction of pump-and-dump schemes on cryptocurrency markets

**Time series, machine learning and deep learning — complete report**

*Module Projet mathématiques, spécialité Infini, 2026–2027. Reproduction and extension of Hu et al. (SIGMOD 2023) [A1]
with the method of Pacheco et al. (ICWSM 2021) [A2], on the data of the authors' repository [3].*

---

## Introduction

A **pump-and-dump** (P&D) is a market manipulation: organisers discreetly buy an illiquid asset, push its price up
through a coordinated campaign (the *pump*), then sell at the top to participants who arrived too late (the *dump*).
[A1] collects 709 such events organised by Telegram channels between 2019 and 2022 and shows that the target coin can
be predicted one hour before the announcement.

The project plays the role of a **market-surveillance service**. Its objectives are to (O1) understand the mechanisms
and signatures of a P&D, (O2) read and reproduce published results, (O3) run an event study, (O4) model financial
series and detect anomalies, (O5) evaluate a highly imbalanced ranking problem without temporal leakage, and (O6)
compare machine-learning and deep-learning models.

**Notation.** $P_{c,t}$ is the price, $V_{c,t}$ the volume and $r_{c,t}=\log P_{c,t}-\log P_{c,t-1}$ the log-return of
coin $c$. An event is $e=(h,c^\star,\tau)$ with channel $h$, target $c^\star$ and pump time $\tau$; $\mathcal H_h(\tau)$ is
the channel's history. For $N$ test events, with $\rho_i$ the rank of the true target among about 200 candidates,
$\mathrm{HR}@k=\frac1N\sum_{i=1}^N\mathbf 1\{\rho_i\le k\}$.

**Protocol.** Chronological split identical to [A1] (train before 2021-01-19, validation until 2021-05-10, test
afterwards); every variable of event $i$ is computable at $\tau_i-1$ h; hyperparameters are chosen on validation and the
test set is used once; results are averaged over several random seeds with their standard deviation.

**How this report was produced.** Each section below is an independent notebook (`notebooks/`) that loads the raw
data itself; the market data are re-downloaded from Binance's public archive. The code is hidden in this PDF and
visible in the per-step PDFs. Numbers come from the **QUICK** configuration (reduced numbers of seeds, permutation
draws and tuning trials) unless the run was made with `PND_QUICK=0`; every statement printed next to a result is
generated from the computed values.

**Ethics.** The project serves detection and investor protection; using the models to take part in a P&D is
forbidden. Channel identifiers are anonymised (`Ch-001`, ...). The repository's data have no licence and are not
redistributed.
"""

VALIDATION_MD = r"""# Validation against [A1]

This chapter gathers, from the tables saved by each step, every quantity that [A1] reports and that this project
re-estimated. A claim is marked *reproduced* when our estimate is within 10 % (relative) of the paper's, *close* within
25 %, and *differs* otherwise; for hit ratios the tolerance is ±0.05 in absolute value."""

VALIDATION_CODE = r'''
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from pnd_paper import settings
from pnd_paper.data import pd_log

RES = settings.RESULTS
def tab(nb, name, **kw):
    p = RES / nb / "tables" / f"{name}.csv"
    return pd.read_csv(p, **kw) if p.is_file() else None

def verdict(ours, theirs, abs_tol=None):
    if ours is None or theirs is None or not np.isfinite(ours) or not np.isfinite(theirs):
        return "n/a"
    if abs_tol is not None:
        d = abs(ours - theirs)
        return "reproduced" if d <= abs_tol else "close" if d <= 2 * abs_tol else "differs"
    rel = abs(ours - theirs) / abs(theirs)
    return "reproduced" if rel <= 0.10 else "close" if rel <= 0.25 else "differs"

F = settings.PAPER_FACTS
rows = []
log = pd_log.load_pd_log()
rows.append(("samples in the P&D log", F["samples"], len(log), "I0"))
evt = tab("I0_data_acquisition", "events_by_tolerance", index_col=0)
if evt is not None:
    rows.append(("P&D events (relayed rows merged, 1 h tolerance)", F["events"], float(evt.loc["1 hour", "events"]), "I0"))
    rows.append(("Binance events (1 h tolerance)", F["binance_events"], float(evt.loc["1 hour", "Binance events"]), "I0"))
rows.append(("channels", F["channels"], log.channel_id.nunique(), "I0"))
rows.append(("coins", F["coins"], log.coin.nunique(), "I0"))
pc = log.coin.value_counts()
rows.append(("re-pump probability (share of coins pumped more than once)", F["repump_probability"], float((pc > 1).mean()), "I2"))
ds = tab("IIB_machine_learning", "dataset_sizes", index_col=0)
if ds is not None:
    for s, key in (("train", "train_pos"), ("valid", "valid_pos"), ("test", "test_pos")):
        rows.append((f"positive samples, {s} split", F[key], float(ds.loc[s, "events"]), "IIB"))
    rows.append(("positive ratio", F["positive_ratio"], float(ds.loc[["train", "valid", "test"], "positives"].sum() / ds.loc[["train", "valid", "test"], "rows"].sum()), "IIB"))
x60 = tab("I3_event_study", "x60_vs_paper", index_col=0)
if x60 is not None:
    rows.append(("mean pre-pump return, x = 60 h (raw)", F["pre_pump_return_x60"], float(x60.iloc[0]["mean"]), "I3"))
facts = pd.DataFrame(rows, columns=["quantity", "[A1]", "this study", "notebook"])
facts["verdict"] = [verdict(o, t) for o, t in zip(facts["this study"], facts["[A1]"])]
display(facts.style.format({"[A1]": "{:.4g}", "this study": "{:.4g}"}).hide(axis="index"))

hr_rows = []
for nb, names in (("IIB_machine_learning", {"LR": "LR", "RF": "RF"}),
                  ("IIC_deep_learning", {"DNN (mean)": "DNN", "LSTM": "LSTM", "GRU": "GRU", "TCN": "TCN", "SNN": "SNN", "SNN_S": "SNNS"})):
    m = tab(nb, "test_mean", index_col=0)
    s = tab(nb, "test_std", index_col=0)
    if m is None:
        continue
    for ours, theirs in names.items():
        if ours not in m.index:
            continue
        for k in (1, 10, 30):
            hr_rows.append({"model": ours, "metric": f"HR@{k}", "[A1]": settings.PAPER_TABLE5.loc[f"HR@{k}", theirs],
                            "ours (mean)": m.loc[ours, f"HR@{k}"], "ours (sd)": s.loc[ours, f"HR@{k}"]})
for nb in ("IIB_machine_learning",):
    m = tab(nb, "test_mean", index_col=0)
    if m is not None:
        for extra in [i for i in m.index if i.startswith("LightGBM") or i == "LR-softmax"]:
            for k in (1, 10, 30):
                hr_rows.append({"model": extra, "metric": f"HR@{k}", "[A1]": np.nan, "ours (mean)": m.loc[extra, f"HR@{k}"],
                                "ours (sd)": tab(nb, "test_std", index_col=0).loc[extra, f"HR@{k}"]})
hr = pd.DataFrame(hr_rows)
if len(hr):
    hr["verdict"] = [verdict(o, t, abs_tol=0.05) for o, t in zip(hr["ours (mean)"], hr["[A1]"])]
    wide = hr.pivot_table(index="model", columns="metric", values=["[A1]", "ours (mean)"], sort=False)
    display(Markdown("**Test-set hit ratios: this study versus [A1] Tables 5–6** (ours: mean over seeds)."))
    display(wide.style.format("{:.3f}", na_rep="—"))
    best = hr[hr.metric == "HR@10"].sort_values("ours (mean)", ascending=False).iloc[0]
    display(Markdown(f"Best HR@10 in this study: **{best['model']}** ({best['ours (mean)']:.3f}); best in [A1]: SNN_S (0.623). "
                     f"Verdicts on the hit ratios: " + ", ".join(f"{k}: {v}" for k, v in hr.verdict.value_counts().items()) + "."))
'''

CONCLUSION = r"""# Conclusion

**What was reproduced.** The data match the description of the brief and of [A1]: 1 335 samples, 108 channels,
the Binance/BTC modelling population and almost exactly the paper's 648 / 100 / 200 split. The event study confirms a
significant pre-pump drift that peaks around 60 hours before the announcement, the central empirical fact behind
[A1]'s claim that targets are predictable, although our average return at $x=60$ is lower than the paper's 9.5 %.
Intra-channel homogeneity, described only qualitatively in [A1], passes a permutation test; the coordination network
recovers groups of channels that relay each other.

**Prediction.** The validation chapter compares our hit ratios with the paper's. Ranking the target among about 200
candidates one hour ahead is possible far beyond chance with every model; channel-history and market features carry
most of the signal (SHAP and ablations, II.B). Sequence models (II.C) and gradient boosting are the strongest models,
and the statistical tests indicate whether their differences are significant on 199 test events, a small sample on
which differences of a few points are rarely conclusive.

**Detection.** A CUSUM on GARCH residuals raises an alarm before the announcement for a sizeable share of pumps, with a
lead of only a few hours; volume-based detectors need a calibration that respects serial dependence or they flood
the analyst with false alarms.

## Limitations

* Hourly data blur the last hour before the pump; minute data were used only for the anatomy of 20 pumps.
* The QUICK configuration uses fewer seeds, permutations and tuning trials than the brief requires; the full run
  (`PND_QUICK=0 make all`) tightens confidence intervals but uses the same code.
* CoinGecko statistics are the authors' snapshots (targets and candidates from two files); missingness was checked
  for label leakage but cannot be re-downloaded without an API key.
* The pre-trained embedding shipped by the authors could not be mapped to coins; SNN$_S$ uses embeddings re-learned
  from the authors' Telegram corpus, restricted to messages before the test period.

## Ethics and the cost of false positives

A surveillance system acts on its alarms: each alarm triggers a review, possibly a suspicious-order report (STOR, Art.
92 MiCA) or trading restrictions. **False positives** have real costs: analysts' time, the reputation of coins and
projects wrongly suspected, delayed orders of honest investors, and alarm fatigue that makes true alarms easier to
miss. A ranking model with HR@10 around 0.6 still points to nine innocent coins for every target in its top ten, so
its output must support a human investigation, never trigger sanctions automatically. Conversely, publishing the
ranking of likely targets would itself be market-moving and could be exploited by manipulators; the models and their
outputs must remain internal to the surveillance function. Channel identities are anonymised throughout.

## Team contributions (annex)

| Role | Part I | Part II |
|---|---|---|
| E1 Data and repository | repository structure, Binance and CoinGecko data, I.2 | ranking dataset (II.B.1), reproducibility (`make all`) |
| E2 Bibliography and report | I.1, regulation (MAR, MiCA) | report integration, ethics, bonus II.D |
| E3 Econometrics | I.3 event study | II.A.1 ARMA–GARCH |
| E4 Networks and tests | I.5 coordination network, null model | II.A.2 CUSUM, Isolation Forest, final significance tests |
| E5 Machine learning | I.4 intra-channel homogeneity | II.B LR, RF, LightGBM, imbalance, SHAP |
| E6 Deep learning | repository SNN (TensorFlow 1) | II.C LSTM, GRU, TCN, SNN in PyTorch, ablations |

## References

1. S. Hu, Z. Zhang, S. Lu, B. He, Z. Li. Sequence-based target coin prediction for cryptocurrency pump-and-dump. *Proc.
   ACM Manag. Data* 1(1), art. 6, 2023. arXiv:2204.12929.
2. D. Pacheco, P.-M. Hui, C. Torres-Lugo, B. T. Truong, A. Flammini, F. Menczer. Uncovering coordinated networks on
   social media: methods and case studies. *ICWSM* 15:455–466, 2021.
3. S. Hu et al. Pump-and-Dump-Detection-on-Cryptocurrency, GitHub repository.
4. J. Xu, B. Livshits. The anatomy of a cryptocurrency pump-and-dump scheme. *USENIX Security*, 2019.
5. T. Bollerslev. Generalized autoregressive conditional heteroskedasticity. *J. Econometrics* 31(3):307–327, 1986.
6. V. D. Blondel et al. Fast unfolding of communities in large networks. *J. Stat. Mech.*, P10008, 2008.
7. E. S. Page. Continuous inspection schemes. *Biometrika* 41:100–115, 1954.
8. F. T. Liu, K. M. Ting, Z.-H. Zhou. Isolation forest. *ICDM*, 2008.
9. S. M. Lundberg, S.-I. Lee. A unified approach to interpreting model predictions. *NeurIPS*, 2017.
10. F. X. Diebold, R. S. Mariano. Comparing predictive accuracy. *J. Business & Economic Statistics* 13:253–263, 1995.
11. C. J. Hutto, E. Gilbert. VADER: a parsimonious rule-based model for sentiment analysis of social media text.
    *ICWSM*, 2014.
12. Regulation (EU) No 596/2014 (MAR); Regulation (EU) 2023/1114 (MiCA); Commission Delegated Regulation (EU) 2016/522.
"""


def demote(md: str) -> str:
    return re.sub(r"^(#{1,5}) ", lambda m: "#" + m.group(1) + " ", md, flags=re.M)


def build() -> Path:
    nb = nbformat.v4.new_notebook()
    cells = [nbformat.v4.new_markdown_cell(INTRO)]
    for title, steps in PARTS:
        cells.append(nbformat.v4.new_markdown_cell(f"# {title}"))
        for name in steps:
            path = HERE / f"{name}.ipynb"
            if not path.is_file():
                cells.append(nbformat.v4.new_markdown_cell(f"## {name}\n\n*(notebook not executed)*"))
                continue
            step = nbformat.read(path, as_version=4)
            for c in step.cells:
                c = nbformat.from_dict(c)
                if c.cell_type == "markdown":
                    c.source = demote(c.source)
                c.pop("id", None)
                cells.append(c)
    val = nbformat.v4.new_notebook()
    val.cells = [nbformat.v4.new_markdown_cell(VALIDATION_MD), nbformat.v4.new_code_cell(VALIDATION_CODE)]
    val.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    NotebookClient(val, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
    cells += val.cells
    cells.append(nbformat.v4.new_markdown_cell(CONCLUSION))
    for i, c in enumerate(cells):
        c["id"] = f"cell-{i:04d}"
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    nbformat.write(nb, OUT)
    return OUT


if __name__ == "__main__":
    print(build())
