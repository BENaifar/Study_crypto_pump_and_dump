"""Reporting helpers: dynamic Markdown, figure/table saving and one chart style for every notebook."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from pnd_paper import settings

# Validated categorical palette (fixed order, never cycled), sequential blue, diverging blue <-> red.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#2a78d6", "#86b6ef", "#f0efec", "#f19c9b", "#e34948"]
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def setup_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.labelsize": 9, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.prop_cycle": mpl.cycler(color=SERIES),
        "lines.linewidth": 2.0, "lines.markersize": 5, "legend.frameon": False, "legend.fontsize": 8,
        "figure.dpi": 110, "savefig.dpi": 150, "figure.figsize": (7.5, 3.6), "font.size": 9,
        "axes.axisbelow": True,
    })


def seq_cmap():
    return mpl.colors.LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)


def div_cmap():
    return mpl.colors.LinearSegmentedColormap.from_list("div_blue_red", DIVERGING)


class Reporter:
    """Saves figures/tables under ``results/<notebook>/`` and renders Markdown built from computed values."""

    def __init__(self, notebook: str) -> None:
        self.dir = settings.RESULTS / notebook
        (self.dir / "figures").mkdir(parents=True, exist_ok=True)
        (self.dir / "tables").mkdir(parents=True, exist_ok=True)
        setup_style()

    def fig(self, fig, name: str) -> Path:
        path = self.dir / "figures" / f"{name}.png"
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        return path

    def table(self, df: pd.DataFrame, name: str, index: bool = True) -> Path:
        path = self.dir / "tables" / f"{name}.csv"
        df.to_csv(path, index=index)
        return path


def md(text: str) -> None:
    """Display Markdown (LaTeX allowed) built from computed values."""
    from IPython.display import Markdown, display

    display(Markdown(text))


def show(df: pd.DataFrame, floatfmt: str = "{:.4g}", caption: str | None = None) -> None:
    """Display a table with compact number formatting."""
    from IPython.display import display

    sty = df.style.format(lambda v: floatfmt.format(v) if isinstance(v, float) else v)
    if caption:
        sty = sty.set_caption(caption)
    display(sty)


def close(fig) -> None:
    plt.show()
    plt.close(fig)
