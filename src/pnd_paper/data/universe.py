"""Symbol universe and candidate sets C_i.

The universe is the set of BTC-quoted Binance symbols the authors considered as negatives
(``neg_binance_required_file_name.txt``) plus every Binance/BTC pump target of the log. For an event at time tau the
candidate set C_i holds the universe symbols that were trading in the 72 hours before tau - 1h (the information
horizon of the brief): a symbol is a candidate when its bar at tau - 1h exists and at least ``min_bars`` of the 72
preceding hourly bars exist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pnd_paper import settings

PANEL_START = pd.Timestamp("2018-11-01", tz="UTC")
PANEL_END = pd.Timestamp("2022-02-28 23:00", tz="UTC")


def authors_negative_symbols() -> list[str]:
    names = pd.read_csv(settings.NEG_REQUIRED_FILES, names=["f"]).f.str.rsplit("-", n=1).str[0]
    return sorted(s for s in names.unique() if s.endswith("BTC"))


def universe(events: pd.DataFrame) -> list[str]:
    return sorted(set(authors_negative_symbols()) | set(events.symbol))


def last_bar_before(tau: pd.Series | pd.Timestamp, horizon_h: int = 1):
    """Open time of the last *complete* hourly bar available at ``tau - horizon_h`` hours."""
    return (tau - pd.Timedelta(hours=horizon_h)).floor("h") - pd.Timedelta(hours=1) if isinstance(tau, pd.Timestamp) \
        else (tau - pd.Timedelta(hours=horizon_h)).dt.floor("h") - pd.Timedelta(hours=1)


def candidate_sets(events: pd.DataFrame, close: pd.DataFrame, window_h: int = 72, min_bars: int = 60) -> pd.DataFrame:
    """Long table (event_id, symbol, label) of candidates; label = 1 for the pump target."""
    present = close.notna()
    counts = present.rolling(window_h, min_periods=1).sum()
    rows = []
    for ev in events.itertuples():
        t = last_bar_before(ev.timestamp)
        if t not in close.index:
            continue
        ok = present.loc[t] & (counts.loc[t] >= min_bars)
        syms = ok.index[ok.values]
        rows.append(pd.DataFrame({"event_id": ev.event_id, "symbol": syms, "label": (syms == ev.symbol).astype(np.int8)}))
    return pd.concat(rows, ignore_index=True)
