r"""The (event, candidate) ranking table used by the II.B and II.C notebooks.

Every feature is computable at tau - 1h (information horizon of the brief). Notation: t_i is the open time of the last
complete hourly bar available at tau_i - 1h, P_{c,t} the close price (in BTC), Q_{c,t} the quote volume (BTC).

Four families (brief, II.B.1):

* **coin** (CoinGecko, 3 days before the pump): log market cap, price, volume (USD and BTC), log Twitter followers,
  log Reddit subscribers, log Alexa rank;
* **market** for x in {1,3,6,12,24,48,60,72}: return r^{(x)} = log P_{t} - log P_{t-x}, log quote volume
  log(1 + sum_{s=t-x+1}^{t} Q_s), relative volume log(mean_x Q / mean_{[t-30d, t-72h]} Q); plus 24h volatility,
  24h taker-buy ratio and the largest hourly return of the last 72h;
* **channel history** (pumps strictly before tau - 1h): number of past pumps of the channel, of candidate c by
  the channel and by all channels, recency (log hours since the last pump of c);
* **interaction**: Euclidean distance between the candidate's standardised coin vector x_c and the barycentre
  \bar x_h of the channel's past targets, and |log mcap_c - mean log mcap of past targets|.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pnd_paper import settings
from pnd_paper.data import coingecko
from pnd_paper.data.universe import candidate_sets, last_bar_before

COIN_FEATURES = ["g_log_mcap_usd", "g_log_mcap_btc", "g_log_price_usd", "g_log_price_btc", "g_log_vol_usd",
                 "g_log_vol_btc", "g_log_twitter", "g_log_reddit", "g_log_alexa"]
BARY_BASIS = ["g_log_mcap_btc", "g_log_vol_btc", "g_log_price_btc", "g_log_twitter", "g_log_reddit"]


def market_feature_names(windows=settings.WINDOWS_H) -> list[str]:
    names = []
    for x in windows:
        names += [f"m_ret_{x}h", f"m_logqv_{x}h", f"m_relvol_{x}h"]
    return names + ["m_vol_24h", "m_taker_24h", "m_maxret_72h", "m_log_price"]


HISTORY_FEATURES = ["h_ch_n_prev", "h_ch_cand_count", "h_ch_cand_recency", "h_gl_cand_count", "h_gl_cand_30d",
                    "h_gl_cand_recency", "h_ch_has_history"]
INTERACTION_FEATURES = ["i_bary_dist", "i_mcap_gap"]


def _market_features(cands: pd.DataFrame, events: pd.DataFrame, panels: dict[str, pd.DataFrame], windows) -> pd.DataFrame:
    close = panels["close"]
    index = close.index
    sym_pos = {s: j for j, s in enumerate(close.columns)}
    logp = np.log(close.values)
    qv = np.nan_to_num(panels["quote_volume"].values)
    tb = np.nan_to_num(panels["taker_buy_quote"].values)
    cq = np.vstack([np.zeros((1, qv.shape[1])), np.cumsum(qv, axis=0)])
    ctb = np.vstack([np.zeros((1, tb.shape[1])), np.cumsum(tb, axis=0)])
    ret = np.diff(logp, axis=0, prepend=np.nan)
    t_of = dict(zip(events.event_id, last_bar_before(events.timestamp)))
    out = np.full((len(cands), len(market_feature_names(windows))), np.nan)
    for eid, g in cands.groupby("event_id", sort=False):
        t = index.get_loc(t_of[eid])
        cols = np.array([sym_pos[s] for s in g.symbol])
        rows = g.index.values
        k = 0

        def qsum(a, b, cols=cols):  # sum of Q over bars a..b inclusive
            return cq[b + 1, cols] - cq[a, cols]
        base_a, base_b = max(t - 30 * 24, 0), t - 72
        base = qsum(base_a, base_b) / max(base_b - base_a + 1, 1)
        for x in windows:
            out[rows, k] = logp[t, cols] - logp[t - x, cols]
            s = qsum(t - x + 1, t)
            out[rows, k + 1] = np.log1p(s)
            with np.errstate(divide="ignore", invalid="ignore"):
                out[rows, k + 2] = np.log((s / x + 1e-12) / (base + 1e-12))
            k += 3
        with np.errstate(invalid="ignore"):
            out[rows, k] = np.nanstd(ret[t - 23:t + 1, cols], axis=0)
            q24 = qsum(t - 23, t)
            out[rows, k + 1] = np.where(q24 > 0, (ctb[t + 1, cols] - ctb[t - 23, cols]) / np.where(q24 > 0, q24, 1), np.nan)
            out[rows, k + 2] = np.nanmax(ret[t - 71:t + 1, cols], axis=0)
        out[rows, k + 3] = logp[t, cols]
    return pd.DataFrame(out, columns=market_feature_names(windows), index=cands.index)


def _coin_features(cands: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    gk = coingecko.load_all()
    ev = events.set_index("event_id")
    pump_date = cands.event_id.map(ev.timestamp.dt.tz_localize(None).dt.normalize())
    coin = cands.symbol.str[:-3]
    is_target = cands.label == 1
    left = pd.DataFrame({"coin": coin, "pump_date": pump_date, "source": np.where(is_target, "targets", "candidates")},
                        index=cands.index)
    merged = left.merge(gk, on=["coin", "pump_date", "source"], how="left")
    # fall back to the other file when a coin/date pair is only present there
    fallback = left.merge(gk.drop(columns="source").drop_duplicates(["coin", "pump_date"]), on=["coin", "pump_date"], how="left")
    feats = pd.DataFrame(index=cands.index)
    mapping = {"g_log_mcap_usd": "market_cap_usd", "g_log_mcap_btc": "market_cap_btc", "g_log_price_usd": "price_usd",
               "g_log_price_btc": "price_btc", "g_log_vol_usd": "volume_usd", "g_log_vol_btc": "volume_btc",
               "g_log_twitter": "twitter_followers", "g_log_reddit": "reddit_subscribers", "g_log_alexa": "alexa_rank"}
    for f, src in mapping.items():
        v = merged[src].values
        v = np.where(np.isnan(v), fallback[src].values, v)
        with np.errstate(divide="ignore", invalid="ignore"):
            feats[f] = np.log(np.where(v > 0, v, np.nan)) if "price" in src else np.log1p(np.where(v >= 0, v, np.nan))
    return feats


def _history_features(cands: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    ev = events[["event_id", "channel_id", "symbol", "timestamp"]].sort_values("timestamp")
    horizon = ev.set_index("event_id").timestamp - pd.Timedelta(hours=1)
    out = np.zeros((len(cands), len(HISTORY_FEATURES)))
    big = np.log1p(24 * 365 * 5.0)
    ts = ev.timestamp.values
    for eid, g in cands.groupby("event_id", sort=False):
        h = horizon[eid]
        me = ev[ev.event_id == eid].iloc[0]
        past = ev[ts < h.to_datetime64()]
        past_ch = past[past.channel_id == me.channel_id]
        ch_counts = past_ch.symbol.value_counts()
        gl_counts = past.symbol.value_counts()
        gl_30 = past[past.timestamp >= h - pd.Timedelta(days=30)].symbol.value_counts()
        last_ch = past_ch.groupby("symbol").timestamp.max()
        last_gl = past.groupby("symbol").timestamp.max()
        syms = g.symbol
        hours_ch = (h - syms.map(last_ch)).dt.total_seconds() / 3600
        hours_gl = (h - syms.map(last_gl)).dt.total_seconds() / 3600
        rows = g.index.values
        out[rows, 0] = len(past_ch)
        out[rows, 1] = syms.map(ch_counts).fillna(0).values
        out[rows, 2] = np.log1p(hours_ch).fillna(big).values
        out[rows, 3] = syms.map(gl_counts).fillna(0).values
        out[rows, 4] = syms.map(gl_30).fillna(0).values
        out[rows, 5] = np.log1p(hours_gl).fillna(big).values
        out[rows, 6] = float(len(past_ch) > 0)
    return pd.DataFrame(out, columns=HISTORY_FEATURES, index=cands.index)


def _interaction_features(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Distance to the barycentre of the channel's past targets (standardised with train-split statistics)."""
    train = df.split == "train"
    mu = df.loc[train, BARY_BASIS].mean()
    sd = df.loc[train, BARY_BASIS].std().replace(0, 1)
    z = ((df[BARY_BASIS] - mu) / sd)
    tgt = df[df.label == 1][["event_id"]].join(z).set_index("event_id")
    ev = events.set_index("event_id").sort_values("timestamp")
    out = pd.DataFrame(np.nan, index=df.index, columns=INTERACTION_FEATURES)
    for eid, g in df.groupby("event_id", sort=False):
        me = ev.loc[eid]
        prev = ev[(ev.channel_id == me.channel_id) & (ev.timestamp < me.timestamp - pd.Timedelta(hours=1))].index
        prev = [p for p in prev if p in tgt.index]
        if not prev:
            continue
        bary = tgt.loc[prev].mean(skipna=True)
        zc = z.loc[g.index]
        out.loc[g.index, "i_bary_dist"] = np.sqrt(((zc - bary) ** 2).sum(axis=1, min_count=1)).values
        out.loc[g.index, "i_mcap_gap"] = (zc["g_log_mcap_btc"] - bary["g_log_mcap_btc"]).abs().values
    return out


def build(events: pd.DataFrame, panels: dict[str, pd.DataFrame], windows=settings.WINDOWS_H) -> pd.DataFrame:
    """Full ranking table: one row per (event, candidate); events whose target is not a candidate are dropped."""
    cands = candidate_sets(events, panels["close"])
    has_target = cands.groupby("event_id").label.max()
    cands = cands[cands.event_id.isin(has_target[has_target == 1].index)].reset_index(drop=True)
    ev = events.set_index("event_id")
    base = cands.assign(channel=cands.event_id.map(ev.channel), channel_id=cands.event_id.map(ev.channel_id),
                        timestamp=cands.event_id.map(ev.timestamp), split=cands.event_id.map(ev.split))
    df = pd.concat([base, _coin_features(cands, events), _market_features(cands, events, panels, windows),
                    _history_features(cands, events)], axis=1)
    df = pd.concat([df, _interaction_features(df, events)], axis=1)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c[:2] in ("g_", "m_", "h_", "i_")]


def channel_history(events: pd.DataFrame, max_len: int = 8) -> dict[int, list[int]]:
    """For each event, the ids of the channel's previous pumps (most recent first, before tau - 1h), at most L."""
    ev = events.sort_values("timestamp")
    out: dict[int, list[int]] = {}
    for _, g in ev.groupby("channel_id"):
        ids = g.event_id.tolist()
        times = g.timestamp.tolist()
        for k, (eid, t) in enumerate(zip(ids, times)):
            prev = [ids[j] for j in range(k - 1, -1, -1) if times[j] < t - pd.Timedelta(hours=1)]
            out[eid] = prev[:max_len]
    return out
