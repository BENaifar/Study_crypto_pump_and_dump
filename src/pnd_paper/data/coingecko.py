"""CoinGecko statistics shipped by the authors (fetched three days before each pump).

* targets: ``coinDate2Statistics_pred3d.json`` keyed ``COIN_<pump date>`` (the API call used pump date - 3 days);
* candidates: ``neg_coin_gecko_statistics.txt`` keyed ``COIN_<pump date - 3 days>``.

Both are returned as one tidy frame keyed by (coin, pump_date) with the same columns, so a feature is always
measured three days before the pump.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from pnd_paper import settings

COLUMNS = ["market_cap_usd", "market_cap_btc", "price_usd", "price_btc", "volume_usd", "volume_btc",
           "twitter_followers", "reddit_subscribers", "alexa_rank"]


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


def load_targets(path=settings.GECKO_TARGETS) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for key, v in data.items():
        coin, date = key.rsplit("_", 1)
        md = v.get("market_data") or {}
        cd = v.get("community_data") or {}
        pi = v.get("public_interest_stats") or {}

        def g(field, cur, md=md):
            return _num((md.get(field) or {}).get(cur))
        rows.append({"coin": coin.upper(), "pump_date": pd.Timestamp(date), "market_cap_usd": g("market_cap", "usd"),
                     "market_cap_btc": g("market_cap", "btc"), "price_usd": g("current_price", "usd"),
                     "price_btc": g("current_price", "btc"), "volume_usd": g("total_volume", "usd"),
                     "volume_btc": g("total_volume", "btc"), "twitter_followers": _num(cd.get("twitter_followers")),
                     "reddit_subscribers": _num(cd.get("reddit_subscribers")), "alexa_rank": _num(pi.get("alexa_rank"))})
    return pd.DataFrame(rows)


def load_candidates(path=settings.GECKO_CANDIDATES) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=["key", *COLUMNS], na_values=["None"])
    df[["coin", "date"]] = df.key.str.rsplit("_", n=1, expand=True)
    df["coin"] = df.coin.str.upper()
    df["pump_date"] = pd.to_datetime(df.date, format="%Y%m%d") + pd.Timedelta(days=3)
    for c in COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["coin", "pump_date", *COLUMNS]].drop_duplicates(["coin", "pump_date"])


def load_all() -> pd.DataFrame:
    """Targets and candidates in one frame (column ``source`` says which file); market cap 0 is treated as missing."""
    t = load_targets().assign(source="targets")
    c = load_candidates().assign(source="candidates")
    df = pd.concat([t, c], ignore_index=True)
    for col in ("market_cap_usd", "market_cap_btc"):
        df.loc[df[col] <= 0, col] = np.nan
    return df
