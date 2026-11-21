"""The authors' cleaned P&D log (read-only) and the event table derived from it."""

from __future__ import annotations

import pandas as pd

from pnd_paper import settings


def load_pd_log(path=settings.PD_LOG) -> pd.DataFrame:
    """All 1 335 rows: channel_id, session_id, coin, exchange, pair, timestamp (UTC)."""
    df = pd.read_csv(path, sep="\t", dtype={"channel_id": str, "session_id": str, "coin": str})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["exchange"] = df["exchange"].str.strip().str.lower()
    df["pair"] = df["pair"].str.strip().str.upper()
    df["coin"] = df["coin"].str.strip().str.upper()
    return df


def channel_labels(channel_ids) -> dict[str, str]:
    """Stable anonymised labels ``Ch-001``... ordered by first appearance id (ethics rule of the brief)."""
    uniq = sorted(pd.unique(pd.Series(list(channel_ids), dtype=str)))
    return {c: f"Ch-{i + 1:03d}" for i, c in enumerate(uniq)}


def binance_btc_events(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Pump events on Binance with a BTC quote, de-duplicated on (channel, coin, minute).

    Adds ``event_id``, ``symbol`` (e.g. ``APPCBTC``), ``channel`` (anonymised) and ``split``.
    """
    df = load_pd_log() if df is None else df
    ev = df[(df.exchange == "binance") & (df.pair == "BTC")].copy()
    ev["minute"] = ev.timestamp.dt.floor("min")
    ev = ev.drop_duplicates(["channel_id", "coin", "minute"]).drop(columns="minute")
    ev = ev.sort_values(["timestamp", "channel_id"]).reset_index(drop=True)
    ev["event_id"] = range(len(ev))
    ev["symbol"] = ev.coin + "BTC"
    ev["channel"] = ev.channel_id.map(channel_labels(df.channel_id))
    ev["split"] = settings.split_of(ev.timestamp).values
    return ev
