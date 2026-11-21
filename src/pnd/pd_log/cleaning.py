"""Fill missing exchange / pair in the raw P&D log. [orig: Data/Telegram/PDlog_clean.py]"""

from __future__ import annotations

import logging

import pandas as pd

from pnd.common.io import ensure_parent
from pnd.common.seeding import set_global_seed
from pnd.config import Config
from pnd.pipeline.registry import register

logger = logging.getLogger(__name__)

PD_LOG_COLUMNS = ["channel_id", "session_id", "coin", "exchange", "pair", "timestamp"]


def read_raw_pd_log(path) -> pd.DataFrame:
    """[orig: PDlog_clean.py:L4-L5]"""
    df = pd.read_csv(path, header=None, sep="\t", names=PD_LOG_COLUMNS)
    df["timestamp"] = df.timestamp.apply(pd.to_datetime)
    return df


def clean_pd_log(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Q13. Returns ``(df, exchange_fail_cnt, pair_fail_cnt)``. [orig: PDlog_clean.py:L6-L51]"""
    coin_time_to_exchange = {}
    coin_time_to_pair = {}

    for idx, row in df[df["pair"] != "?"].iterrows():
        coin_time_to_pair[row.coin + "_" + row.timestamp.strftime("%Y%m%d")] = row.pair

    for idx, row in df[df["exchange"] != "?"].iterrows():
        coin_time_to_exchange[row.coin + "_" + row.timestamp.strftime("%Y%m%d")] = row.exchange

    exchange_fail_cnt = 0
    pair_fail_cnt = 0

    for i, r in df.iterrows():
        if df.loc[i, "exchange"] == "?":
            try:
                exchange = coin_time_to_exchange[df.loc[i, "coin"] + "_" + df.loc[i, "timestamp"].strftime("%Y%m%d")]
                df.loc[i, "exchange"] = exchange

            except:
                exchange_fail_cnt += 1

        if df.loc[i, "pair"] == "?":
            try:
                pair = coin_time_to_pair[df.loc[i, "coin"] + "_" + df.loc[i, "timestamp"].strftime("%Y%m%d")]
                df.loc[i, "pair"] = pair

            except:
                pair_fail_cnt += 1

    for i in range(len(df)):
        # QUIRK(Q13): df.loc[i - 1] raises KeyError at i == 0 when row 0 still has "?" [orig: L38, L42]
        if df.loc[i, "exchange"] == "?":
            if (df.loc[i - 1, "exchange"] != "?" and df.loc[i - 1, "channel_id"] == df.loc[i, "channel_id"]):
                df.loc[i, "exchange"] = df.loc[i - 1, "exchange"]

        if df.loc[i, "pair"] == "?":
            if (df.loc[i - 1, "pair"] != "?" and df.loc[i - 1, "channel_id"] == df.loc[i, "channel_id"]):
                df.loc[i, "pair"] = df.loc[i - 1, "pair"]

    for i in range(len(df)):

        # QUIRK(Q13): remaining unknowns default to BTC / binance [orig: L47-L51]
        if df.loc[i, "pair"] == "?":
            df.loc[i, "pair"] = "BTC"

        if df.loc[i, "exchange"] == "?":
            df.loc[i, "exchange"] = "binance"

    return df, exchange_fail_cnt, pair_fail_cnt


@register
class CleanPdLogStage:
    """``PD_logs_raw.txt`` -> ``PD_logs_cleaned.txt``."""

    name = "clean-pd-log"

    def run(self, cfg: Config) -> None:
        set_global_seed(cfg.seed)
        p = cfg.paths.pd_log
        df = read_raw_pd_log(p.raw)
        df, exchange_fail_cnt, pair_fail_cnt = clean_pd_log(df)
        logger.info("exchange_fail_cnt=%d pair_fail_cnt=%d", exchange_fail_cnt, pair_fail_cnt)
        df.to_csv(ensure_parent(p.cleaned), index=False, sep="\t")
        logger.info("saved %d rows to %s", len(df), p.cleaned)
