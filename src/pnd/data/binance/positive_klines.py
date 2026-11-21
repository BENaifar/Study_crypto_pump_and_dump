"""Download klines for each P&D row. [orig: Data/Binance/kline_collect_Binance.py]"""

from __future__ import annotations

import logging

import pandas as pd

from pnd.common.seeding import set_global_seed
from pnd.common.timeutils import last_month_timestamp
from pnd.config import Config
from pnd.data.binance import archive
from pnd.pipeline.registry import register

logger = logging.getLogger(__name__)


@register
class FetchKlinesStage:
    """Q14: only the month BEFORE each pump is downloaded; the pump month itself never is."""

    name = "fetch-klines"

    def __init__(self, limit: int | None = None) -> None:
        self.limit = limit

    def run(self, cfg: Config) -> None:
        set_global_seed(cfg.seed)
        p = cfg.paths.klines
        k = cfg.klines
        df = pd.read_csv(p.pd_log, sep="\t")
        df["timestamp"] = df.timestamp.apply(pd.to_datetime)
        if self.limit is not None:
            df = df.head(self.limit)  # smoke-test aid only (plan §4.5)

        for i, row in df.iterrows():
            logger.info("\n%s", row)
            # QUIRK(Q14): bare except swallows every error, including Q15's NameError [orig: L21-L41]
            try:
                last_month_ts = last_month_timestamp(row.timestamp, k.last_month_day_threshold,
                                                     k.last_month_days_long, k.last_month_days_short)

                # QUIRK(Q14): only last month's archive is requested [orig: L28-L38]
                archive.download_monthly_klines(trading_type="spot",
                                                symbols=[row.coin + row.pair],
                                                num_symbols=1,
                                                intervals=[k.interval],
                                                years=[last_month_ts.year],
                                                months=[last_month_ts.month],
                                                start_date=None,
                                                end_date=None,
                                                folder="",
                                                checksum=0,
                                                store_directory=p.store_directory,
                                                )

            except:
                continue
