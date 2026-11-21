"""Unzip the archives and prepend last month to the pump month. [orig: Data/Binance/data_process.py]"""

from __future__ import annotations

import logging
import os
import zipfile
from pathlib import Path

import pandas as pd

from pnd.common.io import ensure_dir
from pnd.common.seeding import set_global_seed
from pnd.common.timeutils import last_month_timestamp
from pnd.config import Config
from pnd.data.binance.constants import KLINE_COLUMNS
from pnd.pipeline.registry import register

logger = logging.getLogger(__name__)


def unzip(src_file, dest_dir):
    """ungz zip file"""
    zf = zipfile.ZipFile(src_file)
    zf.extractall(path=dest_dir)
    zf.close()


@register
class ProcessKlinesStage:
    """Q17: walks ``./data`` from cwd ``Data/`` (= datasets/data), which is not where the zips are."""

    name = "process-klines"

    def run(self, cfg: Config) -> None:
        set_global_seed(cfg.seed)
        p = cfg.paths.process_klines
        k = cfg.klines

        dest_dir = Path(p.unzip_dir)
        # QUIRK(Q17): walk dir is <Data>/data, the archives live under <Data>/Binance/data [orig: L14-L20]
        for root, dirs, files in os.walk(p.walk_dir):
            for file in files:
                if file.endswith(".zip"):

                    unzip(os.path.join(root, file), dest_dir)

        logger.info("pause")
        dest_dir = ensure_dir(Path(p.concat_dir))  # D1

        # concatenate the last month data with current month data
        df = pd.read_csv(p.pd_log, sep="\t")
        df["timestamp"] = df.timestamp.apply(pd.to_datetime)

        columns = KLINE_COLUMNS
        written = 0

        for i, row in df.iterrows():

            last_month_ts = last_month_timestamp(row.timestamp, k.last_month_day_threshold,
                                                 k.last_month_days_long, k.last_month_days_short)

            try:
                current_month_file_name = df.loc[i, "coin"] + df.loc[i, "pair"] + "-1m-" + df.loc[i, "timestamp"].strftime("%Y-%m") + ".csv"
                current_month_statistics = pd.read_csv(Path(p.unzip_dir) / current_month_file_name, names=columns)
                last_month_file_name = df.loc[i, "coin"] + df.loc[i, "pair"] + "-1m-" + last_month_ts.strftime("%Y-%m") + ".csv"
                last_month_statistics = pd.read_csv(Path(p.unzip_dir) / last_month_file_name, names=columns)

                current_month_statistics = pd.concat([last_month_statistics, current_month_statistics], axis=0)
                current_month_statistics.to_csv(os.path.join(dest_dir, current_month_file_name), index=False)
                written += 1

            except:
                continue
        logger.info("concat files written: %d", written)
