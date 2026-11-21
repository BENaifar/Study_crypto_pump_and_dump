"""CoinGecko pre-3-day history (opt-in, never run by tests). [orig: TargetCoinPrediction/FeatGeneration/tmp.py]"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta

import pandas as pd

from pnd.common.seeding import set_global_seed
from pnd.common.timeutils import to_unix_ms
from pnd.config import Config
from pnd.data.coingecko.repository import load_symbol_map
from pnd.pipeline.registry import register

logger = logging.getLogger(__name__)


def build_symbol2id(cg, symbol2coinId: dict) -> dict:
    """CoinGecko coin list overridden by the manual Binance map. [orig: tmp.py:L14-L22]"""
    coin_list = cg.get_coins_list()

    symbol2id = {}
    for c in coin_list:
        symbol2id[c["symbol"]] = c["id"]

    for s in symbol2coinId.keys():
        symbol2id[s.lower()] = symbol2coinId[s]
    return symbol2id


@register
class FetchCoinGeckoStage:
    """Q35: fetches only the 16 hard-coded symbols and never saves the result."""

    name = "fetch-coingecko"

    def run(self, cfg: Config) -> None:
        set_global_seed(cfg.seed)
        try:
            from pycoingecko import CoinGeckoAPI
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError("CoinGecko fetching needs the optional extra: uv sync --extra coingecko") from exc

        p = cfg.paths.coingecko_fetch
        symbol2coinId = load_symbol_map(p.symbol_map)
        cg = CoinGeckoAPI()
        symbol2id = build_symbol2id(cg, symbol2coinId)

        df = pd.read_csv(p.pd_log, sep="\t")
        df["timestamp"] = df.timestamp.apply(pd.to_datetime)
        df["timestamp_unix"] = to_unix_ms(df["timestamp"])

        coin_date_to_timestamp = {}
        for idx, row in df.iterrows():
            attack_date = row.timestamp.strftime("%Y%m%d")
            attack_time = row.timestamp
            try:
                coin_date_to_timestamp[row.coin + "_" + attack_date].append(attack_time)
            except:
                coin_date_to_timestamp[row.coin + "_" + attack_date] = [attack_time]

        error_key = []
        coin_date_to_statistics_pre3d = {}

        for i in range(len(coin_date_to_timestamp.keys())):
            key = list(coin_date_to_timestamp.keys())[i]
            coin_symbol, date = key.split("_")
            # QUIRK(Q35): only the hard-coded patch list of symbols [orig: tmp.py:L43-L45]
            if coin_symbol not in list(cfg.coingecko_fetch.symbols):
                continue
            try:
                coin_id = symbol2id[coin_symbol.lower()]
                # QUIRK(Q35/Q21): fetched at pump date - 3 days, stored under the pump date key [orig: L48-L50]
                d = datetime.strptime(date, "%Y%m%d") + timedelta(days=-3)
                H = cg.get_coin_history_by_id(coin_id, date=d.strftime("%d-%m-%Y"))
                coin_date_to_statistics_pre3d[key] = H
            except:
                error_key.append(key)
            logger.info(str(i) + ":" + key)
            time.sleep(cfg.coingecko_fetch.sleep_s)

        logger.info("pause")

        # QUIRK(Q35): serialised but never written to disk [orig: tmp.py:L58]
        str_json = json.dumps(coin_date_to_statistics_pre3d)
        logger.info("pause (json length %d, errors %d)", len(str_json), len(error_key))
