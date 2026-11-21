"""Features of the pumped coins (positive samples). [orig: TargetCoinPrediction/FeatGeneration/pos_sample_process.py]

D4: the module-level ``CoinGeckoAPI().get_coins_list()`` call of the original is not executed (unused result).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pnd.common.io import ensure_parent
from pnd.common.seeding import set_global_seed
from pnd.common.timeutils import to_unix_ms
from pnd.config import Config
from pnd.data.coingecko.repository import load_pre3d_json
from pnd.features.market_stats import pre_pump_statistics, write_window_features
from pnd.pipeline.registry import register

logger = logging.getLogger(__name__)


def add_kline_features(df: pd.DataFrame, concat_dir: Path, fcfg) -> dict:
    """After-pump and pre-pump kline features. [orig: pos_sample_process.py:L79-L162]"""
    debug_cnt1 = 0
    debug_cnt2 = 0
    debug_idx1 = []
    debug_idx2 = []

    for i in range(len(df)):

        if df.loc[i, "exchange"] != "binance":
            continue
        try:
            file_name = df.loc[i, "coin"] + df.loc[i, "pair"] + "-1m-" + df.loc[i, "timestamp"].strftime("%Y-%m") + ".csv"
            # QUIRK(Q18): read from ../../Binance/data/concat (= datasets/legacy_root/...) [orig: L85]
            statistics = pd.read_csv(Path(concat_dir) / file_name)
            debug_cnt1 += 1

        except:
            debug_idx1.append(i)
            continue

        # QUIRK(Q18): computed but unused [orig: L92-L97]
        statistics["open_scale"] = statistics["open"] * 10 ** 8
        statistics["close_scale"] = statistics["close"] * 10 ** 8
        statistics["high_scale"] = statistics["high"] * 10 ** 8
        statistics["low_scale"] = statistics["low"] * 10 ** 8
        statistics["maker_buy_base_asset_volume"] = statistics["volume"] - statistics["taker_buy_base_asset_volume"]
        statistics["maker_buy_quote_asset_volume"] = statistics["quote_asset_volume"] - statistics["taker_buy_quote_asset_volume"]

        # QUIRK(Q18): after-pump block is NOT inside a try (errors abort the stage) [orig: L99-L122]
        idx = np.max(statistics[statistics.open_time < df.loc[i, "timestamp_unix"]].index)

        insider_price = (statistics.loc[idx - fcfg.insider_offset_min, "high"]
                         + statistics.loc[idx - fcfg.insider_offset_min, "low"]) / 2
        outsider_price = statistics.loc[idx, "open"]
        highest_price = np.max(statistics.loc[range(idx, idx + fcfg.highest_window_min), "high"].values)
        outsider_avg_price = (statistics.loc[idx, "high"] + statistics.loc[idx, "low"]) / 2
        outsider_next_avg_price = (statistics.loc[idx + 1, "high"] + statistics.loc[idx + 1, "low"]) / 2

        insider_return = highest_price / insider_price
        outsider_return = highest_price / outsider_price
        outsider_avg_return = highest_price / outsider_avg_price
        outsider_next_avg_return = outsider_next_avg_price / outsider_avg_price

        # after pump
        df.loc[i, "insider_price"] = insider_price
        df.loc[i, "outsider_price"] = outsider_price
        df.loc[i, "highest_price"] = highest_price
        df.loc[i, "outsider_avg_price"] = outsider_avg_price
        df.loc[i, "outsider_next_avg_price"] = outsider_next_avg_price

        df.loc[i, "insider_return"] = insider_return
        df.loc[i, "outsider_return"] = outsider_return
        df.loc[i, "outsider_avg_return"] = outsider_avg_return
        df.loc[i, "outsider_next_avg_return"] = outsider_next_avg_return

        # before pump
        idx = np.max(statistics[statistics.open_time < df.loc[i, "timestamp_unix"]].index)
        # QUIRK(Q18): pre-pump anchor 30 minutes before the last candle before the pump [orig: L126]
        idx = idx - fcfg.pre_anchor_offset_min

        try:

            pre_price_list, pre_volume_list, pre_volume_tb_list, pre_volume_q_list, pre_volume_tb_q_list = \
                pre_pump_statistics(statistics, idx, bucket_num=fcfg.bucket_num, bucket_size_min=fcfg.bucket_size_min)
            write_window_features(df, i, pre_price_list, pre_volume_list, pre_volume_tb_list, pre_volume_q_list,
                                  pre_volume_tb_q_list, W=list(fcfg.windows))

            debug_cnt2 += 1

        except:
            debug_idx2.append(i)
            continue

    return {"debug_cnt1": debug_cnt1, "debug_cnt2": debug_cnt2,
            "debug_idx1": len(debug_idx1), "debug_idx2": len(debug_idx2)}


def add_coingecko_features(df: pd.DataFrame, coin_date_to_statistics_pre3d: dict) -> dict:
    """Q21. [orig: pos_sample_process.py:L168-L209]"""
    debug_cnt3 = 0
    debug_idx3 = []
    for i in range(len(df)):

        # QUIRK(Q21): key uses the PUMP date although the stats were fetched at date-3 [orig: L172]
        key = df.loc[i, "coin"] + "_" + df.loc[i, "timestamp"].strftime("%Y%m%d")
        try:
            market_cap_usd = coin_date_to_statistics_pre3d[key]["market_data"]["market_cap"]["usd"]
            market_cap_btc = coin_date_to_statistics_pre3d[key]["market_data"]["market_cap"]["btc"]
            price_usd = coin_date_to_statistics_pre3d[key]["market_data"]["current_price"]["usd"]
            price_btc = coin_date_to_statistics_pre3d[key]["market_data"]["current_price"]["btc"]
            volume_usd = coin_date_to_statistics_pre3d[key]["market_data"]["total_volume"]["usd"]
            volume_btc = coin_date_to_statistics_pre3d[key]["market_data"]["total_volume"]["btc"]

            twitter_follower = coin_date_to_statistics_pre3d[key]["community_data"]["twitter_followers"]
            reddit_subscriber = coin_date_to_statistics_pre3d[key]["community_data"]["reddit_subscribers"]
            alexa_rank = coin_date_to_statistics_pre3d[key]["public_interest_stats"]["alexa_rank"]

            if market_cap_usd != 0:
                df.loc[i, "pre_3d_market_cap_usd"] = market_cap_usd

            if market_cap_btc != 0:
                df.loc[i, "pre_3d_market_cap_btc"] = market_cap_btc

            df.loc[i, "pre_3d_price_usd"] = price_usd
            df.loc[i, "pre_3d_price_btc"] = price_btc

            df.loc[i, "pre_3d_volume_usd"] = volume_usd
            df.loc[i, "pre_3d_volume_btc"] = volume_btc

            if twitter_follower:
                df.loc[i, "pre_3d_twitter_index"] = twitter_follower

            if reddit_subscriber:
                df.loc[i, "pre_3d_reddit_index"] = reddit_subscriber

            if alexa_rank:
                df.loc[i, "pre_3d_alexa_index"] = alexa_rank

            debug_cnt3 += 1
        except:
            debug_idx3.append(key)
            continue
    return {"debug_cnt3": debug_cnt3, "debug_idx3": len(debug_idx3)}


@register
class PositiveSamplesStage:
    """``PD_logs_cleaned.txt`` + concat klines + CoinGecko JSON -> ``pump_sample_raw.csv``."""

    name = "positive-samples"

    def run(self, cfg: Config) -> None:
        set_global_seed(cfg.seed)
        p = cfg.paths.positive_samples

        # train/validation/test must be split by time, otherwise there is leakage (original comment)
        df = pd.read_csv(p.pd_log, sep="\t")
        df["timestamp"] = df.timestamp.apply(pd.to_datetime)
        # QUIRK(Q18): ms via astype(int) / 10**6 [orig: L77]
        df["timestamp_unix"] = to_unix_ms(df["timestamp"])

        kline_debug = add_kline_features(df, p.concat_dir, cfg.features)
        logger.info("kline features: %s", kline_debug)

        coin_date_to_statistics_pre3d = load_pre3d_json(p.coingecko_json)
        gecko_debug = add_coingecko_features(df, coin_date_to_statistics_pre3d)
        logger.info("coingecko features: %s", gecko_debug)

        logger.info("pause")
        # QUIRK(Q21/Q22): saved in the featgen cwd, gen_feature reads feature/pump_sample_raw.csv [orig: L212]
        df.to_csv(ensure_parent(p.output), index=False)
        logger.info("saved %d rows to %s", len(df), p.output)
