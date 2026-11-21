"""Features of the non-pumped coins (negative samples). [orig: TargetCoinPrediction/FeatGeneration/neg_sample_process.py]

D3: ``from Statistics.utility import download_monthly_klines`` (a module that never existed) is replaced by
:mod:`pnd.data.binance.archive`; ``from pos_sample_process import pre_pump_statistics`` by :mod:`pnd.features.market_stats`.
D4: the module-level ``CoinGeckoAPI().get_coins_list()`` call (unused) is not executed.
"""

from __future__ import annotations

import logging
import os
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from pnd.common.seeding import set_global_seed
from pnd.common.timeutils import to_unix_ms
from pnd.config import Config
from pnd.data.binance.archive import download_monthly_klines
from pnd.data.binance.constants import KLINE_COLUMNS
from pnd.data.binance.symbols import BINANCE_SYMBOLS
from pnd.data.coingecko.repository import load_neg_gecko_txt, load_symbol_map
from pnd.features.market_stats import pre_pump_statistics, write_window_features
from pnd.pipeline.registry import register

logger = logging.getLogger(__name__)


def _list_csv_files(directory: Path) -> list[str]:
    file_name_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                file_name_list.append(file)
    return file_name_list


def load_or_build_required_files(df: pd.DataFrame, required_files: Path, binance_symbol_list):
    """Q23a: read the shipped list; only if that raises, build and write it. [orig: L37-L51]"""
    require_binance_file_names = set()
    try:
        require_binance_file_names = pd.read_csv(required_files, names=["file_name"]).file_name.values
    except:
        for i in range(len(df)):
            if df.loc[i, "exchange"] != "binance":
                continue
            for symbol in binance_symbol_list:
                if symbol == df.loc[i, "coin"]:
                    continue
                file_name = symbol + df.loc[i, "pair"] + "-" + df.loc[i, "timestamp"].strftime("%Y%m")
                require_binance_file_names.add(file_name)

        with open(required_files, "w", encoding="utf-8") as f:  # D13
            for file_name in require_binance_file_names:
                f.write(file_name + "\n")
    return require_binance_file_names


def download_branch(require_binance_file_names, store_directory: Path) -> None:
    """Q23b. [orig: L53-L88]"""
    last_month_file_names = set()
    for i in range(0, len(require_binance_file_names)):
        file_name = require_binance_file_names[i]
        symbol = file_name.split("-")[0]
        date = datetime.strptime(file_name.split("-")[1], "%Y%m")
        download_monthly_klines(trading_type="spot",
                                symbols=[symbol],
                                num_symbols=1,
                                intervals=["1m"],
                                years=[date.year],
                                months=[date.month],
                                start_date=None,
                                end_date=None,
                                folder="",
                                checksum=0,
                                store_directory=store_directory,
                                )
        # QUIRK(Q23b): `date` is always day 1 (parsed from %Y%m) -> the 15-day branch [orig: L70-L73]
        if date.day > 3:
            last_month_date = date - timedelta(days=31)
        else:
            last_month_date = date - timedelta(days=15)

        # QUIRK(Q23b): uses `date`, not `last_month_date`, so the name is always in the required set [orig: L75-L76]
        last_month_file_name = symbol + "-" + date.strftime("%Y%m")
        if last_month_file_name not in last_month_file_names and last_month_file_name not in require_binance_file_names:
            last_month_file_names.add(last_month_file_name)
            download_monthly_klines(trading_type="spot",
                                    symbols=[symbol],
                                    num_symbols=1,
                                    intervals=["1m"],
                                    years=[last_month_date.year],
                                    months=[last_month_date.month],
                                    start_date=None,
                                    end_date=None,
                                    folder="",
                                    checksum=0,
                                    store_directory=store_directory,
                                    )


def zip_branch(zip_walk_dir: Path, unzip_dir: Path) -> list[str]:
    """Q23c. [orig: L90-L105]"""
    def unzip(src_file, dest_dir):
        """ungz zip file"""
        zf = zipfile.ZipFile(src_file)
        zf.extractall(path=dest_dir)
        zf.close()

    dest_dir = unzip_dir
    fail_file_list = []
    for root, dirs, files in os.walk(zip_walk_dir):
        for file in files:
            if file.endswith(".zip"):
                try:
                    unzip(os.path.join(root, file), dest_dir)
                except:
                    fail_file_list.append(file)
    return fail_file_list


def concat_branch(require_binance_file_names, read_unzip_dir: Path, concat_output_dir: Path) -> set:
    """Q23c: reads ../../Binance/data/unzip, writes ../../Data/Binance/data/concat. [orig: L107-L131]"""
    dest_dir = concat_output_dir
    columns = KLINE_COLUMNS

    # concatenate the last month data with current month data
    fail_file_set = set()
    for file_name in require_binance_file_names:
        symbol = file_name.split("-")[0]
        date = datetime.strptime(file_name.split("-")[1], "%Y%m")
        if date.day > 3:
            last_month_date = date - timedelta(days=31)
        else:
            last_month_date = date - timedelta(days=15)
        try:
            current_month_file_name = symbol + "-1m-" + date.strftime("%Y-%m") + ".csv"
            current_month_statistics = pd.read_csv(Path(read_unzip_dir) / current_month_file_name, names=columns)
            last_month_file_name = symbol + "-1m-" + last_month_date.strftime("%Y-%m") + ".csv"
            last_month_statistics = pd.read_csv(Path(read_unzip_dir) / last_month_file_name, names=columns)
            current_month_statistics = pd.concat([last_month_statistics, current_month_statistics], axis=0)
            current_month_statistics.to_csv(os.path.join(dest_dir, current_month_file_name), index=False)
        except:
            fail_file_set.add(file_name)
            continue
    return fail_file_set


def gecko_statistics_branch(df, binance_symbol_list, symbol_map_path: Path, concat_list_dir: Path,
                            gecko_txt: Path, sleep_s: float) -> None:
    """Q23d (network, opt-in via negative_samples.gecko_statistics). [orig: L133-L205]"""
    from pycoingecko import CoinGeckoAPI

    symbol2coinId = load_symbol_map(symbol_map_path)
    cg = CoinGeckoAPI()
    coin_list = cg.get_coins_list()
    symbol2id = {}
    for c in coin_list:
        symbol2id[c["symbol"]] = c["id"]
    for s in symbol2coinId.keys():
        symbol2id[s.lower()] = symbol2coinId[s]

    file_name_list = _list_csv_files(concat_list_dir)

    unattack_coin_date = set()
    count = 0
    for i in range(len(df)):
        if df.loc[i, "exchange"] != "binance":
            continue
        for symbol in binance_symbol_list:
            if symbol == df.loc[i, "coin"]:
                continue

            file_name = symbol + df.loc[i, "pair"] + "-1m-" + df.loc[i, "timestamp"].strftime("%Y-%m") + ".csv"
            if file_name in file_name_list:
                pre_3d = df.loc[i, "timestamp"] + timedelta(days=-3)
                key = symbol + "_" + pre_3d.strftime("%Y%m%d")
                unattack_coin_date.add(key)
                count += 1

    logger.info("pause")
    debug_cnt1 = 0
    debug_cnt2 = 0
    debug_key1 = []
    debug_key2 = []
    with open(gecko_txt, "w", encoding="utf-8") as f:  # D7, D13
        for key in unattack_coin_date:
            symbol = key.split("_")[0]
            date = key.split("_")[1]
            # QUIRK(Q14/Q23d): negatives use symbol2coinId only, not symbol2id [orig: L178-L183]
            try:
                coin_id = symbol2coinId[symbol]
            except:
                debug_cnt1 += 1
                debug_key1.append(symbol)
                continue
            date = datetime.strptime(date, "%Y%m%d").strftime("%d-%m-%Y")
            try:
                H = cg.get_coin_history_by_id(coin_id, date=date)
                market_cap_usd = H["market_data"]["market_cap"]["usd"]
                market_cap_btc = H["market_data"]["market_cap"]["btc"]
                price_usd = H["market_data"]["current_price"]["usd"]
                price_btc = H["market_data"]["current_price"]["btc"]
                volume_usd = H["market_data"]["total_volume"]["usd"]
                volume_btc = H["market_data"]["total_volume"]["btc"]
                twitter_follower = H["community_data"]["twitter_followers"]
                reddit_subscriber = H["community_data"]["reddit_subscribers"]
                alexa_rank = H["public_interest_stats"]["alexa_rank"]

                statistic_data = [key, market_cap_usd, market_cap_btc, price_usd, price_btc, volume_usd, volume_btc,
                                  twitter_follower, reddit_subscriber, alexa_rank]
                f.write("\t".join(map(lambda x: str(x), statistic_data)) + "\n")

            except:
                debug_cnt2 += 1
                debug_key2.append(key)
                continue

            time.sleep(sleep_s)


def build_negative_frame(df: pd.DataFrame, binance_symbol_list, concat_list_dir: Path) -> pd.DataFrame:
    """Q23e: every other Binance symbol with a concat file for the pump month. [orig: L209-L249]"""
    file_name_list = _list_csv_files(concat_list_dir)

    neg_sample_df_symbol = []
    neg_sample_df_pair = []
    neg_sample_df_timestamp = []
    neg_sample_df_channel = []
    neg_sample_df_pre3d = []

    for i in range(len(df)):
        if df.loc[i, "exchange"] != "binance":
            continue
        for symbol in binance_symbol_list:
            if symbol == df.loc[i, "coin"]:
                continue
            file_name = symbol + df.loc[i, "pair"] + "-1m-" + df.loc[i, "timestamp"].strftime("%Y-%m") + ".csv"
            if file_name in file_name_list:
                neg_sample_df_channel.append(df.loc[i, "channel_id"])
                neg_sample_df_symbol.append(symbol)
                neg_sample_df_pair.append(df.loc[i, "pair"])
                neg_sample_df_timestamp.append(df.loc[i, "timestamp"])
                pre3d = df.loc[i, "timestamp"] + timedelta(days=-3)
                neg_sample_df_pre3d.append(pre3d.strftime("%Y%m%d"))

    neg_sample_df = pd.DataFrame({
        "channel_id": neg_sample_df_channel,
        "coin": neg_sample_df_symbol,
        "pair": neg_sample_df_pair,
        "timestamp": neg_sample_df_timestamp,
        "pre3d": neg_sample_df_pre3d
    })

    neg_sample_df["timestamp"] = neg_sample_df.timestamp.apply(pd.to_datetime)
    neg_sample_df["timestamp_unix"] = to_unix_ms(neg_sample_df["timestamp"])
    return neg_sample_df


def add_negative_kline_features(neg_sample_df: pd.DataFrame, statistics_dir: Path, fcfg) -> None:
    """[orig: L253-L309]"""
    for i in range(len(neg_sample_df)):
        try:
            file_name = neg_sample_df.loc[i, "coin"] + neg_sample_df.loc[i, "pair"] + "-1m-" + neg_sample_df.loc[i, "timestamp"].strftime("%Y-%m") + ".csv"  # noqa: E501
            # QUIRK(Q23e): listed from Data/Binance/data/concat but read from <repo>/Binance/data/concat [orig: L256]
            statistics = pd.read_csv(Path(statistics_dir) / file_name)
        except:
            continue

        statistics["open_scale"] = statistics["open"] * 10 ** 8
        statistics["close_scale"] = statistics["close"] * 10 ** 8
        statistics["high_scale"] = statistics["high"] * 10 ** 8
        statistics["low_scale"] = statistics["low"] * 10 ** 8
        statistics["maker_buy_base_asset_volume"] = statistics["volume"] - statistics["taker_buy_base_asset_volume"]
        statistics["maker_buy_quote_asset_volume"] = statistics["quote_asset_volume"] - statistics[
            "taker_buy_quote_asset_volume"]

        # before pump
        idx = np.max(statistics[statistics.open_time < neg_sample_df.loc[i, "timestamp_unix"]].index)
        idx = idx - fcfg.pre_anchor_offset_min

        debug_cnt2 = 0
        debug_idx2 = []

        try:
            pre_price_list, pre_volume_list, pre_volume_tb_list, pre_volume_q_list, pre_volume_tb_q_list = pre_pump_statistics(
                statistics, idx, bucket_num=fcfg.bucket_num, bucket_size_min=fcfg.bucket_size_min)
            write_window_features(neg_sample_df, i, pre_price_list, pre_volume_list, pre_volume_tb_list,
                                  pre_volume_q_list, pre_volume_tb_q_list, W=list(fcfg.windows))

            debug_cnt2 += 1

        except:
            debug_idx2.append(i)
            continue


@register
class NegativeSamplesStage:
    """Q23: builds the negative frame and merges CoinGecko stats; the result is NEVER saved (Q23f)."""

    name = "negative-samples"

    def run(self, cfg: Config) -> None:
        set_global_seed(cfg.seed)
        p = cfg.paths.negative_samples
        flags = cfg.negative_samples
        binance_symbol_list = list(BINANCE_SYMBOLS)

        # train/validation/test must be split by time, otherwise there is leakage (original comment)
        df = pd.read_csv(p.pd_log, sep="\t")
        df["timestamp"] = df.timestamp.apply(pd.to_datetime)
        df["timestamp_unix"] = to_unix_ms(df["timestamp"])

        require_binance_file_names = load_or_build_required_files(df, p.required_files, binance_symbol_list)

        if flags.download_flag:
            download_branch(require_binance_file_names, p.download_store_directory)

        if flags.zip_flag:
            zip_branch(p.zip_walk_dir, p.unzip_dir)

        if flags.concat_flag:
            concat_branch(require_binance_file_names, p.concat_read_unzip_dir, p.concat_output_dir)

        if flags.gecko_statistics:
            gecko_statistics_branch(df, binance_symbol_list, p.symbol_map, p.concat_list_dir, p.gecko_txt,
                                    flags.gecko_sleep_s)

        # generate neg_sample by combining binance price history and coingecko statistics
        neg_sample_df = build_negative_frame(df, binance_symbol_list, p.concat_list_dir)
        logger.info("pause (negative candidates: %d)", len(neg_sample_df))

        add_negative_kline_features(neg_sample_df, p.statistics_dir, cfg.features)
        logger.info("pause")

        neg_coin_gecko_df = load_neg_gecko_txt(p.gecko_txt)

        neg_coin_gecko_df[["coin", "date"]] = neg_coin_gecko_df.key.str.split("_", expand=True)

        neg_coin_gecko_df_new = pd.merge(left=neg_sample_df, right=neg_coin_gecko_df, how='left',
                                         left_on=["coin", "pre3d"], right_on=["coin", "date"], sort=False)

        # QUIRK(Q23f): the merged frame is never written; gen_feature expects feature/neg_sample_raw.csv [orig: L321]
        logger.info("pause (neg_coin_gecko_df_new shape: %s, not saved)", neg_coin_gecko_df_new.shape)
