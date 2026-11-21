"""Build the train / test sample files. [orig: TargetCoinPrediction/FeatGeneration/gen_feature.py]"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pnd.common.io import ensure_parent
from pnd.common.seeding import set_global_seed
from pnd.common.timeutils import PLATFORM_INT
from pnd.config import Config
from pnd.features.columns import COLUMN_LIST
from pnd.pipeline.registry import register

logger = logging.getLogger(__name__)

PRE_3D_COLUMNS = ['pre_3d_market_cap_usd', 'pre_3d_market_cap_btc', 'pre_3d_price_usd', 'pre_3d_price_btc',
                  'pre_3d_volume_usd', 'pre_3d_volume_btc', 'pre_3d_twitter_index', 'pre_3d_reddit_index',
                  'pre_3d_alexa_index']
OTHER_COLUMNS = ['pre_3d_alexa_index', 'pre_3d_market_cap_btc', 'pre_3d_market_cap_usd', 'pre_3d_reddit_index',
                 'pre_3d_twitter_index']


def load_and_filter(pos_input, neg_input) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Q25 filters. [orig: gen_feature.py:L8-L18]"""
    pos_df = pd.read_csv(pos_input)
    pos_df = pos_df[pos_df["exchange"] == "binance"]
    pos_df = pos_df[pos_df["pair"] == "BTC"]
    pos_df = pos_df[pos_df["pre_1h_price"].notna()]

    # QUIRK(Q24): neg_sample_raw.csv is never produced by the negative stage [orig: L13]
    neg_df = pd.read_csv(neg_input, na_values="None")
    neg_df = neg_df[neg_df["pair"] == "BTC"]
    neg_df = neg_df[neg_df["pre_1h_price"].notna()]

    pos_df["label"] = 1
    neg_df["label"] = 0
    return pos_df, neg_df


def impute_per_class(pos_df: pd.DataFrame, neg_df: pd.DataFrame) -> None:
    """Q25: mean imputation computed separately per class (label leakage). [orig: L21-L27]"""
    for column in PRE_3D_COLUMNS:
        mean_val = pos_df[column].mean()
        pos_df[column].fillna(mean_val, inplace=True)

    for column in PRE_3D_COLUMNS:
        mean_val = neg_df[column].mean()
        neg_df[column].fillna(mean_val, inplace=True)


def feature_groups(pos_df: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[str]]:
    """Q26: groups by substring of the positive columns. [orig: L29-L41]"""
    price_columns = []
    return_columns = []
    volume_columns = []

    for column in pos_df.columns:
        if "price" in column:
            price_columns.append(column)
        if "return" in column:
            return_columns.append(column)
        if "volume" in column:
            volume_columns.append(column)

    other_columns = list(OTHER_COLUMNS)
    return price_columns, return_columns, volume_columns, other_columns


def normalize(pos_df, neg_df, columns) -> tuple[list[str], list[str]]:
    """Q26: z-score; pos-only for columns absent from neg, else pos+neg incl. test rows. [orig: L70-L89]"""
    column1 = []
    column2 = []

    for column in columns:

        if column not in neg_df.columns:
            # after pump features
            mean = pos_df[column].mean()
            std = pos_df[column].std()
            pos_df[column] = (pos_df[column] - mean) / std

            column1.append(column)
        else:
            df = pd.concat([pos_df[column], neg_df[column]], axis=0)
            mean = df.mean()
            std = df.std()
            pos_df[column] = (pos_df[column] - mean) / std
            neg_df[column] = (neg_df[column] - mean) / std

            column2.append(column)
    return column1, column2


def map_ids(pos_df, neg_df) -> tuple[dict, dict]:
    """Q27: id 0 is reserved for "0", others follow np.unique order. [orig: L92-L119]"""
    coin_df = pd.concat([pos_df["coin"], neg_df["coin"]], axis=0)
    channel_df = pd.concat([pos_df["channel_id"], neg_df["channel_id"]], axis=0)

    # channel_id & coin_id mapping
    coin_array = np.unique(coin_df.values)
    channel_array = np.unique(channel_df.values)
    coin2id = {}
    id2coin = {}
    channel2id = {}
    coin2id["0"] = 0
    id2coin[0] = "0"
    for i in range(len(coin_array)):
        coin2id[coin_array[i]] = i + 1
        id2coin[i + 1] = coin_array[i]

    id2channel = {}
    channel2id["0"] = 0
    id2channel[0] = "0"

    for i in range(len(channel_array)):
        channel2id[channel_array[i]] = i + 1
        id2channel[i + 1] = channel_array[i]

    pos_df["coin_id"] = pos_df["coin"].apply(lambda x: coin2id[x])
    pos_df["channel_id_new"] = pos_df["channel_id"].apply(lambda x: channel2id[x])

    neg_df["coin_id"] = neg_df["coin"].apply(lambda x: coin2id[x])
    neg_df["channel_id_new"] = neg_df["channel_id"].apply(lambda x: channel2id[x])
    return coin2id, channel2id


def build_sequences(pos_df: pd.DataFrame, columns: list[str], max_len: int = 50) -> pd.DataFrame:
    """Q28: per-channel history of earlier pumps, features joined WITHOUT a delimiter. [orig: L130-L171]"""
    # sequence-ise !!!
    for idx, row in pos_df.iterrows():
        feature = []
        for column in columns:
            feature.append(row[column])

        # QUIRK(Q28): "".join -> values glued together with no separator [orig: L135]
        feature_str = "".join(map(lambda x: str(x), feature))
        pos_df.loc[idx, "feature"] = feature_str

    X_pos = pd.merge(left=pos_df[["channel_id", "timestamp_unix"]],
                     right=pos_df[["channel_id", "coin_id", "feature", "timestamp_unix"]],
                     how='left', on=["channel_id"], sort=False)
    X_pos = X_pos[X_pos.timestamp_unix_x > X_pos.timestamp_unix_y]
    X_pos = X_pos.rename(columns={"timestamp_unix_x": "timestamp_unix_target",
                                  "timestamp_unix_y": "timestamp_unix_seq",
                                  "feature": "feature_seq",
                                  "coin_id": "coin_id_seq"})

    def udf(df):
        def takeFirst(elem):
            return elem[0]
        feature_seq = []
        coin_id_seq = []
        X = list(zip(df.timestamp_unix_seq, df.coin_id_seq, df.feature_seq))
        X.sort(key=takeFirst, reverse=True)
        length = 0
        for x in X:  # set max length to 50
            coin_id_seq.append(str(x[1]))
            feature_seq.append(str(x[2]))
            length += 1
            if length >= max_len:
                break
        return np.array(
            [[df.iloc[0]["channel_id"], df.iloc[0]["timestamp_unix_target"], str(length), "\t".join(coin_id_seq),
              "\t".join(feature_seq)]])

    X_pos_final = X_pos.groupby(["channel_id", "timestamp_unix_target"]).apply(udf)
    pos_seq_feat = pd.DataFrame(np.concatenate(X_pos_final.values, axis=0),
                                columns=["channel_id", "timestamp_unix", "length", "coin_id_seq", "feature_seq"])

    # D12: int64 instead of the platform int
    pos_seq_feat.channel_id = pos_seq_feat.channel_id.astype(PLATFORM_INT)
    pos_seq_feat.timestamp_unix = pos_seq_feat.timestamp_unix.astype(PLATFORM_INT)
    return pos_seq_feat


def attach_sequences(pos_df, neg_df, pos_seq_feat) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Q28: negatives inherit the sequence of the pump with the same (channel, time). [orig: L173-L187]"""
    pos_sample_base = pd.merge(left=pos_df, right=pos_seq_feat, how="left", on=["channel_id", "timestamp_unix"])
    neg_sample_base = pd.merge(left=neg_df, right=pos_seq_feat, how="left", on=["channel_id", "timestamp_unix"])

    column_list = COLUMN_LIST

    pos_sample_base = pos_sample_base[column_list]
    neg_sample_base = neg_sample_base[column_list]

    pos_sample_base.loc[pos_sample_base.length.isna(), "length"] = 0
    pos_sample_base.loc[pos_sample_base.coin_id_seq.isna(), "coin_id_seq"] = "0"
    pos_sample_base.loc[pos_sample_base.feature_seq.isna(), "feature_seq"] = "0"

    neg_sample_base.loc[neg_sample_base.length.isna(), "length"] = 0
    neg_sample_base.loc[neg_sample_base.coin_id_seq.isna(), "coin_id_seq"] = "0"
    neg_sample_base.loc[neg_sample_base.feature_seq.isna(), "feature_seq"] = "0"
    return pos_sample_base, neg_sample_base


def split_and_shuffle(pos_sample_base, neg_sample_base, split_timestamp_unix) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Q29: single time split, unseeded-in-original shuffle (seeded via D5). [orig: L190-L220]"""
    test_pos_sample_base = pos_sample_base[pos_sample_base.timestamp_unix >= split_timestamp_unix]
    train_pos_sample_base = pos_sample_base[pos_sample_base.timestamp_unix < split_timestamp_unix]

    test_neg_sample_base = neg_sample_base[neg_sample_base.timestamp_unix >= split_timestamp_unix]
    train_neg_sample_base = neg_sample_base[neg_sample_base.timestamp_unix < split_timestamp_unix]

    train_sample = pd.concat([train_pos_sample_base, train_neg_sample_base], axis=0)
    test_sample = pd.concat([test_pos_sample_base, test_neg_sample_base], axis=0)

    train_sample = train_sample.reset_index(drop=True)
    test_sample = test_sample.reset_index(drop=True)

    # shuffle
    # QUIRK(Q29): np.random.permutation (unseeded in the original; D5 seeds numpy globally)
    train_sample = train_sample.loc[np.random.permutation(len(train_sample))]
    train_sample = train_sample.reset_index(drop=True)

    test_sample = test_sample.loc[np.random.permutation(len(test_sample))]
    test_sample = test_sample.reset_index(drop=True)

    column_list = COLUMN_LIST
    train_sample = train_sample[column_list]
    test_sample = test_sample[column_list]

    # D12: int64 instead of the platform int
    train_sample.coin_id = train_sample.coin_id.astype(PLATFORM_INT)
    train_sample.channel_id_new = train_sample.channel_id_new.astype(PLATFORM_INT)

    test_sample.coin_id = test_sample.coin_id.astype(PLATFORM_INT)
    test_sample.channel_id_new = test_sample.channel_id_new.astype(PLATFORM_INT)
    return train_sample, test_sample


def build_dataset(pos_df, neg_df, split_timestamp_unix, max_len=50) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Whole gen_feature.py pipeline on in-memory frames (after loading/filtering)."""
    impute_per_class(pos_df, neg_df)
    price_columns, return_columns, volume_columns, other_columns = feature_groups(pos_df)
    columns = price_columns + return_columns + volume_columns + other_columns
    normalize(pos_df, neg_df, columns)
    map_ids(pos_df, neg_df)
    pos_seq_feat = build_sequences(pos_df, columns, max_len)
    pos_sample_base, neg_sample_base = attach_sequences(pos_df, neg_df, pos_seq_feat)
    return split_and_shuffle(pos_sample_base, neg_sample_base, split_timestamp_unix)


@register
class BuildDatasetStage:
    """``feature/pump_sample_raw.csv`` + ``feature/neg_sample_raw.csv`` -> train / test sample CSVs."""

    name = "build-dataset"

    def run(self, cfg: Config) -> None:
        set_global_seed(cfg.seed)
        p = cfg.paths.dataset
        pos_df, neg_df = load_and_filter(p.pos_input, p.neg_input)
        train_sample, test_sample = build_dataset(pos_df, neg_df, cfg.dataset.split_timestamp_unix,
                                                  cfg.dataset.max_seq_len_build)

        # store
        # QUIRK(Q29): written in the featgen cwd; run_train reads feature/train_sample.csv [orig: L223-L224]
        train_sample.to_csv(ensure_parent(p.train_output), index=False, header=False)
        test_sample.to_csv(ensure_parent(p.test_output), index=False, header=False)
        logger.info("train=%d test=%d rows", len(train_sample), len(test_sample))
        # do not forget to manually split the training and testing set by time (original comment)
