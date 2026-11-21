"""Loaders for the authors' shipped CoinGecko files (fixed inputs)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

NEG_GECKO_COLUMNS = ["key", "pre_3d_market_cap_usd", "pre_3d_market_cap_btc", "pre_3d_price_usd", "pre_3d_price_btc",
                     "pre_3d_volume_usd", "pre_3d_volume_btc", 'pre_3d_twitter_index', 'pre_3d_reddit_index',
                     'pre_3d_alexa_index']


def load_pre3d_json(path: Path) -> dict:
    """``coinDate2Statistics_pred3d.json``: key ``COIN_YYYYMMDD`` (pump date) -> CoinGecko history at date-3."""
    with open(path, "r", encoding="utf-8") as f:  # D13
        return json.load(f)


def load_symbol_map(path: Path) -> dict:
    """``binanceSymbol2CoinId.json``: Binance symbol -> CoinGecko id."""
    with open(path, "r", encoding="utf-8") as f:  # D13
        return json.load(f)


def load_neg_gecko_txt(path: Path) -> pd.DataFrame:
    """[orig: neg_sample_process.py:L313-L315]"""
    return pd.read_csv(path, delimiter="\t", names=NEG_GECKO_COLUMNS)
