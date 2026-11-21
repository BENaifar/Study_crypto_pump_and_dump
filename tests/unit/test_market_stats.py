import numpy as np
import pandas as pd
import pytest

from pnd.features.market_stats import pre_pump_statistics, write_window_features
from tests.conftest import ORIG_FEATGEN, load_original_function


def frame(n=5000, price=2.0, volume=1.0):
    return pd.DataFrame({"high": [price] * n, "low": [price] * n, "volume": [volume] * n,
                         "taker_buy_base_asset_volume": [0.5] * n, "quote_asset_volume": [price] * n,
                         "taker_buy_quote_asset_volume": [1.0] * n})


def test_constant_price():
    p, v, tb, q, qtb = pre_pump_statistics(frame(), 4500)
    assert len(p) == 73
    assert all(x == pytest.approx(2.0) for x in p)
    assert v[0] == pytest.approx(60.0)


def test_zero_volume_falls_back_to_mean():
    f = frame(volume=0.0)
    p, *_ = pre_pump_statistics(f, 4500)
    assert p[0] == pytest.approx(2.0)


def test_bucket_boundaries():
    f = frame()
    f.loc[4440:4499, "volume"] = 10.0  # rows [idx-60, idx) = bucket 0
    _, v, *_ = pre_pump_statistics(f, 4500)
    assert v[0] == pytest.approx(600.0) and v[1] == pytest.approx(60.0)


def test_identical_to_original():
    orig = load_original_function(ORIG_FEATGEN / "pos_sample_process.py", "pre_pump_statistics", {"np": np})
    rng = np.random.default_rng(0)
    f = frame()
    f["high"] = rng.uniform(1, 2, len(f))
    f["low"] = f["high"] - 0.1
    f["volume"] = rng.uniform(0, 3, len(f))
    assert pre_pump_statistics(f, 4800) == orig(f, 4800)


def test_window_feature_offsets_q20():
    df = pd.DataFrame(index=[0])
    prices = [float(i + 1) for i in range(73)]
    vols = [1.0] * 73
    write_window_features(df, 0, prices, vols, vols, vols, vols, W=[1, 3])
    assert df.loc[0, "pre_1h_return"] == pytest.approx(prices[0] / prices[1] - 1.0)
    assert df.loc[0, "pre_1h_price"] == prices[0]
    assert df.loc[0, "pre_3h_price"] == prices[2]
    assert "pre_1h_volume_tb_avg" not in df.columns
    assert "pre_3h_volume_tb_avg" in df.columns
