"""Pre-pump kline statistics shared by positive and negative samples.

[orig: TargetCoinPrediction/FeatGeneration/pos_sample_process.py:L24-L65 (pre_pump_statistics),
       pos_sample_process.py:L130-L156 == neg_sample_process.py:L276-L303 (window features, textually identical)]
"""

from __future__ import annotations

import numpy as np


def pre_pump_statistics(statistics, idx, bucket_num=72, bucket_size_min=60):
    """Q19: ``bucket_num + 1`` hourly buckets ending at ``idx`` (volume-weighted mid price)."""
    price_list = []
    volume_list = []
    volume_tb_list = []
    volume_q_list = []
    volume_tb_q_list = []

    for j in range(bucket_num + 1):
        C = 0
        V = 0
        V_tb = 0
        V_q = 0
        V_q_tb = 0
        prices = []

        for i in range(idx - (j + 1) * bucket_size_min, idx - j * bucket_size_min):
            p = float(statistics.loc[i, "high"] + statistics.loc[i, "low"]) / 2
            v = float(statistics.loc[i, "volume"])
            v_tb = float(statistics.loc[i, "taker_buy_base_asset_volume"])
            v_q = float(statistics.loc[i, "quote_asset_volume"])
            v_q_tb = float(statistics.loc[i, "taker_buy_quote_asset_volume"])

            C += p * v
            V += v
            V_tb += v_tb
            V_q += v_q
            V_q_tb += v_q_tb
            prices.append(p)

        if V != 0:
            AggregatedPrice = C / V

        else:
            AggregatedPrice = np.mean(prices)

        price_list.append(AggregatedPrice)
        volume_list.append(V)
        volume_tb_list.append(V_tb)
        volume_q_list.append(V_q)
        volume_tb_q_list.append(V_q_tb)

    return price_list, volume_list, volume_tb_list, volume_q_list, volume_tb_q_list


def write_window_features(df, i, pre_price_list, pre_volume_list, pre_volume_tb_list, pre_volume_q_list,
                          pre_volume_tb_q_list, W=(1, 3, 6, 12, 24, 36, 48, 60, 72)):
    """Q20: writes the window features into ``df`` row ``i`` in the original order (partial writes possible)."""
    for w in W:

        # QUIRK(Q20): return uses p[w] while price uses p[w-1] [orig: pos L135-L136]
        df.loc[i, "pre_" + str(w) + "h_return"] = pre_price_list[0] / pre_price_list[w] - 1.0
        df.loc[i, "pre_" + str(w) + "h_price"] = pre_price_list[w - 1]
        df.loc[i, "pre_" + str(w) + "h_price_avg"] = np.mean(pre_price_list[:w])
        df.loc[i, "pre_" + str(w) + "h_volume"] = np.sum(pre_volume_list[w - 1])
        df.loc[i, "pre_" + str(w) + "h_volume_avg"] = np.mean(pre_volume_list[:w])
        df.loc[i, "pre_" + str(w) + "h_volume_sum"] = np.sum(pre_volume_list[:w])

        df.loc[i, "pre_" + str(w) + "h_volume_tb"] = pre_volume_tb_list[w - 1]
        # QUIRK(Q20): *_avg / *_sum only for w > 1 [orig: pos L143-L156]
        if w > 1:
            df.loc[i, "pre_" + str(w) + "h_volume_tb_avg"] = np.mean(pre_volume_tb_list[:w])
            df.loc[i, "pre_" + str(w) + "h_volume_tb_sum"] = np.sum(pre_volume_tb_list[:w])

        df.loc[i, "pre_" + str(w) + "h_volume_quote"] = pre_volume_q_list[w - 1]
        if w > 1:
            df.loc[i, "pre_" + str(w) + "h_volume_quote_avg"] = np.mean(pre_volume_q_list[:w])
            df.loc[i, "pre_" + str(w) + "h_volume_quote_sum"] = np.sum(pre_volume_q_list[:w])

        df.loc[i, "pre_" + str(w) + "h_volume_quote_tb"] = pre_volume_tb_q_list[w - 1]

        if w > 1:
            df.loc[i, "pre_" + str(w) + "h_volume_quote_tb_avg"] = np.mean(pre_volume_tb_q_list[:w])
            df.loc[i, "pre_" + str(w) + "h_volume_quote_tb_sum"] = np.sum(pre_volume_tb_q_list[:w])
