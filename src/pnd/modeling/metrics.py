"""HitRatio@k per (channel, timestamp) list. [orig: TargetCoinPrediction/SeqModel/run_train.py:L46-L91]"""

import numpy as np
import pandas as pd

from pnd.common.timeutils import PLATFORM_INT


def legacy_int_frame(data: dict) -> pd.DataFrame:
    """D14: emulate ``pd.DataFrame(data, dtype=int)`` as pandas 1.x executed it (verified on pandas 1.5.3).

    Numeric columns are cast with ``astype`` (floats are TRUNCATED); columns that cannot be cast (byte strings
    from TF) silently stay ``object``. pandas >= 2.0 raises instead.
    """
    columns = {}
    for name, values in data.items():
        arr = np.asarray(values)
        if arr.dtype.kind in "biuf":
            columns[name] = arr.astype(PLATFORM_INT)  # D12: int64
        else:
            columns[name] = np.asarray(values, dtype=object)
    return pd.DataFrame(columns)


def HitRatio_calculation(channel_ids, coin_ids, timestamps, labels, pre_probas):
    # QUIRK(Q33): dtype=int truncates the probabilities to 0, so every candidate ties [orig: L47-L54]
    test_df = legacy_int_frame(
        {"channel_id": channel_ids,
         "coin": coin_ids,
         "timestamp": timestamps,
         "label": labels,
         "y_pred_proba": pre_probas}
    )

    def hitrate(k):
        def udf(df):
            def takeFirst(elem):
                return elem[0]

            X = list(zip(df.y_pred_proba, df.label))
            X.sort(key=takeFirst, reverse=True)
            labels = []
            for i in range(k):
                x = X[i]
                labels.append(x[1])

            return np.array(
                [[df.iloc[0]["channel_id"], df.iloc[0]["timestamp"], np.sum(labels)]])

        x_test = test_df[["channel_id", "timestamp", "label", "y_pred_proba"]].\
            groupby(["channel_id", "timestamp"]).apply(udf)

        test_hitrate = pd.DataFrame(np.concatenate(x_test.values, axis=0),
                                    columns=["channel_id", "timestamp", "label_num"])

        test_hitrate.label_num = test_hitrate.label_num.astype(PLATFORM_INT)
        return test_hitrate.label_num.mean()

    HR1 = hitrate(1)
    HR3 = hitrate(3)
    HR5 = hitrate(5)
    HR10 = hitrate(10)
    HR20 = hitrate(20)
    # QUIRK(Q33): the sixth metric is HR@50, printed by the caller as "HitRate@30" [orig: L89, L234]
    HR50 = hitrate(50)

    return HR1, HR3, HR5, HR10, HR20, HR50
