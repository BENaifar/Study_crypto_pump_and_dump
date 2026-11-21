import pytest

from pnd.modeling.metrics import HitRatio_calculation


def test_int_truncation_q33():
    # two lists (channel 1 @ t=10 and channel 2 @ t=20), 50 candidates each so HR@50 is defined
    channel_ids, coins, timestamps, labels, probas = [], [], [], [], []
    for ch, ts in ((1, 10), (2, 20)):
        for c in range(50):
            channel_ids.append(ch)
            coins.append(c)
            timestamps.append(ts)
            labels.append(1 if c == 49 else 0)
            # the positive has the highest probability, but dtype=int truncates everything to 0
            probas.append(0.99 if c == 49 else 0.01)
    hr = HitRatio_calculation(channel_ids, coins, timestamps, labels, probas)
    # after truncation all scores tie at 0 -> stable sort keeps input order -> positive (last) only in top-50
    assert hr[:5] == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert hr[5] == pytest.approx(1.0)
