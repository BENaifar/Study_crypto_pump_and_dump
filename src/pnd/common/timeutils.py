"""Time helpers shared by the Binance and feature stages."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

# D12: the authors ran on Linux/macOS where numpy's default ``int`` is int64. On Windows with numpy<2 it is
# int32, which overflows millisecond timestamps. Every ``astype(int)`` / ``dtype=int`` of the original uses this.
PLATFORM_INT = np.int64


def to_unix_ms(series: pd.Series) -> pd.Series:
    """Q18: ``(timestamp.astype(int) / (10 ** 6)).astype(int)`` -> milliseconds. [orig: pos_sample_process.py:L77]"""
    return (series.astype(PLATFORM_INT) / (10 ** 6)).astype(PLATFORM_INT)


def last_month_timestamp(ts, threshold: int = 3, long: int = 31, short: int = 15):
    """Q14/Q23b: ``ts - 31 days`` if ``ts.day > 3`` else ``ts - 15 days``. [orig: kline_collect_Binance.py:L22-L26]"""
    if ts.day > threshold:
        return ts - timedelta(days=long)
    return ts - timedelta(days=short)
