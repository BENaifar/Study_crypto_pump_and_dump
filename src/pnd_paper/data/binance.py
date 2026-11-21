"""Binance spot klines from the public archive https://data.binance.vision (monthly zip files).

Every archive is cached under ``datasets/paper_cache/binance/<interval>/<SYMBOL>/``; a ``.missing`` marker records a
404 so it is never requested twice. Hourly data is consolidated into wide panels (hours x symbols) stored as
parquet, which is what the notebooks use.
"""

from __future__ import annotations

import io
import logging
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from pnd_paper import settings

logger = logging.getLogger(__name__)

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
KLINE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades",
                 "taker_buy_base", "taker_buy_quote", "ignore"]
PANEL_FIELDS = ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_quote", "trades")


def archive_path(symbol: str, interval: str, year: int, month: int) -> Path:
    return settings.CACHE / "binance" / interval / symbol / f"{symbol}-{interval}-{year}-{month:02d}.zip"


def download_month(symbol: str, interval: str, year: int, month: int, timeout: float = 30.0) -> Path | None:
    """Download one monthly archive (cached). Returns the local path, or None if Binance has no such file."""
    path = archive_path(symbol, interval, year, month)
    missing = path.with_suffix(".missing")
    if path.is_file():
        return path
    if missing.is_file():
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{symbol}/{interval}/{path.name}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                data = r.read()
            tmp = path.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(path)
            return path
        except urllib.error.HTTPError as e:
            if e.code == 404:
                missing.touch()
                return None
            if attempt == 2:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 2:
                raise
    return None


def month_range(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[int, int]]:
    months = pd.period_range(start.tz_localize(None) if start.tzinfo else start,
                             end.tz_localize(None) if end.tzinfo else end, freq="M")
    return [(p.year, p.month) for p in months]


def download_many(jobs: list[tuple[str, str, int, int]], workers: int = 16) -> pd.DataFrame:
    """Download (symbol, interval, year, month) jobs in parallel. Returns one row per job with ``found``."""
    def run(job):
        try:
            return (*job, download_month(*job) is not None, "")
        except Exception as e:  # network trouble is reported, not fatal
            return (*job, False, repr(e))

    with ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(run, jobs))
    return pd.DataFrame(rows, columns=["symbol", "interval", "year", "month", "found", "error"])


def read_archive(path: Path) -> pd.DataFrame:
    """One monthly archive as a DataFrame with UTC ``open_time`` (handles ms and µs epochs, optional header)."""
    with zipfile.ZipFile(path) as z:
        raw = z.read(z.namelist()[0])
    first = raw[:20].split(b",")[0]
    df = pd.read_csv(io.BytesIO(raw), header=0 if not first.strip().isdigit() else None, names=KLINE_COLUMNS)
    t = df["open_time"].astype("int64")
    unit = "us" if t.iloc[0] > 10**14 else "ms"
    df["open_time"] = pd.to_datetime(t, unit=unit, utc=True)
    return df.drop(columns=["close_time", "ignore"])


def load_klines(symbol: str, interval: str, start: pd.Timestamp, end: pd.Timestamp, download: bool = True) -> pd.DataFrame:
    """Klines of one symbol on [start, end] (UTC), indexed by open time."""
    frames = []
    for y, m in month_range(start, end):
        p = download_month(symbol, interval, y, m) if download else archive_path(symbol, interval, y, m)
        if p is not None and Path(p).is_file():
            frames.append(read_archive(p))
    if not frames:
        return pd.DataFrame(columns=KLINE_COLUMNS[1:6]).rename_axis("open_time")
    df = pd.concat(frames).drop_duplicates("open_time").set_index("open_time").sort_index()
    return df.loc[start:end]


def build_hourly_panel(symbols: list[str], start: pd.Timestamp, end: pd.Timestamp, name: str = "hourly",
                       workers: int = 16, rebuild: bool = False) -> dict[str, pd.DataFrame]:
    """Wide hourly panels {field: DataFrame(hours x symbols)} for ``symbols`` on [start, end], cached as parquet."""
    out_dir = settings.CACHE / "panels" / name
    if not rebuild and all((out_dir / f"{f}.parquet").is_file() for f in PANEL_FIELDS):
        return load_panel(name)
    jobs = [(s, "1h", y, m) for s in symbols for (y, m) in month_range(start, end)]
    report = download_many(jobs, workers=workers)
    errors = report[report.error != ""]
    if len(errors):
        logger.warning("%d archives failed to download (first: %s)", len(errors), errors.error.iloc[0])
    index = pd.date_range(start.floor("h"), end.floor("h"), freq="h", tz="UTC")
    fields: dict[str, dict[str, pd.Series]] = {f: {} for f in PANEL_FIELDS}

    def read_symbol(sym: str):
        paths = [archive_path(sym, "1h", y, m) for (y, m) in month_range(start, end)]
        frames = [read_archive(p) for p in paths if p.is_file()]
        if not frames:
            return sym, None
        return sym, pd.concat(frames).drop_duplicates("open_time").set_index("open_time").sort_index()

    with ThreadPoolExecutor(8) as pool:
        for sym, df in pool.map(read_symbol, symbols):
            if df is None:
                continue
            df = df.reindex(index)
            for f in PANEL_FIELDS:
                fields[f][sym] = df[f].astype("float64")
    out_dir.mkdir(parents=True, exist_ok=True)
    panels = {}
    for f in PANEL_FIELDS:
        panel = pd.DataFrame(fields[f], index=index)
        panel.columns.name = "symbol"
        panel.to_parquet(out_dir / f"{f}.parquet")
        panels[f] = panel
    report.to_csv(out_dir / "download_report.csv", index=False)
    return panels


def load_panel(name: str = "hourly") -> dict[str, pd.DataFrame]:
    out_dir = settings.CACHE / "panels" / name
    return {f: pd.read_parquet(out_dir / f"{f}.parquet") for f in PANEL_FIELDS}


def mid_price(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """(high + low) / 2, the mid price used by [A1]."""
    return (panels["high"] + panels["low"]) / 2.0


def log_returns(close: pd.DataFrame) -> pd.DataFrame:
    """r_t = log P_t - log P_{t-1} (NaN where a bar is missing)."""
    return np.log(close).diff()
