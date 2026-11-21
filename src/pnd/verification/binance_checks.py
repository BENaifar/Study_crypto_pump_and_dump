"""[new, read-only] Audit the authors' P&D log and CoinGecko numbers against Binance klines.

This stage does NOT feed the pipeline. It downloads BOTH the pump month and the previous month into its own
folder (``paths.verification.store_directory``) so that it does not inherit Q14 (pump month never downloaded).
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pnd.common.io import ensure_dir
from pnd.common.seeding import set_global_seed
from pnd.common.timeutils import last_month_timestamp, to_unix_ms
from pnd.config import Config
from pnd.data.binance import archive
from pnd.data.binance.constants import KLINE_COLUMNS
from pnd.data.coingecko.repository import load_pre3d_json
from pnd.pd_log.cleaning import PD_LOG_COLUMNS
from pnd.pipeline.registry import register

logger = logging.getLogger(__name__)

PUMP_RATIO_THRESHOLD = 1.05
WINDOW_MIN = 60


@dataclass
class RowCheck:
    """One audited P&D row."""

    row: int
    channel_id: int
    coin: str
    pair: str
    timestamp: str
    symbol: str
    symbol_trading_now: bool | None = None
    archive_prev_found: bool = False
    archive_current_found: bool = False
    pump_ratio: float | None = None
    pump_visible: bool | None = None
    minutes_to_peak: int | None = None
    pair_or_exchange_defaulted: bool = False
    cg_price_btc: float | None = None
    binance_vwap_pre3d: float | None = None
    cg_vs_binance_rel_err: float | None = None
    error: str | None = field(default=None)


def fetch_trading_symbols() -> set[str]:
    """Spot symbols whose status is TRADING (exchangeInfo also lists delisted symbols with status BREAK)."""
    response = urllib.request.urlopen("https://api.binance.com/api/v3/exchangeInfo").read()
    return {s["symbol"] for s in json.loads(response)["symbols"] if s.get("status") == "TRADING"}


def _zip_path(store: Path, symbol: str, year: int, month: int, interval: str = "1m") -> Path:
    return (store / archive.get_path("spot", "klines", "monthly", symbol, interval)
            / f"{symbol.upper()}-{interval}-{year}-{month:02d}.zip")


def _download_month(store: Path, symbol: str, ts) -> Path | None:
    """Download one monthly archive (errors such as Q15's NameError are swallowed). Return the zip if present."""
    try:
        # the ported download_file writes a progress bar to stdout (Q15); keep the audit output readable
        with contextlib.redirect_stdout(io.StringIO()):
            archive.download_monthly_klines(trading_type="spot", symbols=[symbol], num_symbols=1, intervals=["1m"],
                                            years=[ts.year], months=[ts.month], start_date=None, end_date=None,
                                            folder="", checksum=0, store_directory=store)
    except Exception as exc:  # network / NameError without content-length
        logger.debug("download %s %s-%s failed: %r", symbol, ts.year, ts.month, exc)
    path = _zip_path(store, symbol, ts.year, ts.month)
    return path if path.is_file() and path.stat().st_size > 0 else None


def _read_klines(zip_path: Path) -> pd.DataFrame:
    df = pd.read_csv(zip_path, names=KLINE_COLUMNS, compression="zip")
    # newer archives use microseconds; the 2019-2022 ones use milliseconds
    if len(df) and df["open_time"].iloc[0] > 10 ** 14:
        df["open_time"] = df["open_time"] // 1000
    return df


def pump_window_stats(klines: pd.DataFrame, ts_ms: int) -> tuple[float, int]:
    """max(high[t, t+60)) / mid(t-60) and the minute offset of the peak."""
    before = klines.index[klines.open_time < ts_ms]
    if len(before) == 0:
        raise ValueError("no candle before the pump time")
    t_idx = int(before.max())
    if t_idx - WINDOW_MIN < 0:
        raise ValueError("not enough history before the pump")
    mid_pre = (klines.loc[t_idx - WINDOW_MIN, "high"] + klines.loc[t_idx - WINDOW_MIN, "low"]) / 2
    window = klines.loc[t_idx:t_idx + WINDOW_MIN - 1, "high"]
    if mid_pre <= 0 or window.empty:
        raise ValueError("degenerate price window")
    return float(window.max() / mid_pre), int(window.values.argmax())


def daily_vwap(klines: pd.DataFrame, day_start: datetime) -> float | None:
    start = int(day_start.timestamp() * 1000)
    end = start + 24 * 3600 * 1000
    day = klines[(klines.open_time >= start) & (klines.open_time < end)]
    volume = day["volume"].sum()
    if volume <= 0:
        return None
    return float(day["quote_asset_volume"].sum() / volume)


def _cg_price_btc(cg_json: dict, key: str) -> float | None:
    try:
        return float(cg_json[key]["market_data"]["current_price"]["btc"])
    except (KeyError, TypeError, ValueError):
        return None


def _fmt_stats(series: pd.Series) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return "n/a (no values)"
    q1, med, q3 = np.percentile(s, [25, 50, 75])
    return f"median {med:.4f} (IQR {q1:.4f} to {q3:.4f}, n={len(s)})"


def _pct(series: pd.Series) -> str:
    s = series.dropna()
    if s.empty:
        return "n/a"
    return f"{int(s.sum())}/{len(s)} ({100 * s.mean():.1f}%)"


def write_report(df: pd.DataFrame, report_dir: Path, seed, limit) -> Path:
    """Markdown summary next to the per-row CSV."""
    lines = [
        "# Binance verification of the authors' P&D log",
        "",
        f"- Run: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}; seed: {seed}; "
        f"rows audited: {len(df)}" + (f" (limit {limit})" if limit else ""),
        "- Market data: monthly 1m klines from https://data.binance.vision (pump month + previous month), "
        "trading status (status == TRADING) from api.binance.com/api/v3/exchangeInfo.",
        "- P&D log and CoinGecko values: the authors' shipped files (not re-fetched).",
        f"- `pump_visible`: max high in [t, t+60m) / mid price at t-60m >= {PUMP_RATIO_THRESHOLD}.",
        "",
        "## Availability",
        "",
        "| check | result |",
        "|---|---|",
        f"| symbol still TRADING on Binance spot today | {_pct(df.symbol_trading_now.astype('boolean'))} |",
        f"| previous-month archive exists | {_pct(df.archive_prev_found.astype('boolean'))} |",
        f"| pump-month archive exists | {_pct(df.archive_current_found.astype('boolean'))} |",
        f"| pump visible in klines | {_pct(df.pump_visible.astype('boolean'))} |",
        "",
        "## Distributions",
        "",
        f"- pump_ratio: {_fmt_stats(df.pump_ratio)}",
        f"- minutes_to_peak: {_fmt_stats(df.minutes_to_peak)}",
        f"- CoinGecko price_btc vs Binance VWAP (date-3) relative error: {_fmt_stats(df.cg_vs_binance_rel_err)}",
        "",
        "## Rows whose exchange/pair was '?' in PD_logs_raw.txt (defaulted by Q13) vs explicit rows",
        "",
        "| group | rows | pump visible | archive (pump month) found |",
        "|---|---|---|---|",
    ]
    for flag, name in ((True, "defaulted"), (False, "explicit")):
        g = df[df.pair_or_exchange_defaulted == flag]
        lines.append(f"| {name} | {len(g)} | {_pct(g.pump_visible.astype('boolean'))} | "
                     f"{_pct(g.archive_current_found.astype('boolean'))} |")
    not_visible = df[df.pump_visible == False]  # noqa: E712
    lines += ["", f"## Rows with pump_visible == False ({len(not_visible)})", ""]
    if len(not_visible):
        lines.append("| row | channel | symbol | timestamp | pump_ratio |")
        lines.append("|---|---|---|---|---|")
        for r in not_visible.itertuples():
            lines.append(f"| {r.row} | {r.channel_id} | {r.symbol} | {r.timestamp} | {r.pump_ratio:.4f} |")
    errors = df[df.error.notna()]
    lines += ["", f"## Rows with errors ({len(errors)})", ""]
    if len(errors):
        lines.append("| row | symbol | error |")
        lines.append("|---|---|---|")
        for r in errors.itertuples():
            lines.append(f"| {r.row} | {r.symbol} | {str(r.error).replace('|', '/')} |")
    out = report_dir / "binance_verification.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


@register
class VerifyBinanceStage:
    """Per-row availability / pump visibility / CoinGecko cross-check, written to ``reports/``."""

    name = "verify-binance"

    def __init__(self, limit: int | None = None) -> None:
        self.limit = limit

    def run(self, cfg: Config) -> None:
        set_global_seed(cfg.seed)
        p = cfg.paths.verification
        store = ensure_dir(Path(p.store_directory))
        report_dir = ensure_dir(Path(p.report_dir))

        df = pd.read_csv(p.pd_log, sep="\t")
        df["timestamp"] = df.timestamp.apply(pd.to_datetime)
        df["timestamp_unix"] = to_unix_ms(df["timestamp"])
        raw = pd.read_csv(p.pd_log_raw, header=None, sep="\t", names=PD_LOG_COLUMNS, dtype=str)
        cg_json = load_pre3d_json(p.coingecko_json)

        try:
            listed = fetch_trading_symbols()
        except Exception as exc:
            logger.warning("exchangeInfo unavailable: %r", exc)
            listed = None

        binance_rows = df[df.exchange == "binance"]
        if self.limit is not None:
            binance_rows = binance_rows.head(self.limit)

        results: list[RowCheck] = []
        for n, (i, row) in enumerate(binance_rows.iterrows(), start=1):
            symbol = row.coin + row.pair
            check = RowCheck(row=int(i), channel_id=int(row.channel_id), coin=row.coin, pair=row.pair,
                             timestamp=str(row.timestamp), symbol=symbol)
            check.symbol_trading_now = (symbol in listed) if listed is not None else None
            if i < len(raw):
                check.pair_or_exchange_defaulted = bool(raw.loc[i, "exchange"] == "?" or raw.loc[i, "pair"] == "?")
            try:
                prev_ts = last_month_timestamp(row.timestamp)
                prev_zip = _download_month(store, symbol, prev_ts)
                cur_zip = _download_month(store, symbol, row.timestamp)
                check.archive_prev_found = prev_zip is not None
                check.archive_current_found = cur_zip is not None
                frames = [_read_klines(z) for z in (prev_zip, cur_zip) if z is not None]
                klines = pd.concat(frames, axis=0).reset_index(drop=True) if frames else None

                if cur_zip is not None and klines is not None:
                    ratio, peak = pump_window_stats(klines, int(row.timestamp_unix))
                    check.pump_ratio = ratio
                    check.minutes_to_peak = peak
                    check.pump_visible = ratio >= PUMP_RATIO_THRESHOLD

                key = row.coin + "_" + row.timestamp.strftime("%Y%m%d")
                check.cg_price_btc = _cg_price_btc(cg_json, key)
                if check.cg_price_btc is not None and row.pair == "BTC" and klines is not None:
                    pre3d = (row.timestamp - timedelta(days=3)).to_pydatetime()
                    day_start = datetime(pre3d.year, pre3d.month, pre3d.day, tzinfo=timezone.utc)
                    vwap = daily_vwap(klines, day_start)
                    check.binance_vwap_pre3d = vwap
                    if vwap:
                        check.cg_vs_binance_rel_err = abs(check.cg_price_btc - vwap) / vwap
            except Exception as exc:
                check.error = repr(exc)
            results.append(check)
            if n % 25 == 0:
                logger.info("audited %d/%d rows", n, len(binance_rows))

        out = pd.DataFrame([asdict(r) for r in results])
        csv_path = report_dir / "binance_verification.csv"
        out.to_csv(csv_path, index=False)
        md_path = write_report(out, report_dir, cfg.seed, self.limit)
        logger.info("wrote %s and %s", csv_path, md_path)
