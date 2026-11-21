"""Network tests (run manually: ``uv run pytest -m network``)."""

import pytest

from pnd.data.binance.positive_klines import FetchKlinesStage
from pnd.verification.binance_checks import VerifyBinanceStage

pytestmark = pytest.mark.network


def test_fetch_klines_downloads_only_previous_months(make_cfg, tmp_path):
    FetchKlinesStage(limit=3).run(make_cfg(f"paths.klines.store_directory={tmp_path}"))
    zips = sorted(p.name for p in tmp_path.rglob("*.zip"))
    assert zips, "expected at least one archive"
    # first rows of the shipped log: APPC 2019-06-04, REQ 2019-06-08, ADX 2019-07-05 -> months 2019-05, 2019-05, 2019-06
    assert all(name.endswith(("2019-05.zip", "2019-06.zip")) for name in zips)
    assert not any(name.endswith("2019-07.zip") for name in zips)  # Q14: pump month never fetched


def test_verify_binance_writes_reports(make_cfg, tmp_path):
    cfg = make_cfg(f"paths.verification.store_directory={tmp_path / 'store'}",
                   f"paths.verification.report_dir={tmp_path / 'reports'}")
    VerifyBinanceStage(limit=3).run(cfg)
    assert (tmp_path / "reports" / "binance_verification.csv").is_file()
    assert (tmp_path / "reports" / "binance_verification.md").is_file()
