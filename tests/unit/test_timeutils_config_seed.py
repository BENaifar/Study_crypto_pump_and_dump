import numpy as np
import pandas as pd
import pytest

from pnd.common.seeding import set_global_seed
from pnd.common.timeutils import last_month_timestamp, to_unix_ms


def test_last_month_rule_boundaries():
    assert last_month_timestamp(pd.Timestamp("2021-03-03")) == pd.Timestamp("2021-02-16")
    assert last_month_timestamp(pd.Timestamp("2021-03-04")) == pd.Timestamp("2021-02-01")


def test_to_unix_ms_matches_split_constant():
    s = pd.Series(["2021-05-09 17:00:21+00:00"]).apply(pd.to_datetime)
    assert int(to_unix_ms(s).iloc[0]) == 1620579621000


def test_config_overrides_and_paths(make_cfg, repo_root):
    cfg = make_cfg("seed=7", "sessions.gap_hours=24")
    assert cfg.seed == 7 and cfg.sessions.gap_hours == 24
    assert cfg.paths.pd_log.raw == (repo_root / "datasets/telegram/Labeled/PD_logs_raw.txt").resolve()
    with pytest.raises(KeyError):
        make_cfg("sessions.nope=1")


def test_config_is_frozen(cfg):
    with pytest.raises(AttributeError):
        cfg.sessions.gap_hours = 1


def test_seed_reproducible_and_none_untouched():
    set_global_seed(1)
    a = np.random.permutation(10)
    set_global_seed(1)
    b = np.random.permutation(10)
    assert (a == b).all()
    np.random.seed(5)
    state = np.random.get_state()[1].copy()
    set_global_seed(None)
    assert (np.random.get_state()[1] == state).all()
