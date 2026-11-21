from pnd.filtering.keyword_filter import KeywordFilter

SKIP = ["", "the", "we", "min", "coin", "am", "now", "pump", "buy", "profit", "profits", "for", "a", "in", "sure",
        "get", "less"]


def make_filter():
    return KeywordFilter(crypto_symbol_list=["btc", "xyz"], exchange_symbol_list=["binance"],
                         keyword_list=["target", "buy", "pump"], skip_tokens=SKIP)


def test_buy_and_pump_never_match_q01():
    f = make_filter()
    assert f.pre_select({"message": "Buy PUMP now"}) is False
    assert f.keyword_match == []


def test_target_matches():
    f = make_filter()
    assert f.pre_select({"message": "Target 50%"}) is True
    assert f.keyword_match == ["target"]


def test_first_match_wins_q01():
    f = make_filter()
    assert f.pre_select({"message": "btc binance target"}) is True
    assert f.crypto_match == ["btc"]
    assert f.exchange_match == [] and f.keyword_match == []


def test_split_characters():
    f = make_filter()
    assert f.pre_select({"message": "go#xyz$"}) is True
    assert f.crypto_match == ["xyz"]
