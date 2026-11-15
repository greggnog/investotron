import pytest

from investotron.identifiers import Cik, Ticker


def test_ticker_parse_normalises() -> None:
    assert Ticker.parse("  brk.b ") == Ticker("BRK.B")


@pytest.mark.parametrize("raw", ["", "aapl", "1ABC", "TOOLONGTICKER", "A B"])
def test_ticker_rejects_invalid(raw: str) -> None:
    with pytest.raises(ValueError, match="ticker"):
        Ticker(raw)


def test_cik_padding() -> None:
    assert Cik(320193).padded == "0000320193"


@pytest.mark.parametrize("value", [0, -1, 10_000_000_000])
def test_cik_rejects_out_of_range(value: int) -> None:
    with pytest.raises(ValueError, match="CIK"):
        Cik(value)
