from pathlib import Path

import pytest

from investotron.identifiers import Ticker
from investotron.universe import load_universe, resolve_tickers


def test_load_universe_skips_comments_and_blanks(tmp_path: Path) -> None:
    path = tmp_path / "u.txt"
    path.write_text("# header\nAAPL\n\n  msft  # big\nAAPL\n", encoding="utf-8")
    assert load_universe(path) == [Ticker("AAPL"), Ticker("MSFT"), Ticker("AAPL")]


def test_load_universe_reports_bad_line(tmp_path: Path) -> None:
    path = tmp_path / "u.txt"
    path.write_text("AAPL\nnot a ticker\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"u\.txt:2"):
        load_universe(path)


def test_resolve_dedupes_in_order(tmp_path: Path) -> None:
    path = tmp_path / "u.txt"
    path.write_text("MSFT\nAAPL\n", encoding="utf-8")
    assert resolve_tickers([Ticker("AAPL")], path) == [Ticker("AAPL"), Ticker("MSFT")]


def test_resolve_needs_something() -> None:
    with pytest.raises(ValueError, match="at least one"):
        resolve_tickers([], None)


def test_starter_universe_is_valid() -> None:
    tickers = load_universe(Path(__file__).parents[1] / "universes" / "starter.txt")
    assert len(tickers) == len(set(tickers)) >= 30
