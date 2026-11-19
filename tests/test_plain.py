from datetime import UTC, date, datetime

import pytest

from investotron.paper import Direction, Rule, Status
from investotron.plain import (
    MIN_LIVE_TRADES,
    PLAIN_ENV,
    day,
    expectation,
    idea,
    pending,
    plain_enabled,
    scoreboard,
    trade,
    wrap,
)

SHORT = Rule("short-102", "1.02", Direction.SHORT, horizon=20)
LONG_BEAT = Rule("long-beat", "2.02", Direction.LONG, horizon=5, surprise="beat")
STARTED = datetime(2026, 11, 16, 18, 2, tzinfo=UTC)


def test_plain_enabled() -> None:
    assert plain_enabled(True, {})
    assert plain_enabled(False, {PLAIN_ENV: "1"})
    assert not plain_enabled(False, {PLAIN_ENV: "0"})


def test_day_uses_new_york_date() -> None:
    late_evening_utc = datetime(2026, 11, 17, 2, 0, tzinfo=UTC)  # Mon 9 pm in New York
    assert day(late_evening_utc) == "Mon 16 Nov 2026"
    assert day(date(2026, 11, 18)) == "Wed 18 Nov 2026"


def test_idea_describes_event_action_and_period() -> None:
    text = idea(SHORT)
    assert "agreement has been cancelled" in text
    assert "bet against its stock" in text
    assert "20 trading days (about a month)" in text
    assert "do worse than the stock market" in text
    beat = idea(LONG_BEAT)
    assert "profits clearly better than its recent trend" in beat
    assert "buy its stock" in beat


@pytest.mark.parametrize(
    ("stats", "words"),
    [
        (None, "no past cases"),
        ({"n": 334, "mean": -0.0023, "hit_rate": 0.49, "t_stat": -0.46}, "flipping a coin"),
        ({"n": 48, "mean": 0.0487, "hit_rate": 0.67, "t_stat": 2.29}, "does less well afterwards"),
        ({"n": 48, "mean": -0.03, "hit_rate": 0.3, "t_stat": -2.5}, "keep losing"),
        ({"n": 3, "mean": 0.01, "hit_rate": 0.6, "t_stat": None}, "too few cases"),
    ],
)
def test_expectation(stats: dict[str, float] | None, words: str) -> None:
    assert words in expectation(stats)


def test_expectation_reads_numbers_plainly() -> None:
    text = expectation({"n": 48, "mean": 0.0487, "hit_rate": 0.67, "t_stat": 2.29})
    assert "in 48 past cases this bet gained 4.9% per trade" in text
    assert "67 times out of 100" in text


def trade_row(status: Status, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "ticker": "ULTA",
        "known_at": datetime(2026, 11, 24, 21, 5, tzinfo=UTC),
        "entry_session": date(2026, 11, 25),
        "exit_session": date(2026, 11, 27),
        "sessions_held": 2,
        "status": status.value,
        "beta": 1.0,
        "stock_return": 0.031,
        "bench_return": 0.008,
        "pnl": 0.023,
    }
    return row | overrides


def test_open_long_trade_that_is_winning() -> None:
    text = trade(LONG_BEAT, trade_row(Status.OPEN), expected=-0.0023)
    assert "bought at the open on Wed 25 Nov 2026" in text
    assert "After 2 of 5 trading days" in text
    assert "ULTA rose 3.1% and the market rose 0.8%" in text
    assert "it did 2.3% better than the market" in text
    assert "The bet is winning 2.3% so far" in text
    assert "past trades like this averaged -0.2%" in text


def test_closed_short_trade_explains_beta_and_wins_when_stock_lags() -> None:
    row = trade_row(Status.CLOSED, beta=1.5, stock_return=-0.05, bench_return=0.01, pnl=0.065)
    text = trade(SHORT, row, expected=None)
    assert "bet against it at the open" in text
    assert "Ended at the close on Fri 27 Nov 2026" in text
    assert "ULTA fell 5.0% and the market rose 1.0%" in text
    assert "about 1.5 times as much as the market" in text
    assert "it did 6.5% worse than the market. The bet won 6.5%." in text


def test_interrupted_losing_trade() -> None:
    text = trade(LONG_BEAT, trade_row(Status.INTERRUPTED, pnl=-0.04), expected=None)
    assert "stopped trading part-way" in text
    assert "The bet lost 4.0%." in text


def test_pending_names_new_york_time() -> None:
    row = {"ticker": "ULTA", "known_at": datetime(2026, 11, 24, 21, 5, tzinfo=UTC)}
    text = pending(LONG_BEAT, row)
    assert "news on Tue 24 Nov 2026 at 16:05 New York time" in text
    assert "will buy it at the next market open" in text


def test_scoreboard() -> None:
    assert "no finished trades yet. 2 still in progress." in scoreboard(STARTED, None, None, 2)
    backtest = {"mean": 0.0487}
    few = {"n": 3, "mean": 0.02, "hit_rate": 2 / 3}
    text = scoreboard(STARTED, few, backtest, 0)
    assert "3 finished trade(s), 2 won, gaining 2.0% per trade" in text
    assert "(past cases: +4.9%)" in text
    assert "Too few to judge" in text
    many = {"n": MIN_LIVE_TRADES, "mean": -0.01, "hit_rate": 0.4}
    assert "not working the way it did" in scoreboard(STARTED, many, backtest, 0)
    holding = {"n": MIN_LIVE_TRADES, "mean": 0.01, "hit_rate": 0.6}
    assert "holding up" in scoreboard(STARTED, holding, backtest, 0)


def test_wrap_respects_width_and_indent() -> None:
    lines = wrap("word " * 60, "    ").splitlines()
    assert len(lines) > 1
    assert all(line.startswith("    ") and len(line) <= 92 for line in lines)
