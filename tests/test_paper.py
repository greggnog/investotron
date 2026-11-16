import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import cast

import polars as pl
import pytest

from investotron.event_study import Model, aligned_prices
from investotron.identifiers import Ticker
from investotron.paper import (
    Direction,
    Rule,
    RuleChangedError,
    RuleError,
    Status,
    load_rulebook,
    pending_signals,
    rule_trades,
    summarize_trades,
    sync_lock,
)
from investotron.prices import DailyBar
from investotron.storage import prices_frame

SHORT_102 = Rule("short-102", "1.02", Direction.SHORT, horizon=2)
T0 = datetime(2024, 1, 1, tzinfo=UTC)

# --- rules ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"name": "Bad Name"}, "lower-case"),
        ({"item": "1.2"}, "should look like"),
        ({"horizon": 0}, "horizon"),
        ({"surprise": "beat"}, "only applies"),
        ({"item": "2.02", "surprise": "huge"}, "surprise must be"),
    ],
)
def test_rule_validation(kwargs: dict[str, object], message: str) -> None:
    spec: dict[str, object] = {"name": "r", "item": "1.02", "direction": "long", "horizon": 5}
    with pytest.raises(RuleError, match=message):
        Rule.from_toml(spec | kwargs)


def test_rule_from_toml_rejects_unknown_and_missing_settings() -> None:
    with pytest.raises(RuleError, match="unknown setting"):
        Rule.from_toml({"name": "r", "item": "1.02", "direction": "long", "horizon": 5, "x": 1})
    with pytest.raises(RuleError, match="missing horizon"):
        Rule.from_toml({"name": "r", "item": "1.02", "direction": "long"})
    with pytest.raises(RuleError, match="long' or 'short"):
        Rule.from_toml({"name": "r", "item": "1.02", "direction": "up", "horizon": 5})


def test_fingerprint_ignores_name_but_not_behaviour() -> None:
    renamed = Rule("other-name", "1.02", Direction.SHORT, horizon=2)
    longer = Rule("short-102", "1.02", Direction.SHORT, horizon=3)
    assert SHORT_102.fingerprint() == renamed.fingerprint()
    assert SHORT_102.fingerprint() != longer.fingerprint()


def write_rules(tmp_path: Path, body: str) -> Path:
    (tmp_path / "u.txt").write_text("AAPL\n", encoding="utf-8")
    path = tmp_path / "rules.toml"
    path.write_text('universe = "u.txt"\n' + body, encoding="utf-8")
    return path


def test_load_rulebook(tmp_path: Path) -> None:
    path = write_rules(
        tmp_path,
        '[[rule]]\nname = "a"\nitem = "2.02"\ndirection = "long"\nhorizon = 5\nsurprise = "beat"\n',
    )
    book = load_rulebook(path)
    assert book.universe == (tmp_path / "u.txt").resolve()
    assert book.benchmark == Ticker("SPY")
    assert book.rules[0].surprise == "beat"


def test_load_rulebook_rejects_duplicates_and_empty(tmp_path: Path) -> None:
    rule = '[[rule]]\nname = "a"\nitem = "1.02"\ndirection = "long"\nhorizon = 5\n'
    with pytest.raises(RuleError, match="duplicate"):
        load_rulebook(write_rules(tmp_path, rule + rule))
    with pytest.raises(RuleError, match="no \\[\\[rule\\]\\]"):
        load_rulebook(write_rules(tmp_path, ""))


# --- lock --------------------------------------------------------------------------


def test_lock_starts_new_rules_once(tmp_path: Path) -> None:
    lock = tmp_path / "rules.lock.json"
    entries, started = sync_lock((SHORT_102,), lock, T0)
    assert started == ["short-102"]
    assert entries["short-102"].started_at == T0

    later = T0 + timedelta(days=3)
    entries, started = sync_lock((SHORT_102,), lock, later)
    assert started == []
    assert entries["short-102"].started_at == T0  # start date never moves


def test_lock_refuses_a_changed_rule(tmp_path: Path) -> None:
    lock = tmp_path / "rules.lock.json"
    sync_lock((SHORT_102,), lock, T0)
    edited = Rule("short-102", "1.02", Direction.SHORT, horizon=20)
    with pytest.raises(RuleChangedError, match="give the new version a new name"):
        sync_lock((edited,), lock, T0 + timedelta(days=1))


def test_lock_keeps_removed_rules(tmp_path: Path) -> None:
    lock = tmp_path / "rules.lock.json"
    sync_lock((SHORT_102,), lock, T0)
    other = Rule("long-202", "2.02", Direction.LONG, horizon=5)
    sync_lock((other,), lock, T0 + timedelta(days=1))
    assert set(json.loads(lock.read_text(encoding="utf-8"))) == {"short-102", "long-202"}


def test_lock_needs_aware_time(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        sync_lock((SHORT_102,), tmp_path / "l.json", datetime(2024, 1, 1))  # noqa: DTZ001


# --- trades ------------------------------------------------------------------------

DAYS = [date(2024, 1, d) for d in (8, 9, 10, 11, 12, 15, 16)]  # Mon..Fri, Mon, Tue


def priced(days: list[date], closes: list[float]) -> pl.DataFrame:
    """Each session opens at the previous close, so there are no overnight gaps."""
    opens = [closes[0], *closes[:-1]]
    return prices_frame(
        [
            DailyBar(Ticker("AAPL"), d, o, max(o, c), min(o, c), c, 1, o, c)
            for d, o, c in zip(days, opens, closes, strict=True)
        ]
    )


def prices_for(closes: list[float], days: list[date] = DAYS) -> pl.DataFrame:
    return aligned_prices(priced(days, closes), priced(days, [400.0] * len(days)), Model.MARKET)


def filings(*rows: tuple[str, list[str], datetime]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"ticker": "AAPL", "accession_number": a, "items": i, "known_at": k, "surprise": None}
            for a, i, k in rows
        ],
        schema={
            "ticker": pl.String,
            "accession_number": pl.String,
            "items": pl.List(pl.String),
            "known_at": pl.Datetime("us", "UTC"),
            "surprise": pl.String,
        },
    )


def after_close(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, 21, 30, tzinfo=UTC)


def test_short_trade_closes_after_horizon_and_profits_from_a_drop() -> None:
    prices = prices_for([100, 100, 90, 80, 80, 80, 80])
    # Filed after Mon close -> enter Tue open (100), hold 2 sessions -> exit Wed close (90).
    trades = rule_trades(SHORT_102, filings(("a", ["1.02"], after_close(DAYS[0]))), prices, T0)

    row = trades.row(0, named=True)
    assert row["status"] == Status.CLOSED
    assert (row["entry_session"], row["exit_session"]) == (DAYS[1], DAYS[2])
    assert row["stock_return"] == pytest.approx(-0.10)
    assert row["pnl"] == pytest.approx(0.10)  # short gains when the stock falls
    assert row["live"]


def test_trade_still_running_is_open_and_marked_at_latest_close() -> None:
    prices = prices_for([100, 100, 100, 100, 100, 100, 110])
    trades = rule_trades(SHORT_102, filings(("a", ["1.02"], after_close(DAYS[5]))), prices, T0)
    row = trades.row(0, named=True)
    assert row["status"] == Status.OPEN
    assert row["sessions_held"] == 1
    assert row["pnl"] == pytest.approx(-0.10)


def test_trade_cut_short_by_a_price_break_is_interrupted() -> None:
    days = [*DAYS[:3], date(2024, 3, 4), date(2024, 3, 5)]  # break after Wed Jan 10
    prices = prices_for([100, 100, 95, 1, 1], days)
    trades = rule_trades(SHORT_102, filings(("a", ["1.02"], after_close(days[1]))), prices, T0)
    row = trades.row(0, named=True)
    assert row["status"] == Status.INTERRUPTED
    assert row["exit_session"] == days[2]  # last tradable close before the break


def test_only_matching_filings_trade_and_live_is_split_at_start() -> None:
    prices = prices_for([100] * 7)
    started = after_close(DAYS[2])
    events = filings(
        ("before", ["1.02"], after_close(DAYS[0])),
        ("other-item", ["8.01"], after_close(DAYS[1])),
        ("after", ["1.02", "9.01"], after_close(DAYS[3])),
    )
    trades = rule_trades(SHORT_102, events, prices, started)
    assert dict(trades.select("accession_number", "live").iter_rows()) == {
        "before": False,
        "after": True,
    }


def test_surprise_rule_needs_matching_bucket() -> None:
    beat = Rule("beat", "2.02", Direction.LONG, horizon=1, surprise="beat")
    events = filings(
        ("a", ["2.02"], after_close(DAYS[0])), ("b", ["2.02"], after_close(DAYS[1]))
    ).with_columns(pl.Series("surprise", ["3 beat", "1 miss"]))
    trades = rule_trades(beat, events, prices_for([100] * 7), T0)
    assert trades["accession_number"].to_list() == ["a"]


def test_pending_means_live_and_after_the_markets_latest_open() -> None:
    prices = prices_for([100] * 7)
    market_last_open = cast("datetime", prices["opens_at"].max())
    started = after_close(DAYS[0]) + timedelta(hours=1)
    events = filings(
        ("backtest-era", ["1.02"], after_close(DAYS[-1]) - timedelta(days=30)),
        ("traded", ["1.02"], after_close(DAYS[2])),
        ("new", ["1.02"], after_close(DAYS[-1])),
    )
    pending = pending_signals(SHORT_102, events, market_last_open, started)
    assert pending["accession_number"].to_list() == ["new"]
    assert "new" not in rule_trades(SHORT_102, events, prices, T0)["accession_number"].to_list()


def test_stock_that_stopped_trading_has_nothing_pending() -> None:
    # The stock's data ends on DAYS[3]; the market kept going to DAYS[-1].
    stock_prices = prices_for([100] * 4, DAYS[:4])
    market_last_open = cast("datetime", prices_for([100] * 7)["opens_at"].max())
    events = filings(("acquired", ["1.02"], after_close(DAYS[3])))
    assert pending_signals(SHORT_102, events, market_last_open, T0).is_empty()
    assert rule_trades(SHORT_102, events, stock_prices, T0).is_empty()


def test_summarize_trades_splits_live_and_backtest_and_skips_open() -> None:
    trades = pl.DataFrame(
        {
            "live": [False, False, True, True],
            "status": ["closed", "closed", "closed", "open"],
            "entry_session": [date(2024, 1, 2), date(2024, 2, 5), date(2024, 3, 4), DAYS[0]],
            "pnl": [0.02, 0.04, -0.01, 0.5],
        }
    )
    summary = {r["sample"]: r for r in summarize_trades(trades, 5).iter_rows(named=True)}
    assert summary["backtest"]["n"] == 2
    assert summary["backtest"]["mean"] == pytest.approx(0.03)
    assert summary["live"]["n"] == 1  # the open trade isn't finished yet
