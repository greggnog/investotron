from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

from investotron.event_study import (
    ALL_ITEMS,
    aligned_prices,
    attach_surprises,
    event_reactions,
    event_returns,
    summarize,
    summarize_reactions,
    surprise_bucket,
)
from investotron.identifiers import Ticker
from investotron.prices import DailyBar
from investotron.storage import prices_frame

SESSIONS = [date(2024, 1, d) for d in (2, 3, 4, 5, 8)]  # Tue..Fri, then Mon


def bars(ticker: str, opens: list[float], closes: list[float]) -> pl.DataFrame:
    return prices_frame(
        [
            DailyBar(
                ticker=Ticker(ticker),
                session=s,
                open=o,
                high=max(o, c),
                low=min(o, c),
                close=c,
                volume=1,
                adj_open=o,
                adj_close=c,
            )
            for s, o, c in zip(SESSIONS, opens, closes, strict=True)
        ]
    )


# Stock opens at 100 on day 2 and climbs; the benchmark is flat except +1% on day 2.
STOCK = bars("AAPL", [100, 100, 104, 106, 108], [100, 102, 105, 107, 110])
BENCH = bars("SPY", [400, 400, 404, 404, 404], [400, 404, 404, 404, 404])
PRICES = aligned_prices(STOCK, BENCH)


def filings(*rows: tuple[str, list[str], datetime]) -> pl.DataFrame:
    return pl.DataFrame(
        [{"accession_number": a, "items": i, "known_at": k} for a, i, k in rows],
        schema={
            "accession_number": pl.String,
            "items": pl.List(pl.String),
            "known_at": pl.Datetime("us", "UTC"),
        },
    )


AFTER_CLOSE_JAN2 = datetime(2024, 1, 2, 21, 30, tzinfo=UTC)  # 16:30 ET


def test_after_close_event_enters_next_open() -> None:
    returns = event_returns(filings(("a", ["2.02"], AFTER_CLOSE_JAN2)), PRICES, [1, 2])

    one_day, two_day = returns.iter_rows(named=True)
    assert one_day["entry_session"] == date(2024, 1, 3)
    assert one_day["exit_session"] == date(2024, 1, 3)
    assert one_day["stock_return"] == pytest.approx(0.02)  # 100 -> 102
    assert one_day["bench_return"] == pytest.approx(0.01)  # 400 -> 404
    assert one_day["abnormal_return"] == pytest.approx(0.01)
    assert two_day["exit_session"] == date(2024, 1, 4)
    assert two_day["stock_return"] == pytest.approx(0.05)  # 100 -> 105


def test_event_exactly_at_open_waits_for_next_session() -> None:
    at_open = datetime(2024, 1, 3, 14, 30, tzinfo=UTC)  # 09:30:00 ET
    returns = event_returns(filings(("a", [], at_open)), PRICES, [1])
    assert returns["entry_session"].to_list() == [date(2024, 1, 4)]


def test_events_past_end_of_data_are_dropped() -> None:
    late = datetime(2024, 1, 5, 21, 0, tzinfo=UTC)  # entry Mon 8th, the last session
    after_all = datetime(2024, 1, 8, 22, 0, tzinfo=UTC)
    returns = event_returns(filings(("a", [], late), ("b", [], after_all)), PRICES, [1, 2])
    assert returns.select("accession_number", "horizon").rows() == [("a", 1)]


def test_horizons_must_be_positive() -> None:
    with pytest.raises(ValueError, match="horizons"):
        event_returns(filings(), PRICES, [0])


def test_aligned_prices_keeps_only_shared_sessions() -> None:
    aligned = aligned_prices(STOCK, BENCH.filter(pl.col("session") != date(2024, 1, 4)))
    assert aligned.height == 4
    assert aligned["session_index"].to_list() == [0, 1, 2, 3]


def test_summarize_counts_each_item_and_all() -> None:
    returns = event_returns(
        filings(
            ("a", ["2.02", "9.01"], AFTER_CLOSE_JAN2),
            ("b", ["9.01"], datetime(2024, 1, 3, 21, 30, tzinfo=UTC)),
        ),
        PRICES,
        [1],
    )
    summary = {row["group"]: row for row in summarize(returns).iter_rows(named=True)}

    assert summary[ALL_ITEMS]["n"] == 2
    assert summary["2.02"]["n"] == 1
    assert summary["9.01"]["n"] == 2
    # b: stock 104 -> 105 (+0.96%), bench flat -> abnormal +0.96%; a: +1%
    assert summary["9.01"]["mean"] == pytest.approx((0.01 + 1 / 104) / 2)
    assert summary["9.01"]["hit_rate"] == 1.0
    assert next(iter(summary)) == ALL_ITEMS


def long_prices(n: int, stock_jump_at: int | None = None) -> pl.DataFrame:
    """n weekday sessions from 2024-01-01; the stock wiggles +-1% around the
    benchmark, and optionally gaps +10% overnight into session ``stock_jump_at``."""
    sessions = pl.date_range(date(2024, 1, 1), date(2025, 12, 31), "1d", eager=True)
    weekdays = [d for d in sessions.to_list() if d.weekday() < 5][:n]
    stock_bars: list[DailyBar] = []
    bench_bars: list[DailyBar] = []
    level = 100.0
    for i, day in enumerate(weekdays):
        if i == stock_jump_at:
            level *= 1.10
        close = level * (1.01 if i % 2 else 0.99)
        stock_bars.append(
            DailyBar(
                Ticker("AAPL"),
                day,
                level,
                max(level, close),
                min(level, close),
                close,
                1,
                level,
                close,
            )
        )
        bench_bars.append(DailyBar(Ticker("SPY"), day, 400, 400, 400, 400, 1, 400, 400))
    return aligned_prices(prices_frame(stock_bars), prices_frame(bench_bars))


def test_reaction_includes_overnight_gap_and_is_scaled_by_normal_volatility() -> None:
    prices = long_prices(80, stock_jump_at=70)
    day_69 = prices.row(69, named=True)
    event = filings(("a", ["2.02"], day_69["closes_at"] + timedelta(minutes=30)))

    reaction = event_reactions(event, prices).row(0, named=True)

    assert reaction["entry_session"] == prices.row(70, named=True)["session"]
    assert reaction["reaction_sessions"] == 1
    # 10% gap, from a close 1% above the level to one 1% below it
    assert reaction["reaction"] == pytest.approx(1.10 * 0.99 / 1.01 - 1)
    assert reaction["z"] > 3  # far outside the +-2% daily wiggle


def test_reaction_needs_baseline_history() -> None:
    prices = long_prices(30)  # fewer than MIN_BASELINE_SESSIONS
    event = filings(("a", [], prices.row(20, named=True)["closes_at"]))
    assert event_reactions(event, prices).height == 0


def test_summarize_reactions_by_group() -> None:
    prices = long_prices(80, stock_jump_at=70)
    events = filings(
        ("a", ["2.02"], prices.row(69, named=True)["closes_at"] + timedelta(minutes=30)),
        ("b", ["8.01"], prices.row(60, named=True)["closes_at"] + timedelta(minutes=30)),
    )
    summary = {
        row["group"]: row
        for row in summarize_reactions(event_reactions(events, prices)).iter_rows(named=True)
    }
    assert summary[ALL_ITEMS]["n"] == 2
    assert summary["2.02"]["mean_abs_z"] > summary["8.01"]["mean_abs_z"]


def test_attach_surprises_matches_latest_quarter_within_lag() -> None:
    events = filings(
        ("q4", ["2.02", "9.01"], datetime(2024, 1, 30, 21, 30, tzinfo=UTC)),
        ("q1", ["2.02"], datetime(2024, 5, 2, 20, 30, tzinfo=UTC)),
        (
            "late",
            ["2.02"],
            datetime(2024, 9, 30, 20, 30, tzinfo=UTC),
        ),  # latest quarter is 183 days old
        ("other", ["8.01"], datetime(2024, 5, 3, 20, 30, tzinfo=UTC)),
    )
    surprises = pl.DataFrame(
        {
            "period_end": [date(2023, 12, 31), date(2024, 3, 31)],
            "sue": [1.5, -0.5],
        }
    )
    matched = attach_surprises(events, surprises)
    by_accession = {r["accession_number"]: r for r in matched.iter_rows(named=True)}

    assert set(by_accession) == {"q4", "q1", "late"}  # only earnings 8-Ks
    assert by_accession["q4"]["sue"] == 1.5
    assert by_accession["q1"]["period_end"] == date(2024, 3, 31)
    assert by_accession["late"]["sue"] is None  # Mar 31 quarter is too old by Sep 30


@pytest.mark.parametrize(
    ("sue", "bucket"),
    [(-2.0, "1 miss"), (-1.0, "1 miss"), (0.3, "2 in line"), (1.0, "3 beat"), (None, None)],
)
def test_surprise_bucket(sue: float | None, bucket: str | None) -> None:
    frame = pl.DataFrame({"sue": [sue]}, schema={"sue": pl.Float64})
    assert frame.select(surprise_bucket(pl.col("sue")).alias("b"))["b"][0] == bucket
