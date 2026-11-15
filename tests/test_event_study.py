from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

from investotron.event_study import (
    ALL_ITEMS,
    Model,
    Windows,
    aligned_prices,
    attach_surprises,
    calendar_block,
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
PRICES = aligned_prices(STOCK, BENCH, Model.MARKET)


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
    aligned = aligned_prices(
        STOCK, BENCH.filter(pl.col("session") != date(2024, 1, 4)), Model.MARKET
    )
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
    return aligned_prices(prices_frame(stock_bars), prices_frame(bench_bars), Model.MARKET)


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


def market_linked(n: int, multiplier: float, switch_at: int | None = None) -> pl.DataFrame:
    """Benchmark moves by varied daily returns; the stock moves ``multiplier`` times
    as much (3x from ``switch_at`` on). Opens equal the previous close, so 1-day
    open-to-close returns are just the daily returns."""
    weekdays = [
        d
        for d in pl.date_range(date(2023, 1, 2), date(2025, 12, 31), "1d", eager=True).to_list()
        if d.weekday() < 5
    ][:n]
    stock_bars: list[DailyBar] = []
    bench_bars: list[DailyBar] = []
    stock_level, bench_level = 100.0, 400.0
    for i, day in enumerate(weekdays):
        r = 0.004 * ((i * 7) % 5 - 2)  # -0.8% .. +0.8%
        m = 3.0 if switch_at is not None and i >= switch_at else multiplier
        stock_open, bench_open = stock_level, bench_level
        stock_level *= 1 + m * r
        bench_level *= 1 + r
        for bars_list, ticker, o, c in (
            (stock_bars, "AAPL", stock_open, stock_level),
            (bench_bars, "SPY", bench_open, bench_level),
        ):
            bars_list.append(DailyBar(Ticker(ticker), day, o, max(o, c), min(o, c), c, 1, o, c))
    return aligned_prices(
        prices_frame(stock_bars),
        prices_frame(bench_bars),
        Model.BETA,
        Windows(beta_sessions=60, min_beta_sessions=30),
    )


def test_beta_recovers_the_stocks_market_sensitivity() -> None:
    prices = market_linked(100, multiplier=2.0)
    assert prices["beta"][29] is None  # not enough history yet
    assert prices["beta"][-1] == pytest.approx(2.0)
    assert prices["daily_abnormal"].tail(50).abs().max() == pytest.approx(0, abs=1e-12)


def test_beta_never_uses_later_sessions() -> None:
    full = market_linked(150, multiplier=1.0, switch_at=100)
    truncated = market_linked(100, multiplier=1.0)
    assert full["beta"][:100].to_list() == truncated["beta"].to_list()
    assert full["beta"][99] == pytest.approx(1.0)
    assert full["beta"][-1] > 2  # picks up the change only once it has happened


def test_beta_model_removes_market_exposure_from_abnormal_returns() -> None:
    prices = market_linked(120, multiplier=2.0)
    event = filings(("a", ["8.01"], prices.row(100, named=True)["closes_at"]))

    beta_row = event_returns(event, prices, [1]).row(0, named=True)
    assert beta_row["beta"] == pytest.approx(2.0)
    assert beta_row["abnormal_return"] == pytest.approx(0, abs=1e-12)

    market_prices = market_linked(120, multiplier=2.0).with_columns(pl.lit(1.0).alias("beta"))
    market_row = event_returns(event, market_prices, [1]).row(0, named=True)
    assert market_row["abnormal_return"] == pytest.approx(market_row["bench_return"])


def test_calendar_block_groups_by_week_scaled_to_horizon() -> None:
    days = [
        date(2024, 1, 1),
        date(2024, 1, 5),
        date(2024, 1, 8),
        date(2024, 1, 26),
        date(2024, 1, 29),
    ]
    frame = pl.DataFrame({"d": days})

    def blocks(horizon: int) -> list[int]:
        return frame.select(calendar_block(pl.col("d"), pl.lit(horizon)).alias("b"))["b"].to_list()

    weekly = blocks(1)  # Mon, Fri same week; next Mon new block
    assert weekly[0] == weekly[1] != weekly[2]
    assert blocks(5) == weekly
    # 4-week blocks: 16 consecutive Mondays fall into runs of at most 4, never
    # sharing a block with the Monday 4 weeks later.
    mondays = pl.DataFrame({"d": [date(2024, 1, 1) + timedelta(weeks=i) for i in range(16)]})
    four_weekly = mondays.select(calendar_block(pl.col("d"), pl.lit(20)).alias("b"))["b"]
    assert four_weekly.is_sorted()
    assert four_weekly.value_counts()["count"].max() == 4
    assert all(four_weekly[i] != four_weekly[i + 4] for i in range(12))


def returns_frame(rows: list[tuple[date, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "items": [["8.01"]] * len(rows),
            "horizon": [1] * len(rows),
            "entry_session": [d for d, _ in rows],
            "abnormal_return": [r for _, r in rows],
        },
        schema={
            "items": pl.List(pl.String),
            "horizon": pl.Int32,
            "entry_session": pl.Date,
            "abnormal_return": pl.Float64,
        },
    )


def test_clustered_t_matches_naive_with_one_event_per_week() -> None:
    weeks = [date(2024, 1, 1) + timedelta(weeks=i) for i in range(6)]
    summary = summarize(
        returns_frame(list(zip(weeks, [0.01, 0.03, -0.01, 0.02, 0.0, 0.015], strict=True)))
    )
    row = summary.row(0, named=True)
    assert row["clusters"] == 6
    assert row["t_stat"] == pytest.approx(row["t_naive"])


def test_clustered_t_shrinks_when_events_share_a_week() -> None:
    # Three weeks; within each week all stocks moved together.
    rows = [
        (date(2024, 1, 1) + timedelta(weeks=w), r)
        for w, r in ((0, 0.02), (1, 0.01), (2, -0.005))
        for _ in range(5)
    ]
    row = summarize(returns_frame(rows)).row(0, named=True)
    assert row["n"] == 15
    assert row["clusters"] == 3
    assert abs(row["t_stat"]) < abs(row["t_naive"]) / 2


def test_single_cluster_has_no_clustered_t() -> None:
    row = summarize(returns_frame([(date(2024, 1, 1), 0.01), (date(2024, 1, 2), 0.02)])).row(
        0, named=True
    )
    assert row["t_stat"] is None


def test_events_without_a_session_soon_after_are_dropped() -> None:
    before_history = datetime(2008, 8, 20, 16, 0, tzinfo=UTC)
    returns = event_returns(filings(("old", ["2.02"], before_history)), PRICES, [1])
    assert returns.height == 0
