"""Event study: how did a stock move after each event, compared with the market?

Two questions, two measures:

**Direction (tradable).** :func:`event_returns`.

1. **Entry** is the open of the first session that opens strictly *after* the event's
   ``known_at``. That's the earliest anyone could have traded on it. A filing at
   exactly 9:30:00 misses that day's opening auction, hence "strictly".
2. For a horizon of ``h`` sessions, **exit** is the close of session ``entry + h - 1``,
   so ``h = 1`` means "open to close on the first tradable day".
3. **Abnormal return** = stock return minus benchmark return over the same window
   (a "market-adjusted" model: it assumes beta = 1. Good enough for a first look).

**Size (not tradable, but tells you whether the event mattered).** :func:`event_reactions`.
Measured from the last close *before* the event to the close of the entry session,
so it includes the overnight gap where most of an after-hours event's reaction
happens. Scaled by the stock's normal day-to-day abnormal volatility over the
preceding sessions, giving a z-score. If events carried no news, |z| would average
about 0.8 and exceed 2 about 5% of the time.

Caveats when reading results:

* Windows of nearby events overlap, and events in the same week across stocks share
  market moves, so they aren't independent and t-stats overstate confidence.
* One filing usually has several items (2.02 earnings + 9.01 exhibits), so it's
  counted once under each.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta

import polars as pl

DEFAULT_HORIZONS = (1, 5, 20)
EARNINGS_ITEM = "2.02"
MAX_REPORT_LAG = timedelta(days=75)  # quarter end -> earnings release
ALL_ITEMS = "all"
BASELINE_SESSIONS = 60
MIN_BASELINE_SESSIONS = 40

# Reference values for |z| when an event carries no information (standard normal).
NULL_MEAN_ABS_Z = 0.798
NULL_SHARE_ABOVE_2 = 0.046


def aligned_prices(
    stock: pl.DataFrame,
    benchmark: pl.DataFrame,
    baseline_sessions: int = BASELINE_SESSIONS,
    min_baseline_sessions: int = MIN_BASELINE_SESSIONS,
) -> pl.DataFrame:
    """Sessions both series have, numbered 0..n-1 so horizons count trading days.

    Adds ``daily_abnormal`` (close-to-close stock minus benchmark return) and
    ``baseline_std``: its standard deviation over the ``baseline_sessions`` up to
    and including that session -- i.e. only data available at that session's close.
    """
    daily_abnormal = (pl.col("adj_close") / pl.col("adj_close").shift(1)) - (
        pl.col("bench_adj_close") / pl.col("bench_adj_close").shift(1)
    )
    return (
        stock.select("session", "opens_at", "closes_at", "adj_open", "adj_close")
        .join(
            benchmark.select(
                "session",
                pl.col("adj_open").alias("bench_adj_open"),
                pl.col("adj_close").alias("bench_adj_close"),
            ),
            on="session",
            how="inner",
        )
        .sort("session")
        .with_row_index("session_index")
        .with_columns(daily_abnormal.alias("daily_abnormal"))
        .with_columns(
            pl.col("daily_abnormal")
            .rolling_std(baseline_sessions, min_samples=min_baseline_sessions)
            .alias("baseline_std")
        )
    )


def _with_entry(events: pl.DataFrame, prices: pl.DataFrame) -> pl.DataFrame:
    """Attach the first session opening strictly after ``known_at``."""
    return (
        events.sort("known_at")
        .join_asof(
            prices.select(
                "opens_at",
                pl.col("session_index").alias("entry_index"),
                pl.col("session").alias("entry_session"),
                pl.col("adj_open").alias("entry_open"),
                pl.col("bench_adj_open").alias("bench_entry_open"),
            ),
            left_on="known_at",
            right_on="opens_at",
            strategy="forward",
            allow_exact_matches=False,
            coalesce=False,
        )
        .drop("opens_at")
        .drop_nulls("entry_index")
    )


def event_returns(
    events: pl.DataFrame,
    prices: pl.DataFrame,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> pl.DataFrame:
    """One row per (event, horizon) with stock, benchmark and abnormal returns.

    ``events`` needs ``accession_number``, ``items`` and ``known_at`` (extra
    columns are carried through); ``prices`` comes from :func:`aligned_prices`.
    Events with no later session to enter on, or whose exit is past the end of the
    data, are left out of that horizon.
    """
    if not horizons or min(horizons) < 1:
        raise ValueError(f"horizons must be >= 1, got {list(horizons)}")

    entries = _with_entry(events, prices)
    exits = prices.select(
        pl.col("session_index").alias("exit_index"),
        pl.col("session").alias("exit_session"),
        pl.col("adj_close").alias("exit_close"),
        pl.col("bench_adj_close").alias("bench_exit_close"),
    )
    per_horizon = [
        entries.with_columns(
            pl.lit(h, dtype=pl.Int32).alias("horizon"),
            (pl.col("entry_index") + (h - 1)).alias("exit_index"),
        ).join(exits, on="exit_index", how="inner")
        for h in horizons
    ]

    stock_return = pl.col("exit_close") / pl.col("entry_open") - 1
    bench_return = pl.col("bench_exit_close") / pl.col("bench_entry_open") - 1
    return (
        pl.concat(per_horizon)
        .with_columns(
            stock_return.alias("stock_return"),
            bench_return.alias("bench_return"),
            (stock_return - bench_return).alias("abnormal_return"),
        )
        .drop(
            "entry_index",
            "exit_index",
            "entry_open",
            "bench_entry_open",
            "exit_close",
            "bench_exit_close",
        )
        .sort("known_at", "horizon")
    )


def event_reactions(events: pl.DataFrame, prices: pl.DataFrame) -> pl.DataFrame:
    """One row per event: abnormal move from the last close before it to the close
    of the entry session, and that move as a z-score against normal volatility.

    Events without enough prior history for a baseline are dropped.
    """
    starts = prices.select(
        "closes_at",
        pl.col("session_index").alias("start_index"),
        pl.col("adj_close").alias("start_close"),
        pl.col("bench_adj_close").alias("bench_start_close"),
        "baseline_std",
    )
    ends = prices.select(
        pl.col("session_index").alias("entry_index"),
        pl.col("adj_close").alias("end_close"),
        pl.col("bench_adj_close").alias("bench_end_close"),
    )
    reaction = (pl.col("end_close") / pl.col("start_close")) - (
        pl.col("bench_end_close") / pl.col("bench_start_close")
    )
    sessions = (pl.col("entry_index") - pl.col("start_index")).cast(pl.Float64)
    return (
        _with_entry(events, prices)
        .join_asof(
            starts,
            left_on="known_at",
            right_on="closes_at",
            strategy="backward",  # last close at or before the event
            coalesce=False,
        )
        .drop_nulls("baseline_std")
        .join(ends, on="entry_index", how="inner")
        .with_columns(
            reaction.alias("reaction"),
            sessions.alias("reaction_sessions"),
        )
        .with_columns((pl.col("reaction") / (pl.col("baseline_std") * sessions.sqrt())).alias("z"))
        .select(
            pl.exclude(
                "closes_at",
                "start_index",
                "start_close",
                "bench_start_close",
                "entry_index",
                "end_close",
                "bench_end_close",
                "entry_open",
                "bench_entry_open",
            )
        )
        .sort("known_at")
    )


def _by_group(frame: pl.DataFrame, by: str) -> pl.DataFrame:
    """Rows labelled with ``group``: once as "all", then once per value of ``by``
    (list columns like ``items`` count each element)."""
    grouped = frame.explode(by) if isinstance(frame.schema[by], pl.List) else frame
    everything = frame.with_columns(pl.lit(ALL_ITEMS).alias("group")).drop(by)
    per_value = grouped.with_columns(
        pl.col(by).cast(pl.String).fill_null("none").alias("group")
    ).drop(by)
    return pl.concat([everything, per_value])


def summarize(returns: pl.DataFrame, by: str = "items") -> pl.DataFrame:
    """Abnormal-return statistics per group and horizon, plus an "all" row.

    ``t_stat`` is mean / standard error -- roughly, |t| > 2 suggests the average
    isn't zero by luck, *if* events were independent (see module caveats).
    """
    abnormal = pl.col("abnormal_return")
    return (
        _by_group(returns, by)
        .group_by("group", "horizon")
        .agg(
            pl.len().alias("n"),
            abnormal.mean().alias("mean"),
            abnormal.median().alias("median"),
            (abnormal.mean() / (abnormal.std() / pl.len().sqrt())).alias("t_stat"),
            (abnormal > 0).mean().alias("hit_rate"),
        )
        .sort(pl.col("group") != ALL_ITEMS, "group", "horizon")
    )


def summarize_reactions(reactions: pl.DataFrame, by: str = "items") -> pl.DataFrame:
    """Reaction size per group: compare ``mean_abs_z`` with ~0.8 and
    ``share_above_2`` with ~5%, the values for events that carry no news."""
    abs_z = pl.col("z").abs()
    return (
        _by_group(reactions, by)
        .group_by("group")
        .agg(
            pl.len().alias("n"),
            abs_z.mean().alias("mean_abs_z"),
            abs_z.median().alias("median_abs_z"),
            (abs_z > 2).mean().alias("share_above_2"),  # noqa: PLR2004
            pl.col("reaction").abs().mean().alias("mean_abs_reaction"),
            pl.col("reaction").mean().alias("mean_reaction"),
        )
        .sort(pl.col("group") != ALL_ITEMS, "group")
    )


def surprise_bucket(sue: pl.Expr, threshold: float = 1.0) -> pl.Expr:
    """Label SUE as a miss / in line / beat, using a fixed threshold (not sample
    quantiles, which would quietly use the whole period's data)."""
    return (
        pl.when(sue <= -threshold)
        .then(pl.lit("1 miss"))
        .when(sue >= threshold)
        .then(pl.lit("3 beat"))
        .when(sue.is_not_null())
        .then(pl.lit("2 in line"))
        .otherwise(None)
    )


def attach_surprises(events: pl.DataFrame, surprises: pl.DataFrame) -> pl.DataFrame:
    """Earnings events (item 2.02), each with the SUE of the quarter it reports.

    The quarter is the latest one that ended before the filing, within
    ``MAX_REPORT_LAG``. Earnings 8-Ks with no matching quarter get a null ``sue``.
    """
    quarters = surprises.select(
        (pl.col("period_end").cast(pl.Datetime("us", "UTC")) + pl.duration(days=1)).alias(
            "period_closed_at"
        ),
        "period_end",
        "sue",
    ).sort("period_closed_at")
    return (
        events.filter(pl.col("items").list.contains(EARNINGS_ITEM))
        .sort("known_at")
        .join_asof(
            quarters,
            left_on="known_at",
            right_on="period_closed_at",
            strategy="backward",
            tolerance=MAX_REPORT_LAG,
            coalesce=False,
        )
        .drop("period_closed_at")
    )
