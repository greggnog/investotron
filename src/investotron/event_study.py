"""Event study: how did a stock move after each event, compared with the market?

Two questions, two measures:

**Direction (tradable).** :func:`event_returns`.

1. **Entry** is the open of the first session that opens strictly *after* the event's
   ``known_at``. That's the earliest anyone could have traded on it. A filing at
   exactly 9:30:00 misses that day's opening auction, hence "strictly".
2. For a horizon of ``h`` sessions, **exit** is the close of session ``entry + h - 1``,
   so ``h = 1`` means "open to close on the first tradable day".
3. **Abnormal return** = stock return - beta x benchmark return over the same window.

**Size (not tradable, but tells you whether the event mattered).** :func:`event_reactions`.
Measured from the last close *before* the event to the close of the entry session,
so it includes the overnight gap where most of an after-hours event's reaction
happens. Scaled by the stock's normal day-to-day abnormal volatility over the
preceding sessions, giving a z-score. If events carried no news, |z| would average
about 0.8 and exceed 2 about 5% of the time.

**Beta.** With :attr:`Model.BETA` (the default), beta is the slope of the stock's
daily returns on the benchmark's over the previous ``BETA_SESSIONS`` sessions,
using only closes before the window starts -- so it's point-in-time too. There's no
intercept (alpha): a year of past outperformance is mostly noise, and extrapolating
it would bake momentum into "expected" returns. :attr:`Model.MARKET` fixes beta = 1,
the simpler "market-adjusted" model, kept for comparison.

**Standard errors.** Events aren't independent: earnings seasons put many stocks'
windows in the same weeks, and a 20-day window overlaps the next few weeks' events.
``t_stat`` clusters events by calendar block (one week for horizons up to 5 days,
``ceil(h / 5)`` weeks beyond that), which allows any correlation within a block.
Correlation between neighbouring blocks is still ignored, so it's an improvement,
not a guarantee. ``t_naive`` is the old independent-events version, for comparison.

One filing usually has several items (2.02 earnings + 9.01 exhibits), so it's
counted once under each.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum

import polars as pl

DEFAULT_HORIZONS = (1, 5, 20)
EARNINGS_ITEM = "2.02"
MAX_REPORT_LAG = timedelta(days=75)  # quarter end -> earnings release
# Longest normal gap between an event and the next open (a holiday weekend). Events
# with no session that soon -- before the price history starts, or during a gap in
# it -- are dropped rather than matched to a session days or years later.
MAX_ENTRY_DELAY = timedelta(days=5)
# Consecutive sessions further apart than this mark a break in the price history
# (a trading halt, a delisting, missing data). Returns are never measured across one.
MAX_SESSION_GAP = timedelta(days=5)
ALL_ITEMS = "all"
BASELINE_SESSIONS = 60
MIN_BASELINE_SESSIONS = 40
BETA_SESSIONS = 250  # about a year
MIN_BETA_SESSIONS = 120

# Reference values for |z| when an event carries no information (standard normal).
NULL_MEAN_ABS_Z = 0.798
NULL_SHARE_ABOVE_2 = 0.046

_FIRST_MONDAY = 4  # 1970-01-05, in days since the epoch; calendar blocks start Mondays
_SESSIONS_PER_WEEK = 5


class Model(StrEnum):
    BETA = "beta"
    MARKET = "market"


@dataclass(frozen=True, slots=True)
class Windows:
    """Trailing windows (in sessions) for beta and normal volatility, with the
    minimum history each needs before it's trusted."""

    beta_sessions: int = BETA_SESSIONS
    min_beta_sessions: int = MIN_BETA_SESSIONS
    baseline_sessions: int = BASELINE_SESSIONS
    min_baseline_sessions: int = MIN_BASELINE_SESSIONS


def aligned_prices(
    stock: pl.DataFrame,
    benchmark: pl.DataFrame,
    model: Model = Model.BETA,
    windows: Windows = Windows(),
) -> pl.DataFrame:
    """Sessions both series actually traded, numbered 0..n-1 so horizons count
    trading days.

    Zero-volume sessions are dropped: vendors often carry the last price forward
    through halts and after delistings, and those "prices" were never tradable.
    Sessions are split into ``segment``s wherever the gap between consecutive
    sessions exceeds ``MAX_SESSION_GAP``; the return across a break is left null,
    so it never feeds beta or volatility, and windows can't span segments.

    Adds, all using only data up to that session's close:

    * ``beta`` -- trailing beta of the stock on the benchmark (1.0 for ``Model.MARKET``).
    * ``daily_abnormal`` -- close-to-close stock return minus the *previous* session's
      beta times the benchmark return.
    * ``baseline_std`` -- standard deviation of ``daily_abnormal`` over the last
      ``windows.baseline_sessions``.
    """
    w = windows
    after_break = pl.col("session").diff().dt.total_days() > MAX_SESSION_GAP.days
    stock_daily = (
        pl.when(after_break.not_())
        .then(pl.col("adj_close") / pl.col("adj_close").shift(1) - 1)
        .otherwise(None)
    )
    bench_daily = (
        pl.when(after_break.not_())
        .then(pl.col("bench_adj_close") / pl.col("bench_adj_close").shift(1) - 1)
        .otherwise(None)
    )
    if model is Model.BETA:
        beta = pl.rolling_cov(
            "stock_daily",
            "bench_daily",
            window_size=w.beta_sessions,
            min_samples=w.min_beta_sessions,
        ) / pl.col("bench_daily").rolling_var(w.beta_sessions, min_samples=w.min_beta_sessions)
    else:
        beta = pl.lit(1.0)
    return (
        stock.filter(pl.col("volume") > 0)
        .select("session", "opens_at", "closes_at", "adj_open", "adj_close")
        .join(
            benchmark.filter(pl.col("volume") > 0).select(
                "session",
                pl.col("adj_open").alias("bench_adj_open"),
                pl.col("adj_close").alias("bench_adj_close"),
            ),
            on="session",
            how="inner",
        )
        .sort("session")
        .with_row_index("session_index")
        .with_columns(after_break.fill_null(False).cum_sum().alias("segment"))
        .with_columns(stock_daily.alias("stock_daily"), bench_daily.alias("bench_daily"))
        .with_columns(beta.alias("beta"))
        .with_columns(
            (pl.col("stock_daily") - pl.col("beta").shift(1) * pl.col("bench_daily")).alias(
                "daily_abnormal"
            )
        )
        .with_columns(
            pl.col("daily_abnormal")
            .rolling_std(w.baseline_sessions, min_samples=w.min_baseline_sessions)
            .alias("baseline_std")
        )
    )


def attach_entry(events: pl.DataFrame, prices: pl.DataFrame) -> pl.DataFrame:
    """Attach the first session opening strictly after ``known_at`` (within
    ``MAX_ENTRY_DELAY``), plus the beta known at the close before it (``entry_beta``)."""
    return (
        events.sort("known_at")
        .join_asof(
            prices.select(
                "opens_at",
                pl.col("session_index").alias("entry_index"),
                pl.col("session").alias("entry_session"),
                pl.col("adj_open").alias("entry_open"),
                pl.col("bench_adj_open").alias("bench_entry_open"),
                pl.col("beta").shift(1).alias("entry_beta"),
                pl.col("segment").alias("entry_segment"),
            ),
            left_on="known_at",
            right_on="opens_at",
            strategy="forward",
            allow_exact_matches=False,
            tolerance=MAX_ENTRY_DELAY,
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
    Events with no later session to enter on, no beta yet, or whose exit is past
    the end of the data or across a break in it, are left out of that horizon.
    """
    if not horizons or min(horizons) < 1:
        raise ValueError(f"horizons must be >= 1, got {list(horizons)}")

    entries = attach_entry(events, prices).drop_nulls("entry_beta").rename({"entry_beta": "beta"})
    exits = prices.select(
        pl.col("session_index").alias("exit_index"),
        pl.col("session").alias("exit_session"),
        pl.col("adj_close").alias("exit_close"),
        pl.col("bench_adj_close").alias("bench_exit_close"),
        pl.col("segment").alias("exit_segment"),
    )
    per_horizon = [
        entries.with_columns(
            pl.lit(h, dtype=pl.Int32).alias("horizon"),
            (pl.col("entry_index") + (h - 1)).alias("exit_index"),
        )
        .join(exits, on="exit_index", how="inner")
        .filter(pl.col("entry_segment") == pl.col("exit_segment"))
        for h in horizons
    ]

    stock_return = pl.col("exit_close") / pl.col("entry_open") - 1
    bench_return = pl.col("bench_exit_close") / pl.col("bench_entry_open") - 1
    return (
        pl.concat(per_horizon)
        .with_columns(
            stock_return.alias("stock_return"),
            bench_return.alias("bench_return"),
            (stock_return - pl.col("beta") * bench_return).alias("abnormal_return"),
        )
        .drop(
            "entry_index",
            "exit_index",
            "entry_open",
            "bench_entry_open",
            "exit_close",
            "bench_exit_close",
            "entry_segment",
            "exit_segment",
        )
        .sort("known_at", "horizon")
    )


def event_reactions(events: pl.DataFrame, prices: pl.DataFrame) -> pl.DataFrame:
    """One row per event: abnormal move from the last close before it to the close
    of the entry session, and that move as a z-score against normal volatility.

    Events without enough prior history for a beta and a baseline, or with a break
    in the price history between the start and the entry session, are dropped.
    """
    starts = prices.select(
        "closes_at",
        pl.col("session_index").alias("start_index"),
        pl.col("adj_close").alias("start_close"),
        pl.col("bench_adj_close").alias("bench_start_close"),
        "beta",
        "baseline_std",
        pl.col("segment").alias("start_segment"),
    )
    ends = prices.select(
        pl.col("session_index").alias("entry_index"),
        pl.col("adj_close").alias("end_close"),
        pl.col("bench_adj_close").alias("bench_end_close"),
    )
    stock_move = pl.col("end_close") / pl.col("start_close") - 1
    bench_move = pl.col("bench_end_close") / pl.col("bench_start_close") - 1
    sessions = (pl.col("entry_index") - pl.col("start_index")).cast(pl.Float64)
    return (
        attach_entry(events, prices)
        .join_asof(
            starts,
            left_on="known_at",
            right_on="closes_at",
            strategy="backward",  # last close at or before the event
            coalesce=False,
        )
        .drop_nulls(["baseline_std", "beta"])
        .filter(pl.col("start_segment") == pl.col("entry_segment"))
        .join(ends, on="entry_index", how="inner")
        .with_columns(
            (stock_move - pl.col("beta") * bench_move).alias("reaction"),
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
                "entry_beta",
                "start_segment",
                "entry_segment",
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


def calendar_block(entry_session: pl.Expr, horizon: pl.Expr) -> pl.Expr:
    """Cluster id: Monday-aligned blocks of ``max(1, ceil(horizon / 5))`` weeks, so
    each block is about as long as the return windows that start in it."""
    weeks = pl.max_horizontal(pl.lit(1), (horizon + _SESSIONS_PER_WEEK - 1) // _SESSIONS_PER_WEEK)
    days = entry_session.cast(pl.Int32) - _FIRST_MONDAY
    return days // (weeks * 7)


def clustered_mean(rows: pl.DataFrame, keys: Sequence[str], value: str) -> pl.DataFrame:
    """Mean of ``value`` per ``keys`` with a cluster-robust t-stat.

    ``rows`` needs a ``cluster`` column (see :func:`calendar_block`). Returns ``n``,
    ``clusters``, ``mean``, ``median``, ``t_naive`` (independent rows), ``t_stat``
    (clustered; null with a single cluster) and ``hit_rate`` (share above zero).
    """
    keys = list(keys)
    x = pl.col(value)
    stats = rows.group_by(keys).agg(
        pl.len().alias("n"),
        pl.col("cluster").n_unique().alias("clusters"),
        x.mean().alias("mean"),
        x.median().alias("median"),
        (x.mean() / (x.std() / pl.len().sqrt())).alias("t_naive"),
        (x > 0).mean().alias("hit_rate"),
    )
    # Cluster-robust variance of the mean: sum over clusters of (sum of residuals)^2,
    # with the usual G / (G - 1) small-sample correction.
    cluster_sums = (
        rows.join(stats.select(*keys, "mean"), on=keys)
        .group_by(*keys, "cluster")
        .agg((x - pl.col("mean")).sum().alias("residual_sum"))
        .group_by(keys)
        .agg((pl.col("residual_sum") ** 2).sum().alias("sum_sq"))
    )
    g = pl.col("clusters")
    clustered_se = (pl.col("sum_sq") * g / (g - 1)).sqrt() / pl.col("n")
    return (
        stats.join(cluster_sums, on=keys)
        .with_columns(pl.when(g > 1).then(pl.col("mean") / clustered_se).alias("t_stat"))
        .drop("sum_sq")
    )


def summarize(returns: pl.DataFrame, by: str = "items") -> pl.DataFrame:
    """Abnormal-return statistics per group and horizon, plus an "all" row.

    ``t_stat`` is clustered by calendar block (see module docstring); |t| > 2 is
    roughly "unlikely to be luck" for a *single* pre-chosen row. ``t_naive`` assumes
    independent events and is shown for comparison.
    """
    rows = _by_group(returns, by).with_columns(
        calendar_block(pl.col("entry_session"), pl.col("horizon")).alias("cluster")
    )
    return clustered_mean(rows, ["group", "horizon"], "abnormal_return").sort(
        pl.col("group") != ALL_ITEMS, "group", "horizon"
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
