"""``investotron paper``: recompute every rule's trades and print the report."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import cast

import polars as pl

from investotron import plain
from investotron.datastore import Paths, load_ticker, write_frame
from investotron.event_study import Model
from investotron.explain import Narrator, rule_takeaway
from investotron.paper import (
    LockEntry,
    Rule,
    RuleBook,
    RuleError,
    Status,
    load_rulebook,
    pending_signals,
    rule_trades,
    summarize_trades,
    sync_lock,
)
from investotron.sources.edgar import EIGHT_K_ITEMS
from investotron.universe import load_universe

PCT = 100
RECENT_TRADES = 5
LOCK_FILE_NAME = "rules.lock.json"


@dataclass(slots=True)
class _RuleResults:
    trades: list[pl.DataFrame] = field(default_factory=lambda: [])
    pending: list[pl.DataFrame] = field(default_factory=lambda: [])


@dataclass(frozen=True, slots=True)
class _Split:
    """One rule's trades, cut the ways the report needs."""

    trades: pl.DataFrame
    open_trades: pl.DataFrame
    finished: pl.DataFrame  # live only
    pending: pl.DataFrame
    summary: pl.DataFrame

    @classmethod
    def of(cls, rule: Rule, results: _RuleResults) -> _Split:
        trades = pl.concat(results.trades) if results.trades else pl.DataFrame()
        live = trades.filter(pl.col("live")) if not trades.is_empty() else trades
        is_open = pl.col("status") == Status.OPEN.value
        return cls(
            trades=trades,
            open_trades=live.filter(is_open) if not live.is_empty() else live,
            finished=live.filter(~is_open) if not live.is_empty() else live,
            pending=pl.concat(results.pending) if results.pending else pl.DataFrame(),
            summary=summarize_trades(trades, rule.horizon) if not trades.is_empty() else trades,
        )

    def stats(self, sample: str) -> dict[str, object] | None:
        if self.summary.is_empty():
            return None
        rows = self.summary.filter(pl.col("sample") == sample).to_dicts()
        return rows[0] if rows else None


def _table(frame: pl.DataFrame, indent: str = "     ") -> None:
    with pl.Config(
        tbl_rows=-1,
        tbl_cols=-1,
        tbl_width_chars=200,
        tbl_hide_dataframe_shape=True,
        tbl_hide_column_data_types=True,
    ):
        for line in str(frame).splitlines():
            print(indent + line)


def _describe(rule: Rule) -> str:
    what = EIGHT_K_ITEMS.get(rule.item, "")
    surprise = f", {rule.surprise} only" if rule.surprise else ""
    return (
        f"{rule.direction} for {rule.horizon} sessions after 8-K item {rule.item}"
        f"{f' ({what})' if what else ''}{surprise}"
    )


def _stats_table(summary: pl.DataFrame) -> pl.DataFrame:
    return summary.select(
        "sample",
        pl.col("n").alias("trades"),
        (pl.col("mean") * PCT).round(2).alias("mean P&L %"),
        (pl.col("median") * PCT).round(2).alias("median %"),
        pl.col("t_stat").round(2).alias("t"),
        (pl.col("hit_rate") * PCT).round(0).alias("win %"),
    )


def _positions_table(trades: pl.DataFrame, horizon: int) -> pl.DataFrame:
    return trades.select(
        "ticker",
        pl.col("known_at").dt.date().alias("filed"),
        pl.col("entry_session").alias("entered"),
        pl.col("exit_session").alias("exit/mark"),
        pl.format("{}/{}", pl.col("sessions_held"), pl.lit(horizon)).alias("held"),
        (pl.col("pnl") * PCT).round(2).alias("P&L %"),
        "status",
    )


def _print_rule(
    rule: Rule, entry: LockEntry, results: _RuleResults, now: datetime, narrator: Narrator
) -> None:
    print(f"\n== {rule.name}: {_describe(rule)}")
    narrator.terms("paper trading", rule.direction.value, "horizon", "item")
    days_live = (now - entry.started_at).days
    print(f"   Live since {entry.started_at:%Y-%m-%d %H:%M} UTC ({days_live} days)")

    split = _Split.of(rule, results)
    open_trades, finished, pending = split.open_trades, split.finished, split.pending

    print(
        f"   Live: {finished.height} finished, {open_trades.height} open, "
        f"{pending.height} waiting for the next open"
    )
    narrator.terms("live", "backtest", "status")
    if not pending.is_empty():
        for row in pending.sort("known_at").iter_rows(named=True):
            print(f"     pending: {row['ticker']} filed {row['known_at']:%Y-%m-%d %H:%M} UTC")
    if not open_trades.is_empty():
        print("   Open positions (marked at the latest close):")
        _table(_positions_table(open_trades.sort("entry_session"), rule.horizon))
    if not finished.is_empty():
        print(f"   Latest finished live trades (up to {RECENT_TRADES}):")
        recent = finished.sort("exit_session", descending=True).head(RECENT_TRADES)
        _table(_positions_table(recent, rule.horizon))

    summary = split.summary
    if summary.is_empty():
        print("   No finished trades yet, live or backtest.")
        return
    print("   Finished trades, live vs backtest (backtest = same rule on earlier filings):")
    _table(_stats_table(summary))
    narrator.terms("P&L", "win %", "t")
    stats = {row["sample"]: row for row in summary.iter_rows(named=True)}
    for sentence in rule_takeaway(rule, stats, open_trades.height):
        narrator.say(sentence)


def _print_plain_rule(number: int, rule: Rule, entry: LockEntry, results: _RuleResults) -> None:
    split = _Split.of(rule, results)
    backtest = split.stats("backtest")
    expected = float(cast("float", backtest["mean"])) if backtest else None
    print(f"\n{number}. {rule.name}")
    print(plain.wrap(plain.idea(rule)))
    print()
    print(plain.wrap(plain.expectation(backtest)))
    print()
    print(
        plain.wrap(
            plain.scoreboard(
                entry.started_at, split.stats("live"), backtest, split.open_trades.height
            )
        )
    )
    sections = [
        (
            "Waiting to start:",
            split.pending.sort("known_at") if not split.pending.is_empty() else None,
        ),
        (
            "In progress:",
            split.open_trades.sort("entry_session") if not split.open_trades.is_empty() else None,
        ),
        (
            f"Most recently finished (up to {RECENT_TRADES}):",
            split.finished.sort("exit_session", descending=True).head(RECENT_TRADES)
            if not split.finished.is_empty()
            else None,
        ),
    ]
    for title, frame in sections:
        if frame is None:
            continue
        print(f"\n  {title}")
        for row in frame.iter_rows(named=True):
            text = (
                plain.pending(rule, row)
                if title.startswith("Waiting")
                else plain.trade(rule, row, expected)
            )
            print(plain.wrap(text, "  - ").replace("\n  - ", "\n    "))


def _print_plain(book: RuleBook, entries: dict[str, LockEntry], collected: _Collected) -> None:
    print(
        plain.wrap(
            "Paper trading, in plain English: pretend bets with no real money, to see whether "
            "an idea keeps working on news that arrives from now on.",
            "",
        )
    )
    print(f"Prices are up to the close on {plain.day(max(collected.last_sessions))}.")
    if collected.stopped:
        names = ", ".join(f"{t} (last price {plain.day(d)})" for t, d in collected.stopped)
        print(
            plain.wrap(
                f"Stopped trading early, usually because the company was bought or delisted: "
                f"{names}. Any bet on these ends at the last price.",
                "",
            )
        )
    for number, rule in enumerate(book.rules, start=1):
        _print_plain_rule(number, rule, entries[rule.name], collected.results[rule.name])
    print()
    print(
        plain.wrap(
            "\"Did better than the market\" means the stock's move compared with the market's "
            "over the same days, so a bet can win even when the stock fell, if the market fell "
            "more. No trading fees are counted. Fetch new data each evening after 4 pm New "
            "York time (see the README).",
            "",
        )
    )


def _print_freshness(last_session: date, last_filing: datetime, book: RuleBook) -> None:
    try:
        universe = book.universe.relative_to(Path.cwd())
    except ValueError:
        universe = book.universe
    print(f"Data: prices through {last_session}, filings through {last_filing}.")
    print("Refresh before each run (prices update after each close):")
    print(f"  uv run investotron fetch-edgar --universe {universe} --refresh")
    print(f"  uv run investotron fetch-prices {book.benchmark} --universe {universe} --refresh")


@dataclass(slots=True)
class _Collected:
    results: dict[str, _RuleResults]
    missing: list[str] = field(default_factory=lambda: [])
    stopped: list[tuple[str, date]] = field(default_factory=lambda: [])
    last_sessions: list[date] = field(default_factory=lambda: [])
    last_filings: list[datetime] = field(default_factory=lambda: [])


def _collect(
    book: RuleBook, entries: dict[str, LockEntry], bench: pl.DataFrame, paths: Paths
) -> _Collected:
    """Every rule's trades and pending signals across the universe."""
    market_last_session = cast("date", bench["session"].max())
    market_last_open = cast("datetime", bench["opens_at"].max())
    collected = _Collected({rule.name: _RuleResults() for rule in book.rules})
    for ticker in load_universe(book.universe):
        data = load_ticker(ticker, bench, Model.BETA, paths)
        if data is None:
            collected.missing.append(ticker.value)
            continue
        last_session = cast("date", data.prices["session"].max())
        collected.last_sessions.append(last_session)
        collected.last_filings.append(cast("datetime", data.filings["known_at"].max()))
        if last_session < market_last_session:
            collected.stopped.append((ticker.value, last_session))
        for rule in book.rules:
            started_at = entries[rule.name].started_at
            results = collected.results[rule.name]
            results.trades.append(
                rule_trades(rule, data.filings, data.prices, started_at, market_last_session)
            )
            results.pending.append(
                pending_signals(rule, data.filings, market_last_open, started_at)
            )
    for results in collected.results.values():
        results.trades = [t for t in results.trades if not t.is_empty()]
        results.pending = [p for p in results.pending if not p.is_empty()]
    return collected


def run_paper(
    rules_path: Path,
    paths: Paths,
    now: datetime,
    narrator: Narrator,
    *,
    plain_english: bool = False,
) -> None:
    try:
        book = load_rulebook(rules_path)
        entries, started = sync_lock(book.rules, rules_path.with_name(LOCK_FILE_NAME), now)
        if not paths.prices(book.benchmark).exists():
            sys.exit(f"Missing {book.benchmark} prices -- run `investotron fetch-prices`.")
        bench = pl.read_parquet(paths.prices(book.benchmark))
        collected = _collect(book, entries, bench, paths)
    except (RuleError, OSError) as error:
        sys.exit(str(error))
    for name in started:
        print(f"Started tracking {name} at {now:%Y-%m-%d %H:%M} UTC.")
        print("  Only filings after this moment count as live; commit rules.lock.json.")

    if not collected.last_sessions:
        sys.exit("No tickers in the universe have data yet -- fetch them first.")
    if collected.missing:
        print(f"Skipped (not fetched yet): {', '.join(collected.missing)}")
    all_trades = [t for r in collected.results.values() for t in r.trades]
    if all_trades:
        write_frame(pl.concat(all_trades), paths.data_dir / "paper" / "trades.parquet")
    if plain_english:
        _print_plain(book, entries, collected)
        return
    _print_freshness(max(collected.last_sessions), max(collected.last_filings), book)
    if collected.stopped:
        stopped = ", ".join(f"{t} ({d})" for t, d in collected.stopped)
        print(
            f"Prices end before {book.benchmark}'s latest session for: {stopped}. Stopped "
            "trading (acquired, delisted) or needs a refresh; unfinished trades in these end "
            "at their last price, as interrupted."
        )
        narrator.terms("stopped trading")
    narrator.say(
        "Below, each rule is a simple trading idea being tested without real money. The key "
        "comparison is 'live' (filings since the rule was written down) against 'backtest' "
        "(the same rule on older filings). A rule only means something if it keeps working live."
    )

    for rule in book.rules:
        _print_rule(rule, entries[rule.name], collected.results[rule.name], now, narrator)
    print("\nP&L is hypothetical: beta-hedged vs the benchmark, no costs or slippage.")
