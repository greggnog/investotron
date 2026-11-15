"""Command line entry point: ``uv run investotron <command>``."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx
import polars as pl

from investotron.event_study import (
    ALL_ITEMS,
    DEFAULT_HORIZONS,
    MIN_BASELINE_SESSIONS,
    NULL_MEAN_ABS_Z,
    NULL_SHARE_ABOVE_2,
    aligned_prices,
    attach_surprises,
    event_reactions,
    event_returns,
    summarize,
    summarize_reactions,
    surprise_bucket,
)
from investotron.fundamentals import earnings_surprises, parse_quarterly_eps
from investotron.identifiers import Ticker
from investotron.sources.edgar import EIGHT_K_ITEMS, EdgarClient, parse_filings
from investotron.sources.tiingo import (
    TiingoClient,
    TiingoError,
    TiingoRateLimitError,
    parse_daily_bars,
)
from investotron.storage import (
    eps_frame,
    eps_from_frame,
    filings_frame,
    prices_frame,
    surprises_frame,
    write_raw_json,
)
from investotron.universe import resolve_tickers

USER_AGENT_ENV = "INVESTOTRON_SEC_USER_AGENT"
TIINGO_KEY_ENV = "INVESTOTRON_TIINGO_API_KEY"
DEFAULT_PRICES_START = date(2015, 1, 1)
PCT = 100


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(f"Set {name} first (see .env.example).")
    return value


@dataclass(frozen=True, slots=True)
class Paths:
    data_dir: Path

    def filings(self, ticker: Ticker) -> Path:
        return self.data_dir / "clean" / "edgar_8k" / f"{ticker}.parquet"

    def eps(self, ticker: Ticker) -> Path:
        return self.data_dir / "clean" / "eps_quarterly" / f"{ticker}.parquet"

    def prices(self, ticker: Ticker) -> Path:
        return self.data_dir / "clean" / "prices_daily" / f"{ticker}.parquet"

    def raw(self, *parts: str) -> Path:
        return self.data_dir.joinpath("raw", *parts)

    def results(self, name: str) -> Path:
        return self.data_dir / "results" / name


def _write(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)


# --- fetching ---------------------------------------------------------------


def fetch_edgar(tickers: list[Ticker], refresh: bool, paths: Paths) -> None:
    """8-K filings and quarterly EPS for each ticker."""
    user_agent = _require_env(USER_AGENT_ENV)
    with EdgarClient(user_agent) as edgar:
        for ticker in tickers:
            if not refresh and paths.filings(ticker).exists() and paths.eps(ticker).exists():
                print(f"{ticker}: already fetched (use --refresh to re-download)")
                continue
            try:
                cik = edgar.lookup_cik(ticker)
                submissions = edgar.submissions(cik)
                facts = edgar.company_facts(cik)
            except (LookupError, httpx.HTTPStatusError) as error:
                print(f"{ticker}: skipped -- {error}")
                continue

            write_raw_json(paths.raw("edgar", "submissions", f"CIK{cik.padded}.json"), submissions)
            write_raw_json(paths.raw("edgar", "companyfacts", f"CIK{cik.padded}.json"), facts)
            filings = filings_frame(parse_filings(cik, submissions))
            eps = eps_frame(parse_quarterly_eps(facts))
            _write(filings, paths.filings(ticker))
            _write(eps, paths.eps(ticker))
            print(f"{ticker}: {filings.height} 8-Ks, {eps.height} quarters of EPS")


def fetch_prices(tickers: list[Ticker], start: date, refresh: bool, paths: Paths) -> None:
    api_key = _require_env(TIINGO_KEY_ENV)
    with TiingoClient(api_key) as tiingo:
        for ticker in tickers:
            if not refresh and paths.prices(ticker).exists():
                print(f"{ticker}: already fetched (use --refresh to re-download)")
                continue
            try:
                raw = tiingo.daily_prices(ticker, start)
            except TiingoRateLimitError as error:
                sys.exit(
                    f"Tiingo rate limit reached at {ticker}: {error}\n"
                    "Everything fetched so far is saved; rerun later to continue."
                )
            except (TiingoError, httpx.HTTPStatusError) as error:
                print(f"{ticker}: skipped -- {error}")
                continue

            write_raw_json(paths.raw("tiingo", "daily", f"{ticker}.json"), raw)
            frame = prices_frame(parse_daily_bars(ticker, raw))
            _write(frame, paths.prices(ticker))
            span = f"{frame['session'].min()} .. {frame['session'].max()}" if frame.height else ""
            print(f"{ticker}: {frame.height} sessions {span}")


# --- event study --------------------------------------------------------------


def _show(title: str, frame: pl.DataFrame) -> None:
    print(f"\n{title}")
    with pl.Config(tbl_rows=-1, tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True):
        print(frame)


def _describe(group: pl.Expr) -> pl.Expr:
    return group.replace_strict(EIGHT_K_ITEMS, default="", return_dtype=pl.String)


def _direction_table(summary: pl.DataFrame, label: pl.Expr) -> pl.DataFrame:
    return summary.select(
        "group",
        label.alias("description"),
        pl.col("horizon").alias("days"),
        "n",
        (pl.col("mean") * PCT).round(2).alias("mean %"),
        (pl.col("median") * PCT).round(2).alias("median %"),
        pl.col("t_stat").round(2).alias("t"),
        (pl.col("hit_rate") * PCT).round(0).alias("hit %"),
    )


def _reaction_table(summary: pl.DataFrame, label: pl.Expr) -> pl.DataFrame:
    return summary.select(
        "group",
        label.alias("description"),
        "n",
        pl.col("mean_abs_z").round(2).alias("mean |z|"),
        (pl.col("share_above_2") * PCT).round(1).alias("|z|>2 %"),
        (pl.col("mean_abs_reaction") * PCT).round(2).alias("mean |move| %"),
        (pl.col("mean_reaction") * PCT).round(2).alias("mean move %"),
    )


def event_study(
    tickers: list[Ticker],
    benchmark: Ticker,
    horizons: list[int],
    min_events: int,
    paths: Paths,
) -> None:
    if not paths.prices(benchmark).exists():
        sys.exit(f"Missing {benchmark} prices -- run `investotron fetch-prices {benchmark}`.")
    bench = pl.read_parquet(paths.prices(benchmark))

    returns: list[pl.DataFrame] = []
    reactions: list[pl.DataFrame] = []
    surprised: list[pl.DataFrame] = []
    missing: list[str] = []
    for ticker in tickers:
        if not (paths.filings(ticker).exists() and paths.prices(ticker).exists()):
            missing.append(ticker.value)
            continue
        filings = pl.read_parquet(paths.filings(ticker)).with_columns(
            pl.lit(ticker.value).alias("ticker")
        )
        prices = aligned_prices(pl.read_parquet(paths.prices(ticker)), bench)
        returns.append(event_returns(filings, prices, horizons))
        reactions.append(event_reactions(filings, prices))
        if paths.eps(ticker).exists():
            quarters = eps_from_frame(pl.read_parquet(paths.eps(ticker)))
            surprises = surprises_frame(earnings_surprises(quarters))
            surprised.append(attach_surprises(filings, surprises))

    if not returns:
        sys.exit("No tickers have both 8-K and price data yet.")
    if missing:
        print(f"Skipped (not fetched yet): {', '.join(missing)}")

    all_returns = pl.concat(returns)
    all_reactions = pl.concat(reactions)
    tag = tickers[0].value if len(tickers) == 1 else f"{len(tickers)}_tickers"
    _write(all_returns, paths.results(f"event_returns_{tag}_vs_{benchmark}.parquet"))
    _write(all_reactions, paths.results(f"event_reactions_{tag}_vs_{benchmark}.parquet"))

    n_tickers = all_returns["ticker"].n_unique()
    n_events = all_returns.select("ticker", "accession_number").n_unique()
    print(
        f"{n_tickers} ticker(s), {n_events} 8-Ks ({all_reactions.height} with the "
        f"{MIN_BASELINE_SESSIONS}+ sessions of history the size test needs), "
        f"benchmark {benchmark}"
    )
    keep = (pl.col("group") == ALL_ITEMS) | (pl.col("n") >= min_events)

    _show(
        "1) DIRECTION: abnormal return from the first open after the filing "
        f"(items with n >= {min_events})",
        _direction_table(summarize(all_returns).filter(keep), _describe(pl.col("group"))),
    )
    _show(
        "2) SIZE: move from last close before the filing to the entry-day close, "
        f"vs normal volatility (no-news reference: mean |z| ~{NULL_MEAN_ABS_Z:.1f}, "
        f"|z|>2 ~{NULL_SHARE_ABOVE_2:.0%})",
        _reaction_table(
            summarize_reactions(all_reactions).filter(keep), _describe(pl.col("group"))
        ),
    )

    if not surprised:
        print("\n3) SURPRISE: no EPS data yet -- run `investotron fetch-edgar` for these tickers.")
        return
    earnings = pl.concat(surprised).with_columns(surprise_bucket(pl.col("sue")).alias("surprise"))
    rated = earnings.drop_nulls("surprise")
    print(
        f"\n3) SURPRISE: {earnings.height} earnings 8-Ks, {rated.height} with a SUE "
        "(needs ~2 years of EPS history first)"
    )
    surprise_returns = summarize(
        all_returns.join(
            rated.select("ticker", "accession_number", "surprise"),
            on=["ticker", "accession_number"],
        ),
        by="surprise",
    )
    surprise_reactions = summarize_reactions(
        all_reactions.join(
            rated.select("ticker", "accession_number", "surprise"),
            on=["ticker", "accession_number"],
        ),
        by="surprise",
    )
    label = pl.lit("")
    _show(
        "   Size of the reaction, by surprise (SUE <= -1 miss, >= +1 beat)",
        _reaction_table(surprise_reactions, label),
    )
    _show("   Direction after entry, by surprise", _direction_table(surprise_returns, label))


# --- argument parsing ---------------------------------------------------------


def _add_ticker_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("tickers", type=Ticker.parse, nargs="*")
    parser.add_argument("--universe", type=Path, help="file with one ticker per line")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="investotron")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    commands = parser.add_subparsers(dest="command", required=True)

    edgar = commands.add_parser(
        "fetch-edgar", aliases=["fetch-8k"], help="download 8-K filings and quarterly EPS"
    )
    _add_ticker_args(edgar)
    edgar.add_argument("--refresh", action="store_true", help="re-download existing data")

    prices = commands.add_parser("fetch-prices", help="download daily prices from Tiingo")
    _add_ticker_args(prices)
    prices.add_argument("--refresh", action="store_true", help="re-download existing data")
    prices.add_argument(
        "--start",
        type=date.fromisoformat,
        default=DEFAULT_PRICES_START,
        help=f"first session to fetch (default {DEFAULT_PRICES_START})",
    )

    study = commands.add_parser("event-study", help="returns after 8-Ks, relative to a benchmark")
    _add_ticker_args(study)
    study.add_argument("--benchmark", type=Ticker.parse, default=Ticker("SPY"))
    study.add_argument(
        "--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS), metavar="DAYS"
    )
    study.add_argument("--min-events", type=int, default=5, help="hide groups with fewer events")

    args = parser.parse_args(argv)
    try:
        tickers = resolve_tickers(args.tickers, args.universe)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    paths = Paths(args.data_dir)

    if args.command in ("fetch-edgar", "fetch-8k"):
        fetch_edgar(tickers, args.refresh, paths)
    elif args.command == "fetch-prices":
        fetch_prices(tickers, args.start, args.refresh, paths)
    elif args.command == "event-study":
        event_study(tickers, args.benchmark, args.horizons, args.min_events, paths)
