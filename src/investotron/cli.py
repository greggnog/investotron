"""Command line entry point: ``uv run investotron <command>``."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import polars as pl

from investotron.datastore import Paths, load_ticker, write_frame
from investotron.event_study import (
    ALL_ITEMS,
    BETA_SESSIONS,
    DEFAULT_HORIZONS,
    MIN_BETA_SESSIONS,
    NULL_MEAN_ABS_Z,
    NULL_SHARE_ABOVE_2,
    Model,
    event_reactions,
    event_returns,
    summarize,
    summarize_reactions,
)
from investotron.explain import (
    EXPLAIN_ENV,
    Narrator,
    direction_takeaway,
    explain_enabled,
    multiple_comparisons_takeaway,
    size_takeaway,
    surprise_takeaway,
)
from investotron.fundamentals import parse_quarterly_eps
from investotron.identifiers import Ticker
from investotron.paper_report import run_paper
from investotron.plain import PLAIN_ENV, plain_enabled
from investotron.sources.edgar import EIGHT_K_ITEMS, EdgarClient, parse_filings
from investotron.sources.tiingo import (
    TiingoClient,
    TiingoError,
    TiingoRateLimitError,
    parse_daily_bars,
)
from investotron.storage import (
    eps_frame,
    filings_frame,
    prices_frame,
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


# --- fetching ---------------------------------------------------------------


def fetch_edgar(tickers: list[Ticker], refresh: bool, paths: Paths, narrator: Narrator) -> None:
    """8-K filings and quarterly EPS for each ticker."""
    user_agent = _require_env(USER_AGENT_ENV)
    narrator.say(
        "Downloading each company's 8-K filings and quarterly earnings from the SEC's free "
        "EDGAR database. Each line below says how many 8-Ks and how many quarters of "
        "earnings-per-share figures were saved for that stock."
    )
    narrator.terms("ticker", "8-K", "EPS")
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
            write_frame(filings, paths.filings(ticker))
            write_frame(eps, paths.eps(ticker))
            print(f"{ticker}: {filings.height} 8-Ks, {eps.height} quarters of EPS")


def fetch_prices(
    tickers: list[Ticker], start: date, refresh: bool, paths: Paths, narrator: Narrator
) -> None:
    api_key = _require_env(TIINGO_KEY_ENV)
    narrator.say(
        "Downloading daily prices (open, close, volume) from Tiingo. Each line says how many "
        "trading days were saved for that stock and the date range they cover."
    )
    narrator.terms("ticker", "session")
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
            write_frame(frame, paths.prices(ticker))
            span = f"{frame['session'].min()} .. {frame['session'].max()}" if frame.height else ""
            print(f"{ticker}: {frame.height} sessions {span}")


# --- event study --------------------------------------------------------------


def _show(title: str, frame: pl.DataFrame) -> None:
    print(f"\n{title}")
    with pl.Config(
        tbl_rows=-1,
        tbl_cols=-1,
        tbl_width_chars=200,
        tbl_hide_dataframe_shape=True,
        tbl_hide_column_data_types=True,
    ):
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
        pl.col("t_naive").round(2).alias("t naive"),
        pl.col("t_stat").round(2).alias("t"),
        pl.col("clusters").alias("weeks"),
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


def _print_multiple_comparisons(summary: pl.DataFrame, narrator: Narrator) -> None:
    """How many rows clear |t| > 2, vs how many would by luck alone."""
    rows = summary.filter(pl.col("group") != ALL_ITEMS).drop_nulls("t_stat")
    hits = rows.filter(pl.col("t_stat").abs() > 2).height  # noqa: PLR2004
    expected = rows.height * NULL_SHARE_ABOVE_2
    print(
        f"   {hits} of {rows.height} item rows have |t| > 2; about {expected:.1f} would by "
        "chance alone, and rows at different horizons of the same item aren't independent."
    )
    narrator.terms("multiple comparisons")
    narrator.say(multiple_comparisons_takeaway(hits, rows.height, expected))


@dataclass(frozen=True, slots=True)
class StudyConfig:
    benchmark: Ticker
    horizons: list[int]
    min_events: int
    model: Model


@dataclass(frozen=True, slots=True)
class StudyData:
    returns: pl.DataFrame
    reactions: pl.DataFrame
    earnings: pl.DataFrame | None  # earnings 8-Ks with SUE, if any EPS data exists


def _load_study(tickers: list[Ticker], config: StudyConfig, paths: Paths) -> StudyData:
    if not paths.prices(config.benchmark).exists():
        sys.exit(
            f"Missing {config.benchmark} prices -- run "
            f"`investotron fetch-prices {config.benchmark}`."
        )
    bench = pl.read_parquet(paths.prices(config.benchmark))

    returns: list[pl.DataFrame] = []
    reactions: list[pl.DataFrame] = []
    earnings: list[pl.DataFrame] = []
    missing: list[str] = []
    for ticker in tickers:
        data = load_ticker(ticker, bench, config.model, paths)
        if data is None:
            missing.append(ticker.value)
            continue
        returns.append(event_returns(data.filings, data.prices, config.horizons))
        reactions.append(event_reactions(data.filings, data.prices))
        if data.earnings is not None:
            earnings.append(data.earnings)

    if not returns:
        sys.exit("No tickers have both 8-K and price data yet.")
    if missing:
        print(f"Skipped (not fetched yet): {', '.join(missing)}")
    return StudyData(
        pl.concat(returns), pl.concat(reactions), pl.concat(earnings) if earnings else None
    )


def _print_header(data: StudyData, config: StudyConfig, narrator: Narrator) -> None:
    n_tickers = data.returns["ticker"].n_unique()
    n_events = data.returns.select("ticker", "accession_number").n_unique()
    print(
        f"{n_tickers} ticker(s), {n_events} 8-Ks ({data.reactions.height} with the "
        f"history the size test needs), benchmark {config.benchmark}"
    )
    if config.model is Model.BETA:
        betas = data.returns.unique(["ticker", "accession_number"])["beta"]
        print(
            f"Model: trailing {BETA_SESSIONS}-session beta (median {betas.median():.2f}, "
            f"middle half {betas.quantile(0.25):.2f}..{betas.quantile(0.75):.2f}); "
            f"a stock's first {MIN_BETA_SESSIONS} sessions have no beta yet and are skipped"
        )
    else:
        print("Model: market-adjusted (beta = 1)")
    narrator.say(
        "This looks at every 8-K these companies filed and asks two questions: did the "
        "stock beat or trail the market afterwards (section 1), and how big was the move "
        "compared with a normal day (section 2)? Section 3 does the same for earnings "
        "reports, split by whether results beat or missed."
    )
    narrator.terms("8-K", "item", "benchmark", "beta", "session")


def _print_surprise_section(data: StudyData, narrator: Narrator) -> None:
    if data.earnings is None:
        print("\n3) SURPRISE: no EPS data yet -- run `investotron fetch-edgar` for these tickers.")
        return
    rated = data.earnings.drop_nulls("surprise").height
    print(
        f"\n3) SURPRISE: {data.earnings.height} earnings 8-Ks, {rated} with a SUE "
        "(needs ~2 years of EPS history first)"
    )
    narrator.say(
        "Earnings reports (item 2.02) grouped by how good the results were. The first "
        "table is the immediate reaction, the second is what happened after you could "
        "actually trade."
    )
    narrator.terms("EPS", "SUE", "miss / beat")
    label = pl.lit("")
    reactions = summarize_reactions(data.reactions.drop_nulls("surprise"), by="surprise")
    returns = summarize(data.returns.drop_nulls("surprise"), by="surprise")
    _show(
        "   Size of the reaction, by surprise (SUE <= -1 miss, >= +1 beat)",
        _reaction_table(reactions, label),
    )
    _show("   Direction after entry, by surprise", _direction_table(returns, label))
    for sentence in surprise_takeaway(reactions, returns):
        narrator.say(sentence)


def event_study(
    tickers: list[Ticker], config: StudyConfig, paths: Paths, narrator: Narrator
) -> None:
    data = _load_study(tickers, config, paths)

    tag = tickers[0].value if len(tickers) == 1 else f"{len(tickers)}_tickers"
    tag = f"{tag}_vs_{config.benchmark}_{config.model}"
    write_frame(data.returns, paths.results(f"event_returns_{tag}.parquet"))
    write_frame(data.reactions, paths.results(f"event_reactions_{tag}.parquet"))

    _print_header(data, config, narrator)
    keep = (pl.col("group") == ALL_ITEMS) | (pl.col("n") >= config.min_events)
    direction = summarize(data.returns).filter(keep)
    _show(
        "1) DIRECTION: abnormal return from the first open after the filing "
        f"(items with n >= {config.min_events}; t clustered by calendar weeks)",
        _direction_table(direction, _describe(pl.col("group"))),
    )
    narrator.say(
        "Each row: if you had bought the stock at the first opportunity after this kind "
        "of filing and held it for 'days' trading days, how did it do compared with the "
        "market?"
    )
    narrator.terms("abnormal return", "n", "mean %", "median %", "t naive", "t", "weeks", "hit %")
    for sentence in direction_takeaway(direction, EIGHT_K_ITEMS):
        narrator.say(sentence)
    _print_multiple_comparisons(direction, narrator)
    size = summarize_reactions(data.reactions).filter(keep)
    _show(
        "2) SIZE: move from last close before the filing to the entry-day close, "
        f"vs normal volatility (no-news reference: mean |z| ~{NULL_MEAN_ABS_Z:.1f}, "
        f"|z|>2 ~{NULL_SHARE_ABOVE_2:.0%})",
        _reaction_table(size, _describe(pl.col("group"))),
    )
    narrator.say(
        "Ignoring direction: how much did the price jump around the filing, including the "
        "overnight gap, compared with how much this stock normally moves in a day?"
    )
    narrator.terms("overnight gap", "z", "mean |z|", "|z|>2 %", "mean |move| %", "mean move %")
    for sentence in size_takeaway(size, EIGHT_K_ITEMS):
        narrator.say(sentence)
    _print_surprise_section(data, narrator)


# --- argument parsing ---------------------------------------------------------


def _add_ticker_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("tickers", type=Ticker.parse, nargs="*")
    parser.add_argument("--universe", type=Path, help="file with one ticker per line")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="investotron")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    commands = parser.add_subparsers(dest="command", required=True)
    explain = argparse.ArgumentParser(add_help=False)
    explain.add_argument(
        "--explain",
        action="store_true",
        help=f"explain terms and results in plain English (or set {EXPLAIN_ENV}=1)",
    )

    edgar = commands.add_parser(
        "fetch-edgar",
        aliases=["fetch-8k"],
        help="download 8-K filings and quarterly EPS",
        parents=[explain],
    )
    _add_ticker_args(edgar)
    edgar.add_argument("--refresh", action="store_true", help="re-download existing data")

    prices = commands.add_parser(
        "fetch-prices", help="download daily prices from Tiingo", parents=[explain]
    )
    _add_ticker_args(prices)
    prices.add_argument("--refresh", action="store_true", help="re-download existing data")
    prices.add_argument(
        "--start",
        type=date.fromisoformat,
        default=DEFAULT_PRICES_START,
        help=f"first session to fetch (default {DEFAULT_PRICES_START})",
    )

    study = commands.add_parser(
        "event-study", help="returns after 8-Ks, relative to a benchmark", parents=[explain]
    )
    _add_ticker_args(study)
    study.add_argument("--benchmark", type=Ticker.parse, default=Ticker("SPY"))
    study.add_argument(
        "--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS), metavar="DAYS"
    )
    study.add_argument("--min-events", type=int, default=5, help="hide groups with fewer events")
    study.add_argument(
        "--model",
        type=Model,
        choices=list(Model),
        default=Model.BETA,
        help="expected-return model: trailing beta (default) or beta = 1",
    )

    paper = commands.add_parser(
        "paper",
        help="paper-trade the rules in a rules file and report how they're doing",
        parents=[explain],
    )
    paper.add_argument(
        "--rules", type=Path, default=Path("paper/rules.toml"), help="rules file (TOML)"
    )
    paper.add_argument(
        "--plain",
        action="store_true",
        help=f"short plain-English report with no tables (or set {PLAIN_ENV}=1)",
    )

    args = parser.parse_args(argv)
    narrator = Narrator(explain_enabled(args.explain, os.environ))
    if args.command == "paper":
        run_paper(
            args.rules,
            Paths(args.data_dir),
            datetime.now(UTC),
            narrator,
            plain_english=plain_enabled(args.plain, os.environ),
        )
        return
    try:
        tickers = resolve_tickers(args.tickers, args.universe)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    paths = Paths(args.data_dir)

    if args.command in ("fetch-edgar", "fetch-8k"):
        fetch_edgar(tickers, args.refresh, paths, narrator)
    elif args.command == "fetch-prices":
        fetch_prices(tickers, args.start, args.refresh, paths, narrator)
    elif args.command == "event-study":
        config = StudyConfig(args.benchmark, args.horizons, args.min_events, args.model)
        event_study(tickers, config, paths, narrator)
