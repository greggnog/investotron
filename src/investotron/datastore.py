"""Where data lives on disk, and loading one stock's study-ready data from it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from investotron.event_study import Model, aligned_prices, attach_surprises, surprise_bucket
from investotron.fundamentals import earnings_surprises
from investotron.identifiers import Ticker
from investotron.storage import eps_from_frame, surprises_frame


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


def write_frame(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)


@dataclass(frozen=True, slots=True)
class TickerData:
    filings: pl.DataFrame  # 8-Ks with ``ticker`` and ``surprise`` (null unless a rated 2.02)
    prices: pl.DataFrame  # from aligned_prices
    earnings: pl.DataFrame | None  # earnings 8-Ks with sue/surprise; None without EPS data


def load_ticker(
    ticker: Ticker, benchmark: pl.DataFrame, model: Model, paths: Paths
) -> TickerData | None:
    """One stock's filings and benchmark-aligned prices, or None if not fetched."""
    if not (paths.filings(ticker).exists() and paths.prices(ticker).exists()):
        return None
    filings = pl.read_parquet(paths.filings(ticker)).with_columns(
        pl.lit(ticker.value).alias("ticker")
    )
    prices = aligned_prices(pl.read_parquet(paths.prices(ticker)), benchmark, model)

    if not paths.eps(ticker).exists():
        return TickerData(
            filings.with_columns(pl.lit(None, dtype=pl.String).alias("surprise")), prices, None
        )
    quarters = eps_from_frame(pl.read_parquet(paths.eps(ticker)))
    earnings = attach_surprises(
        filings, surprises_frame(earnings_surprises(quarters))
    ).with_columns(surprise_bucket(pl.col("sue")).alias("surprise"))
    filings = filings.join(
        earnings.select("accession_number", "surprise"), on="accession_number", how="left"
    )
    return TickerData(filings, prices, earnings)
