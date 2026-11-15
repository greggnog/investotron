"""On-disk layers: ``raw/`` holds payloads exactly as downloaded, ``clean/`` holds
typed tables derived from them. Clean data can always be rebuilt from raw."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC
from pathlib import Path
from typing import Any

import polars as pl

from investotron.fundamentals import QuarterlyEps, Surprise
from investotron.point_in_time import Fact
from investotron.prices import DailyBar
from investotron.sources.edgar import Filing


def write_raw_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


FILINGS_SCHEMA = pl.Schema(
    {
        "cik": pl.Int64,
        "accession_number": pl.String,
        "form": pl.String,
        "items": pl.List(pl.String),
        "occurred_at": pl.Datetime("us", "UTC"),
        "known_at": pl.Datetime("us", "UTC"),
        "url": pl.String,
    }
)


def filings_frame(facts: Sequence[Fact[Filing]]) -> pl.DataFrame:
    """One row per filing; timestamps normalised to UTC."""
    rows = [
        {
            "cik": f.value.cik.value,
            "accession_number": f.value.accession_number,
            "form": f.value.form,
            "items": list(f.value.items),
            "occurred_at": f.occurred_at.astimezone(UTC),
            "known_at": f.known_at.astimezone(UTC),
            "url": f.value.url,
        }
        for f in facts
    ]
    return pl.DataFrame(rows, schema=FILINGS_SCHEMA).sort("known_at")


PRICES_SCHEMA = pl.Schema(
    {
        "ticker": pl.String,
        "session": pl.Date,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Int64,
        "adj_open": pl.Float64,
        "adj_close": pl.Float64,
        "opens_at": pl.Datetime("us", "UTC"),
        "closes_at": pl.Datetime("us", "UTC"),
    }
)


def prices_frame(bars: Sequence[DailyBar]) -> pl.DataFrame:
    """One row per session; ``opens_at``/``closes_at`` in UTC, so they compare
    directly with filing ``known_at`` times."""
    rows = [
        {
            "ticker": b.ticker.value,
            "session": b.session,
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume,
            "adj_open": b.adj_open,
            "adj_close": b.adj_close,
            "opens_at": b.opens_at.astimezone(UTC),
            "closes_at": b.closes_at.astimezone(UTC),
        }
        for b in bars
    ]
    return pl.DataFrame(rows, schema=PRICES_SCHEMA).sort("session")


EPS_SCHEMA = pl.Schema(
    {
        "period_start": pl.Date,
        "period_end": pl.Date,
        "eps": pl.Float64,
        "first_filed": pl.Date,
        "derived": pl.Boolean,
    }
)


def eps_frame(quarters: Sequence[QuarterlyEps]) -> pl.DataFrame:
    rows = [
        {
            "period_start": q.period_start,
            "period_end": q.period_end,
            "eps": q.eps,
            "first_filed": q.first_filed,
            "derived": q.derived,
        }
        for q in quarters
    ]
    return pl.DataFrame(rows, schema=EPS_SCHEMA).sort("period_end")


def eps_from_frame(frame: pl.DataFrame) -> list[QuarterlyEps]:
    return [
        QuarterlyEps(
            period_start=row["period_start"],
            period_end=row["period_end"],
            eps=row["eps"],
            first_filed=row["first_filed"],
            derived=row["derived"],
        )
        for row in frame.iter_rows(named=True)
    ]


def surprises_frame(surprises: Sequence[Surprise]) -> pl.DataFrame:
    rows = [
        {"period_end": s.period_end, "eps": s.eps, "eps_year_ago": s.eps_year_ago, "sue": s.sue}
        for s in surprises
    ]
    schema = {
        "period_end": pl.Date,
        "eps": pl.Float64,
        "eps_year_ago": pl.Float64,
        "sue": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)
