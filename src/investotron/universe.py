"""Ticker lists ("universes") to run fetches and studies over."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from investotron.identifiers import Ticker


def load_universe(path: Path) -> list[Ticker]:
    """One ticker per line; ``#`` starts a comment; blank lines are ignored."""
    tickers: list[Ticker] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        try:
            tickers.append(Ticker.parse(text))
        except ValueError as error:
            raise ValueError(f"{path}:{number}: {error}") from error
    return tickers


def resolve_tickers(explicit: Sequence[Ticker], universe: Path | None) -> list[Ticker]:
    """Tickers from the command line plus an optional universe file, deduplicated
    in first-seen order."""
    combined = [*explicit, *(load_universe(universe) if universe else [])]
    if not combined:
        raise ValueError("give at least one ticker or --universe FILE")
    return list(dict.fromkeys(combined))
