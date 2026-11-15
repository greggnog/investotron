"""Small value types so tickers, CIKs, etc. can't be mixed up with plain strings/ints."""

from __future__ import annotations

import re
from dataclasses import dataclass

_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
_MAX_CIK = 9_999_999_999  # CIKs are at most 10 digits


@dataclass(frozen=True, slots=True)
class Ticker:
    """An exchange ticker symbol, always stored upper-case (e.g. ``BRK.B``)."""

    value: str

    def __post_init__(self) -> None:
        if not _TICKER_RE.fullmatch(self.value):
            raise ValueError(f"not a valid ticker: {self.value!r}")

    @classmethod
    def parse(cls, raw: str) -> Ticker:
        """Normalise user input (whitespace, case) before validating."""
        return cls(raw.strip().upper())

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class Cik:
    """SEC Central Index Key: the SEC's permanent id for a filer."""

    value: int

    def __post_init__(self) -> None:
        if not 0 < self.value <= _MAX_CIK:
            raise ValueError(f"not a valid CIK: {self.value}")

    @property
    def padded(self) -> str:
        """Zero-padded 10-digit form used in EDGAR URLs."""
        return f"{self.value:010d}"

    def __str__(self) -> str:
        return self.padded
