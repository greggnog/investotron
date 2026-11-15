"""Daily price bars and when each part of a bar becomes known.

A daily bar is not known all at once: the open is public at 9:30 ET, the close at
16:00 ET. An event published after the close (like most earnings 8-Ks) can first
be traded at the *next* session's open. Bars carry ``opens_at`` / ``closes_at``
so the event study can pick prices by time rather than by date.

Early-close days (e.g. 13:00 ET the day after Thanksgiving) still get 16:00 here.
That only makes ``closes_at`` later than reality, so it can never cause look-ahead.

Adjusted prices are back-adjusted using splits and dividends that happen *later*,
so their absolute level isn't point-in-time. Ratios between two adjusted prices
(i.e. returns) are correct, which is all they should be used for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from investotron.identifiers import Ticker

EASTERN = ZoneInfo("America/New_York")
SESSION_OPEN = time(9, 30)
SESSION_CLOSE = time(16, 0)


@dataclass(frozen=True, slots=True)
class DailyBar:
    ticker: Ticker
    session: date
    open: float
    high: float
    low: float
    close: float
    volume: int
    adj_open: float
    adj_close: float

    def __post_init__(self) -> None:
        prices = (self.open, self.high, self.low, self.close, self.adj_open, self.adj_close)
        if not all(math.isfinite(p) and p > 0 for p in prices):
            raise ValueError(f"{self.ticker} {self.session}: non-positive price in {prices}")
        if self.low > self.high:
            raise ValueError(f"{self.ticker} {self.session}: low {self.low} > high {self.high}")
        if self.volume < 0:
            raise ValueError(f"{self.ticker} {self.session}: negative volume")

    @property
    def opens_at(self) -> datetime:
        return datetime.combine(self.session, SESSION_OPEN, tzinfo=EASTERN)

    @property
    def closes_at(self) -> datetime:
        return datetime.combine(self.session, SESSION_CLOSE, tzinfo=EASTERN)
