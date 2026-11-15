"""Quarterly EPS from SEC XBRL data, and earnings surprise.

Earnings surprise usually means "actual EPS vs analysts' consensus", but consensus
data isn't free. The standard stand-in from the academic literature is
**SUE** (standardised unexpected earnings) with a *seasonal random walk*:

    expected EPS this quarter = EPS in the same quarter last year
    SUE = (actual - expected) / std dev of that difference over prior quarters

So SUE = +2 means "earnings grew year-on-year by twice the usual amount".

Point-in-time notes:

* XBRL EPS comes from the 10-Q/10-K, usually filed a few weeks *after* the
  earnings 8-K. The number itself was public in the 8-K press release, so using it
  at the 8-K's time is fine; the later filing just makes it machine-readable.
* Companies restate. We keep the *first-reported* value for each period, never a
  later revision, since that's what the market saw at the time.
* 10-Ks report the full year, not Q4. Q4 is derived as the annual figure minus
  Q1-Q3. Per-share numbers don't add up exactly when the share count changes, so
  derived Q4s are approximate (flagged with ``derived=True``).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

EPS_TAGS = ("EarningsPerShareDiluted", "EarningsPerShareBasic")
_QUARTER_DAYS = range(80, 101)
_YEAR_DAYS = range(350, 381)
_YEAR_AGO_TOLERANCE = timedelta(days=20)


@dataclass(frozen=True, slots=True)
class QuarterlyEps:
    period_start: date
    period_end: date
    eps: float
    first_filed: date
    derived: bool = False


@dataclass(frozen=True, slots=True)
class Surprise:
    period_end: date
    eps: float
    eps_year_ago: float
    sue: float | None  # None until there's enough history to estimate the spread


@dataclass(frozen=True, slots=True)
class _Reported:
    start: date
    end: date
    value: float
    filed: date

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1


def parse_quarterly_eps(company_facts: Any) -> list[QuarterlyEps]:
    """Quarterly EPS series (oldest first) from a ``companyfacts`` payload.

    Uses diluted EPS, falling back to basic. Returns [] if neither is reported.
    """
    reported = _first_reported(company_facts)
    quarters = {(r.start, r.end): r for r in reported if r.days in _QUARTER_DAYS}
    annuals = [r for r in reported if r.days in _YEAR_DAYS]

    result = [QuarterlyEps(q.start, q.end, q.value, q.filed) for q in quarters.values()]
    quarter_ends = {q.end for q in quarters.values()}
    for year in annuals:
        if year.end in quarter_ends:
            continue  # company reported Q4 directly
        inside = sorted(
            (q for q in quarters.values() if q.start >= year.start and q.end < year.end),
            key=lambda q: q.end,
        )
        if len(inside) != 3:  # noqa: PLR2004 -- Q1..Q3
            continue
        result.append(
            QuarterlyEps(
                period_start=inside[-1].end + timedelta(days=1),
                period_end=year.end,
                eps=year.value - sum(q.value for q in inside),
                first_filed=year.filed,
                derived=True,
            )
        )

    by_end: dict[date, QuarterlyEps] = {}
    for q in sorted(result, key=lambda q: (q.period_end, q.first_filed)):
        by_end.setdefault(q.period_end, q)
    return list(by_end.values())


def _first_reported(company_facts: Any) -> list[_Reported]:
    gaap: dict[str, Any] = company_facts.get("facts", {}).get("us-gaap", {})
    entries: list[dict[str, Any]] = []
    for tag in EPS_TAGS:  # first tag the company reports wins
        if tag in gaap:
            entries = gaap[tag]["units"]["USD/shares"]
            break
    first: dict[tuple[date, date], _Reported] = {}
    for entry in entries:
        if entry.get("form") not in {"10-Q", "10-K"} or "start" not in entry:
            continue  # amendments are revisions; instant facts have no start
        candidate = _Reported(
            start=date.fromisoformat(entry["start"]),
            end=date.fromisoformat(entry["end"]),
            value=float(entry["val"]),
            filed=date.fromisoformat(entry["filed"]),
        )
        key = (candidate.start, candidate.end)
        if key not in first or candidate.filed < first[key].filed:
            first[key] = candidate
    return list(first.values())


def earnings_surprises(
    quarters: list[QuarterlyEps], lookback: int = 8, min_history: int = 4
) -> list[Surprise]:
    """Seasonal-random-walk SUE for each quarter that has a year-ago quarter.

    The spread uses only the ``lookback`` differences *before* this quarter, so a
    quarter's SUE never depends on its own or later results.
    """
    ordered = sorted(quarters, key=lambda q: q.period_end)
    by_end = {q.period_end: q for q in ordered}

    surprises: list[Surprise] = []
    history: list[float] = []
    for quarter in ordered:
        year_ago = _find_year_ago(quarter.period_end, by_end)
        if year_ago is None:
            continue
        difference = quarter.eps - year_ago.eps
        recent = history[-lookback:]
        sue = None
        if len(recent) >= min_history:
            spread = statistics.stdev(recent)
            sue = difference / spread if spread > 0 else None
        surprises.append(Surprise(quarter.period_end, quarter.eps, year_ago.eps, sue))
        history.append(difference)
    return surprises


def _find_year_ago(period_end: date, by_end: dict[date, QuarterlyEps]) -> QuarterlyEps | None:
    target = period_end - timedelta(days=365)
    candidates = [q for end, q in by_end.items() if abs(end - target) <= _YEAR_AGO_TOLERANCE]
    return min(candidates, key=lambda q: abs(q.period_end - target), default=None)
