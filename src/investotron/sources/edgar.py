"""SEC EDGAR: company filings.

Two halves, kept apart on purpose:

* :class:`EdgarClient` does network I/O and returns raw JSON, untouched.
* :func:`parse_filings` is a pure function from raw JSON to :class:`Fact` objects,
  so it can be tested without the network and re-run over saved raw files.

API docs: https://www.sec.gov/search-filings/edgar-application-programming-interfaces
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from datetime import date, datetime, time
from time import monotonic, sleep
from types import TracebackType
from typing import Any, Self, TypedDict
from zoneinfo import ZoneInfo

import httpx

from investotron.identifiers import Cik, Ticker
from investotron.point_in_time import Fact

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# SEC fair-access policy: at most 10 requests per second.
MIN_REQUEST_INTERVAL = 0.11

EASTERN = ZoneInfo("America/New_York")

# Common 8-K item codes. Full list: https://www.sec.gov/fast-answers/answersform8khtm.html
EIGHT_K_ITEMS: dict[str, str] = {
    "1.01": "Material agreement",
    "1.02": "Agreement terminated",
    "2.01": "Acquisition/disposition",
    "2.02": "Earnings",
    "2.03": "New debt obligation",
    "2.05": "Restructuring",
    "2.06": "Material impairment",
    "3.01": "Delisting notice",
    "3.02": "Unregistered share sale",
    "3.03": "Shareholder rights changed",
    "4.02": "Restatement",
    "5.02": "Officer/director change",
    "5.03": "Bylaws/charter amended",
    "5.07": "Shareholder vote results",
    "7.01": "Reg FD disclosure",
    "8.01": "Other events",
    "9.01": "Exhibits",
}

# Functional syntax because the keys are EDGAR's camelCase names.
_RecentFilings = TypedDict(  # noqa: UP013
    "_RecentFilings",
    {
        "accessionNumber": list[str],
        "filingDate": list[str],
        "reportDate": list[str],
        "acceptanceDateTime": list[str],
        "form": list[str],
        "items": list[str],
        "primaryDocument": list[str],
    },
)


@dataclass(frozen=True, slots=True)
class Filing:
    cik: Cik
    accession_number: str
    form: str
    filed_on: date
    items: tuple[str, ...]  # 8-K item codes, e.g. ("2.02", "9.01") = results + exhibits
    primary_document: str

    @property
    def url(self) -> str:
        folder = self.accession_number.replace("-", "")
        return (
            f"https://www.sec.gov/Archives/edgar/data/{self.cik.value}/"
            f"{folder}/{self.primary_document}"
        )


def parse_filings(
    cik: Cik,
    submissions: Any,  # raw JSON from SUBMISSIONS_URL
    forms: Collection[str] = frozenset({"8-K"}),
) -> list[Fact[Filing]]:
    """Turn a submissions payload into point-in-time facts for the given form types.

    ``known_at`` is EDGAR's acceptance time: the moment the filing went public.
    ``occurred_at`` is the report date (the date of the event being disclosed),
    falling back to the filing date when no report date is given.

    Only covers ``filings.recent`` (roughly the last 1,000 filings); older ones
    live in the extra files listed under ``filings.files``.
    """
    recent: _RecentFilings = submissions["filings"]["recent"]
    count = len(recent["accessionNumber"])
    columns = (
        recent["filingDate"],
        recent["reportDate"],
        recent["acceptanceDateTime"],
        recent["form"],
        recent["items"],
        recent["primaryDocument"],
    )
    if any(len(column) != count for column in columns):
        raise ValueError("EDGAR submissions columns have mismatched lengths")

    facts: list[Fact[Filing]] = []
    for i in range(count):
        form = recent["form"][i]
        if form not in forms:
            continue
        filed_on = date.fromisoformat(recent["filingDate"][i])
        report_date = recent["reportDate"][i]
        occurred_on = date.fromisoformat(report_date) if report_date else filed_on
        filing = Filing(
            cik=cik,
            accession_number=recent["accessionNumber"][i],
            form=form,
            filed_on=filed_on,
            items=tuple(item for item in recent["items"][i].split(",") if item),
            primary_document=recent["primaryDocument"][i],
        )
        facts.append(
            Fact(
                value=filing,
                occurred_at=datetime.combine(occurred_on, time.min, tzinfo=EASTERN),
                known_at=_parse_acceptance(recent["acceptanceDateTime"][i]),
            )
        )
    return facts


def _parse_acceptance(raw: str) -> datetime:
    # The API's "...Z" suffix is real UTC. Verified 2026-11-15: accession
    # 0001140361-26-042798 is "2026-11-09T21:30:40.000Z" in the API and
    # "Accepted 2026-11-09 16:30:40" (Eastern, EST) on its EDGAR index page.
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        raise ValueError(f"acceptanceDateTime has no timezone: {raw!r}")
    return parsed


class EdgarClient:
    """Thin HTTP client. SEC requires a User-Agent with a contact email and
    asks for at most 10 requests per second."""

    def __init__(
        self,
        user_agent: str,
        *,
        transport: httpx.BaseTransport | None = None,
        min_interval: float = MIN_REQUEST_INTERVAL,
    ) -> None:
        if "@" not in user_agent:
            raise ValueError(
                "SEC requires a User-Agent with a contact email, e.g. 'investotron you@example.com'"
            )
        self._http = httpx.Client(
            headers={"User-Agent": user_agent},
            timeout=30.0,
            transport=transport,
        )
        self._min_interval = min_interval
        self._last_request = float("-inf")
        self._ciks: dict[str, Cik] | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._http.close()

    def _get_json(self, url: str) -> Any:
        wait = self._last_request + self._min_interval - monotonic()
        if wait > 0:
            sleep(wait)
        self._last_request = monotonic()
        response = self._http.get(url)
        response.raise_for_status()
        return response.json()

    def lookup_cik(self, ticker: Ticker) -> Cik:
        if self._ciks is None:  # one download serves every lookup
            companies: dict[str, dict[str, Any]] = self._get_json(TICKERS_URL)
            self._ciks = {c["ticker"]: Cik(int(c["cik_str"])) for c in companies.values()}
        cik = self._ciks.get(ticker.value)
        if cik is None:
            raise LookupError(f"ticker {ticker} not found in SEC company list")
        return cik

    def submissions(self, cik: Cik) -> Any:
        return self._get_json(SUBMISSIONS_URL.format(cik=cik.padded))

    def company_facts(self, cik: Cik) -> Any:
        """All XBRL financial data the company has filed (EPS, revenue, ...)."""
        return self._get_json(COMPANY_FACTS_URL.format(cik=cik.padded))
