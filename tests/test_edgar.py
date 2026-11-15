from datetime import UTC, date, datetime
from typing import Any

import httpx
import pytest

from investotron.identifiers import Cik, Ticker
from investotron.sources.edgar import EASTERN, EdgarClient, parse_filings
from investotron.storage import filings_frame

CIK = Cik(320193)

SUBMISSIONS: dict[str, Any] = {
    "filings": {
        "recent": {
            "accessionNumber": [
                "0000320193-24-000120",
                "0000320193-24-000119",
                "0000320193-24-0001",
            ],
            "filingDate": ["2024-10-31", "2024-10-31", "2024-08-02"],
            "reportDate": ["2024-10-31", "2024-09-28", ""],
            "acceptanceDateTime": [
                "2024-10-31T20:30:52.000Z",
                "2024-10-31T20:31:00.000Z",
                "2024-08-02T12:00:00.000Z",
            ],
            "form": ["8-K", "10-K", "8-K"],
            "items": ["2.02,9.01", "", "5.07"],
            "primaryDocument": ["aapl-20241031.htm", "aapl-20240928.htm", "x.htm"],
        }
    }
}


def test_parse_filings_keeps_only_8ks_with_both_timestamps() -> None:
    facts = parse_filings(CIK, SUBMISSIONS)

    assert len(facts) == 2
    first = facts[0]
    assert first.value.items == ("2.02", "9.01")
    assert first.value.filed_on == date(2024, 10, 31)
    assert first.occurred_at == datetime(2024, 10, 31, tzinfo=EASTERN)
    assert first.known_at == datetime(2024, 10, 31, 20, 30, 52, tzinfo=UTC)
    assert first.value.url == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019324000120/aapl-20241031.htm"
    )


def test_missing_report_date_falls_back_to_filing_date() -> None:
    facts = parse_filings(CIK, SUBMISSIONS)
    assert facts[1].occurred_at == datetime(2024, 8, 2, tzinfo=EASTERN)


def test_mismatched_columns_rejected() -> None:
    broken: dict[str, Any] = {"filings": {"recent": {**SUBMISSIONS["filings"]["recent"]}}}
    broken["filings"]["recent"]["form"] = ["8-K"]
    with pytest.raises(ValueError, match="mismatched"):
        parse_filings(CIK, broken)


def test_filings_frame_is_utc_and_sorted() -> None:
    frame = filings_frame(parse_filings(CIK, SUBMISSIONS))
    assert frame["accession_number"].to_list() == [
        "0000320193-24-0001",
        "0000320193-24-000120",
    ]
    assert frame.schema["known_at"] == frame.schema["occurred_at"]


def test_lookup_cik_uses_sec_ticker_list() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "@" in request.headers["User-Agent"]
        return httpx.Response(
            200, json={"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
        )

    with EdgarClient("test test@example.com", transport=httpx.MockTransport(handler)) as edgar:
        assert edgar.lookup_cik(Ticker("AAPL")) == CIK
        with pytest.raises(LookupError):
            edgar.lookup_cik(Ticker("ZZZZ"))


def test_client_requires_contact_email() -> None:
    with pytest.raises(ValueError, match="contact email"):
        EdgarClient("investotron")


def test_lookup_cik_downloads_ticker_list_once() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json={"0": {"cik_str": 320193, "ticker": "AAPL"}})

    transport = httpx.MockTransport(handler)
    with EdgarClient("t t@example.com", transport=transport, min_interval=0) as edgar:
        edgar.lookup_cik(Ticker("AAPL"))
        edgar.lookup_cik(Ticker("AAPL"))
    assert len(calls) == 1
