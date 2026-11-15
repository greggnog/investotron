from datetime import UTC, date, datetime
from typing import Any

import httpx
import pytest

from investotron.identifiers import Ticker
from investotron.prices import DailyBar
from investotron.sources.tiingo import (
    TiingoClient,
    TiingoError,
    TiingoRateLimitError,
    parse_daily_bars,
)
from investotron.storage import prices_frame

AAPL = Ticker("AAPL")


def row(day: str, close: float = 101.0) -> dict[str, Any]:
    return {
        "date": f"{day}T00:00:00.000Z",
        "open": 100.0,
        "high": 102.0,
        "low": 99.0,
        "close": close,
        "volume": 1_000,
        "adjOpen": 50.0,
        "adjClose": close / 2,
        "adjHigh": 51.0,
        "adjLow": 49.5,
        "adjVolume": 2_000,
        "divCash": 0.0,
        "splitFactor": 1.0,
    }


def test_parse_sorts_by_session_and_keeps_date_only() -> None:
    bars = parse_daily_bars(AAPL, [row("2024-01-03"), row("2024-01-02")])
    assert [b.session for b in bars] == [date(2024, 1, 2), date(2024, 1, 3)]
    assert bars[0].adj_close == 50.5


def test_duplicate_sessions_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        parse_daily_bars(AAPL, [row("2024-01-02"), row("2024-01-02")])


def test_session_times_follow_daylight_saving() -> None:
    winter, summer = parse_daily_bars(AAPL, [row("2024-01-02"), row("2024-07-01")])
    assert winter.opens_at.astimezone(UTC) == datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    assert winter.closes_at.astimezone(UTC) == datetime(2024, 1, 2, 21, 0, tzinfo=UTC)
    assert summer.closes_at.astimezone(UTC) == datetime(2024, 7, 1, 20, 0, tzinfo=UTC)


def test_after_close_filing_first_tradable_next_open() -> None:
    # Earnings 8-K accepted 16:30 ET: after that day's close, before the next open.
    day1, day2 = parse_daily_bars(AAPL, [row("2024-02-01"), row("2024-02-02")])
    filed = datetime(2024, 2, 1, 21, 30, tzinfo=UTC)
    assert day1.closes_at < filed < day2.opens_at


@pytest.mark.parametrize(
    "changes",
    [{"close": 0.0}, {"low": 105.0}, {"volume": -1}, {"adj_close": float("nan")}],
)
def test_bad_bars_rejected(changes: dict[str, Any]) -> None:
    fields: dict[str, Any] = {
        "ticker": AAPL,
        "session": date(2024, 1, 2),
        "open": 100.0,
        "high": 102.0,
        "low": 99.0,
        "close": 101.0,
        "volume": 10,
        "adj_open": 100.0,
        "adj_close": 101.0,
    }
    with pytest.raises(ValueError, match=r"AAPL 2024-01-02"):
        DailyBar(**(fields | changes))


def test_prices_frame_utc_and_sorted() -> None:
    frame = prices_frame(parse_daily_bars(AAPL, [row("2024-01-03"), row("2024-01-02")]))
    assert frame["session"].to_list() == [date(2024, 1, 2), date(2024, 1, 3)]
    assert frame["opens_at"][0] == datetime(2024, 1, 2, 14, 30, tzinfo=UTC)


def test_client_sends_key_in_header_and_maps_share_class() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=[row("2024-01-02")])

    with TiingoClient("secret", transport=httpx.MockTransport(handler)) as tiingo:
        tiingo.daily_prices(Ticker("BRK.B"), date(2024, 1, 1))

    request = seen[0]
    assert request.headers["Authorization"] == "Token secret"
    assert "secret" not in str(request.url)
    assert request.url.path == "/tiingo/daily/BRK-B/prices"
    assert request.url.params["startDate"] == "2024-01-01"


@pytest.mark.parametrize(
    ("status", "body", "error"),
    [
        (429, {"detail": "hourly limit"}, TiingoRateLimitError),
        (200, {"detail": "Error: ticker not found"}, TiingoError),
    ],
)
def test_client_turns_error_payloads_into_exceptions(
    status: int, body: dict[str, Any], error: type[Exception]
) -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(status, json=body))
    with (
        TiingoClient("k", transport=transport) as tiingo,
        pytest.raises(error, match=body["detail"]),
    ):
        tiingo.daily_prices(AAPL, date(2024, 1, 1))
