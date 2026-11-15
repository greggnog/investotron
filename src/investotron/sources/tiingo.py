"""Tiingo: end-of-day stock prices.

Free API key from https://www.tiingo.com/ (free tier is rate limited, roughly
1,000 requests/day, which is plenty for daily bars). Docs:
https://www.tiingo.com/documentation/end-of-day

Same split as the EDGAR module: the client returns raw JSON, and
:func:`parse_daily_bars` is a pure function from that JSON to typed bars.
"""

from __future__ import annotations

from datetime import date, datetime
from types import TracebackType
from typing import Any, Self, cast

import httpx

from investotron.identifiers import Ticker
from investotron.prices import DailyBar

DAILY_PRICES_URL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"
HTTP_TOO_MANY_REQUESTS = 429


def _tiingo_symbol(ticker: Ticker) -> str:
    # Tiingo writes share classes with a dash (BRK-B) where exchanges use a dot.
    return ticker.value.replace(".", "-")


def parse_daily_bars(ticker: Ticker, raw: Any) -> list[DailyBar]:
    """Turn a Tiingo daily-prices payload into bars, oldest first."""
    rows: list[dict[str, Any]] = raw
    bars = [
        DailyBar(
            ticker=ticker,
            # Tiingo stamps each session as midnight UTC; only the date part means anything.
            session=datetime.fromisoformat(row["date"]).date(),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=int(row["volume"]),
            adj_open=float(row["adjOpen"]),
            adj_close=float(row["adjClose"]),
        )
        for row in rows
    ]
    bars.sort(key=lambda b: b.session)
    sessions = [b.session for b in bars]
    if len(set(sessions)) != len(sessions):
        raise ValueError(f"{ticker}: duplicate sessions in Tiingo response")
    return bars


class TiingoClient:
    def __init__(self, api_key: str, *, transport: httpx.BaseTransport | None = None) -> None:
        if not api_key.strip():
            raise ValueError("Tiingo API key is empty")
        # Send the key as a header, not a query parameter, so it never shows up
        # in URLs, logs or error messages.
        self._http = httpx.Client(
            headers={"Authorization": f"Token {api_key}"},
            timeout=30.0,
            transport=transport,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._http.close()

    def daily_prices(self, ticker: Ticker, start: date, end: date | None = None) -> Any:
        params = {"startDate": start.isoformat(), "format": "json"}
        if end is not None:
            params["endDate"] = end.isoformat()
        response = self._http.get(
            DAILY_PRICES_URL.format(ticker=_tiingo_symbol(ticker)), params=params
        )
        if response.status_code == HTTP_TOO_MANY_REQUESTS:
            raise TiingoRateLimitError(_detail(response))
        response.raise_for_status()
        payload: Any = response.json()
        if not isinstance(payload, list):  # errors come back as {"detail": "..."}
            raise TiingoError(f"{ticker}: {_detail(response)}")
        return cast("list[Any]", payload)


class TiingoError(Exception):
    pass


class TiingoRateLimitError(TiingoError):
    """Free tier allows only so many requests per hour/day; retry later."""


def _detail(response: httpx.Response) -> str:
    try:
        body: Any = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(body, dict):
        return str(cast("dict[str, Any]", body).get("detail", body))
    return str(body)[:200]
