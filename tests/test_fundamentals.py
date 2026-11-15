from datetime import date
from typing import Any

import pytest

from investotron.fundamentals import QuarterlyEps, earnings_surprises, parse_quarterly_eps


def fact(start: str, end: str, val: float, filed: str, form: str = "10-Q") -> dict[str, Any]:
    return {"start": start, "end": end, "val": val, "filed": filed, "form": form}


def company_facts(*entries: dict[str, Any], tag: str = "EarningsPerShareDiluted") -> dict[str, Any]:
    return {"facts": {"us-gaap": {tag: {"units": {"USD/shares": list(entries)}}}}}


FY2023 = company_facts(
    fact("2023-01-01", "2023-03-31", 1.00, "2023-05-01"),
    fact("2023-04-01", "2023-06-30", 1.10, "2023-08-01"),
    fact("2023-01-01", "2023-06-30", 2.10, "2023-08-01"),  # year-to-date: ignored
    fact("2023-07-01", "2023-09-30", 1.20, "2023-11-01"),
    fact("2023-01-01", "2023-12-31", 4.80, "2024-02-15", form="10-K"),
    # Q1 2023 restated in a later filing: the first-reported 1.00 must win
    fact("2023-01-01", "2023-03-31", 0.90, "2024-05-01"),
    fact("2023-01-01", "2023-03-31", 0.50, "2023-06-01", form="10-Q/A"),
)


def test_parse_keeps_first_reported_and_derives_q4() -> None:
    quarters = parse_quarterly_eps(FY2023)

    assert [q.period_end for q in quarters] == [
        date(2023, 3, 31),
        date(2023, 6, 30),
        date(2023, 9, 30),
        date(2023, 12, 31),
    ]
    assert quarters[0].eps == 1.00
    q4 = quarters[-1]
    assert q4.derived
    assert q4.period_start == date(2023, 10, 1)
    assert q4.eps == pytest.approx(4.80 - 1.00 - 1.10 - 1.20)
    assert q4.first_filed == date(2024, 2, 15)


def test_parse_falls_back_to_basic_and_handles_missing() -> None:
    basic = company_facts(
        fact("2023-01-01", "2023-03-31", 1.0, "2023-05-01"), tag="EarningsPerShareBasic"
    )
    assert len(parse_quarterly_eps(basic)) == 1
    assert parse_quarterly_eps({"facts": {}}) == []


def quarter_series(eps: list[float]) -> list[QuarterlyEps]:
    ends = [
        date(2020 + i // 4, 3 * (i % 4) + 3, 30 if (i % 4) in (1, 2) else 31)
        for i in range(len(eps))
    ]
    return [
        QuarterlyEps(date(e.year, e.month - 2, 1), e, v, e) for e, v in zip(ends, eps, strict=True)
    ]


def test_sue_uses_only_prior_quarters() -> None:
    # Year-on-year differences: 0.1, -0.1, 0.1, -0.1 (spread ~0.115), then a +0.5 jump.
    eps = [1.0, 1.0, 1.0, 1.0, 1.1, 0.9, 1.1, 0.9, 1.6]
    surprises = earnings_surprises(quarter_series(eps), min_history=4)

    assert [s.sue for s in surprises[:4]] == [None] * 4  # not enough history yet
    last = surprises[-1]
    assert last.eps_year_ago == 1.1
    assert last.sue == pytest.approx(0.5 / 0.11547, rel=1e-3)


def test_sue_is_none_when_history_is_flat() -> None:
    surprises = earnings_surprises(quarter_series([1.0] * 9))
    assert surprises[-1].sue is None
