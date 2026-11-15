from datetime import UTC, datetime, timedelta

import pytest

from investotron.point_in_time import AsOf, Fact, PointInTimeStore

T0 = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)


def fact(name: str, known_offset_hours: int) -> Fact[str]:
    return Fact(name, occurred_at=T0, known_at=T0 + timedelta(hours=known_offset_hours))


def test_as_of_hides_facts_not_yet_known() -> None:
    store = PointInTimeStore([fact("later", 5), fact("early", 1), fact("middle", 3)])

    view = store.as_of(T0 + timedelta(hours=3))

    assert [f.value for f in view] == ["early", "middle"]  # inclusive, ordered by known_at
    latest = view.latest()
    assert latest is not None
    assert latest.value == "middle"


def test_as_of_before_everything_is_empty() -> None:
    store = PointInTimeStore([fact("a", 1)])
    view = store.as_of(T0)
    assert len(view) == 0
    assert view.latest() is None


def test_naive_datetimes_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        Fact("x", occurred_at=datetime(2024, 1, 1), known_at=T0)  # noqa: DTZ001
    with pytest.raises(ValueError, match="timezone-aware"):
        PointInTimeStore([fact("a", 1)]).as_of(datetime(2024, 1, 1))  # noqa: DTZ001


def test_as_of_cannot_be_built_with_future_facts() -> None:
    with pytest.raises(ValueError, match="not yet known"):
        AsOf(T0, (fact("future", 1),))
