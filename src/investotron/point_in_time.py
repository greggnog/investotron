"""Point-in-time data, the guard against look-ahead bias.

Every fact carries two timestamps:

* ``occurred_at`` -- when the thing happened in the world (e.g. the event an 8-K reports).
* ``known_at``    -- when the market could first have known it (e.g. the moment the
  filing was accepted and published by the SEC).

Backtests must only ever use ``known_at``. To make that hard to get wrong, a
:class:`PointInTimeStore` has no way to list its facts directly: the only read path
is :meth:`PointInTimeStore.as_of`, which returns just the facts known at that moment.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime


def _require_aware(name: str, value: datetime) -> None:
    # Naive datetimes are ambiguous (whose midnight?), and comparing them to aware
    # ones raises. Reject them at the boundary instead of deep inside a backtest.
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware, got naive {value!r}")


@dataclass(frozen=True, slots=True)
class Fact[T]:
    """A value plus when it happened and when it became public."""

    value: T
    occurred_at: datetime
    known_at: datetime

    def __post_init__(self) -> None:
        _require_aware("occurred_at", self.occurred_at)
        _require_aware("known_at", self.known_at)


@dataclass(frozen=True, slots=True)
class AsOf[T]:
    """The facts that were public at ``moment``, ordered by ``known_at``."""

    moment: datetime
    facts: tuple[Fact[T], ...]

    def __post_init__(self) -> None:
        _require_aware("moment", self.moment)
        leaked = [f for f in self.facts if f.known_at > self.moment]
        if leaked:
            raise ValueError(
                f"{len(leaked)} fact(s) were not yet known at {self.moment.isoformat()}"
            )

    def __iter__(self) -> Iterator[Fact[T]]:
        return iter(self.facts)

    def __len__(self) -> int:
        return len(self.facts)

    def latest(self) -> Fact[T] | None:
        """The most recently published fact, if any."""
        return self.facts[-1] if self.facts else None


class PointInTimeStore[T]:
    """An immutable collection of facts that can only be read as of a moment."""

    __slots__ = ("_facts", "_known_ats")

    def __init__(self, facts: Iterable[Fact[T]]) -> None:
        ordered = sorted(facts, key=lambda f: f.known_at)
        self._facts: tuple[Fact[T], ...] = tuple(ordered)
        self._known_ats: tuple[datetime, ...] = tuple(f.known_at for f in ordered)

    def __len__(self) -> int:
        return len(self._facts)

    def as_of(self, moment: datetime) -> AsOf[T]:
        """Facts with ``known_at <= moment``. O(log n) to find the cutoff."""
        _require_aware("moment", moment)
        cutoff = bisect.bisect_right(self._known_ats, moment)
        return AsOf(moment, self._facts[:cutoff])
