"""Paper trading: track simple event rules on filings that arrive *after* each rule
was written down.

A backtest can always be tuned until it looks good. The honest test of a pattern is
how it does on data nobody had seen when the rule was chosen. So:

* Rules live in a TOML file (see ``paper/rules.toml``).
* The first time a rule is seen, its definition's fingerprint and the current time
  are written to a lock file (``rules.lock.json``, meant to be committed). Only
  filings accepted after that moment count as **live** trades.
* Changing a rule after it started is refused: that would rewrite its track record.
  Give the changed rule a new name and it starts a fresh record.
* Trades are recomputed from filings and prices on every run, so there's no
  hand-maintained ledger to drift. The same rule applied to filings *before* its
  start date is reported separately as the **backtest**, for comparison.

P&L is the beta-hedged abnormal return (stock minus beta x benchmark, as in the event
study), signed by direction: a short makes money when the stock underperforms.
Hypothetical: no costs, borrow fees or slippage.
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import polars as pl

from investotron.event_study import EARNINGS_ITEM, attach_entry, calendar_block, clustered_mean
from investotron.identifiers import Ticker

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_ITEM_RE = re.compile(r"^\d\.\d\d$")
MAX_HORIZON = 250
SURPRISE_BUCKETS = {"miss": "1 miss", "in line": "2 in line", "beat": "3 beat"}
_RULE_KEYS = {"name", "item", "direction", "horizon", "surprise"}
_BOOK_KEYS = {"universe", "benchmark", "rule"}


class Direction(StrEnum):
    LONG = "long"
    SHORT = "short"


class Status(StrEnum):
    OPEN = "open"  # holding period not over yet; marked at the latest close
    CLOSED = "closed"
    INTERRUPTED = "interrupted"  # price history broke (halt, delisting) mid-trade


class RuleError(ValueError):
    pass


class RuleChangedError(RuleError):
    pass


@dataclass(frozen=True, slots=True)
class Rule:
    """Trade ``direction`` at the first open after any 8-K with ``item``, hold for
    ``horizon`` sessions. ``surprise`` (earnings only) narrows to misses/beats."""

    name: str
    item: str
    direction: Direction
    horizon: int
    surprise: str | None = None

    def __post_init__(self) -> None:
        if not _NAME_RE.fullmatch(self.name):
            raise RuleError(f"rule name {self.name!r}: use lower-case letters, digits and -")
        if not _ITEM_RE.fullmatch(self.item):
            raise RuleError(f"rule {self.name}: item {self.item!r} should look like '2.02'")
        if not 1 <= self.horizon <= MAX_HORIZON:
            raise RuleError(f"rule {self.name}: horizon must be 1..{MAX_HORIZON} sessions")
        if self.surprise is not None:
            if self.surprise not in SURPRISE_BUCKETS:
                raise RuleError(
                    f"rule {self.name}: surprise must be one of {', '.join(SURPRISE_BUCKETS)}"
                )
            if self.item != EARNINGS_ITEM:
                raise RuleError(f"rule {self.name}: surprise only applies to item {EARNINGS_ITEM}")

    @property
    def sign(self) -> int:
        return 1 if self.direction is Direction.LONG else -1

    def fingerprint(self) -> str:
        """Identifies the rule's *behaviour*; the name isn't part of it."""
        spec = {
            "item": self.item,
            "direction": self.direction.value,
            "horizon": self.horizon,
            "surprise": self.surprise,
        }
        return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]

    def matches(self) -> pl.Expr:
        """Filter for a filings frame with ``items`` (and ``surprise`` buckets)."""
        condition = pl.col("items").list.contains(self.item)
        if self.surprise is not None:
            condition &= pl.col("surprise") == SURPRISE_BUCKETS[self.surprise]
        return condition

    @classmethod
    def from_toml(cls, raw: dict[str, Any]) -> Rule:
        name = str(raw.get("name", "?"))
        unknown = set(raw) - _RULE_KEYS
        if unknown:
            raise RuleError(f"rule {name}: unknown setting(s) {', '.join(sorted(unknown))}")
        missing = {"name", "item", "direction", "horizon"} - set(raw)
        if missing:
            raise RuleError(f"rule {name}: missing {', '.join(sorted(missing))}")
        try:
            direction = Direction(raw["direction"])
        except ValueError:
            raise RuleError(f"rule {name}: direction must be 'long' or 'short'") from None
        horizon = raw["horizon"]
        if not isinstance(horizon, int) or isinstance(horizon, bool):
            raise RuleError(f"rule {name}: horizon must be a whole number of sessions")
        surprise = raw.get("surprise")
        return cls(
            name=name,
            item=str(raw["item"]),
            direction=direction,
            horizon=horizon,
            surprise=None if surprise is None else str(surprise),
        )


@dataclass(frozen=True, slots=True)
class RuleBook:
    universe: Path
    benchmark: Ticker
    rules: tuple[Rule, ...]


def load_rulebook(path: Path) -> RuleBook:
    """Read a rules file. ``universe`` is resolved relative to the rules file."""
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise RuleError(f"{path}: {error}") from error
    unknown = set(raw) - _BOOK_KEYS
    if unknown:
        raise RuleError(f"{path}: unknown setting(s) {', '.join(sorted(unknown))}")
    if "universe" not in raw:
        raise RuleError(f"{path}: missing 'universe'")
    rule_tables = cast("list[dict[str, Any]]", raw.get("rule", []))
    rules = tuple(Rule.from_toml(table) for table in rule_tables)
    names = [r.name for r in rules]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise RuleError(f"{path}: duplicate rule name(s) {', '.join(duplicates)}")
    if not rules:
        raise RuleError(f"{path}: no [[rule]] entries")
    return RuleBook(
        universe=(path.parent / str(raw["universe"])).resolve(),
        benchmark=Ticker.parse(str(raw.get("benchmark", "SPY"))),
        rules=rules,
    )


@dataclass(frozen=True, slots=True)
class LockEntry:
    fingerprint: str
    started_at: datetime


def _read_lock(path: Path) -> dict[str, LockEntry]:
    if not path.exists():
        return {}
    raw = cast("dict[str, dict[str, str]]", json.loads(path.read_text(encoding="utf-8")))
    return {
        name: LockEntry(entry["fingerprint"], datetime.fromisoformat(entry["started_at"]))
        for name, entry in raw.items()
    }


def sync_lock(
    rules: tuple[Rule, ...], lock_path: Path, now: datetime
) -> tuple[dict[str, LockEntry], list[str]]:
    """Start tracking new rules as of ``now``; refuse rules changed since they started.

    Returns the lock entries and the names started on this call. Rules removed from
    the rules file stay in the lock, so a name can't be reused for a different rule.
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    entries = _read_lock(lock_path)
    started: list[str] = []
    for rule in rules:
        entry = entries.get(rule.name)
        if entry is None:
            entries[rule.name] = LockEntry(rule.fingerprint(), now)
            started.append(rule.name)
        elif entry.fingerprint != rule.fingerprint():
            raise RuleChangedError(
                f"rule {rule.name} changed since it started on "
                f"{entry.started_at:%Y-%m-%d}. Editing it would rewrite its track record; "
                "give the new version a new name instead."
            )
    if started:
        payload = {
            name: {"fingerprint": e.fingerprint, "started_at": e.started_at.isoformat()}
            for name, e in sorted(entries.items())
        }
        lock_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return entries, started


def rule_trades(
    rule: Rule, filings: pl.DataFrame, prices: pl.DataFrame, started_at: datetime
) -> pl.DataFrame:
    """Every trade ``rule`` would have made on one stock's ``filings``.

    ``filings`` needs ``ticker``, ``accession_number``, ``items``, ``known_at`` and,
    for surprise rules, ``surprise``; ``prices`` comes from ``aligned_prices``.
    Open trades are marked at the latest close. ``live`` marks trades on filings
    accepted at or after ``started_at``.
    """
    if prices.is_empty():
        return pl.DataFrame()
    last_index = cast("int", prices["session_index"].max())
    segment_ends = prices.group_by("segment").agg(
        pl.col("session_index").max().alias("segment_end")
    )
    exits = prices.select(
        pl.col("session_index").alias("exit_index"),
        pl.col("session").alias("exit_session"),
        pl.col("adj_close").alias("exit_close"),
        pl.col("bench_adj_close").alias("bench_exit_close"),
    )
    target = pl.col("entry_index") + (rule.horizon - 1)
    stock_return = pl.col("exit_close") / pl.col("entry_open") - 1
    bench_return = pl.col("bench_exit_close") / pl.col("bench_entry_open") - 1
    abnormal = stock_return - pl.col("beta") * bench_return
    return (
        attach_entry(filings.filter(rule.matches()), prices)
        .drop_nulls("entry_beta")
        .rename({"entry_beta": "beta"})
        .join(segment_ends, left_on="entry_segment", right_on="segment")
        .with_columns(
            pl.min_horizontal(target, pl.col("segment_end")).alias("exit_index"),
            pl.when(target <= pl.col("segment_end"))
            .then(pl.lit(Status.CLOSED.value))
            .when(pl.col("segment_end") == last_index)
            .then(pl.lit(Status.OPEN.value))
            .otherwise(pl.lit(Status.INTERRUPTED.value))
            .alias("status"),
        )
        .join(exits, on="exit_index")
        .with_columns(
            pl.lit(rule.name).alias("rule"),
            (pl.col("exit_index") - pl.col("entry_index") + 1).alias("sessions_held"),
            stock_return.alias("stock_return"),
            bench_return.alias("bench_return"),
            (abnormal * rule.sign).alias("pnl"),
            (pl.col("known_at") >= started_at).alias("live"),
        )
        .select(
            "rule",
            "ticker",
            "accession_number",
            "known_at",
            "live",
            "status",
            "entry_session",
            "exit_session",
            "sessions_held",
            "beta",
            "stock_return",
            "bench_return",
            "pnl",
        )
    )


def pending_signals(
    rule: Rule, filings: pl.DataFrame, market_last_open: datetime, started_at: datetime
) -> pl.DataFrame:
    """Live matching filings accepted after the market's latest open: they'll be
    entered at the next open, once that session's prices exist.

    ``market_last_open`` should come from the benchmark, not the stock -- a stock
    whose prices stopped earlier (delisted, acquired) has nothing to wait for.
    """
    return filings.filter(
        rule.matches()
        & (pl.col("known_at") >= market_last_open)
        & (pl.col("known_at") >= started_at)
    )


def summarize_trades(trades: pl.DataFrame, horizon: int) -> pl.DataFrame:
    """P&L statistics of finished trades (closed or interrupted), split into
    ``backtest`` and ``live``; ``t_stat`` is clustered by calendar block."""
    finished = trades.filter(pl.col("status") != Status.OPEN.value)
    if finished.is_empty():
        return pl.DataFrame()
    rows = finished.with_columns(
        pl.when(pl.col("live")).then(pl.lit("live")).otherwise(pl.lit("backtest")).alias("sample"),
        calendar_block(pl.col("entry_session"), pl.lit(horizon)).alias("cluster"),
    )
    return clustered_mean(rows, ["sample"], "pnl").sort("sample", descending=True)
