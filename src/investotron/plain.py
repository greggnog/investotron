"""Plain-English paper-trading report (``investotron paper --plain``).

No tables and no jargon: each rule as an idea in everyday words, what its past
record says to expect, and each trade as "the stock did X, the market did Y, so the
bet is winning or losing by Z".
"""

from __future__ import annotations

import textwrap
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from investotron.paper import Direction, Rule, Status
from investotron.sources.edgar import EIGHT_K_ITEMS

PLAIN_ENV = "INVESTOTRON_PLAIN"
EASTERN = ZoneInfo("America/New_York")
MIN_LIVE_TRADES = 20  # below this, a live average says little
_WIDTH = 92
_STRONG_T = 2
_FLAT = 0.0005  # moves smaller than 0.05% read as "flat"
_BETA_CLOSE_TO_ONE = 0.1

_EVENTS = {
    "1.01": "signs an important agreement",
    "1.02": "announces that an important agreement has been cancelled",
    "2.01": "completes buying or selling a business or major assets",
    "2.02": "reports its quarterly results",
    "2.03": "takes on significant new debt",
    "2.05": "announces job cuts or a restructuring",
    "2.06": "writes down the value of some of its assets",
    "3.01": "is warned it may be removed from its stock exchange",
    "3.02": "sells new shares privately",
    "4.02": "says its earlier financial statements can't be relied on",
    "5.02": "announces that an executive or director is joining or leaving",
    "5.07": "announces the results of a shareholder vote",
    "7.01": "publishes an investor update",
    "8.01": "announces other news",
}
_SURPRISE_EVENTS = {
    "beat": "reports quarterly profits clearly better than its recent trend",
    "miss": "reports quarterly profits clearly worse than its recent trend",
    "in line": "reports quarterly profits close to its recent trend",
}
_ROUGHLY = {5: "a week", 10: "two weeks", 20: "a month", 60: "three months"}


def plain_enabled(flag: bool, environ: Mapping[str, str]) -> bool:
    return flag or environ.get(PLAIN_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def wrap(text: str, indent: str = "  ") -> str:
    return textwrap.fill(text, _WIDTH, initial_indent=indent, subsequent_indent=indent)


def day(moment: date | datetime) -> str:
    """'Wed 18 Nov 2026'; datetimes are shown as their New York date."""
    if isinstance(moment, datetime):
        moment = moment.astimezone(EASTERN).date()
    return f"{moment:%a %d %b %Y}"


def _size(x: float) -> str:
    return f"{abs(x) * 100:.1f}%"


def _moved(x: float) -> str:
    if abs(x) < _FLAT:
        return "stayed about flat"
    return f"{'rose' if x > 0 else 'fell'} {_size(x)}"


def _period(horizon: int) -> str:
    if horizon == 1:
        return "one trading day"
    roughly = _ROUGHLY.get(horizon)
    return f"{horizon} trading days" + (f" (about {roughly})" if roughly else "")


def event(rule: Rule) -> str:
    if rule.surprise is not None:
        return _SURPRISE_EVENTS[rule.surprise]
    if rule.item in _EVENTS:
        return _EVENTS[rule.item]
    label = EIGHT_K_ITEMS.get(rule.item, f"item {rule.item}")
    return f"files a report about '{label}'"


def _action(rule: Rule) -> str:
    return "bet against" if rule.direction is Direction.SHORT else "buy"


def _acted(rule: Rule) -> str:
    return "bet against it" if rule.direction is Direction.SHORT else "bought"


def idea(rule: Rule) -> str:
    hope = "worse" if rule.direction is Direction.SHORT else "better"
    return (
        f"The idea: when a company {event(rule)}, {_action(rule)} its stock at the next "
        f"chance and hold for {_period(rule.horizon)}, betting that the stock will do "
        f"{hope} than the stock market as a whole."
    )


def _times_in_100(hit_rate: float) -> str:
    return f"{hit_rate * 100:.0f} times out of 100"


def expectation(backtest: Mapping[str, Any] | None) -> str:
    """What the rule's past record says to expect from now on."""
    if backtest is None:
        return "What to expect: there are no past cases to compare with."
    mean = float(backtest["mean"])
    text = (
        f"What to expect: in {backtest['n']} past cases this bet "
        f"{'gained' if mean >= 0 else 'lost'} {_size(mean)} per trade on average, and paid "
        f"off {_times_in_100(float(backtest['hit_rate']))}."
    )
    t = backtest["t_stat"]
    if t is None:
        return text + " That's too few cases to say whether it was more than luck."
    if abs(t) < 1:
        return text + " That's no better than flipping a coin, so expect roughly break-even."
    if abs(t) < _STRONG_T:
        return text + " That could easily be luck, so don't expect much."
    if mean < 0:
        return text + " It lost fairly consistently, so expect it to keep losing."
    return text + (
        " That's unlikely to be pure luck, but an idea picked because it looked good on "
        "past data usually does less well afterwards. Watching it from now on is the test."
    )


def trade(rule: Rule, row: Mapping[str, Any], expected: float | None) -> str:
    """One trade (open or finished) in a few plain sentences."""
    ticker = row["ticker"]
    held, status = row["sessions_held"], row["status"]
    abnormal = float(row["pnl"]) * rule.sign  # stock vs market, before picking a side
    pnl = float(row["pnl"])
    text = (
        f"{ticker}: news on {day(row['known_at'])}; {_acted(rule)} at the open on "
        f"{day(row['entry_session'])}. "
    )
    if status == Status.OPEN.value:
        text += f"After {held} of {rule.horizon} trading days (to {day(row['exit_session'])}), "
    elif status == Status.INTERRUPTED.value:
        text += (
            f"{ticker} stopped trading part-way, so the bet ended at its last price on "
            f"{day(row['exit_session'])}, after {held} of {rule.horizon} days: "
        )
    else:
        text += f"Ended at the close on {day(row['exit_session'])}: "
    text += f"{ticker} {_moved(float(row['stock_return']))} and the market "
    text += f"{_moved(float(row['bench_return']))}. "
    beta = float(row["beta"])
    if abs(beta - 1) >= _BETA_CLOSE_TO_ONE:
        text += f"{ticker} usually moves about {beta:.1f} times as much as the market, so "
        text += "allowing for that, "
    else:
        text += "So "
    text += f"it did {_size(abnormal)} {'better' if abnormal >= 0 else 'worse'} than the market. "
    verb = "is winning" if status == Status.OPEN.value else "won"
    if pnl < 0:
        verb = "is losing" if status == Status.OPEN.value else "lost"
    text += f"The bet {verb} {_size(pnl)}" + ("" if status != Status.OPEN.value else " so far")
    if expected is not None:
        text += f" (past trades like this averaged {expected * 100:+.1f}%)"
    return text + "."


def pending(rule: Rule, row: Mapping[str, Any]) -> str:
    known_at: datetime = row["known_at"]
    return (
        f"{row['ticker']}: news on {day(known_at)} at {known_at.astimezone(EASTERN):%H:%M} "
        f"New York time. The rule will {_action(rule)} it at the next market open; it shows "
        "up as a trade once that day's prices have been fetched."
    )


def scoreboard(
    started: datetime,
    live: Mapping[str, Any] | None,
    backtest: Mapping[str, Any] | None,
    in_progress: int,
) -> str:
    """How the rule has done since it started, compared with its past."""
    waiting = f" {in_progress} still in progress." if in_progress else ""
    since = f"Since we started watching on {day(started)}:"
    if live is None:
        return f"{since} no finished trades yet.{waiting}"
    n = int(live["n"])
    mean = float(live["mean"])
    wins = round(float(live["hit_rate"]) * n)
    text = (
        f"{since} {n} finished trade(s), {wins} won, {'gaining' if mean >= 0 else 'losing'} "
        f"{_size(mean)} per trade on average"
    )
    if backtest is not None:
        text += f" (past cases: {float(backtest['mean']) * 100:+.1f}%)"
    text += f".{waiting} "
    if n < MIN_LIVE_TRADES:
        return text + f"Too few to judge yet: it takes about {MIN_LIVE_TRADES} or more."
    if backtest is not None and (mean >= 0) != (float(backtest["mean"]) >= 0):
        return text + "So far it is not working the way it did in the past."
    return text + "So far it is holding up."
