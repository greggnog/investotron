"""Plain-English mode (``--explain``): define terms and spell out what results mean.

Two kinds of output, both silent unless explaining is on:

* :meth:`Narrator.terms` prints glossary entries, each at most once per run, right
  before the table that uses them.
* :meth:`Narrator.say` prints a sentence about *these* results, built by the
  ``*_takeaway`` functions below from the same numbers the tables show.

Turn it on per command with ``--explain``, or for every command by setting the
environment variable ``INVESTOTRON_EXPLAIN=1``.
"""

from __future__ import annotations

import textwrap
from collections.abc import Mapping
from typing import Any

import polars as pl

from investotron.paper import Direction, Rule

EXPLAIN_ENV = "INVESTOTRON_EXPLAIN"
_WIDTH = 92
_PREFIX = "   | "

GLOSSARY: dict[str, str] = {
    # Filings and data
    "8-K": "A report US public companies must file with the SEC within four business "
    "days of something important happening: earnings, a deal signed or cancelled, an "
    "executive leaving, and so on.",
    "item": "Each 8-K lists numbered 'items' saying what kind of news it contains, e.g. "
    "2.02 = quarterly results (earnings), 1.02 = an important agreement was terminated, "
    "5.02 = an executive or director joined or left. One filing can have several items; "
    "9.01 just means 'documents attached', so it rides along with almost everything.",
    "ticker": "The short code a stock trades under, e.g. AAPL for Apple.",
    "EPS": "Earnings per share: the company's profit for the quarter divided by its "
    "number of shares.",
    "session": "One trading day. US markets open at 9:30 and close at 16:00 New York "
    "time; 'days' in these tables always means trading days, not calendar days.",
    "overnight gap": "The jump between one day's closing price and the next day's "
    "opening price. News released after the close (most earnings are) shows up here, "
    "before anyone can trade on it during normal hours.",
    # Returns
    "benchmark": "What each stock is compared against. SPY is a fund that tracks the "
    "S&P 500, the 500 largest US companies, so it stands for 'the market'.",
    "beta": "How strongly a stock tends to move with the market. Beta 1.5 means it "
    "typically moves 1.5% when the market moves 1%. Here it's measured from the "
    "stock's previous year of prices.",
    "abnormal return": "How much better or worse the stock did than expected given what "
    "the market did over the same days (stock return minus beta x market return). It "
    "strips out 'the whole market went up' so only the stock's own move is left.",
    "mean %": "The average abnormal return across all the cases in the row.",
    "median %": "The middle case: half did better, half did worse. When it's very "
    "different from the mean, a few extreme cases are pulling the average.",
    "hit %": "The share of cases where the stock beat the market (abnormal return above "
    "zero). 50% is a coin flip.",
    "n": "How many cases (filings or trades) the row is based on. Small numbers "
    "(under ~30) give unreliable averages.",
    # Statistics
    "t": "A measure of how unlikely the average would be if there were really no "
    "effect. Roughly: below 1 means 'no sign of anything', 1-2 means 'weak, easily "
    "luck', above 2 means 'unlikely to be luck if this was the one thing you set out "
    "to test'. Negative t means the average is negative.",
    "t naive": "The same measure computed as if every case were independent. It's "
    "shown for comparison; the 't' column is the one to trust.",
    "weeks": "How many separate calendar weeks the cases fall in. Cases in the same week "
    "share whatever the market did that week, so they count as one 'cluster' when "
    "computing t. Many cases but few weeks means less real evidence than n suggests.",
    "multiple comparisons": "If you test many things, some will look significant by "
    "pure luck: about 1 row in 20 shows |t| above 2 even if nothing is going on (so "
    "about 2 of 45). A result only stands out if there are clearly more than that.",
    "z": "A move measured in units of the stock's normal daily wobble. z = 3 means the "
    "move was three times bigger than a typical day's. On ordinary days |z| averages "
    "about 0.8 and is above 2 only about 5% of the time.",
    "mean |z|": "The average size of the move in z units, ignoring direction. Compare "
    "with ~0.8 for a normal day.",
    "|z|>2 %": "The share of cases with a move more than twice the normal daily size. "
    "Compare with ~5% for normal days.",
    "mean |move| %": "The average size of the move in percent, ignoring whether it was up or down.",
    "mean move %": "The average move in percent, with direction: up and down moves "
    "cancel out, so this shows whether the news was good or bad on average.",
    # Earnings surprise
    "SUE": "Standardised unexpected earnings: how much this quarter's EPS beat or missed "
    "the same quarter a year earlier, compared with how much that usually varies for "
    "this company. A rough stand-in for 'did results beat expectations?'.",
    "miss / beat": "SUE of -1 or below counts as a miss, +1 or above as a beat, anything "
    "between as 'in line'.",
    # Paper trading
    "long": "Betting the stock does better than the market: you buy it and profit if "
    "it outperforms.",
    "short": "Betting the stock does worse than the market: you profit if it "
    "underperforms. (In real life this means borrowing shares and selling them.)",
    "horizon": "How many trading days each trade is held before it's closed.",
    "paper trading": "Recording the trades a rule would make, without real money, to see "
    "whether it works on new data.",
    "live": "Trades on filings that arrived after the rule was written down. These are "
    "the honest test, because nobody could have tuned the rule to them.",
    "backtest": "The same rule applied to past filings. Useful for comparison, but it "
    "flatters the rule: the rule was chosen because it looked good on this data.",
    "status": "pending = filed after the last market open, enters at the next open; "
    "open = holding period not over, valued at the latest close; closed = held for the "
    "full horizon; interrupted = the stock stopped trading part-way (e.g. acquired).",
    "P&L": "Profit and loss of the hypothetical trade, as the abnormal return: for a "
    "short, positive P&L means the stock lagged the market. No trading costs included.",
    "win %": "The share of finished trades with positive P&L.",
    "stopped trading": "A stock whose prices end before the market's latest day, usually "
    "because it was acquired or delisted, or because its data needs refreshing.",
}


class Narrator:
    """Prints explanations only when ``enabled``; otherwise every call is a no-op."""

    __slots__ = ("_seen", "enabled")

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._seen: set[str] = set()

    def say(self, text: str) -> None:
        if not self.enabled:
            return
        print(textwrap.fill(text, _WIDTH, initial_indent=_PREFIX, subsequent_indent=_PREFIX))

    def terms(self, *keys: str) -> None:
        """Define terms not already defined earlier in this run."""
        if not self.enabled:
            return
        new = [k for k in keys if k not in self._seen]
        if not new:
            return
        print(_PREFIX + "Terms:")
        for key in new:
            self._seen.add(key)
            print(
                textwrap.fill(
                    f"{key}: {GLOSSARY[key]}",
                    _WIDTH,
                    initial_indent=_PREFIX + "  - ",
                    subsequent_indent=_PREFIX + "    ",
                )
            )


def explain_enabled(flag: bool, environ: Mapping[str, str]) -> bool:
    return flag or environ.get(EXPLAIN_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


# --- sentences about specific results ------------------------------------------


def pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def t_meaning(t: float | None) -> str:
    if t is None:
        return "too few separate weeks to judge"
    size = abs(t)
    if size < 1:
        return "no sign this is more than noise"
    if size < 2:  # noqa: PLR2004
        return "a weak signal that could easily be luck"
    if size < 3:  # noqa: PLR2004
        return "unlikely to be luck if it were the only thing tested"
    return "strong evidence, unlikely to be luck"


def _better_or_worse(x: float) -> str:
    return "better" if x >= 0 else "worse"


def direction_takeaway(summary: pl.DataFrame, labels: Mapping[str, str]) -> list[str]:
    """Sentences about the direction table (columns from ``event_study.summarize``)."""
    sentences: list[str] = []
    overall = summary.filter((pl.col("group") == "all") & (pl.col("horizon") == 1))
    if not overall.is_empty():
        row = overall.row(0, named=True)
        sentences.append(
            f"Across all {row['n']} filings, buying at the first chance and selling at that "
            f"day's close did {pct(row['mean'])} vs the market on average, with "
            f"{row['hit_rate']:.0%} beating it: {t_meaning(row['t_stat'])}."
        )
    items = summary.filter((pl.col("group") != "all") & pl.col("t_stat").is_not_null())
    if items.is_empty():
        return sentences
    best = items.sort(pl.col("t_stat").abs(), descending=True).row(0, named=True)
    label = labels.get(best["group"], "")
    sentences.append(
        f"The strongest row: after item {best['group']}"
        f"{f' ({label})' if label else ''} filings, stocks did {abs(best['mean']) * 100:.2f}% "
        f"{_better_or_worse(best['mean'])} than the market over {best['horizon']} trading "
        f"days on average ({best['n']} cases, t = {best['t_stat']:.2f}): "
        f"{t_meaning(best['t_stat'])}."
    )
    return sentences


def multiple_comparisons_takeaway(hits: int, rows: int, expected: float) -> str:
    if hits <= expected * 1.5:
        verdict = (
            "That's about what luck alone produces, so treat individual 'significant' "
            "rows with suspicion."
        )
    else:
        verdict = "That's more than luck would explain, though not by itself proof of anything."
    return (
        f"{hits} of {rows} rows cleared |t| > 2, where about {expected:.0f} would by chance. "
        + verdict
    )


def size_takeaway(summary: pl.DataFrame, labels: Mapping[str, str]) -> list[str]:
    """Sentences about the reaction-size table (``summarize_reactions`` columns)."""
    items = summary.filter(pl.col("group") != "all")
    if items.is_empty():
        return []
    biggest = items.sort("mean_abs_z", descending=True).row(0, named=True)
    label = labels.get(biggest["group"], biggest["group"])
    sentences = [
        f"{label} filings move stocks the most: on average {biggest['mean_abs_z']:.1f} times "
        f"a normal day's move (about {biggest['mean_abs_reaction'] * 100:.1f}% up or down), "
        f"and {biggest['share_above_2']:.0%} of them were big moves (|z| > 2), versus about "
        "5% on an ordinary day."
    ]
    quiet = items.filter(pl.col("mean_abs_z") < 1.0)
    if not quiet.is_empty():
        names = ", ".join(labels.get(g, g) for g in quiet["group"].to_list())
        sentences.append(f"These barely move the price beyond a normal day: {names}.")
    return sentences


def surprise_takeaway(reactions: pl.DataFrame, returns: pl.DataFrame) -> list[str]:
    """Sentences comparing misses and beats (``group`` = surprise bucket)."""
    by_group = {r["group"]: r for r in reactions.iter_rows(named=True)}
    miss, beat = by_group.get("1 miss"), by_group.get("3 beat")
    if miss is None or beat is None:
        return []
    sentences = [
        f"Earnings surprises predict the immediate reaction: misses moved "
        f"{pct(miss['mean_reaction'])} and beats {pct(beat['mean_reaction'])} on average, "
        "mostly in the overnight gap before regular trading opens."
    ]
    later = returns.filter((pl.col("group") == "3 beat") & (pl.col("horizon") == pl.max("horizon")))
    if not later.is_empty():
        row = later.row(0, named=True)
        sentences.append(
            f"After the open, beats went on to do {pct(row['mean'])} over {row['horizon']} "
            f"days ({t_meaning(row['t_stat'])}), so by the time you could trade, the news "
            "was already in the price."
        )
    return sentences


_SURPRISE_WORDS = {
    "miss": " where earnings missed",
    "in line": " where earnings came in about as expected",
    "beat": " where earnings beat",
}


def rule_takeaway(rule: Rule, stats: Mapping[str, Mapping[str, Any]], open_count: int) -> list[str]:
    """Sentences about one paper-trading rule; ``stats`` maps sample -> stats row."""
    bet = "worse" if rule.direction is Direction.SHORT else "better"
    which = _SURPRISE_WORDS.get(rule.surprise or "", "")
    sentences = [
        f"This rule bets that after an item {rule.item} filing{which}, the stock will do "
        f"{bet} than the market over the next {rule.horizon} trading days."
    ]
    backtest, live = stats.get("backtest"), stats.get("live")
    if backtest is not None:
        sentences.append(
            f"On past filings it would have made {pct(backtest['mean'])} per trade on average "
            f"over {backtest['n']} trades, winning {backtest['hit_rate']:.0%} of the time "
            f"({t_meaning(backtest['t_stat'])})."
        )
    if live is None:
        waiting = f" {open_count} trade(s) are still open." if open_count else ""
        sentences.append(
            "No live trades have finished yet, so there's no honest verdict." + waiting
        )
    else:
        comparison = ""
        if backtest is not None:
            held_up = (live["mean"] > 0) == (backtest["mean"] > 0)
            comparison = (
                " So far that's in the same direction as the backtest."
                if held_up
                else " So far that's the opposite of the backtest."
            )
        sentences.append(
            f"Live, it has made {pct(live['mean'])} per trade over {live['n']} finished "
            f"trades ({t_meaning(live['t_stat'])}).{comparison}"
        )
        if live["n"] < 30:  # noqa: PLR2004
            sentences.append("With under 30 live trades, it's still too early to conclude much.")
    return sentences
