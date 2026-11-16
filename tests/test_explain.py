import re
from pathlib import Path

import polars as pl
import pytest

from investotron.explain import (
    EXPLAIN_ENV,
    GLOSSARY,
    Narrator,
    direction_takeaway,
    explain_enabled,
    multiple_comparisons_takeaway,
    rule_takeaway,
    size_takeaway,
    t_meaning,
)
from investotron.paper import Direction, Rule

SRC = Path(__file__).parents[1] / "src" / "investotron"


def test_disabled_narrator_prints_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    narrator = Narrator(enabled=False)
    narrator.say("hello")
    narrator.terms("8-K")
    assert capsys.readouterr().out == ""


def test_terms_are_defined_once_per_run(capsys: pytest.CaptureFixture[str]) -> None:
    narrator = Narrator(enabled=True)
    narrator.terms("8-K", "beta")
    narrator.terms("beta")
    out = capsys.readouterr().out
    assert out.count("beta:") == 1
    assert "8-K:" in out


def test_long_text_is_wrapped(capsys: pytest.CaptureFixture[str]) -> None:
    Narrator(enabled=True).say("word " * 60)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) > 1
    assert all(len(line) <= 92 for line in lines)


@pytest.mark.parametrize(
    ("flag", "env", "expected"),
    [
        (True, {}, True),
        (False, {}, False),
        (False, {EXPLAIN_ENV: "1"}, True),
        (False, {EXPLAIN_ENV: "yes"}, True),
        (False, {EXPLAIN_ENV: "0"}, False),
    ],
)
def test_explain_enabled(flag: bool, env: dict[str, str], expected: bool) -> None:
    assert explain_enabled(flag, env) is expected


def test_every_term_used_in_output_has_a_glossary_entry() -> None:
    used: set[str] = set()
    for path in (SRC / "cli.py", SRC / "paper_report.py"):
        for call in re.findall(r"narrator\.terms\((.*?)\)", path.read_text(encoding="utf-8"), re.S):
            used.update(re.findall(r'"([^"]+)"', call))
    used.update(d.value for d in Direction)  # paper report passes rule.direction.value
    assert used, "found no narrator.terms calls"
    assert sorted(used - set(GLOSSARY)) == []


@pytest.mark.parametrize(
    ("t", "words"),
    [
        (None, "too few"),
        (0.4, "no sign"),
        (-1.5, "weak"),
        (2.3, "only thing tested"),
        (4, "strong"),
    ],
)
def test_t_meaning(t: float | None, words: str) -> None:
    assert words in t_meaning(t)


def test_direction_takeaway_describes_overall_and_strongest_row() -> None:
    summary = pl.DataFrame(
        {
            "group": ["all", "1.02", "5.07"],
            "horizon": [1, 20, 20],
            "n": [3459, 48, 330],
            "mean": [0.0007, -0.0487, 0.0148],
            "hit_rate": [0.49, 0.33, 0.53],
            "t_stat": [0.64, -2.29, 1.84],
        }
    )
    overall, strongest = direction_takeaway(summary, {"1.02": "Agreement terminated"})
    assert "3459 filings" in overall
    assert "1.02 (Agreement terminated)" in strongest
    assert "4.87% worse" in strongest
    assert "48 cases" in strongest


def test_multiple_comparisons_takeaway_judges_against_chance() -> None:
    assert "about what luck alone produces" in multiple_comparisons_takeaway(1, 45, 2.1)
    assert "more than luck" in multiple_comparisons_takeaway(9, 45, 2.1)


def test_size_takeaway_names_biggest_and_quiet_items() -> None:
    summary = pl.DataFrame(
        {
            "group": ["all", "2.02", "5.07"],
            "mean_abs_z": [2.0, 3.56, 0.73],
            "mean_abs_reaction": [0.05, 0.0863, 0.022],
            "share_above_2": [0.31, 0.62, 0.05],
        }
    )
    biggest, quiet = size_takeaway(summary, {"2.02": "Earnings", "5.07": "Shareholder vote"})
    assert biggest.startswith("Earnings filings move stocks the most")
    assert "3.6 times" in biggest
    assert "Shareholder vote" in quiet


SHORT = Rule("short-102", "1.02", Direction.SHORT, horizon=20)
BACKTEST = {"n": 48, "mean": 0.0487, "hit_rate": 0.67, "t_stat": 2.29}


def test_rule_takeaway_before_any_live_trades() -> None:
    sentences = rule_takeaway(SHORT, {"backtest": BACKTEST}, open_count=2)
    assert "do worse than the market over the next 20 trading days" in sentences[0]
    assert "+4.87% per trade" in sentences[1]
    assert "No live trades have finished yet" in sentences[2]
    assert "2 trade(s) are still open" in sentences[2]


def test_rule_takeaway_compares_live_with_backtest() -> None:
    live = {"n": 5, "mean": -0.01, "hit_rate": 0.4, "t_stat": -0.5}
    sentences = rule_takeaway(SHORT, {"backtest": BACKTEST, "live": live}, open_count=0)
    assert "opposite of the backtest" in sentences[2]
    assert "too early" in sentences[3]


def test_rule_takeaway_mentions_surprise_filter() -> None:
    beat = Rule("long-beat", "2.02", Direction.LONG, horizon=5, surprise="beat")
    assert "filing where earnings beat, the stock will do better" in rule_takeaway(beat, {}, 0)[0]
