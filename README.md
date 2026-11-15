# Investotron

A research tool for event studies: when something public happens to a company
(an SEC filing, a news spike, ...), what do its stock returns do afterwards,
compared with the market?

It tests whether a signal *ever worked* historically. It is not a trading system
and doesn't tell you what to buy — paper-trade anything before trusting it.

## Setup

Needs [uv](https://docs.astral.sh/uv/) (`pip install uv`, or `winget install astral-sh.uv`).

```bash
uv sync
cp .env.example .env   # then put your own name/email in it
```

SEC asks automated clients to identify themselves, so set
`INVESTOTRON_SEC_USER_AGENT` (in your shell, or load `.env`) before fetching.

## Use

Every command takes tickers, a `--universe` file (one ticker per line), or both.
Fetches skip tickers already downloaded unless you pass `--refresh`, so an
interrupted run (e.g. a Tiingo rate limit) can just be rerun.

```bash
uv run investotron fetch-edgar --universe universes/starter.txt
uv run investotron fetch-prices SPY --universe universes/starter.txt
uv run investotron event-study --universe universes/starter.txt
```

- **fetch-edgar** (alias `fetch-8k`): 8-K filings and quarterly EPS from SEC XBRL
  data, to `data/clean/edgar_8k/` and `data/clean/eps_quarterly/`. Raw payloads
  are kept untouched under `data/raw/`.
- **fetch-prices**: daily bars from Tiingo (needs `INVESTOTRON_TIINGO_API_KEY`), from
  2015-01-01 unless `--start` says otherwise. Bars carry `opens_at`/`closes_at` in UTC
  so they compare directly with a filing's `known_at`.
- **event-study**: three tables, pooled over all tickers:
  1. *Direction*: abnormal return from the first open after each 8-K, by item code.
  2. *Size*: move from the last close before the filing to the entry-day close, as a
     z-score against the stock's normal volatility.
  3. *Surprise*: earnings 8-Ks split by SUE (year-on-year EPS change vs its usual
     spread). See `fundamentals.py` for why this stands in for analyst consensus.

  Per-event rows go to `data/results/` for digging into.

## Checks

```bash
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run pytest
```

## Design

**Point-in-time data** (`point_in_time.py`). Every fact has `occurred_at` (when it
happened) and `known_at` (when the market could have known). A
`PointInTimeStore` can only be read through `as_of(moment)`, so code can't
accidentally see the future. This is the main defence against look-ahead bias.

**Layers.** `sources/` does I/O and returns raw payloads; parsing is a pure function
from raw payload to `Fact`s; `storage.py` writes `raw/` (exactly as downloaded) and
`clean/` (typed tables, always rebuildable from raw).

**Value types** (`identifiers.py`). `Ticker` and `Cik` are validated, frozen types
rather than bare `str`/`int`, so they can't be swapped by mistake.

**Strict tooling.** pyright in strict mode; ruff with `DTZ` on, which flags naive
datetimes.

## Roadmap

1. ~~Fetch 8-K filings with acceptance timestamps~~
2. ~~Verify EDGAR `acceptanceDateTime` timezone~~ (it's real UTC)
3. ~~Daily prices for the stock and a market benchmark (e.g. SPY)~~
4. ~~Event study: average 1/5/20-day abnormal returns after an 8-K, split by item code~~
   (`uv run investotron event-study AAPL`)
5. ~~Many tickers, reaction size, earnings surprise (SUE)~~
6. Fitted beta instead of assuming beta = 1
7. Standard errors that account for overlapping windows / same-day events
8. Survivorship: include delisted companies
