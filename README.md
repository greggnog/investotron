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

**New to the market jargon?** Add `--explain` to any command for plain-English
definitions of the terms and a sentence or two on what your numbers mean. To leave it
on while you learn, set the environment variable instead, and remove it when you no
longer need it:

```powershell
$env:INVESTOTRON_EXPLAIN = "1"        # this terminal only
Remove-Item Env:INVESTOTRON_EXPLAIN   # back to the compact output
```

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

### Paper trading

```bash
uv run investotron fetch-edgar --universe universes/starter.txt --refresh
uv run investotron fetch-prices SPY --universe universes/starter.txt --refresh
uv run investotron paper
```

Rules live in [`paper/rules.toml`](paper/rules.toml): "go long/short at the first open
after an 8-K with item X, hold N sessions", optionally limited to earnings misses or
beats. The report shows, per rule: open positions marked at the latest close, filings
waiting for the next open, recent finished trades, and live vs backtest statistics.

What keeps it honest:

- The first run after a rule is added writes its fingerprint and start time to
  `paper/rules.lock.json`. Only filings accepted after that count as **live**; the
  same rule on earlier filings is the **backtest**, shown for comparison. Commit the
  lock file so the start dates are on record.
- Editing a rule that has started is refused. Copy it under a new name instead, which
  starts a fresh record.
- Trades are recomputed from the data on every run, so there's no ledger to drift.
  Stocks whose prices stop early (acquired, delisted) are flagged, and trades cut
  short by a break in prices are marked `interrupted` at the last tradable close.

P&L is the beta-hedged abnormal return, signed by direction, with no costs, borrow
fees or slippage. It's a test of whether a pattern holds up on data nobody had seen,
not a record of tradable profits.

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

**Price-data guards** (`event_study.aligned_prices`). Zero-volume sessions are
dropped, because vendors carry the last price forward through halts and after
delistings, and those prices were never tradable. A gap of more than 5 days between
sessions splits the history into segments: the return across the gap feeds neither
beta nor volatility, and no return window or reaction may span it.

**Strict tooling.** pyright in strict mode; ruff with `DTZ` on, which flags naive
datetimes.

## Known limitation: survivorship bias

`universes/starter.txt` lists companies that exist today. Companies that went
bankrupt, were acquired or were delisted since 2015 are missing, and they are not a
random sample: they include most of the worst outcomes. Results computed on today's
survivors look better than reality.

How much that matters depends on the question:

- **Returns over long horizons, or of a portfolio** (would buying X have made money?):
  biased upwards. Don't trust these numbers from this universe.
- **Event reactions measured against each stock's own history** (reaction size,
  surprise direction, abnormal returns over days): much less affected, because each
  event is compared with the same stock's normal behaviour just before it. Still, the
  sample excludes the kinds of firms that later fail, so treat results as "for
  companies that survived".

Fixing it properly needs a **point-in-time universe**: membership decided by a rule as
of each date (e.g. every US common stock above some size), including the companies
that later disappeared. That requires data this project doesn't have for free:

- **Prices for delisted stocks.** Tested 2026-11-15: Tiingo has some (TWTR, clean to
  its 2022-10-28 delisting) but not all (FRC returns 404). Where it has them they need
  cleaning: SIVB shows 197 zero-volume sessions repeating $106.04 after SVB's failure,
  then a 456-day gap, then OTC trades around $0.006. The guards above handle the
  frozen prices and gaps; the loss at delisting itself (the move nobody could trade
  at) still has to be estimated, as research databases do with a "delisting return".
- **Ticker-to-CIK mapping for dead companies.** SEC's `company_tickers.json` only
  lists current tickers, so a delisted ticker can't be matched to its filings without
  name matching or a paid mapping.
- Paid sources with point-in-time coverage include Norgate, EODHD and CRSP.

## Roadmap

1. ~~Fetch 8-K filings with acceptance timestamps~~
2. ~~Verify EDGAR `acceptanceDateTime` timezone~~ (it's real UTC)
3. ~~Daily prices for the stock and a market benchmark (e.g. SPY)~~
4. ~~Event study: average 1/5/20-day abnormal returns after an 8-K, split by item code~~
   (`uv run investotron event-study AAPL`)
5. ~~Many tickers, reaction size, earnings surprise (SUE)~~
6. ~~Fitted beta instead of assuming beta = 1~~ (`--model beta`, the default; `--model market` for beta = 1)
7. ~~Standard errors that account for overlapping windows / same-day events~~
   (clustered by calendar week; neighbouring weeks are still assumed independent)
8. ~~Survivorship~~: price-data guards added; a full point-in-time universe needs
   paid data -- see "Known limitation: survivorship bias" above
