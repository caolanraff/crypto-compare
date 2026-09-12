# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A CLI tool that compares up to 5 cryptocurrencies (price history, exchange pairings, financials, ratings, community and developer stats) and publishes the result as a single PDF report. Data comes from the CoinGecko API (`pycoingecko`). Which coins to compare, the report title, and the output filename are set in an `.ini` config file (default: `config/default.ini`), so different configs can produce different reports.

## Commands

```bash
# install
brew install poetry
poetry install

# run (from repo root)
poetry run python -m src.cli.main --config config/default.ini

# test (pytest + pytest-mock; runs with coverage over src/report and src/utils)
poetry run pytest

# lint / format / type-check (each hook runs in its own isolated env — third-party
# runtime deps aren't installed there, so mypy only meaningfully checks first-party code)
poetry run pre-commit run --all-files
```

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs `pre-commit run --all-files` and `pytest` on every push/PR to `main`.

## Architecture

The codebase is layered so that data shaping (pure, unit-testable) is separate from network I/O and PDF file writing:

- **[src/cli/main.py](src/cli/main.py)** — entrypoint. Parses args, loads the config, and orchestrates one linear pipeline: download data → build each report section → merge into the final PDF. Each `get_*`/`plot_*`/`create_*` call returns the file path(s) it wrote, appended to a `saved_files` list that's handed to `pdf.merge_pdfs()` at the end (no module-level global state).
- **[src/utils/data.py](src/utils/data.py)** — `download_coingecko_data()`: the only network-calling function. Resolves each coin name to a CoinGecko id and returns `(coin_by_id, coin_market_chart_by_id)` dicts keyed by coin name.
- **[src/report/calcs.py](src/report/calcs.py)** — pure functions (`calculate_*_dataframe`, `calculate_summary_paragraphs`) that turn the raw CoinGecko dicts into pandas DataFrames/lists ready for rendering. No file or network I/O, so these are the most heavily unit-tested part of the codebase (see `tests/report/test_calcs.py`). Fields CoinGecko doesn't report for a coin, and ratios with a zero/missing denominator, are deliberately left as `None` rather than raising `KeyError`/`ZeroDivisionError`.
- **[src/report/report.py](src/report/report.py)** — one function per report page/section (`get_summary_info`, `plot_price_data`, `get_exchange_data`, `get_financial_data`, `get_ratings_data`, `get_community_data`, `get_developer_data`, `create_title_page`). Each calls the matching `calcs.py` function, then `utils/pdf.py` to render it, and returns the resulting file path(s).
- **[src/utils/pdf.py](src/utils/pdf.py)** — generic PDF-building layer with no CoinGecko-specific knowledge: table rendering (`df_to_pdf`, with optional cell highlighting and automatic pagination beyond `max_rows`), paragraph pages (`save_paragraphs_to_pdf`), and merging (`merge_pdfs`). Filenames are derived from each page's title via `convert_to_snake_case()` in [src/utils/util.py](src/utils/util.py).

Each `calculate_*_dataframe` function in `calcs.py` follows the same shape: pull a nested dict per coin out of `coin_by_id`, select a fixed set of keys, apply per-field formatting (rounding, `,`-separators, `%` suffixes via the `_safe_*` helpers), pivot into a `Metric` column, and merge across coins into one wide DataFrame.

## Rate limits

CoinGecko's free API allows 10-30 calls/minute. A single run makes 3 calls per coin, so running the tool repeatedly in quick succession can fail with rate-limit errors.
