# crypto-compare

![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Compare up to 5 cryptocurrencies side by side, published as a single PDF report.

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Configuration](#configuration)
- [Rate Limit](#rate-limit)
- [Samples](#samples)
- [License](#license)

## Overview

Data is pulled live from the [CoinGecko](https://www.coingecko.com/) API for each configured coin and assembled into a report covering:

- 90-day price changes and price correlation
- Exchange pairings
- Financial metrics (market cap, volume, ATH/ATL, supply, etc.)
- Community sentiment ratings
- Community engagement (social media, Reddit, Telegram)
- Developer activity (GitHub stats)

## Getting Started

### Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/)

### Installation

```bash
brew install poetry
poetry install
```

## Usage

### CLI reference

```bash
poetry run python -m src.cli.main --help
```

```
usage: main.py [-h] [--path PATH] [--config CONFIG]

options:
  -h, --help       show this help message and exit
  --path PATH      directory path
  --config CONFIG  config file
```

### Examples

```bash
poetry run python -m src.cli.main --config config/default.ini
```

## Configuration

The cryptocurrencies to compare, the report title, and the output filename are all set in a config file (default: `config/default.ini`). Create additional config files to produce different reports on an ad hoc basis.

| Section | Key | Description |
| --- | --- | --- |
| `Input` | `coins` | Comma-separated coin names to compare (max 5), e.g. `ethereum,cardano,solana` |
| `Text` | `title` | Subtitle shown on the report's title page |
| `Output` | `file` | Output PDF filename |

## Rate Limit

The CoinGecko free API has a rate limit of 10-30 calls per minute. This tool makes 3 calls per coin per run, so running it too many times in quick succession may cause requests to fail.

## Samples

Sample output file: [CryptoCompare.pdf](data/output/CryptoCompare.pdf)

## License

[MIT](LICENSE)
