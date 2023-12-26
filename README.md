# crypto-compare
The Crypto Compare tool allows the comparison of up to 5 different cryptocurrencies, published in a PDF report.
It compares several categories including:
- price changes
- exchange pairings
- financials
- ratings
- community data
- developer data

The cryptocurrencies to be compared can be defined in the config/default.ini file, along with the report title and output file name.
Multiple configs can be created to generate various reports.

### Samples
Sample output file: [CryptoCompare.pdf](data/output/CryptoCompare.pdf)

### Getting started
Clone the project and install poetry:
```
$ brew install poetry
$ poetry install
```

### Usage
```
$ poetry run python src/main.py --help
usage: main.py [-h] [--config CONFIG]

optional arguments:
  -h, --help       show this help message and exit
  --config CONFIG  config file
```

### Examples
```
poetry run python src/main.py --config config/default.ini
```

### Rate Limit
Note that the Coingecko API has a rate limit of 10-30 calls per minute on the free version.
The script makes 3 calls per execution, and therefore may fail if run too many times in a row.
