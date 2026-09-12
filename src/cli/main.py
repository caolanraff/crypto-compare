"""
Crypto Comparison Tool.

This script allows the comparison of up to 5 different cryptocurrencies, published in a PDF report.

Author: Caolan Rafferty
Date: 2023-08-12
"""

import argparse
import configparser
import logging
import os
import sys
import warnings
from typing import Any

from src.report.report import (
    create_title_page,
    get_community_data,
    get_developer_data,
    get_exchange_data,
    get_financial_data,
    get_ratings_data,
    get_summary_info,
    plot_price_data,
)
from src.utils.data import download_coingecko_data
from src.utils.pdf import merge_pdfs

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def parse_arguments() -> Any:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="./", type=str, help="directory path")
    parser.add_argument(
        "--config", default="config/default.ini", type=str, help="config file"
    )
    return parser.parse_args()


def main(args: Any, config: configparser.ConfigParser) -> None:
    """Generate the report."""
    coins = config.get("Input", "coins").split(",")
    if len(coins) > 5:
        logging.warning("Maximum 5 coins at a time")
        sys.exit(1)

    output_dir = f"{args.path}/data/output"
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Running report for: {coins}")
    logging.info("Downloading coingecko data")
    coin_by_id, coin_market_chart_by_id = download_coingecko_data(coins)

    saved_files = []
    logging.info("Creating title page")
    saved_files.append(create_title_page(config.get("Text", "title"), output_dir))
    logging.info("Getting summary information")
    saved_files.append(get_summary_info(coins, coin_by_id, output_dir))
    logging.info("Plotting price data")
    saved_files.extend(plot_price_data(coins, coin_market_chart_by_id, output_dir))
    logging.info("Getting exchange data")
    saved_files.extend(get_exchange_data(coins, coin_by_id, output_dir))
    logging.info("Getting financial data")
    saved_files.extend(get_financial_data(coins, coin_by_id, output_dir))
    logging.info("Getting ratings data")
    saved_files.extend(get_ratings_data(coins, coin_by_id, output_dir))
    logging.info("Getting community data")
    saved_files.extend(get_community_data(coins, coin_by_id, output_dir))
    logging.info("Getting developer data")
    saved_files.extend(get_developer_data(coins, coin_by_id, output_dir))

    merge_pdfs(saved_files, f"{output_dir}/{config.get('Output', 'file')}")
    logging.info("Complete")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    args = parse_arguments()
    config = configparser.ConfigParser()
    config.read(f"{args.path}/{args.config}")
    main(args, config)
