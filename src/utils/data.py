"""
Functions for retrieving cryptocurrency data from CoinGecko.

Author: Caolan Rafferty
Date: 2023-08-12
"""

import logging
import sys
from typing import Dict, List, Tuple

import pandas as pd
from pycoingecko import CoinGeckoAPI

from src.utils.types import CoinData

cg = CoinGeckoAPI()


def download_coingecko_data(
    coins: List[str],
) -> Tuple[Dict[str, CoinData], Dict[str, CoinData]]:
    """
    Download coin metadata and 90-day market chart history for each coin from CoinGecko.

    Parameters:
    coins (List[str]): Coin names as they appear in the config, e.g. "ethereum".

    Returns:
    Tuple[Dict[str, CoinData], Dict[str, CoinData]]: Coin metadata and market chart data,
    each keyed by coin name.
    """
    coin_list = pd.DataFrame(cg.get_coins_list())
    coin_by_id: Dict[str, CoinData] = {}
    coin_market_chart_by_id: Dict[str, CoinData] = {}

    for name in coins:
        matches = coin_list.loc[coin_list["name"].str.lower().isin([name])]
        if matches.empty:
            logging.fatal(f"{name} is not available")
            sys.exit(1)
        try:
            coin_id = matches["id"].values[0]
            coin_by_id[name] = cg.get_coin_by_id(coin_id)
            coin_market_chart_by_id[name] = cg.get_coin_market_chart_by_id(
                id=coin_id, vs_currency="usd", days="90"
            )
        except Exception as e:
            logging.fatal(f"Unable to get data for {name}: {e}")
            sys.exit(1)

    return coin_by_id, coin_market_chart_by_id
