"""
Calculations required for report generation.

Author: Caolan Rafferty
Date: 2023-08-12
"""

from typing import Any, Dict, List

import pandas as pd

from src.utils.types import CoinData, Frame

_FINANCIAL_USD_KEYS = [
    "current_price",
    "market_cap",
    "total_volume",
    "high_24h",
    "low_24h",
    "ath",
    "ath_change_percentage",
    "ath_date",
    "atl",
    "atl_change_percentage",
    "atl_date",
    "total_value_locked",
]
_FINANCIAL_KEYS = _FINANCIAL_USD_KEYS + [
    "price_change_percentage_24h",
    "price_change_percentage_7d",
    "price_change_percentage_30d",
    "price_change_percentage_60d",
    "price_change_percentage_1y",
    "total_supply",
    "max_supply",
    "circulating_supply",
    "mcap_to_tvl_ratio",
    "fdv_to_tvl_ratio",
]
_FINANCIAL_ROUND = [
    "ath",
    "ath_change_percentage",
    "atl",
    "atl_change_percentage",
    "price_change_percentage_24h",
    "price_change_percentage_7d",
    "price_change_percentage_30d",
    "price_change_percentage_60d",
    "price_change_percentage_1y",
    "total_supply",
    "circulating_supply",
    "nvt_ratio",
    "fully_diluted_value",
    "fdv_to_market_cap",
]
_FINANCIAL_SEPARATED = [
    "current_price",
    "market_cap",
    "total_volume",
    "high_24h",
    "low_24h",
    "ath",
    "atl",
    "total_supply",
    "max_supply",
    "circulating_supply",
    "fully_diluted_value",
]
_FINANCIAL_PERCENT = [
    "ath_change_percentage",
    "atl_change_percentage",
    "price_change_percentage_24h",
    "price_change_percentage_7d",
    "price_change_percentage_30d",
    "price_change_percentage_60d",
    "price_change_percentage_1y",
]

_RATINGS_KEYS = ["sentiment_votes_up_percentage", "sentiment_votes_down_percentage"]

_COMMUNITY_KEYS = [
    "facebook_likes",
    "twitter_followers",
    "reddit_average_posts_48h",
    "reddit_average_comments_48h",
    "reddit_subscribers",
    "reddit_accounts_active_48h",
    "telegram_channel_user_count",
]
_COMMUNITY_ROUND = [
    "reddit_average_posts_48h",
    "reddit_average_comments_48h",
    "reddit_subscribers",
    "reddit_accounts_active_48h",
]

_DEVELOPER_KEYS = [
    "forks",
    "stars",
    "subscribers",
    "total_issues",
    "closed_issues",
    "pull_requests_merged",
    "pull_request_contributors",
    "commit_count_4_weeks",
]


def _safe_round(value: Any, ndigits: int) -> Any:
    """Round a numeric value, leaving a missing (None) value unchanged."""
    return round(value, ndigits) if isinstance(value, (int, float)) else value


def _safe_percent(value: Any) -> Any:
    """Append a '%' suffix, leaving a missing (None) value unchanged."""
    return f"{value}%" if value is not None else None


def _safe_separated(value: Any) -> Any:
    """Format a number with thousands separators, leaving a missing (None) value unchanged."""
    return format(value, ",") if value is not None else None


def _tidy_metric_labels(df: Frame) -> Frame:
    """Turn snake_case metric keys into title-cased row labels, and title-case the columns."""
    df["Metric"] = df["Metric"].str.replace("_", " ")
    df["Metric"] = df["Metric"].str.replace("percentage", "%")
    df["Metric"] = df["Metric"].str.title()
    return df.rename(columns=lambda x: x.title())


def calculate_price_dataframe(
    coins: List[str], coin_market_chart_by_id: Dict[str, CoinData]
) -> Frame:
    """
    Build a wide DataFrame of cumulative 90-day price returns for each coin, indexed by time.

    Parameters:
    coins (List[str]): Coin names to include.
    coin_market_chart_by_id (Dict[str, CoinData]): CoinGecko market chart data by coin name.

    Returns:
    Frame: DataFrame with a "time" column and one cumulative-return column per coin.
    """
    price_data = pd.DataFrame(columns=["time"])

    for coin in coins:
        df = pd.DataFrame(
            coin_market_chart_by_id[coin]["prices"], columns=["time", coin]
        )
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df["time"] = df["time"].round("h")
        df[coin] = df[coin].pct_change().cumsum()
        price_data = pd.merge(price_data, df, on="time", how="outer")

    return price_data.sort_values(by=["time"]).reset_index(drop=True)


def calculate_exchange_dataframe(
    coins: List[str], coin_by_id: Dict[str, CoinData]
) -> Frame:
    """
    Build a table of the top 10 exchanges by combined number of trading pairs across the given coins.

    Parameters:
    coins (List[str]): Coin names to include.
    coin_by_id (Dict[str, CoinData]): CoinGecko coin metadata by coin name.

    Returns:
    Frame: DataFrame with one row per exchange and one column per coin.
    """
    exchange_data = pd.DataFrame(columns=["Exchange"])

    for coin in coins:
        tickers = pd.DataFrame(coin_by_id[coin]["tickers"])
        exchanges = tickers["market"].apply(lambda market: market["name"])
        counts = exchanges.value_counts().rename_axis("Exchange").reset_index(name=coin)
        exchange_data = pd.merge(exchange_data, counts, on="Exchange", how="outer")

    exchange_data["total"] = exchange_data.iloc[:, 1:].sum(axis=1)
    exchange_data = exchange_data.sort_values(by=["total"], ascending=False)
    exchange_data = exchange_data.drop(columns=["total"]).head(10)
    exchange_data["Exchange"] = exchange_data["Exchange"].str.replace(
        " Exchange", "", regex=False
    )
    return exchange_data.rename(columns=lambda x: x.title())


def calculate_financial_dataframe(
    coins: List[str], coin_by_id: Dict[str, CoinData]
) -> Frame:
    """
    Build a table of financial metrics for each coin.

    Fields CoinGecko doesn't report for a given coin (e.g. `total_value_locked` for non-DeFi
    coins) are left blank rather than raising, and ratios with a zero or missing denominator
    (e.g. NVT ratio when `total_volume` is 0) are left blank instead of raising a
    ZeroDivisionError.

    Parameters:
    coins (List[str]): Coin names to include.
    coin_by_id (Dict[str, CoinData]): CoinGecko coin metadata by coin name.

    Returns:
    Frame: DataFrame with one row per metric and one column per coin.
    """
    financial_data = pd.DataFrame(columns=["Metric"])

    for coin in coins:
        market_data = dict(coin_by_id[coin]["market_data"])
        for key in _FINANCIAL_USD_KEYS:
            value = market_data.get(key)
            market_data[key] = value["usd"] if isinstance(value, dict) else value
        dic: Dict[str, Any] = {key: market_data.get(key) for key in _FINANCIAL_KEYS}

        dic["nvt_ratio"] = (
            dic["market_cap"] / dic["total_volume"]
            if dic["market_cap"] is not None and dic["total_volume"]
            else None
        )
        dic["fully_diluted_value"] = (
            dic["current_price"] * dic["total_supply"]
            if dic["current_price"] is not None and dic["total_supply"] is not None
            else None
        )
        dic["fdv_to_market_cap"] = (
            dic["fully_diluted_value"] / dic["market_cap"]
            if dic["fully_diluted_value"] is not None and dic["market_cap"]
            else None
        )

        for key in ("ath_date", "atl_date"):
            dic[key] = dic[key][:10] if dic[key] else None
        for key in _FINANCIAL_ROUND:
            dic[key] = _safe_round(dic[key], 2)
        for key in _FINANCIAL_SEPARATED:
            dic[key] = _safe_separated(dic[key])
        for key in _FINANCIAL_PERCENT:
            dic[key] = _safe_percent(dic[key])

        df = pd.DataFrame(list(dic.items()), columns=["Metric", coin])
        financial_data = pd.merge(financial_data, df, on="Metric", how="outer")

    return _tidy_metric_labels(financial_data)


def calculate_ratings_dataframe(
    coins: List[str], coin_by_id: Dict[str, CoinData]
) -> Frame:
    """
    Build a table of community sentiment ratings for each coin.

    Parameters:
    coins (List[str]): Coin names to include.
    coin_by_id (Dict[str, CoinData]): CoinGecko coin metadata by coin name.

    Returns:
    Frame: DataFrame with one row per metric and one column per coin.
    """
    ratings_data = pd.DataFrame(columns=["Metric"])

    for coin in coins:
        data = coin_by_id[coin]
        dic = {
            key: _safe_percent(_safe_round(data.get(key), 2)) for key in _RATINGS_KEYS
        }
        df = pd.DataFrame(list(dic.items()), columns=["Metric", coin])
        ratings_data = pd.merge(ratings_data, df, on="Metric", how="outer")

    return _tidy_metric_labels(ratings_data)


def calculate_community_dataframe(
    coins: List[str], coin_by_id: Dict[str, CoinData]
) -> Frame:
    """
    Build a table of community engagement metrics for each coin.

    Parameters:
    coins (List[str]): Coin names to include.
    coin_by_id (Dict[str, CoinData]): CoinGecko coin metadata by coin name.

    Returns:
    Frame: DataFrame with one row per metric and one column per coin.
    """
    community_data = pd.DataFrame(columns=["Metric"])

    for coin in coins:
        data = coin_by_id[coin]["community_data"]
        dic: Dict[str, Any] = {key: data.get(key) for key in _COMMUNITY_KEYS}
        for key in _COMMUNITY_ROUND:
            dic[key] = _safe_round(dic[key], 2)
        for key in _COMMUNITY_KEYS:
            dic[key] = _safe_separated(dic[key])
        df = pd.DataFrame(list(dic.items()), columns=["Metric", coin])
        community_data = pd.merge(community_data, df, on="Metric", how="outer")

    community_data["Metric"] = community_data["Metric"].str.replace("_", " ")
    community_data["Metric"] = community_data["Metric"].str.replace("average", "avg")
    community_data["Metric"] = community_data["Metric"].str.replace("channel", "")
    community_data["Metric"] = community_data["Metric"].str.title()
    return community_data.rename(columns=lambda x: x.title())


def calculate_developer_dataframe(
    coins: List[str], coin_by_id: Dict[str, CoinData]
) -> Frame:
    """
    Build a table of developer activity metrics for each coin.

    A coin with no recorded GitHub issues would otherwise raise a ZeroDivisionError when
    computing the closed-issues percentage; that metric is left blank instead.

    Parameters:
    coins (List[str]): Coin names to include.
    coin_by_id (Dict[str, CoinData]): CoinGecko coin metadata by coin name.

    Returns:
    Frame: DataFrame with one row per metric and one column per coin.
    """
    developer_data = pd.DataFrame(columns=["Metric"])

    for coin in coins:
        data = coin_by_id[coin]["developer_data"]
        dic: Dict[str, Any] = {key: data.get(key) for key in _DEVELOPER_KEYS}
        dic["closed_issues_%"] = (
            _safe_percent(
                _safe_round(100 * dic["closed_issues"] / dic["total_issues"], 2)
            )
            if dic["closed_issues"] is not None and dic["total_issues"]
            else None
        )
        for key in _DEVELOPER_KEYS:
            dic[key] = _safe_separated(dic[key])
        df = pd.DataFrame(list(dic.items()), columns=["Metric", coin])
        developer_data = pd.merge(developer_data, df, on="Metric", how="outer")

    developer_data["Metric"] = developer_data["Metric"].str.replace("_", " ")
    developer_data["Metric"] = developer_data["Metric"].str.title()
    return developer_data.rename(columns=lambda x: x.title())


def calculate_summary_paragraphs(
    coins: List[str], coin_by_id: Dict[str, CoinData]
) -> List[str]:
    """
    Build the summary paragraph text (description + homepage link) for each coin.

    Parameters:
    coins (List[str]): Coin names to include.
    coin_by_id (Dict[str, CoinData]): CoinGecko coin metadata by coin name.

    Returns:
    List[str]: One paragraph of text per coin, in the same order as `coins`.
    """
    paragraphs = []

    for coin in coins:
        data = coin_by_id[coin]
        description = data["description"]["en"].split("\r")[0]
        homepages = [url for url in data.get("links", {}).get("homepage", []) if url]
        link = f" Link: {homepages[0]}" if homepages else ""
        paragraphs.append(description + link)

    return paragraphs
