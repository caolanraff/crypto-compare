from typing import Any, Dict, List

import pandas as pd

from src.report.calcs import (
    calculate_community_dataframe,
    calculate_developer_dataframe,
    calculate_exchange_dataframe,
    calculate_financial_dataframe,
    calculate_price_dataframe,
    calculate_ratings_dataframe,
    calculate_summary_paragraphs,
)


def _market_chart(prices: List[List[float]]) -> Dict[str, Any]:
    return {"prices": prices}


def test_calculate_price_dataframe() -> None:
    chart = {
        "bitcoin": _market_chart([[0, 100.0], [3_600_000, 110.0], [7_200_000, 121.0]]),
        "ethereum": _market_chart([[0, 10.0], [3_600_000, 12.0], [7_200_000, 9.0]]),
    }

    result = calculate_price_dataframe(["bitcoin", "ethereum"], chart)

    assert list(result.columns) == ["time", "bitcoin", "ethereum"]
    assert len(result) == 3
    # first row has no prior price to compute a pct_change from
    assert pd.isna(result["bitcoin"].iloc[0])
    assert round(result["bitcoin"].iloc[1], 2) == 0.1
    assert round(result["bitcoin"].iloc[2], 2) == 0.2


def test_calculate_exchange_dataframe() -> None:
    coin_by_id = {
        "bitcoin": {
            "tickers": [
                {"market": {"name": "Binance"}},
                {"market": {"name": "Binance"}},
                {"market": {"name": "Coinbase Exchange"}},
            ]
        }
    }

    result = calculate_exchange_dataframe(["bitcoin"], coin_by_id)

    assert list(result["Exchange"]) == ["Binance", "Coinbase"]
    assert list(result["Bitcoin"]) == [2, 1]


def test_calculate_financial_dataframe_handles_missing_and_zero_fields() -> None:
    coin_by_id = {
        "bitcoin": {
            "market_data": {
                "current_price": {"usd": 100.0},
                "market_cap": {"usd": 0},
                "total_volume": {"usd": 0},
                "high_24h": {"usd": 110.0},
                "low_24h": {"usd": 90.0},
                "ath": {"usd": 200.0},
                "ath_change_percentage": {"usd": -50.0},
                "ath_date": {"usd": "2021-11-10T00:00:00.000Z"},
                "atl": {"usd": 1.0},
                "atl_change_percentage": {"usd": 9900.0},
                "atl_date": {"usd": "2013-07-06T00:00:00.000Z"},
                "total_value_locked": None,
                "price_change_percentage_24h": 1.0,
                "price_change_percentage_7d": 2.0,
                "price_change_percentage_30d": 3.0,
                "price_change_percentage_60d": 4.0,
                "price_change_percentage_1y": 5.0,
                "total_supply": None,
                "max_supply": 21_000_000,
                "circulating_supply": 19_000_000,
                "mcap_to_tvl_ratio": None,
                "fdv_to_tvl_ratio": None,
            }
        }
    }

    result = calculate_financial_dataframe(["bitcoin"], coin_by_id)
    values = dict(zip(result["Metric"], result["Bitcoin"]))

    # missing total_supply -> fully diluted value (and its ratio) are left blank, not 0
    assert values["Fully Diluted Value"] is None
    assert values["Fdv To Market Cap"] is None
    # zero total_volume -> NVT ratio left blank instead of raising ZeroDivisionError
    assert values["Nvt Ratio"] is None
    assert values["Ath Date"] == "2021-11-10"
    assert values["Current Price"] == "100.0"


def test_calculate_ratings_dataframe() -> None:
    coin_by_id = {
        "bitcoin": {
            "sentiment_votes_up_percentage": 80.456,
            "sentiment_votes_down_percentage": 19.544,
        }
    }

    result = calculate_ratings_dataframe(["bitcoin"], coin_by_id)
    values = dict(zip(result["Metric"], result["Bitcoin"]))

    assert values["Sentiment Votes Up %"] == "80.46%"


def test_calculate_community_dataframe_handles_missing_fields() -> None:
    coin_by_id = {
        "bitcoin": {
            "community_data": {
                "facebook_likes": None,
                "twitter_followers": 1_000_000,
                "reddit_average_posts_48h": 1.234,
                "reddit_average_comments_48h": 5.678,
                "reddit_subscribers": 5_000_000,
                "reddit_accounts_active_48h": 10_000,
                "telegram_channel_user_count": None,
            }
        }
    }

    result = calculate_community_dataframe(["bitcoin"], coin_by_id)
    values = dict(zip(result["Metric"], result["Bitcoin"]))

    assert values["Facebook Likes"] is None
    assert values["Twitter Followers"] == "1,000,000"


def test_calculate_developer_dataframe_handles_zero_issues() -> None:
    coin_by_id = {
        "bitcoin": {
            "developer_data": {
                "forks": 30_000,
                "stars": 70_000,
                "subscribers": 3_000,
                "total_issues": 0,
                "closed_issues": 0,
                "pull_requests_merged": 10_000,
                "pull_request_contributors": 800,
                "commit_count_4_weeks": 50,
            }
        }
    }

    result = calculate_developer_dataframe(["bitcoin"], coin_by_id)
    values = dict(zip(result["Metric"], result["Bitcoin"]))

    # total_issues == 0 -> percentage left blank instead of raising ZeroDivisionError
    assert values["Closed Issues %"] is None


def test_calculate_summary_paragraphs() -> None:
    coin_by_id = {
        "bitcoin": {
            "description": {"en": "Digital gold.\rMore detail here."},
            "links": {"homepage": ["https://bitcoin.org", ""]},
        },
        "ethereum": {
            "description": {"en": "Smart contract platform."},
            "links": {"homepage": []},
        },
    }

    result = calculate_summary_paragraphs(["bitcoin", "ethereum"], coin_by_id)

    assert result[0] == "Digital gold. Link: https://bitcoin.org"
    assert result[1] == "Smart contract platform."
