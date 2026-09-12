from typing import Any

import pytest

from src.utils.data import download_coingecko_data


def test_download_coingecko_data(mocker: Any) -> None:
    mocker.patch(
        "src.utils.data.cg.get_coins_list",
        return_value=[
            {"id": "bitcoin", "name": "Bitcoin"},
            {"id": "ethereum", "name": "Ethereum"},
        ],
    )
    mocker.patch(
        "src.utils.data.cg.get_coin_by_id",
        side_effect=lambda coin_id: {"id": coin_id},
    )
    mocker.patch(
        "src.utils.data.cg.get_coin_market_chart_by_id",
        side_effect=lambda id, vs_currency, days: {"prices": [], "for": id},
    )

    coin_by_id, coin_market_chart_by_id = download_coingecko_data(["bitcoin"])

    assert coin_by_id == {"bitcoin": {"id": "bitcoin"}}
    assert coin_market_chart_by_id == {"bitcoin": {"prices": [], "for": "bitcoin"}}


def test_download_coingecko_data_unknown_coin(mocker: Any) -> None:
    mocker.patch(
        "src.utils.data.cg.get_coins_list",
        return_value=[{"id": "bitcoin", "name": "Bitcoin"}],
    )

    with pytest.raises(SystemExit) as exc_info:
        download_coingecko_data(["dogecoin"])

    assert exc_info.value.code == 1


def test_download_coingecko_data_api_error(mocker: Any) -> None:
    mocker.patch(
        "src.utils.data.cg.get_coins_list",
        return_value=[{"id": "bitcoin", "name": "Bitcoin"}],
    )
    mocker.patch("src.utils.data.cg.get_coin_by_id", side_effect=RuntimeError("boom"))

    with pytest.raises(SystemExit) as exc_info:
        download_coingecko_data(["bitcoin"])

    assert exc_info.value.code == 1
