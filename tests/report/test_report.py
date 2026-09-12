from typing import Any, Dict

import pandas as pd

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


def test_create_title_page(mocker: Any) -> None:
    mock_fpdf = mocker.patch("src.report.report.FPDF")
    pdf_output = mock_fpdf.return_value

    file_path = create_title_page("Ethereum Killers", "/tmp")

    assert file_path == "/tmp/title.pdf"
    pdf_output.output.assert_called_once_with("/tmp/title.pdf")


def test_get_summary_info(mocker: Any) -> None:
    mock_calc = mocker.patch(
        "src.report.report.calculate_summary_paragraphs",
        return_value=["paragraph 1", "paragraph 2"],
    )
    mock_save = mocker.patch(
        "src.report.report.save_paragraphs_to_pdf", return_value="/tmp/summary.pdf"
    )

    coin_by_id: Dict[str, Any] = {"bitcoin": {}, "ethereum": {}}
    file_path = get_summary_info(["bitcoin", "ethereum"], coin_by_id, "/tmp")

    assert file_path == "/tmp/summary.pdf"
    mock_calc.assert_called_once_with(["bitcoin", "ethereum"], coin_by_id)
    mock_save.assert_called_once_with(
        "Summary", ["bitcoin", "ethereum"], ["paragraph 1", "paragraph 2"], "/tmp"
    )


def test_plot_price_data(mocker: Any) -> None:
    price_data = pd.DataFrame(
        {
            "time": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "bitcoin": [0.0, 0.1],
            "ethereum": [0.0, 0.05],
        }
    )
    mocker.patch("src.report.report.calculate_price_dataframe", return_value=price_data)
    mocker.patch("src.report.report.plt.savefig")
    mocker.patch("src.report.report.plt.close")
    mock_pdf_pages = mocker.patch("src.report.report.PdfPages")

    result = plot_price_data(["bitcoin", "ethereum"], {}, "/tmp")

    assert result == ["/tmp/price_chart.pdf", "/tmp/price_heatmap.pdf"]
    mock_pdf_pages.assert_called_once_with("/tmp/price_heatmap.pdf")


def test_get_exchange_data(mocker: Any) -> None:
    df = pd.DataFrame({"Exchange": ["Binance"], "Bitcoin": [10]})
    mock_calc = mocker.patch(
        "src.report.report.calculate_exchange_dataframe", return_value=df
    )
    mock_pdf = mocker.patch(
        "src.report.report.df_to_pdf", return_value=["/tmp/exchange_no_of_pairs_1.pdf"]
    )

    coin_by_id: Dict[str, Any] = {"bitcoin": {}}
    result = get_exchange_data(["bitcoin"], coin_by_id, "/tmp")

    assert result == ["/tmp/exchange_no_of_pairs_1.pdf"]
    mock_calc.assert_called_once_with(["bitcoin"], coin_by_id)
    mock_pdf.assert_called_once_with("Exchange No. of Pairs", df, "/tmp")


def test_get_financial_data(mocker: Any) -> None:
    df = pd.DataFrame({"Metric": ["Current Price"], "Bitcoin": ["100"]})
    mocker.patch("src.report.report.calculate_financial_dataframe", return_value=df)
    mock_pdf = mocker.patch(
        "src.report.report.df_to_pdf", return_value=["/tmp/financial_data_1.pdf"]
    )

    result = get_financial_data(["bitcoin"], {"bitcoin": {}}, "/tmp")

    assert result == ["/tmp/financial_data_1.pdf"]
    mock_pdf.assert_called_once_with("Financial Data", df, "/tmp")


def test_get_ratings_data(mocker: Any) -> None:
    df = pd.DataFrame({"Metric": ["Sentiment Votes Up %"], "Bitcoin": ["80%"]})
    mocker.patch("src.report.report.calculate_ratings_dataframe", return_value=df)
    mock_pdf = mocker.patch(
        "src.report.report.df_to_pdf", return_value=["/tmp/ratings_data_1.pdf"]
    )

    result = get_ratings_data(["bitcoin"], {"bitcoin": {}}, "/tmp")

    assert result == ["/tmp/ratings_data_1.pdf"]
    mock_pdf.assert_called_once_with("Ratings Data", df, "/tmp")


def test_get_community_data(mocker: Any) -> None:
    df = pd.DataFrame({"Metric": ["Twitter Followers"], "Bitcoin": ["1,000"]})
    mocker.patch("src.report.report.calculate_community_dataframe", return_value=df)
    mock_pdf = mocker.patch(
        "src.report.report.df_to_pdf", return_value=["/tmp/community_data_1.pdf"]
    )

    result = get_community_data(["bitcoin"], {"bitcoin": {}}, "/tmp")

    assert result == ["/tmp/community_data_1.pdf"]
    mock_pdf.assert_called_once_with("Community Data", df, "/tmp")


def test_get_developer_data(mocker: Any) -> None:
    df = pd.DataFrame({"Metric": ["Forks"], "Bitcoin": ["30,000"]})
    mocker.patch("src.report.report.calculate_developer_dataframe", return_value=df)
    mock_pdf = mocker.patch(
        "src.report.report.df_to_pdf", return_value=["/tmp/developer_data_1.pdf"]
    )

    result = get_developer_data(["bitcoin"], {"bitcoin": {}}, "/tmp")

    assert result == ["/tmp/developer_data_1.pdf"]
    mock_pdf.assert_called_once_with("Developer Data", df, "/tmp")
