"""
Functions required to generate the report.

Author: Caolan Rafferty
Date: 2023-08-12
"""

from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from matplotlib.backends.backend_pdf import PdfPages

from src.report.calcs import (
    calculate_community_dataframe,
    calculate_developer_dataframe,
    calculate_exchange_dataframe,
    calculate_financial_dataframe,
    calculate_price_dataframe,
    calculate_ratings_dataframe,
    calculate_summary_paragraphs,
)
from src.utils.pdf import df_to_pdf, save_paragraphs_to_pdf
from src.utils.types import CoinData

plt.style.use("seaborn-v0_8")

CHART_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def create_title_page(subtitle: str, output_dir: str) -> str:
    """
    Create a title page for the PDF report.

    Parameters:
    subtitle (str): The report subtitle, taken from the config's [Text] title.
    output_dir (str): The path to the output directory.

    Returns:
    str: The file path of the created PDF.
    """
    pdf_output = FPDF()
    pdf_output.add_page()
    pdf_output.set_font("Arial", "B", 36)
    pdf_output.cell(0, 80, "Crypto Compare", 0, 1, "C")
    pdf_output.set_font("Arial", "", 24)
    pdf_output.cell(0, 20, subtitle, 0, 1, "C")
    file = f"{output_dir}/title.pdf"
    pdf_output.output(file)
    return file


def get_summary_info(
    coins: List[str], coin_by_id: Dict[str, CoinData], output_dir: str
) -> str:
    """
    Build and save the summary page.

    Parameters:
    coins (List[str]): Coin names to include.
    coin_by_id (Dict[str, CoinData]): CoinGecko coin metadata by coin name.
    output_dir (str): The path to the output directory.

    Returns:
    str: The file path of the created PDF.
    """
    paragraphs = calculate_summary_paragraphs(coins, coin_by_id)
    return save_paragraphs_to_pdf("Summary", coins, paragraphs, output_dir)


def plot_price_data(
    coins: List[str], coin_market_chart_by_id: Dict[str, CoinData], output_dir: str
) -> List[str]:
    """
    Plot 90-day cumulative price returns and a price-correlation heatmap.

    Parameters:
    coins (List[str]): Coin names to include.
    coin_market_chart_by_id (Dict[str, CoinData]): CoinGecko market chart data by coin name.
    output_dir (str): The path to the output directory.

    Returns:
    List[str]: The file paths of the created PDFs (price chart, then heatmap).
    """
    price_data = calculate_price_dataframe(coins, coin_market_chart_by_id)
    saved_files = []

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_prop_cycle(color=CHART_PALETTE)
    for coin in coins:
        ax.plot(price_data["time"], price_data[coin], label=coin)
    plt.title("90 Day Price Changes", fontsize=12, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    fig.legend(loc="upper right")
    file = f"{output_dir}/price_chart.pdf"
    plt.savefig(file)
    plt.close(fig)
    saved_files.append(file)

    heat = price_data[coins].corr(method="pearson")
    sns_plot = sns.heatmap(
        heat, cmap="Blues", annot=True, fmt=".2f", annot_kws={"fontsize": 8}
    )
    sns_plot.figure.set_size_inches(10, 7)
    sns_plot.set_title("Price Correlation", fontsize=12, fontweight="bold")
    file = f"{output_dir}/price_heatmap.pdf"
    pp = PdfPages(file)
    pp.savefig(sns_plot.figure)
    pp.close()
    plt.close(sns_plot.figure)
    saved_files.append(file)

    return saved_files


def get_exchange_data(
    coins: List[str], coin_by_id: Dict[str, CoinData], output_dir: str
) -> List[str]:
    """
    Build and save the exchange-pairings page.

    Parameters:
    coins (List[str]): Coin names to include.
    coin_by_id (Dict[str, CoinData]): CoinGecko coin metadata by coin name.
    output_dir (str): The path to the output directory.

    Returns:
    List[str]: The file path(s) of the created PDF(s).
    """
    exchange_data = calculate_exchange_dataframe(coins, coin_by_id)
    return df_to_pdf("Exchange No. of Pairs", exchange_data, output_dir)


def get_financial_data(
    coins: List[str], coin_by_id: Dict[str, CoinData], output_dir: str
) -> List[str]:
    """
    Build and save the financial-data page.

    Parameters:
    coins (List[str]): Coin names to include.
    coin_by_id (Dict[str, CoinData]): CoinGecko coin metadata by coin name.
    output_dir (str): The path to the output directory.

    Returns:
    List[str]: The file path(s) of the created PDF(s).
    """
    financial_data = calculate_financial_dataframe(coins, coin_by_id)
    return df_to_pdf("Financial Data", financial_data, output_dir)


def get_ratings_data(
    coins: List[str], coin_by_id: Dict[str, CoinData], output_dir: str
) -> List[str]:
    """
    Build and save the ratings-data page.

    Parameters:
    coins (List[str]): Coin names to include.
    coin_by_id (Dict[str, CoinData]): CoinGecko coin metadata by coin name.
    output_dir (str): The path to the output directory.

    Returns:
    List[str]: The file path(s) of the created PDF(s).
    """
    ratings_data = calculate_ratings_dataframe(coins, coin_by_id)
    return df_to_pdf("Ratings Data", ratings_data, output_dir)


def get_community_data(
    coins: List[str], coin_by_id: Dict[str, CoinData], output_dir: str
) -> List[str]:
    """
    Build and save the community-data page.

    Parameters:
    coins (List[str]): Coin names to include.
    coin_by_id (Dict[str, CoinData]): CoinGecko coin metadata by coin name.
    output_dir (str): The path to the output directory.

    Returns:
    List[str]: The file path(s) of the created PDF(s).
    """
    community_data = calculate_community_dataframe(coins, coin_by_id)
    return df_to_pdf("Community Data", community_data, output_dir)


def get_developer_data(
    coins: List[str], coin_by_id: Dict[str, CoinData], output_dir: str
) -> List[str]:
    """
    Build and save the developer-data page.

    Parameters:
    coins (List[str]): Coin names to include.
    coin_by_id (Dict[str, CoinData]): CoinGecko coin metadata by coin name.
    output_dir (str): The path to the output directory.

    Returns:
    List[str]: The file path(s) of the created PDF(s).
    """
    developer_data = calculate_developer_dataframe(coins, coin_by_id)
    return df_to_pdf("Developer Data", developer_data, output_dir)
