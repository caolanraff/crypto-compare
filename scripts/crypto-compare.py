"""
Crypto Compare Tool

This script..

Author: Caolan Rafferty
Date: 2023-08-12
"""

from pycoingecko import CoinGeckoAPI
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pdfrw
import logging
import os
import configparser
import argparse
from fpdf import FPDF

parser = argparse.ArgumentParser()
parser.add_argument("--path", default="./", type=str, help="directory path")
parser.add_argument(
    "--config", default="config/default.ini", type=str, help="config file"
)
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

config = configparser.ConfigParser()
config.read(args.path + "/" + args.config)

plt.style.use('seaborn')
palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

files = []
output_dir = args.path + "/data/output/"

cg = CoinGeckoAPI()
# TODO: get from config
coins = ['ethereum', 'cardano', 'solana']
coin_by_id = {}
coin_market_chart_by_id = {}


def save_dataframe_to_pdf(
        title,
        df,
        file,
        highlight_columns=None,
        thresholds=None,
        operators=None,
        highlight_colour=None,
):
    """
    Save a DataFrame as a PDF file with optional highlighting of cells based on specified conditions.

    Args:
        title (str): Title of the PDF document.
        df (pandas.DataFrame): The DataFrame to be saved as a PDF.
        file (str): The path and filename of the PDF file to be created.
        highlight_columns (list, optional): List of column names to be highlighted. Defaults to None.
        thresholds (list, optional): List of threshold values for highlighting. Defaults to None.
        operators (list, optional): List of comparison operators ('>' or '<') for highlighting. Defaults to None.
        highlight_colour (str, optional): The colour for highlighting the cells. Defaults to None.
    Returns:
        None
    """
    max_rows = 14
    if len(df) > max_rows:
        dfs = np.array_split(df, np.ceil(len(df) / max_rows))
        for i, sub_df in enumerate(dfs):
            new_file = f"{file[:-4]}_{i}.pdf"
            save_dataframe_to_pdf(
                title,
                sub_df,
                new_file,
                highlight_columns,
                thresholds,
                operators,
                highlight_colour,
            )
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold", ha="left")
        else:
            cell.set_text_props(ha="left")
            if highlight_columns and thresholds and operators and highlight_colour:
                for i, col_name in enumerate(highlight_columns):
                    try:
                        col_index = df.columns.get_loc(col_name)
                    except KeyError:
                        raise ValueError(f"Column '{col_name}' not found in dataframe")
                    if col == col_index:
                        cell_value = float(cell.get_text().get_text())
                        if operators[i] == ">" and cell_value > thresholds[i]:
                            cell.set_facecolor(highlight_colour)
                        elif operators[i] == "<" and cell_value < thresholds[i]:
                            cell.set_facecolor(highlight_colour)

    ax.set_title(title, fontsize=12, fontweight="bold", y=0.9)
    pp = PdfPages(output_dir + file)
    pp.savefig(fig, bbox_inches="tight")
    pp.close()
    files.append(output_dir + file)


def merge_pdfs(input_files, output_file):
    """
    Merge multiple PDF files into a single PDF file.

    Args:
        input_files (list): A list of input file paths (strings) representing the PDF files to be merged.
        output_file (str): The output file path (string) where the merged PDF file will be saved.
    Returns:
        None
    """
    logging.info("Merging files")
    pdf_output = pdfrw.PdfWriter()
    for file_name in input_files:
        pdf_input = pdfrw.PdfReader(file_name)
        for page in pdf_input.pages:
            pdf_output.addpage(page)
        os.remove(file_name)
    pdf_output.write(output_file)


def download_coingecko_data():
    """
    Downloading coingecko data
    """
    logging.info("Downloading coingecko data")
    for i in coins:
        coin_by_id[i] = cg.get_coin_by_id(i)
        coin_market_chart_by_id[i] = cg.get_coin_market_chart_by_id(id=i,vs_currency='usd',days='90')


def create_title_page():
    """
    Create a title page for a PDF document with specified information.
    """
    logging.info("Creating title page")
    pdf_output = FPDF()
    pdf_output.add_page()
    title = "Crypto Compare"
    subtitle = "Ethereum Killers"
    pdf_output.set_font("Arial", "B", 36)
    pdf_output.cell(0, 80, title, 0, 1, "C")
    pdf_output.set_font("Arial", "", 24)
    pdf_output.cell(0, 20, subtitle, 0, 1, "C")
    files.append(output_dir + "title.pdf")
    pdf_output.output(files[-1])


def get_summary_info():
    """
    Getting summary information.
    """
    logging.info("Getting summary information")
    pdf_output = FPDF()
    pdf_output.add_page()
    pdf_output.set_font("Arial", "", 24)
    pdf_output.cell(0, 20, 'Summary', 0, 1, "C")

    for i in coins:
        pdf_output.set_font("Arial", "", 16)
        pdf_output.cell(0, 20, i, 0, 1, "C")
        data = coin_by_id[i]
        description = data['description']['en']
        description = description.split('\r')[0]
        pdf_output.cell(0, 20, description, 0, 1, "C")
        link = 'Link: '+ data['links']['homepage'][0]
        pdf_output.cell(0, 20, link, 0, 1, "C")

    files.append(output_dir + "summary.pdf")
    pdf_output.output(files[-1])

def get_summary_info():
    """
    Getting summary information.
    """
    logging.info("Getting summary information")
    summary_data = pd.DataFrame(columns=["Coin", "Description", "Link"])

    for i in coins:
        df = coin_by_id[i]
        description = df['description']['en']
        description = ""
        link = df['links']['homepage'][0]
        df = pd.DataFrame({"Coin": [i], "Description": [description], "Link": [link]})
        summary_data = pd.concat([summary_data, df], ignore_index=True)

    save_dataframe_to_pdf("Summary", summary_data, "summary.pdf")


def plot_price_data():
    """
    Plot price data
    """
    logging.info("Plotting price data")
    price_data = pd.DataFrame(columns=["time"])

    for i in coins:
        df = coin_market_chart_by_id[i]
        df = pd.DataFrame(df["prices"], columns=["time", i])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df["time"] = df['time'].round("H")
        price_data = pd.merge(price_data, df, on="time", how="outer")

    price_data = price_data.sort_values(by=["time"])
    scaler = MinMaxScaler()
    price_data[coins] = scaler.fit_transform(price_data[coins])

    fig, ax = plt.subplots(figsize=(20,10))
    ax.set_prop_cycle(color=palette)

    for i in coins:
        ax.plot(price_data["time"], price_data[i], label=i)

    plt.title("90 Day Price Changes", fontsize=12, fontweight="bold")
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.set_yscale('linear')
    fig.legend(loc="upper right")
    files.append(output_dir + "price_chart.pdf")
    plt.savefig(files[-1])

    plt.clf()
    heat = price_data.corr(method='pearson')
    sns_plot = sns.heatmap(heat, cmap="Blues", annot=True, fmt=".2f", annot_kws={"fontsize":8})
    sns_plot.figure.set_size_inches(10, 7)
    sns_plot.set_title("Price Correlation", fontsize=12, fontweight="bold")
    files.append(output_dir + "price_heatmap.pdf")
    pp = PdfPages(files[-1])
    pp.savefig(sns_plot.figure)
    pp.close()


def plot_volume_data():
    """
    Plot volume data
    """
    logging.info("Plotting volume data")
    volume_data = pd.DataFrame(columns=['time'])

    for i in coins:
        df = coin_market_chart_by_id[i]
        df = pd.DataFrame(df['total_volumes'], columns=['time', i])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df['time'] = df['time'].round('H')
        volume_data = pd.merge(volume_data, df, on='time', how='outer')

    volume_data = volume_data.sort_values(by=['time'])
    scaler = MinMaxScaler()
    volume_data[coins] = scaler.fit_transform(volume_data[coins])

    fig, ax = plt.subplots(figsize=(20,10))
    ax.set_prop_cycle(color=palette)

    for i in coins:
        ax.plot(volume_data['time'], volume_data[i], label=i)

    plt.title("90 Day Volume Changes", fontsize=12, fontweight="bold")
    ax.set_xlabel('Time')
    ax.set_ylabel('Volume')
    ax.set_yscale('log')
    fig.legend(loc="upper right")
    files.append(output_dir + "volume_chart.pdf")
    plt.savefig(files[-1])


def tm_sorter(column):
    """Sort function"""
    teams = ['Binance', 'Binance US', 'Bitstamp','Bitfinex','Bittrex','Coinbase Exchange','Crypto.com Exchange','FTX',
             'Gate.io', 'Gemini', 'Huobi Global', 'Huobi Korea', 'Kraken', 'KuCoin', 'OKEx', 'Poloniex', 'Uniswap (v3)',
             'Upbit']
    correspondence = {team: order for order, team in enumerate(teams)}
    return column.map(correspondence)


def get_exchange_data():
    """
    Get exchange data
    """
    logging.info("Getting exchange data")
    exchange_data = pd.DataFrame(columns=['Exchange'])

    for i in coins:
        df = coin_by_id[i]
        df = pd.DataFrame(df['tickers'])
        lst = []
        for j in range(df['market'].count()):
            lst.append(df['market'][j]['name'])

        df['Exchange'] = lst
        df = df.groupby(['Exchange']).size().to_frame(i)
        exchange_data = pd.merge(exchange_data, df, on='Exchange', how='outer')

    exchange_data = exchange_data.sort_values(by='Exchange', key=tm_sorter)
    exchange_data.reset_index(drop=True, inplace=True)
    exchange_data = exchange_data.head(20)
    save_dataframe_to_pdf("Exchange No. of Pairs", exchange_data, "exchange.pdf")


def get_financial_data():
    """
    Getting financial data.
    """
    logging.info("Getting financial data")

    k1 = ['current_price','market_cap','total_volume','high_24h','low_24h','ath','ath_change_percentage','ath_date','atl','atl_change_percentage','atl_date','total_value_locked']
    k2 = k1 + ['price_change_24h','price_change_percentage_24h','price_change_percentage_7d','price_change_percentage_14d','price_change_percentage_30d','price_change_percentage_60d','price_change_percentage_200d','price_change_percentage_1y','total_supply','max_supply','circulating_supply','mcap_to_tvl_ratio','fdv_to_tvl_ratio']
    add_round = ['ath_change_percentage', 'atl', 'atl_change_percentage', 'price_change_24h','price_change_percentage_24h','price_change_percentage_7d','price_change_percentage_14d','price_change_percentage_30d','price_change_percentage_60d','price_change_percentage_200d','price_change_percentage_1y', 'circulating_supply', 'fully_diluted_value']
    add_seperator = ['market_cap','total_volume','high_24h','low_24h','ath','atl','price_change_24h','total_supply','max_supply','circulating_supply','fully_diluted_value']
    add_percent = ['ath_change_percentage', 'atl_change_percentage', 'price_change_percentage_24h', 'price_change_percentage_7d', 'price_change_percentage_14d', 'price_change_percentage_30d', 'price_change_percentage_60d', 'price_change_percentage_200d', 'price_change_percentage_1y']

    financial_data = pd.DataFrame(columns=['metric'])

    for i in coins:
        df = coin_by_id[i]['market_data']
        for j in k1:
            r = df[j]
            if isinstance(r, dict):
                df[j] = r['usd']
            else:
                df[j] = r
        dic = {key: df[key] for key in k2}
        dic['nvt_ratio'] = dic['market_cap'] / dic['total_volume']
        dic['fully_diluted_value'] = dic['current_price'] * dic['total_supply'] if dic['total_supply'] is not None else 0
        dic['fdv_to_market_cap'] = dic['fully_diluted_value'] / dic['market_cap']
        for c in add_round:
            dic[c] = round(dic[c], 4)
        for c in add_seperator:
            try:
                dic[c] = format(dic[c],",")
            except:
                pass
        for c in add_percent:
            dic[c] = str(dic[c])+'%'
        df = pd.DataFrame(list(dic.items()), columns=['metric', i])
        financial_data = pd.merge(financial_data, df, on='metric', how='outer')

    save_dataframe_to_pdf("Financial Data", financial_data, "financial.pdf")


def get_ratings_data():
    """
    Get ratings data.
    """
    logging.info("Getting title page")

    keys = ['sentiment_votes_up_percentage','sentiment_votes_down_percentage','coingecko_rank','coingecko_score','developer_score','community_score','liquidity_score','public_interest_score']
    add_round = keys
    add_percent = ['sentiment_votes_up_percentage', 'sentiment_votes_down_percentage']

    ratings_data = pd.DataFrame(columns=['metric'])

    for i in coins:
        df = coin_by_id[i]
        dic = {key: df[key] for key in keys}
        for c in add_round:
            dic[c] = round(dic[c], 2)
        for c in add_percent:
            dic[c] = str(dic[c])+'%'
        df = pd.DataFrame(list(dic.items()), columns=['metric', i])
        ratings_data = pd.merge(ratings_data, df, on='metric', how='outer')

    save_dataframe_to_pdf("Ratings Data", ratings_data, "ratings.pdf")


def get_community_data():
    """
    Get community data.
    """
    logging.info("Getting community page")

    keys = ['facebook_likes','twitter_followers','reddit_average_posts_48h','reddit_average_comments_48h','reddit_subscribers','reddit_accounts_active_48h','telegram_channel_user_count']
    add_round = ['reddit_average_posts_48h','reddit_average_comments_48h','reddit_subscribers','reddit_accounts_active_48h']
    add_seperator = keys

    community_data = pd.DataFrame(columns=['metric'])

    for i in coins:
        df = coin_by_id[i]['community_data']
        dic = {key: df[key] for key in keys}
        for c in add_round:
            dic[c] = round(dic[c], 2)
        for c in add_seperator:
            try:
                dic[c] = format(dic[c],",")
            except:
                pass
        df = pd.DataFrame(list(dic.items()), columns=['metric', i])
        community_data = pd.merge(community_data, df, on='metric', how='outer')

    save_dataframe_to_pdf("Community Data", community_data, "community.pdf")


def get_developer_data():
    """
    Get Developer data.
    """
    logging.info("Getting developer page")

    keys = ['forks','stars','subscribers','total_issues','closed_issues','pull_requests_merged','pull_request_contributors','commit_count_4_weeks']
    add_round = ['closed_issues_percent']
    add_seperator = keys

    developer_data = pd.DataFrame(columns=['metric'])

    for i in coins:
        df = coin_by_id[i]['developer_data']
        dic = {key: df[key] for key in keys}
        dic['closed_issues_percent'] = 100 * dic['closed_issues'] / dic['total_issues']
        for c in add_round:
            dic[c] = round(dic[c], 2)
        for c in add_seperator:
            try:
                dic[c] = format(dic[c],",")
            except:
                pass
        df = pd.DataFrame(list(dic.items()), columns=['metric', i])
        developer_data = pd.merge(developer_data, df, on='metric', how='outer')

    save_dataframe_to_pdf("Developer Data", developer_data, "developer.pdf")


def main():
    download_coingecko_data()
    create_title_page()
    get_summary_info()
    plot_price_data()
    plot_volume_data()
    get_exchange_data()
    get_financial_data()
    get_ratings_data()
    get_community_data()
    get_developer_data()
    merge_pdfs(files, output_dir + 'CryptoCompare.pdf')


if __name__ == "__main__":
    main()

