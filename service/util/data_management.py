import csv
import json
import os

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date
from typing import Tuple, List

from ..impl.portfolio import Portfolio
from ..impl.stats_models import StatsModels
from ..impl.sector import Sector
from ..impl.user import User
from ..impl.config import aws_settings
from . import helpers, console_handler, plot_functions, settings
import boto3


######################################################################################
# update all tables
def update_all_tables(stocksSymbols, numOfYearsHistory):  # build DB for withdraw
    today = date.today()
    formatted_date = today.strftime("%Y-%m-%d")
    update_closing_prices_tables(formatted_date, stocksSymbols, numOfYearsHistory)
    update_data_frame_tables(formatted_date, stocksSymbols)
    # TODO : upload to external DB


def update_closing_prices_tables(formatted_date_today, stocksSymbols, numOfYearsHistory):
    with open(settings.BUCKET_REPOSITORY + "lastUpdatedClosingPrice.txt", "r") as file:
        lastUpdatedDateClosingPrices = file.read().strip()
    if lastUpdatedDateClosingPrices != formatted_date_today:
        helpers.convert_data_to_tables(settings.BUCKET_REPOSITORY, settings.CLOSING_PRICES_FILE_NAME,
                                        stocksSymbols,
                                        numOfYearsHistory, saveToCsv=True)

        with open(settings.BUCKET_REPOSITORY + "lastUpdatedClosingPrice.txt", "w") as file:
            file.write(formatted_date_today)


def update_data_frame_tables(formatted_date_today, stocksSymbols):
    with open(settings.BUCKET_REPOSITORY + "lastUpdatedDftables.txt", "r") as file:
        lastUpdatedDftables = file.read().strip()
    if lastUpdatedDftables != formatted_date_today:
        sectorsList = helpers.set_sectors(stocksSymbols, mode='regular')
        closingPricesTable = get_closing_prices_table(mode='regular')
        pct_change_table = closingPricesTable.pct_change()
        # without maching learning
        # Markowitz
        update_three_level_data_frame_tables(machingLearningOpt=0, modelName="Markowitz",
                                             stocksSymbols=stocksSymbols, sectorsList=sectorsList,
                                             closingPricesTable=closingPricesTable, pct_change_table=pct_change_table)
        # Gini
        update_three_level_data_frame_tables(machingLearningOpt=0, modelName="Gini",
                                             stocksSymbols=stocksSymbols, sectorsList=sectorsList,
                                             closingPricesTable=closingPricesTable, pct_change_table=pct_change_table)

        # Including maching learning
        pct_change_table, annual_return, excepted_returns = helpers.update_daily_change_with_machine_learning(
            pct_change_table, closingPricesTable.index)
        # Markowitz
        update_three_level_data_frame_tables(machingLearningOpt=1, modelName="Markowitz",
                                             stocksSymbols=stocksSymbols, sectorsList=sectorsList,
                                             closingPricesTable=closingPricesTable, pct_change_table=pct_change_table)
        # Gini
        update_three_level_data_frame_tables(machingLearningOpt=1, modelName="Gini",
                                             stocksSymbols=stocksSymbols, sectorsList=sectorsList,
                                             closingPricesTable=closingPricesTable, pct_change_table=pct_change_table)

        with open(settings.BUCKET_REPOSITORY + "lastUpdatedDftables.txt", "w") as file:
            file.write(formatted_date_today)


def update_three_level_data_frame_tables(machingLearningOpt, modelName, stocksSymbols, sectorsList,
                                         closingPricesTable, pct_change_table):
    # high risk
    update_specific_data_frame_table(is_machine_learning=machingLearningOpt, model_name=modelName,
                                     stocks_symbols=stocksSymbols,
                                     sectors=sectorsList, levelOfRisk="high", max_percent_commodity=1,
                                     max_percent_stocks=1, closing_prices_table=closingPricesTable,
                                     pct_change_table=pct_change_table)
    # medium risk
    update_specific_data_frame_table(is_machine_learning=machingLearningOpt, model_name=modelName,
                                     stocks_symbols=stocksSymbols,
                                     sectors=sectorsList, levelOfRisk="medium",
                                     max_percent_commodity=settings.LIMIT_PERCENT_MEDIUM_RISK_COMMODITY,
                                     max_percent_stocks=settings.LIMIT_PERCENT_MEDIUM_RISK_STOCKS,
                                     closing_prices_table=closingPricesTable, pct_change_table=pct_change_table)
    # low risk
    update_specific_data_frame_table(is_machine_learning=machingLearningOpt, model_name=modelName,
                                     stocks_symbols=stocksSymbols,
                                     sectors=sectorsList, levelOfRisk="low",
                                     max_percent_commodity=settings.LIMIT_PERCENT_LOW_RISK_COMMODITY,
                                     max_percent_stocks=settings.LIMIT_PERCENT_LOW_RISK_STOCKS,
                                     closing_prices_table=closingPricesTable, pct_change_table=pct_change_table)


def update_specific_data_frame_table(is_machine_learning, model_name, stocks_symbols, sectors, levelOfRisk,
                                     max_percent_commodity, max_percent_stocks, closing_prices_table, pct_change_table):
    if is_machine_learning:
        locationForSaving = settings.MACHINE_LEARNING_LOCATION
    else:
        locationForSaving = settings.NON_MACHINE_LEARNING_LOCATION

    if max_percent_commodity <= 0:
        stock_sectors = helpers.setStockSectors(stocks_symbols, sectors)
        filtered_stocks = []
        for i in range(len(stock_sectors)):
            if stock_sectors[i] != "US commodity":
                filtered_stocks.append(stocks_symbols[i])
            else:
                closing_prices_table = closing_prices_table.drop(stocks_symbols[i], axis=1)
                pct_change_table = pct_change_table.drop(stocks_symbols[i], axis=1)
        stocks_symbols = filtered_stocks

    stats_models = StatsModels(
        stocks_symbols=stocks_symbols,
        sectors=sectors,
        closing_prices_table=closing_prices_table,
        pct_change_table=pct_change_table,
        num_por_simulation=settings.NUM_POR_SIMULATION,
        min_num_por_simulation=settings.MIN_NUM_POR_SIMULATION,
        max_percent_commodity=max_percent_commodity,
        max_percent_stocks=max_percent_stocks,
        model_name=model_name,
        is_machine_learning=is_machine_learning,
    )
    df = stats_models.get_df()
    df.to_csv(locationForSaving + model_name + '_df_' + levelOfRisk + '.csv')
    print('updated data frame Table(machine learning:' + str(is_machine_learning) +
          ', model name:' + model_name +
          ', level of risk:' + str(levelOfRisk))


def connect_to_s3() -> boto3.client:
    s3 = boto3.resource(service_name='s3',
                        region_name=aws_settings.region_name,
                        aws_secret_access_key=aws_settings.aws_secret_access_key,
                        aws_access_key_id=aws_settings.aws_access_key_id)

    s3_client = boto3.client('s3', aws_access_key_id=aws_settings.aws_access_key_id,
                             aws_secret_access_key=aws_settings.aws_secret_access_key,
                             region_name=aws_settings.region_name)
    return s3_client


def upload_file_to_s3(file_path, bucket_name, s3_object_key, s3_client):
    # Local folder path to upload
    local_folder_path = 'path/to/your/local/folder'

    """for root, dirs, files in os.walk(local_folder_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            s3_object_key = os.path.relpath(local_file_path, local_folder_path)
            upload_file_to_s3(local_file_path, bucket_name, s3_object_key)

    s3_client.upload_file(file_path, bucket_name, s3_object_key)"""


##################################################################
def create_new_user_portfolio(stocks_symbols: List, investment_amount: int, is_machine_learning: int,
                              model_option: int, risk_level: int, extended_data_from_db: Tuple) -> Portfolio:
    sectors, sectors, closing_prices_table, three_best_portfolios, _, \
        pct_change_table, _ = extended_data_from_db

    final_portfolio = three_best_portfolios[risk_level - 1]
    if risk_level == 1:
        # drop from stocks_symbols the stocks that are in Us Commodity sector
        stocks_symbols = helpers.drop_stocks_from_us_commodity_sector(
            stocks_symbols, helpers.set_stock_sectors(stocks_symbols, sectors)
        )

    portfolio = Portfolio(
        stocks_symbols=stocks_symbols,
        sectors=sectors,
        risk_level=risk_level,
        starting_investment_amount=investment_amount,
        selected_model=model_option,
        is_machine_learning=is_machine_learning
    )

    portfolio.update_stocks_data(
        closing_prices_table, pct_change_table, final_portfolio.iloc[0][3:],
        final_portfolio.iloc[0][0], final_portfolio.iloc[0][1], final_portfolio.iloc[0][2]
    )
    return portfolio


def download_data_for_research(num_of_years_history: int) -> None:
    stocks_symbols = []

    usa_stocks_list = read_csv_file(settings.RESEARCH_LOCATION + 'stocks_list.csv')  # usa stocks list
    usa_bonds_list = read_csv_file(settings.RESEARCH_LOCATION + 'bonds_list.csv')  # usa bonds list
    jsonData = get_json_data(settings.INDICES_LIST_JSON_NAME)
    israel_indexes_list = [item['indexId'] for item in jsonData['indicesList']['result']]  # israel indexes list
    # TODO - usa indexes list
    # TODO - israel stocks list
    # TODO - US commodity list

    helpers.convert_data_to_tables(settings.RESEARCH_LOCATION,
                                    'usa_stocks_closing_prices', usa_stocks_list, num_of_years_history, saveToCsv=True)

    helpers.convert_data_to_tables(settings.RESEARCH_LOCATION,
                                    'usa_bonds_closing_prices', usa_bonds_list, num_of_years_history, saveToCsv=True)

    helpers.convert_data_to_tables(settings.RESEARCH_LOCATION,
                                    'israel_indexes_closing_prices', israel_indexes_list, num_of_years_history,
                                    saveToCsv=True)

    stocks_symbols.extend(usa_stocks_list)
    stocks_symbols.extend(usa_bonds_list)
    stocks_symbols.extend(israel_indexes_list)
    helpers.convert_data_to_tables(settings.RESEARCH_LOCATION,
                                    'all_closing_prices', stocks_symbols, num_of_years_history, saveToCsv=True)


def get_stocks_data_for_research_by_group(group_of_stocks: str):
    if group_of_stocks == "nasdaq":  # TODO -fix
        nasdaq_tickers = yf.Tickers('^IXIC').tickers
        # Convert the dictionary of Ticker objects to a list of Ticker objects
        ticker_list = list(nasdaq_tickers.values())
        tickers = [ticker.ticker for ticker in ticker_list]
        filtered_stocks = tickers[
            (tickers['marketCap'] >= 1e9)  # &  # Stocks with a market capitalization of at least $1 billion
            # (tickers['ipoyear'] <= 2020)  # Stocks that went public before or in 2020
        ]
        # Get the ticker symbols for the filtered stocks
        tickers = filtered_stocks['symbol'].to_list()
    elif group_of_stocks == "sp500":  # TODO -fix
        sp500_tickers = yf.Tickers('^GSPC').tickers
        # Convert the dictionary of Ticker objects to a list of Ticker objects
        ticker_list = list(sp500_tickers.values())
        tickers = [ticker.ticker for ticker in ticker_list]
        filtered_stocks = tickers[
            (tickers['marketCap'] >= 1e9)  # &  # Stocks with a market capitalization of at least $1 billion
        ]
        # Get the ticker symbols for the filtered stocks
        tickers = filtered_stocks['symbol'].to_list()
    elif group_of_stocks == "dowjones":  # TODO -fix
        dowjones_tickers = yf.Tickers('^DJI').tickers
        # Convert the dictionary of Ticker objects to a list of Ticker objects
        ticker_list = list(dowjones_tickers.values())
        tickers = [ticker.ticker for ticker in ticker_list]
        filtered_stocks = tickers[
            (tickers['marketCap'] >= 1e9)  # &  # Stocks with a market capitalization of at least $1 billion
        ]
        # Get the ticker symbols for the filtered stocks
        tickers = filtered_stocks['symbol'].to_list()
    elif group_of_stocks == "TA35":  # TODO -fix
        TA35_tickers = yf.Tickers('^TA35.TA').tickers
        # Convert the dictionary of Ticker objects to a list of Ticker objects
        ticker_list = list(TA35_tickers.values())
        tickers = [ticker.ticker for ticker in ticker_list]
        """filtered_stocks = tickers[
            (tickers['marketCap'] >= 1e9) # &  # Stocks with a market capitalization of at least $1 billion
        ]"""
        # Get the ticker symbols for the filtered stocks
        tickers = tickers['symbol'].to_list()
    elif group_of_stocks == "TA90":  # TODO -fix
        TA90_tickers = yf.Tickers('^TA90.TA').tickers
        # Convert the dictionary of Ticker objects to a list of Ticker objects
        ticker_list = list(TA90_tickers.values())
        tickers = [ticker.ticker for ticker in ticker_list]
        """filtered_stocks = tickers[
            (tickers['marketCap'] >= 1e9) # &  # Stocks with a market capitalization of at least $1 billion
        ]"""
        # Get the ticker symbols for the filtered stocks
        tickers = tickers['symbol'].to_list()
    elif group_of_stocks == "TA125":  # TODO -fix
        TA125_tickers = yf.Tickers('^TA125.TA').tickers
        # Convert the dictionary of Ticker objects to a list of Ticker objects
        ticker_list = list(TA125_tickers.values())
        tickers = [ticker.ticker for ticker in ticker_list]
        """filtered_stocks = tickers[
            (tickers['marketCap'] >= 1e9) # &  # Stocks with a market capitalization of at least $1 billion
        ]"""
        # Get the ticker symbols for the filtered stocks
        tickers = tickers['symbol'].to_list()

    elif group_of_stocks == "usa_bonds":
        tickers = read_csv_file(settings.RESEARCH_LOCATION + 'usa_stocks_closing_prices.csv')

    elif group_of_stocks == "usa_stocks":
        tickers = read_csv_file(settings.RESEARCH_LOCATION + 'usa_stocks_closing_prices.csv')

    elif group_of_stocks == "israel_indexes":
        tickers = read_csv_file(settings.RESEARCH_LOCATION + 'israel_indexes_closing_prices.csv')

    elif group_of_stocks == "all":
        tickers = read_csv_file(settings.RESEARCH_LOCATION + 'all_closing_prices.csv')


    else:
        tickers = settings.STOCKS_SYMBOLS
        start, end = helpers.get_from_and_to_dates(10)
        data = yf.download(tickers, start, end=end)
        data = data.set_index(pd.to_datetime(data.index))
        # TODO

    return tickers


############################################################################################################
# UTILITY FUNCTIONS
############################################################################################################
# database utility functions:
def get_extended_data_from_db(stocks_symbols: list, is_machine_learning: int, model_option: int, mode: str):
    """
    Get extended data information from DB (CSV tables)
    """
    if mode == 'regular':
        sectors_data = get_json_data(settings.SECTORS_JSON_NAME)
    else:
        sectors_data = get_json_data('../../' + settings.SECTORS_JSON_NAME)
    sectors: list = helpers.set_sectors(stocks_symbols, mode)
    closing_prices_table: pd.DataFrame = get_closing_prices_table(mode=mode)
    # TODO : GET STOCKS SYMBOLS FROM TABLES INDEXES
    df = get_three_level_df_tables(is_machine_learning, settings.MODEL_NAME[model_option - 1], mode=mode)
    three_best_portfolios = helpers.get_best_portfolios(df, model_name=settings.MODEL_NAME[model_option - 1])
    best_stocks_weights_column = helpers.get_best_weights_column(stocks_symbols, sectors, three_best_portfolios,
                                                                  closing_prices_table.pct_change())
    three_best_stocks_weights = helpers.get_three_best_weights(three_best_portfolios)
    three_best_sectors_weights = helpers.get_three_best_sectors_weights(sectors,
                                                                         three_best_stocks_weights)
    pct_change_table: pd = closing_prices_table.pct_change()
    yields: list = update_pct_change_table(best_stocks_weights_column, pct_change_table)

    return sectors_data, sectors, closing_prices_table, three_best_portfolios, three_best_sectors_weights, \
        pct_change_table, yields


# Tables according to stocks symbols
def get_closing_prices_table(mode: str) -> pd.DataFrame:
    if mode == 'regular':
        closing_prices_table = pd.read_csv(
            settings.BUCKET_REPOSITORY + 'closing_prices.csv', index_col=0
        )
    else:
        closing_prices_table = pd.read_csv(
            '../../' + settings.BUCKET_REPOSITORY + 'closing_prices.csv', index_col=0
        )
    closing_prices_table = closing_prices_table.iloc[1:]
    closing_prices_table = closing_prices_table.apply(pd.to_numeric, errors='coerce')

    return closing_prices_table


def get_three_level_df_tables(is_machine_learning: int, model_name, mode: str) -> list:
    """
    Get the three level df tables according to machine learning option and model name
    """
    low_risk_df_table = get_df_table(is_machine_learning, model_name, "low", mode=mode)
    medium_risk_df_table = get_df_table(is_machine_learning, model_name, "medium", mode=mode)
    high_risk_df_table = get_df_table(is_machine_learning, model_name, "high", mode=mode)

    return [low_risk_df_table, medium_risk_df_table, high_risk_df_table]


def get_df_table(is_machine_learning: int, model_name, level_of_risk: str, mode: str) -> pd.DataFrame:
    """
    get specific df table from csv file according to machine learning option, model name and level of risk
    """
    if is_machine_learning:
        if mode == 'regular':
            df_table = pd.read_csv(
                settings.MACHINE_LEARNING_LOCATION + model_name + '_df_' + level_of_risk + '.csv'
            )
        else:
            df_table = pd.read_csv(
                '../../' + settings.MACHINE_LEARNING_LOCATION + model_name + '_df_' + level_of_risk + '.csv'
            )
    else:
        if mode == 'regular':
            df_table = pd.read_csv(
                settings.NON_MACHINE_LEARNING_LOCATION + model_name + '_df_' + level_of_risk + '.csv'
            )
        else:
            df_table = pd.read_csv(
                '../../' + settings.NON_MACHINE_LEARNING_LOCATION + model_name + '_df_' + level_of_risk + '.csv'
            )
    df_table = df_table.iloc[:, 1:]
    df_table = df_table.apply(pd.to_numeric, errors='coerce')
    return df_table


def get_all_users() -> List:
    """
    Get all users with their portfolios details from json file
    """
    json_data = get_json_data(settings.USERS_JSON_NAME)
    num_of_user = len(json_data['usersList'])
    users_data = json_data['usersList']
    users: List = [] * num_of_user
    for name in users_data.items():
        users.append(get_user_from_db(name))

    return users


def get_user_from_db(user_name: str):
    """
    Get specific user by his name with his portfolio details from json file
    """
    json_data = get_json_data(settings.USERS_JSON_NAME)
    if user_name not in json_data['usersList']:
        print("User not found")
        return None
    user_data = json_data['usersList'][user_name][0]
    starting_investment_amount = user_data['startingInvestmentAmount']
    is_machine_learning = user_data['machineLearningOpt']
    selected_model = user_data['selectedModel']
    risk_level = user_data['levelOfRisk']
    stocks_symbols = user_data['stocksSymbols']
    stocks_weights = user_data['stocksWeights']
    annual_returns = user_data['annualReturns']
    annual_volatility = user_data['annualVolatility']
    annual_sharpe = user_data['annualSharpe']
    sectors = helpers.set_sectors(stocks_symbols)

    closing_prices_table: pd.DataFrame = get_closing_prices_table(mode='regular')
    portfolio: Portfolio = Portfolio(
        stocks_symbols=stocks_symbols,
        sectors=sectors,
        risk_level=risk_level,
        starting_investment_amount=starting_investment_amount,
        selected_model=selected_model,
        is_machine_learning=is_machine_learning,
    )
    pct_change_table: pd = closing_prices_table.pct_change()
    pct_change_table.dropna(inplace=True)
    weighted_sum = np.dot(stocks_weights, pct_change_table.T)
    pct_change_table["weighted_sum_" + str(risk_level)] = weighted_sum
    yield_column: str = "yield_" + str(risk_level)
    pct_change_table[yield_column] = weighted_sum
    pct_change_table[yield_column] = makes_yield_column(pct_change_table[yield_column], weighted_sum)
    portfolio.update_stocks_data(closing_prices_table, pct_change_table, stocks_weights, annual_returns,
                                 annual_volatility, annual_sharpe)
    curr_user = User(user_name, portfolio)

    return curr_user


def find_user_in_list(user_name: str, users: list):
    for curr_user in users:
        if curr_user.getUserName() == user_name:
            return curr_user
    return None


def get_num_of_users_in_db() -> int:
    json_data = get_json_data(settings.USERS_JSON_NAME)
    return len(json_data['usersList'])


def update_pct_change_table(best_stocks_weights_column, pct_change_table):
    [weighted_low, weighted_medium, weighted_high] = best_stocks_weights_column
    pct_change_table.dropna(inplace=True)
    pct_change_table["weighted_sum_1"] = weighted_low
    pct_change_table["weighted_sum_2"] = weighted_medium
    pct_change_table["weighted_sum_3"] = weighted_high
    pct_change_table["yield_1"] = weighted_low
    pct_change_table["yield_2"] = weighted_medium
    pct_change_table["yield_3"] = weighted_high
    yield_low = makes_yield_column(pct_change_table["yield_1"], weighted_low)
    yield_medium = makes_yield_column(pct_change_table["yield_2"], weighted_medium)
    yield_high = makes_yield_column(pct_change_table["yield_3"], weighted_high)
    pct_change_table["yield_1"] = yield_low
    pct_change_table["yield_2"] = yield_medium
    pct_change_table["yield_3"] = yield_high

    return [yield_low, yield_medium, yield_high]


def read_csv_file(file_path):
    rows_list = []
    with open(file_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            if type(row) == list:
                if row[0] == 'Symbol':
                    continue
                rows_list.append(row)
            else:
                if row == 'Symbol':
                    continue
                rows_list.append(row)
    return rows_list


def get_level_of_risk_by_score(count: int) -> int:
    if count <= 4:
        return 1
    elif count <= 7:
        return 2
    elif count <= 9:
        return 3
    else:
        raise ValueError


def creates_json_file(json_obj, name_product: str) -> None:
    # Open a file in write mode
    parts: list = name_product.split("/")
    last_element: str = parts[-1]
    with open("impl/config/" + last_element + ".json", "w") as f:
        json.dump(json_obj, f)  # Use the `dump()` function to write the JSON data to the file


# impl utility functions
def set_sectors(stocks_symbols):
    return helpers.set_sectors(stocks_symbols, mode='regular')


def makes_yield_column(_yield, weighted_sum_column):
    return helpers.makes_yield_column(_yield, weighted_sum_column)


def get_json_data(name):
    return helpers.get_json_data(name)


def get_from_and_to_date(num_of_years):  # TODO FIX RETURN TUPLE
    return helpers.get_from_and_to_dates(num_of_years)


def get_score_by_answer_from_user(string_to_show: str) -> int:
    return console_handler.get_score_by_answer_from_user(string_to_show)


# plot functions
def plot_three_portfolios_graph(three_best_portfolios: list, three_best_sectors_weights, sectors: list,
                                pct_change_table, mode: str, sub_folder: str = '00/'):
    min_variance_port = three_best_portfolios[0]
    sharpe_portfolio = three_best_portfolios[1]
    max_returns = three_best_portfolios[2]
    plt_instance_three_graph = plot_functions.plot_three_portfolios_graph(min_variance_port, sharpe_portfolio,
                                                                          max_returns,
                                                                          three_best_sectors_weights, sectors,
                                                                          pct_change_table)
    if mode == 'regular':
        fully_qualified_name = settings.GRAPH_IMAGES + sub_folder + 'three_portfolios'
    else:
        fully_qualified_name = '../../' + settings.GRAPH_IMAGES + sub_folder + 'three_portfolios'
    plot_functions.save_graphs(plt_instance_three_graph, fully_qualified_name)

    return plt_instance_three_graph


def plot_distribution_of_stocks(stock_names, pct_change_table):
    plt_instance = plot_functions.plot_distribution_of_stocks(stock_names, pct_change_table)

    return plt_instance


def plot_distribution_of_portfolio(distribution_graph, mode: str, sub_folder: str = '00/'):
    plt_instance = plot_functions.plot_distribution_of_portfolio(distribution_graph)
    if mode == 'regular':
        fully_qualified_name = settings.GRAPH_IMAGES + sub_folder + 'distribution_graph'
    else:
        fully_qualified_name = '../../' + settings.GRAPH_IMAGES + sub_folder + 'distribution_graph'
    plot_functions.save_graphs(plt_instance, fully_qualified_name)

    return plt_instance


def plot_stat_model_graph(stocks_symbols: list, is_machine_learning: int, model_option: int,
                          num_of_years_history=settings.NUM_OF_YEARS_HISTORY) -> None:
    sectors: list = set_sectors(stocks_symbols)

    closing_prices_table: pd.DataFrame = get_closing_prices_table(mode='regular')
    pct_change_table = closing_prices_table.pct_change()
    # TODO - get part of closing_prices_table according to num_of_years_history

    if is_machine_learning == 1:
        pct_change_table, annual_return, excepted_returns = helpers.update_daily_change_with_machine_learning(
            pct_change_table, closing_prices_table.index)

    if model_option == "Markowitz":
        model_name = "Markowitz"
    else:
        model_name = "Gini"

    stats_models = StatsModels(
        stocks_symbols=stocks_symbols,
        sectors=sectors,
        closing_prices_table=closing_prices_table,
        pct_change_table=pct_change_table,
        num_por_simulation=settings.NUM_POR_SIMULATION,
        min_num_por_simulation=settings.MIN_NUM_POR_SIMULATION,
        max_percent_commodity=1,
        max_percent_stocks=1,
        model_name=model_name,
        is_machine_learning=is_machine_learning,
    )
    df = stats_models.get_df()
    three_best_portfolios = helpers.get_best_portfolios(df=[df, df, df], model_name=model_option)
    three_best_stocks_weights = helpers.get_three_best_weights(three_best_portfolios)
    three_best_sectors_weights = helpers.get_three_best_sectors_weights(sectors, three_best_stocks_weights)
    min_variance_port = three_best_portfolios[0]
    sharpe_portfolio = three_best_portfolios[1]
    max_returns = three_best_portfolios[2]
    max_vols = stats_models.get_max_vols()
    df = stats_models.get_df()

    if model_option == "Markowitz":
        plt_instance = plot_functions.plot_markowitz_graph(sectors, three_best_sectors_weights, min_variance_port,
                                                           sharpe_portfolio, max_returns, max_vols, df)
    else:
        plt_instance = plot_functions.plot_gini_graph(sectors, three_best_sectors_weights, min_variance_port,
                                                      sharpe_portfolio, max_returns, max_vols, df)

    plot_functions.save_graphs(plt_instance, settings.GRAPH_IMAGES + model_option + '_all_option')  # TODO plot at site


def save_user_portfolio(curr_user: User) -> None:
    # Creating directories
    curr_user_directory = settings.USER_IMAGES + curr_user.name
    try:
        os.mkdir(curr_user_directory)  # Creates 'static/img/user/<USER_ID>' folder
    except FileExistsError:  # Ignore the exception
        pass

    # get data from user
    portfolio: Portfolio = curr_user.portfolio
    stocks_weights: List[float] = portfolio.stocks_weights
    stocks_symbols: List[str] = portfolio.stocks_symbols
    sectors_weights: list[float] = portfolio.get_sectors_weights()
    sectors_names: list[str] = portfolio.get_sectors_names()
    table = portfolio.pct_change_table
    stats_details_tuple = portfolio.get_portfolio_stats()
    sectors: List[Sector] = portfolio.sectors
    description: List[str] = helpers.get_stocks_descriptions(stocks_symbols)

    # pie chart of sectors & sectors weights
    plt_sectors_component = plot_functions.plot_portfolio_component(curr_user.name,
                                                                    sectors_weights,
                                                                    sectors_names)
    plot_functions.save_graphs(plt_sectors_component, file_name=curr_user_directory + '/sectors_component')

    # pie chart of stocks & stocks weights , TODO: show as tables instead of pie chart
    plt_stocks_component = plot_functions.plot_portfolio_component_stocks(curr_user.name,
                                                                          stocks_weights,
                                                                          stocks_symbols,
                                                                          description[1:]
                                                                          )
    plot_functions.save_graphs(plt_stocks_component, file_name=curr_user_directory + '/stocks_component')

    # total yield graph with sectors weights
    table['yield__selected_percent'] = (table["yield_selected"] - 1) * 100
    df, returns_forecast, returns_annual = helpers.analyze_with_machine_learning_linear_regression(
        table['yield__selected_percent'],
        table.index, closing_prices_mode=True)
    df['yield__selected_percent'] = df['col']
    df['yield__selected_percent_forecast'] = df["Forecast"]

    plt_yield_graph = plot_functions.plot_investment_portfolio_yield(curr_user.name,
                                                                     df,
                                                                     stats_details_tuple,
                                                                     sectors)
    plot_functions.save_graphs(plt_yield_graph, file_name=curr_user_directory + '/yield_graph')


# research functions plot
def save_user_specific_stock(user_name: str, stock: str, operation: str, plt_instance) -> None:
    # Creating directories
    curr_user_directory = settings.USER_IMAGES + user_name + '/research'
    try:
        os.mkdir(curr_user_directory)  # Creates 'static/img/user/<USER_ID>' folder
    except FileExistsError:  # Ignore the exception
        pass
    # Saving files
    plot_functions.save_graphs(plt_instance, file_name=curr_user_directory + '/' + stock + '_' + operation)


def forecast_specific_stock(stock: str, machine_learning_model, num_of_years_history: int):
    plt = None
    file_name = settings.CLOSING_PRICES_FILE_NAME
    table = helpers.convert_data_to_tables(settings.BUCKET_REPOSITORY, file_name,
                                            [stock], num_of_years_history, saveToCsv=False)
    if machine_learning_model == settings.MACHINE_LEARNING_MODEL[0]:
        df, annual_return, excepted_returns = helpers.analyze_with_machine_learning_linear_regression(table,
                                                                                                       table.index,
                                                                                                       closing_prices_mode=True)
    elif machine_learning_model == settings.MACHINE_LEARNING_MODEL[1]:
        df, annual_return, excepted_returns = helpers.analyze_with_machine_learning_arima(table, table.index,
                                                                                           closing_prices_mode=True)
    elif machine_learning_model == settings.MACHINE_LEARNING_MODEL[2]:
        df, annual_return, excepted_returns = helpers.analyze_with_machine_learning_gbm(table, table.index,
                                                                                         closing_prices_mode=True)
    elif machine_learning_model == settings.MACHINE_LEARNING_MODEL[3]:
        df, annual_return, excepted_returns, plt = helpers.analyze_with_machine_learning_prophet(table, table.index,
                                                                                                  closing_prices_mode=True)

    plt_instance = plot_functions.plot_price_forecast(stock, df, annual_return, plt)
    return plt_instance


def plotbb_strategy_stock(stock_name: str, start="2009-01-01", end="2023-01-01"):
    stock_prices = yf.download(stock_name, start, end)
    stock_prices['MA50'] = stock_prices['Adj Close'].rolling(window=50).mean()
    stock_prices['50dSTD'] = stock_prices['Adj Close'].rolling(window=50).std()
    stock_prices['Upper'] = stock_prices['MA50'] + (stock_prices['50dSTD'] * 2)
    stock_prices['Lower'] = stock_prices['MA50'] - (stock_prices['50dSTD'] * 2)

    stock_prices = stock_prices.dropna()
    stock_prices = stock_prices.iloc[51:]

    buy_price, sell_price, bb_signal = helpers.implement_bb_strategy(stock_prices['Adj Close'],
                                                                      stock_prices['Lower'], stock_prices['Upper'])
    plt_instance = plot_functions.plotbb_strategy_stock(stock_prices, buy_price, sell_price)
    return plt_instance


def plotbb_strategy_portfolio(pct_change_table, new_portfolio):  # TODO
    plt_instance = plot_functions.plotbb_strategy_portfolio(pct_change_table, new_portfolio)

    return plt_instance


def find_good_stocks(group_of_stocks="usa_stocks", filter_option=False):  # TODO - fix
    max_returns_stocks_list = None
    min_volatility_stocks_list = None
    max_sharpest_stocks_list = None

    tickers_table = get_stocks_data_for_research_by_group(group_of_stocks)
    tickers_df = pd.DataFrame(tickers_table)
    tickers_df.iloc[2:3] = np.nan
    tickers_df.dropna(inplace=True)
    tickers_df.index.name = "Date"
    data = tickers_df.set_index(0)
    data.columns = data.iloc[0]
    data = data.iloc[1:]
    data = pd.DataFrame(data, columns=data.columns)
    data = data.rename_axis('Date')
    data = data.apply(pd.to_numeric, errors='coerce')

    data_pct_change = data.pct_change()
    data_pct_change.fillna(value=-0.0, inplace=True)

    # Convert the index to datetime
    data_pct_change.index = pd.to_datetime(data_pct_change.index)
    annual_returns = ((data_pct_change + 1).resample('Y').prod() - 1) * 100
    total_profit_return = ((data_pct_change + 1).prod() - 1) * 100
    total_volatility = data_pct_change.std() * np.sqrt(254)
    annual_volatility = data_pct_change.groupby(pd.Grouper(freq='Y')).std()
    sharpe = annual_returns / annual_volatility

    # forcast from total:
    returns_annual_forcecast = (((1 + data_pct_change.mean()) ** 254) - 1) * 100
    cov_daily = data_pct_change.cov()
    cov_annual = cov_daily * 254
    volatility_forecast = (data_pct_change.std() * np.sqrt(254)) * 100
    sharpe_forecast = returns_annual_forcecast / volatility_forecast

    total_sharpe = total_profit_return / total_volatility

    # sort total
    total_profit_return_sorted = total_profit_return.sort_values(ascending=False).head(10)
    total_volatility_sorted = total_volatility.sort_values(ascending=True).head(10)
    total_sharpe_sorted = total_sharpe.sort_values(ascending=False).head(10)

    print("total_profit_return_sorted")
    print(total_profit_return_sorted)
    print("total_volatility_sorted")
    print(total_volatility_sorted)
    print("total_sharpe_sorted")
    print(total_sharpe_sorted)

    # sort last year
    annual_sharpe_sorted = sharpe[-1].sort_values().head(10)
    annual_returns_sorted = annual_returns[-1].sort_values().head(10)
    annual_volatility_sorted = annual_volatility[-1].sort_values().head(10)
    annual_sharpe_sorted = annual_sharpe_sorted[-1].sort_values().head(10)
    print("annual_returns_sorted")
    print(annual_returns_sorted)
    print("annual_volatility_sorted")
    print(annual_volatility_sorted)
    print("annual_sharpe_sorted")
    print(annual_sharpe_sorted)

    # sort forecast
    returns_annual_forcecast_sorted = returns_annual_forcecast.sort_values().head(10)
    volatility_forecast_sorted = volatility_forecast.sort_values().head(10)
    sharpe_forecast_sorted = sharpe_forecast.sort_values().head(10)

    """max_sharpest_stocks_list = sharpe_sorted.head(10)
    max_returns_stocks_list = returns_sorted.head(10)
    min_volatility_stocks_list = min_volatility_sorted.tail(10)"""

    # plot_functions.plot_top_stocks(helpers.scan_good_stocks())
    # plot_functions.plot(plt_instance)  # TODO plot at site
    # plot_functions.plot(plt_instance)  # TODO plot at site

    return max_returns_stocks_list, min_volatility_stocks_list, max_sharpest_stocks_list


def plot_image(file_name) -> None:
    plot_functions.plot_image(file_name)


# console functions
def main_menu() -> None:
    console_handler.main_menu()


def expert_menu() -> None:
    console_handler.expert_menu()


def selected_menu_option() -> int:
    return console_handler.selected_menu_option()


def get_name() -> str:
    return console_handler.get_name()


def get_num_of_years_history() -> int:
    return console_handler.get_num_of_years_history()


def get_machine_learning_option() -> int:
    return console_handler.get_machine_learning_option()


def get_model_option() -> int:
    return console_handler.get_model_option()


def get_machine_learning_model() -> str:
    option: int = console_handler.get_machine_learning_mdoel()
    return settings.MACHINE_LEARNING_MODEL[option - 1]


def get_group_of_stocks_option() -> int:
    return console_handler.get_group_of_stocks_option()


def get_investment_amount():
    return console_handler.get_investment_amount()
