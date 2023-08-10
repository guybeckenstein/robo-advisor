import csv
import json
import os
import numpy as np
import pandas as pd
from datetime import date
from typing import Tuple, List

from pip._internal.utils.misc import tabulate

from ..impl.portfolio import Portfolio
from ..impl.stats_models import StatsModels
from ..impl.sector import Sector
from ..impl.user import User
from ..impl.config import aws_settings
from . import helpers, console_handler, plot_functions, settings
import boto3


######################################################################################
# update DB tables
def update_all_tables(numOfYearsHistory):  # build DB for withdraw
    today = date.today()
    formatted_date = today.strftime("%Y-%m-%d")
    collections_json_data = get_collection_json_data()
    for i, collection in enumerate(collections_json_data):
        curr_collection = collections_json_data[str(i + 1)][0]
        stocksSymbols = curr_collection['stocksSymbols']
        __path = settings.BASIC_STOCK_COLLECTION_REPOSITORY + str(str(i + 1)) + '/'  # where to save the DB
        update_closing_prices_tables(formatted_date, stocksSymbols, numOfYearsHistory, __path)
        update_data_frame_tables(formatted_date, curr_collection, __path, collections_json_data, str(i + 1))


def update_closing_prices_tables(formatted_date_today, stocksSymbols, numOfYearsHistory, __path):
    with open(__path + "lastUpdatedClosingPrice.txt", "r") as file:
        lastUpdatedDateClosingPrices = file.read().strip()
    if lastUpdatedDateClosingPrices != formatted_date_today:
        helpers.convert_data_to_tables(__path, settings.CLOSING_PRICES_FILE_NAME,
                                       stocksSymbols,
                                       numOfYearsHistory, saveToCsv=True)

        with open(__path + "lastUpdatedClosingPrice.txt", "w") as file:
            file.write(formatted_date_today)


def update_data_frame_tables(formatted_date_today, collection_json_data, __path, models_data, collection_num):
    stocksSymbols = collection_json_data['stocksSymbols']


    with open(__path + "lastUpdatedDftables.txt", "r") as file:
        lastUpdatedDftables = file.read().strip()
    if lastUpdatedDftables != formatted_date_today:
        sectorsList = helpers.set_sectors(stocksSymbols, mode='regular')
        closingPricesTable = get_closing_prices_table(__path, mode='regular')
        pct_change_table = closingPricesTable.pct_change()

        # without maching learning
        # Markowitz
        update_three_level_data_frame_tables(machingLearningOpt=0, modelName="Markowitz",
                                             stocksSymbols=stocksSymbols, sectorsList=sectorsList,
                                             closingPricesTable=closingPricesTable, pct_change_table=pct_change_table,
                                             __path=__path, models_data=models_data, collection_num=collection_num)
        # Gini
        update_three_level_data_frame_tables(machingLearningOpt=0, modelName="Gini",
                                             stocksSymbols=stocksSymbols, sectorsList=sectorsList,
                                             closingPricesTable=closingPricesTable, pct_change_table=pct_change_table,
                                             __path=__path, models_data=models_data, collection_num=collection_num)

        # Including maching learning
        pct_change_table, annual_return, excepted_returns = helpers.update_daily_change_with_machine_learning(
            pct_change_table, closingPricesTable.index, models_data)
        # Markowitz
        update_three_level_data_frame_tables(machingLearningOpt=1, modelName="Markowitz",
                                             stocksSymbols=stocksSymbols, sectorsList=sectorsList,
                                             closingPricesTable=closingPricesTable, pct_change_table=pct_change_table,
                                             __path=__path, models_data=models_data, collection_num=collection_num)
        # Gini
        update_three_level_data_frame_tables(machingLearningOpt=1, modelName="Gini",
                                             stocksSymbols=stocksSymbols, sectorsList=sectorsList,
                                             closingPricesTable=closingPricesTable, pct_change_table=pct_change_table,
                                             __path=__path, models_data=models_data, collection_num=collection_num)

        with open(__path + "lastUpdatedDftables.txt", "w") as file:
            file.write(formatted_date_today)


def update_three_level_data_frame_tables(machingLearningOpt, modelName, stocksSymbols, sectorsList,
                                         closingPricesTable, pct_change_table, __path, models_data, collection_num):
    collection_data = models_data[collection_num][0]
    LIMIT_PERCENT_MEDIUM_RISK_STOCKS = collection_data['LIMIT_PERCENT_MEDIUM_RISK_STOCKS']
    LIMIT_PERCENT_MEDIUM_RISK_COMMODITY = collection_data['LIMIT_PERCENT_MEDIUM_RISK_COMMODITY']
    LIMIT_PERCENT_LOW_RISK_STOCKS = collection_data['LIMIT_PERCENT_LOW_RISK_STOCKS']
    LIMIT_PERCENT_LOW_RISK_COMMODITY = collection_data['LIMIT_PERCENT_LOW_RISK_COMMODITY']
    # high risk
    update_specific_data_frame_table(is_machine_learning=machingLearningOpt, model_name=modelName,
                                     stocks_symbols=stocksSymbols,
                                     sectors=sectorsList, levelOfRisk="high", max_percent_commodity=1,
                                     max_percent_stocks=1, closing_prices_table=closingPricesTable,
                                     pct_change_table=pct_change_table,
                                     __path=__path, models_data=models_data)
    # medium risk
    update_specific_data_frame_table(is_machine_learning=machingLearningOpt, model_name=modelName,
                                     stocks_symbols=stocksSymbols,
                                     sectors=sectorsList, levelOfRisk="medium",
                                     max_percent_commodity=LIMIT_PERCENT_MEDIUM_RISK_COMMODITY,
                                     max_percent_stocks=LIMIT_PERCENT_MEDIUM_RISK_STOCKS,
                                     closing_prices_table=closingPricesTable, pct_change_table=pct_change_table,
                                     __path=__path, models_data=models_data)
    # low risk
    update_specific_data_frame_table(is_machine_learning=machingLearningOpt, model_name=modelName,
                                     stocks_symbols=stocksSymbols,
                                     sectors=sectorsList, levelOfRisk="low",
                                     max_percent_commodity=LIMIT_PERCENT_LOW_RISK_COMMODITY,
                                     max_percent_stocks=LIMIT_PERCENT_LOW_RISK_STOCKS,
                                     closing_prices_table=closingPricesTable, pct_change_table=pct_change_table,
                                     __path=__path, models_data=models_data)


def update_specific_data_frame_table(is_machine_learning, model_name, stocks_symbols, sectors, levelOfRisk,
                                     max_percent_commodity, max_percent_stocks, closing_prices_table, pct_change_table,
                                     __path, models_data):
    NUM_POR_SIMULATION = models_data["models_data"]['NUM_POR_SIMULATION']
    MIN_NUM_POR_SIMULATION = models_data["models_data"]['MIN_NUM_POR_SIMULATION']
    GINI_V_VALUE = models_data["models_data"]['GINI_V_VALUE']

    if is_machine_learning:
        locationForSaving = __path + settings.MACHINE_LEARNING_LOCATION
    else:
        locationForSaving = __path + settings.NON_MACHINE_LEARNING_LOCATION

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
        num_por_simulation=NUM_POR_SIMULATION,
        min_num_por_simulation=MIN_NUM_POR_SIMULATION,
        max_percent_commodity=max_percent_commodity,
        max_percent_stocks=max_percent_stocks,
        model_name=model_name,
        gini_value=GINI_V_VALUE

    )
    df = stats_models.get_df()
    df.to_csv(locationForSaving + model_name + '_df_' + levelOfRisk + '.csv')
    print('updated data frame Table(machine learning:' + str(is_machine_learning) +
          ', model name:' + model_name +
          ', level of risk:' + str(levelOfRisk))


##################################################################
# operations
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


############################################################################################################
# UTILITY FUNCTIONS
############################################################################################################
# database utility functions:
def get_extended_data_from_db(stocks_symbols: list, is_machine_learning: int, model_option: int,
                              stocks_collection_number, mode: str):
    """
    Get extended data information from DB (CSV tables)
    """

    __path = settings.BASIC_STOCK_COLLECTION_REPOSITORY + stocks_collection_number + '/'
    if mode == 'regular':
        sectors_data = get_json_data(settings.SECTORS_JSON_NAME)
    else:
        sectors_data = get_json_data('../../' + settings.SECTORS_JSON_NAME)
    sectors: list = helpers.set_sectors(stocks_symbols, mode)
    closing_prices_table: pd.DataFrame = get_closing_prices_table(__path, mode=mode)
    df = get_three_level_df_tables(is_machine_learning, settings.MODEL_NAME[model_option - 1], __path, mode=mode)
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
def get_closing_prices_table(__path, mode: str) -> pd.DataFrame:
    if mode == 'regular':
        closing_prices_table = pd.read_csv(
            __path + 'closing_prices.csv', index_col=0
        )
    else:
        closing_prices_table = pd.read_csv(
            '../../' + __path + 'closing_prices.csv', index_col=0
        )
    closing_prices_table = closing_prices_table.iloc[1:]
    closing_prices_table = closing_prices_table.apply(pd.to_numeric, errors='coerce')

    return closing_prices_table


def get_three_level_df_tables(is_machine_learning: int, model_name, collection_path: str, mode: str) -> list:
    """
    Get the three level df tables according to machine learning option and model name
    """
    low_risk_df_table = get_df_table(is_machine_learning, model_name, "low", collection_path, mode=mode)
    medium_risk_df_table = get_df_table(is_machine_learning, model_name, "medium", collection_path, mode=mode)
    high_risk_df_table = get_df_table(is_machine_learning, model_name, "high", collection_path, mode=mode)

    return [low_risk_df_table, medium_risk_df_table, high_risk_df_table]


def get_df_table(is_machine_learning: int, model_name, level_of_risk: str, collection_path: str,
                 mode: str) -> pd.DataFrame:
    """
    get specific df table from csv file according to machine learning option, model name and level of risk
    """
    basic_path = collection_path
    if is_machine_learning:
        basic_path += settings.MACHINE_LEARNING_LOCATION
    else:
        basic_path += settings.NON_MACHINE_LEARNING_LOCATION
    if mode == 'regular':
        df_table = pd.read_csv(basic_path + model_name + '_df_' + level_of_risk + '.csv'
                               )
    else:
        df_table = pd.read_csv(
            '../../' + basic_path + model_name + '_df_' + level_of_risk + '.csv'
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
    try:
        stocks_collection_number = user_data['stocksCollectionNumber']
    except:
        stocks_collection_number = "1"  # default
    sectors = helpers.set_sectors(stocks_symbols)

    __path = settings.BASIC_STOCK_COLLECTION_REPOSITORY + stocks_collection_number + '/'

    closing_prices_table: pd.DataFrame = get_closing_prices_table(__path=__path, mode='regular')
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


def get_collection_json_data():
    return get_json_data(settings.COLLECTION_JSON_NAME)['collections']


def get_stocks_symbols_from_collection(stocks_collection_number) -> List:
    """
    Get all stocks symbols from json file
    """
    json_data = get_collection_json_data()
    stocks_symbols = json_data[str(stocks_collection_number)][0]['stocksSymbols']
    return stocks_symbols


def get_models_data_from_collections_file():  # TODO - maybe use from admin
    json_data = get_collection_json_data()
    return json_data["models_data"]


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


def update_models_data_settings(NUM_POR_SIMULATION,
                                MIN_NUM_POR_SIMULATION,
                                RECORD_PERCENT_TO_PREDICT,
                                TEST_SIZE_MACHINE_LEARNING,
                                SELECTED_ML_MODEL_FOR_BUILD,
                                GINI_V_VALUE):
    models_data = get_models_data_from_collections_file()
    models_data['NUM_POR_SIMULATION'] = NUM_POR_SIMULATION
    models_data['MIN_NUM_POR_SIMULATION'] = MIN_NUM_POR_SIMULATION
    models_data['RECORD_PERCENT_TO_PREDICT'] = RECORD_PERCENT_TO_PREDICT
    models_data['TEST_SIZE_MACHINE_LEARNING'] = TEST_SIZE_MACHINE_LEARNING
    models_data['SELECTED_ML_MODEL_FOR_BUILD'] = SELECTED_ML_MODEL_FOR_BUILD
    models_data['GINI_V_VALUE'] = GINI_V_VALUE


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
                          num_of_years_history=settings.NUM_OF_YEARS_HISTORY, models_data: dict = None,
                          closing_prices_table_path: str = "") -> None:
    sectors: list = set_sectors(stocks_symbols)
    NUM_POR_SIMULATION = models_data["models_data"]['NUM_POR_SIMULATION']
    MIN_NUM_POR_SIMULATION = models_data["models_data"]['MIN_NUM_POR_SIMULATION']
    GINI_V_VALUE = models_data["models_data"]['GINI_V_VALUE']
    closing_prices_table: pd.DataFrame = get_closing_prices_table(closing_prices_table_path,
                                                                  mode='regular')
    pct_change_table = closing_prices_table.pct_change()
    if num_of_years_history != settings.NUM_OF_YEARS_HISTORY:
        pct_change_table = pct_change_table.tail(num_of_years_history * 252)

    if is_machine_learning == 1:
        pct_change_table, annual_return, excepted_returns = helpers.update_daily_change_with_machine_learning(
            pct_change_table, models_data, closing_prices_table.index)

    if model_option == "Markowitz":
        model_name = "Markowitz"
    else:
        model_name = "Gini"

    stats_models = StatsModels(
        stocks_symbols=stocks_symbols,
        sectors=sectors,
        closing_prices_table=closing_prices_table,
        pct_change_table=pct_change_table,
        num_por_simulation=NUM_POR_SIMULATION,
        min_num_por_simulation=MIN_NUM_POR_SIMULATION,
        max_percent_commodity=1,
        max_percent_stocks=1,
        model_name=model_name,
        gini_value=GINI_V_VALUE,
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
    models_data = get_models_data_from_collections_file()
    RECORD_PERCENT_TO_PREDICT = models_data['RECORD_PERCENT_TO_PREDICT']
    TEST_SIZE_MACHINE_LEARNING = models_data['TEST_SIZE_MACHINE_LEARNING']

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
        table.index,
        RECORD_PERCENT_TO_PREDICT,
        TEST_SIZE_MACHINE_LEARNING,
        closing_prices_mode=True)
    df['yield__selected_percent'] = df['col']
    df['yield__selected_percent_forecast'] = df["Forecast"]

    plt_yield_graph = plot_functions.plot_investment_portfolio_yield(curr_user.name,
                                                                     df,
                                                                     stats_details_tuple,
                                                                     sectors)
    plot_functions.save_graphs(plt_yield_graph, file_name=curr_user_directory + '/yield_graph')


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


def get_investment_amount() -> int:
    return console_handler.get_investment_amount()


def get_collection_number() -> str:
    stocks_collections = get_stocks_collections_from_json_file()
    return console_handler.get_collection_number(stocks_collections)


def get_stocks_collections_from_json_file():
    collections_data = get_collection_json_data()
    stocks_collections = {}
    for i in range(1, len(collections_data)):
        stocks_symbols_list = collections_data[str(i)][0]['stocksSymbols']
        stocks_description_list = helpers.get_stocks_descriptions(stocks_symbols_list, is_reverse_mode=False)
        stocks_collections[str(i)] = [stocks_symbols_list, stocks_description_list]
    return stocks_collections
