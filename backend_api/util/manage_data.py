import json
import os

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Tuple

from backend_api.api import portfolio, stats_models, user
from backend_api.util import api_util, console_handler, plot_functions, settings
from backend_api.util.settings import STATIC_GRAPH_FILES_LOCATION, STATIC_USER_FILES_LOCATION


######################################################################################
# 1
def create_new_user_portfolio(stocks_symbols: list, investment_amount: int, is_machine_learning: int,
                              model_option: int, level_of_risk: int, extendedDataFromDB:Tuple) -> portfolio.Portfolio:
    sectors_data, sectors, closing_prices_table, three_best_portfolios, _, \
        pct_change_table, _ = extendedDataFromDB

    final_portfolio = three_best_portfolios[level_of_risk - 1]
    if level_of_risk == 1:
        # drop from stocks_symbols the stocks that are in Us Commodity sector
        stocks_symbols = api_util.drop_stocks_from_us_commodity_sector(
            stocks_symbols, api_util.set_stock_sectors(stocks_symbols, sectors)
        )

    new_portfolio = portfolio.Portfolio(level_of_risk, investment_amount, stocks_symbols, sectors_data, model_option,
                                        is_machine_learning)

    new_portfolio.update_stocks_data(closing_prices_table, pct_change_table, final_portfolio.iloc[0][3:],
                                     final_portfolio.iloc[0][0], final_portfolio.iloc[0][1], final_portfolio.iloc[0][2])
    return new_portfolio


#############################################################################################################
# 3 - plot user portfolio -# TODO plot at site

def plot_user_portfolio(curr_user: user.User) -> None:
    try:
        os.mkdir(STATIC_USER_FILES_LOCATION + curr_user.get_name())
    except FileExistsError:
        pass
    # pie chart of sectors weights
    sectors_component_plt = curr_user.plot_sectors_component()
    sectors_component_plt.savefig(
        STATIC_USER_FILES_LOCATION + curr_user.get_name() + '/sectors_component.png', format='png'
    )
    # chart of stocks weights
    stocks_component_plt = curr_user.plot_stocks_component()
    stocks_component_plt.savefig(
        STATIC_USER_FILES_LOCATION + curr_user.get_name() + '/stocks_component.png', format='png'
    )
    yield_graph_plt = curr_user.plot_yield_component()
    yield_graph_plt.savefig(
        STATIC_USER_FILES_LOCATION + curr_user.get_name() + '/yield_graph.png', format='png'
    )


#############################################################################################################
# 4- EXPERT OPTIONS:
#############################################################################################################
# EXPERT - 1


def forecast_specific_stock(stock: str, is_data_come_from_tase: bool, num_of_years_history: int) -> None:
    if is_data_come_from_tase:
        df = pd.DataFrame(stock["indexEndOfDay"]["result"])
        df["tradeDate"] = pd.to_datetime(df["tradeDate"])
        df.set_index("tradeDate", inplace=True)
        df, col = price_forecast(df, settings.RECORD_PERCENT_TO_PREDICT, 1)
        plt_instance = plot_price_forecast(stock, df, 1)

    else:
        yf.pdr_override()
        start_date, end_date = get_from_and_to_date(num_of_years_history)
        df = yf.download(stock, start=start_date, end=end_date)
        df, col = price_forecast(df, settings.RECORD_PERCENT_TO_PREDICT, 0)
        plt_instance = plot_price_forecast(stock, df, 0)
    plot_functions.plot(plt_instance)  # TODO plot at site


#############################################################################################################
# EXPERT -2


def plotbb_strategy_stock(stock_name: str, start="2009-01-01", end="2023-01-01") -> None:
    stock_prices = yf.download(stock_name, start, end)
    stock_prices['MA50'] = stock_prices['Adj Close'].rolling(window=50).mean()
    stock_prices['50dSTD'] = stock_prices['Adj Close'].rolling(window=50).std()
    stock_prices['Upper'] = stock_prices['MA50'] + (stock_prices['50dSTD'] * 2)
    stock_prices['Lower'] = stock_prices['MA50'] - (stock_prices['50dSTD'] * 2)

    stock_prices = stock_prices.dropna()
    stock_prices = stock_prices.iloc[51:]

    buy_price, sell_price, bb_signal = api_util.implement_bb_strategy(stock_prices['Adj Close'],
                                                                      stock_prices['Lower'], stock_prices['Upper'])
    plt_instance = plot_functions.plotbb_strategy_stock(stock_prices, buy_price, sell_price)
    plot_functions.plot(plt_instance)  # TODO plot at site


#############################################################################################################
# EXPERT -4


def find_best_stocks() -> None:
    plt_instance = plot_functions.plot_top_stocks(api_util.find_best_stocks())
    plot_functions.plot(plt_instance)  # TODO plot at site


def scan_good_stocks() -> None:
    plt_instance = plot_functions.plot_top_stocks(api_util.scan_good_stocks())
    plot_functions.plot(plt_instance)  # TODO plot at site


############################################################################################################
# EXPERT- 5&6


def plot_stat_model_graph(stocks_symbols: list, is_machine_learning: int, model_option: str) -> None:
    sectors: list = set_sectors(stocks_symbols)

    closing_prices_table: pd.DataFrame = get_closing_prices_table(is_machine_learning, mode='regular')

    if model_option == "Markowitz":
        curr_stats_models = stats_models.StatsModels(stocks_symbols, sectors, closing_prices_table,
                                                     settings.NUM_POR_SIMULATION, settings.MIN_NUM_POR_SIMULATION,
                                                     1, 1, "Markowitz")
    else:
        curr_stats_models = stats_models.StatsModels(stocks_symbols, sectors, closing_prices_table,
                                                     settings.NUM_POR_SIMULATION, settings.MIN_NUM_POR_SIMULATION,
                                                     1, 1, "Gini")
    df = curr_stats_models.get_df()
    three_best_portfolios: list = api_util.get_best_portfolios(df, model_name=settings.MODEL_NAME[model_option - 1])
    three_best_stocks_weights = api_util.get_three_best_weights(three_best_portfolios)
    three_best_sectors_weights = api_util.get_three_best_sectors_weights(sectors, settings.STOCKS_SYMBOLS,
                                                                         three_best_stocks_weights)

    min_variance_port = three_best_portfolios[0]
    sharpe_portfolio = three_best_portfolios[1]
    max_returns = three_best_portfolios[2]
    max_vols = curr_stats_models.get_max_vols()
    df = curr_stats_models.get_df()

    if model_option == "Markowitz":
        plt_instance = plot_functions.plot_markowitz_graph(sectors, three_best_sectors_weights, min_variance_port,
                                                           sharpe_portfolio, max_returns, max_vols, df)
    else:
        plt_instance = plot_functions.plot_gini_graph(sectors, three_best_sectors_weights, min_variance_port,
                                                      sharpe_portfolio, max_returns, max_vols, df)

    plot_functions.plot(plt_instance)  # TODO plot at site


############################################################################################################
# UTILITY FUNCTIONS
############################################################################################################
# database utility functions:
def get_extended_data_from_db(stocks_symbols: list, is_machine_learning: int, model_option: int, mode: str):
    """
    Get extended data information from DB (CSV tables)
    """
    if mode == 'regular':
        sectors_data = get_json_data(settings.SECTORS_LOCATION)
    else:
        sectors_data = get_json_data('../../' + settings.SECTORS_LOCATION)
    sectors: list = api_util.set_sectors(stocks_symbols, mode)
    closing_prices_table: pd.DataFrame = get_closing_prices_table(is_machine_learning=is_machine_learning, mode=mode)
    df = get_three_level_df_tables(is_machine_learning, settings.MODEL_NAME[model_option - 1], mode=mode)
    three_best_portfolios = api_util.get_best_portfolios(df, model_name=settings.MODEL_NAME[model_option - 1])
    best_stocks_weights_column = api_util.get_best_weights_column(stocks_symbols, sectors, three_best_portfolios,
                                                                  closing_prices_table.pct_change())
    three_best_stocks_weights = api_util.get_three_best_weights(three_best_portfolios)
    three_best_sectors_weights = api_util.get_three_best_sectors_weights(sectors, settings.STOCKS_SYMBOLS,
                                                                         three_best_stocks_weights)
    pct_change_table: pd = closing_prices_table.pct_change()
    yields: list = update_pct_change_table(best_stocks_weights_column, pct_change_table)

    return sectors_data, sectors, closing_prices_table, three_best_portfolios, three_best_sectors_weights, \
        pct_change_table, yields


# Tables according to stocks symbols
def get_closing_prices_table(is_machine_learning: int, mode: str) -> pd.DataFrame:
    if is_machine_learning == 1:
        if mode == 'regular':
            closing_prices_table = pd.read_csv(
                settings.MACHINE_LEARNING_LOCATION + 'closing_prices.csv', index_col=0
            )
        else:
            closing_prices_table = pd.read_csv(
                '../../' + settings.MACHINE_LEARNING_LOCATION + 'closing_prices.csv', index_col=0
            )
    else:
        if mode == 'regular':
            closing_prices_table = pd.read_csv(
                settings.NON_MACHINE_LEARNING_LOCATION + 'closing_prices.csv', index_col=0
            )
        else:
            closing_prices_table = pd.read_csv(
                '../../' + settings.NON_MACHINE_LEARNING_LOCATION + 'closing_prices.csv', index_col=0
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


def get_all_users() -> list:
    """
    Get all users with their portfolios details from json file
    """
    json_data = get_json_data(settings.USERS_JSON_NAME)
    num_of_user = len(json_data['usersList'])
    users_data = json_data['usersList']
    users: list = [] * num_of_user
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
    is_machine_learning = user_data['is_machine_learning']
    selected_model = user_data['selectedModel']
    level_of_risk = user_data['levelOfRisk']
    stocks_symbols = user_data['stocks_symbols']
    stocks_weights = user_data['stocksWeights']
    annual_returns = user_data['annualReturns']
    annual_volatility = user_data['annualVolatility']
    annual_sharpe = user_data['annualSharpe']
    sectors_data = get_json_data("backend_api/api/resources/sectors")  # universal from file

    closing_prices_table: pd.DataFrame = get_closing_prices_table(int(is_machine_learning), mode='regular')
    user_portfolio: portfolio.Portfolio = portfolio.Portfolio(level_of_risk, starting_investment_amount, stocks_symbols,
                                                              sectors_data, selected_model, is_machine_learning)
    pct_change_table: pd = closing_prices_table.pct_change()
    pct_change_table.dropna(inplace=True)
    weighted_sum = np.dot(stocks_weights, pct_change_table.T)
    pct_change_table["weighted_sum_" + str(level_of_risk)] = weighted_sum
    yield_column: str = "yield_" + str(level_of_risk)
    pct_change_table[yield_column] = weighted_sum
    pct_change_table[yield_column] = makes_yield_column(pct_change_table[yield_column], weighted_sum)
    user_portfolio.update_stocks_data(closing_prices_table, pct_change_table, stocks_weights, annual_returns,
                                      annual_volatility, annual_sharpe)
    curr_user = user.User(user_name, user_portfolio)

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


def get_data_from_form(three_best_portfolios: list, three_best_sectors_weights, sectors: list, yields: list,
                       pct_change_table, mode: str) -> int:
    count = 0

    # question 1
    string_to_show = "for how many years do you want to invest?\n" + "0-1 - 1\n""1-3 - 2\n""3-100 - 3\n"
    count += get_score_by_answer_from_user(string_to_show)

    # question 2
    string_to_show = "Which distribution do you prefer?\nlow risk - 1, medium risk - 2, high risk - 3 ?\n"
    plot_distribution_of_portfolio(yields, mode=mode)
    count += get_score_by_answer_from_user(string_to_show)

    # question 3
    string_to_show = "Which graph do you prefer?\nsoftest - 1, sharpest - 2, max return - 3 ?\n"
    plot_three_portfolios_graph(three_best_portfolios, three_best_sectors_weights, sectors, pct_change_table, mode=mode)
    count += get_score_by_answer_from_user(string_to_show)

    return get_level_of_risk_by_score(count)


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
    with open("api/resources/" + last_element + ".json", "w") as f:
        json.dump(json_obj, f)  # Use the `dump()` function to write the JSON data to the file


# api utility functions
def set_sectors(stocks_symbols):
    return api_util.set_sectors(stocks_symbols)


def makes_yield_column(_yield, weighted_sum_column):
    return api_util.makes_yield_column(_yield, weighted_sum_column)


def price_forecast(df, record_percentage_to_predict, is_data_from_tase):
    return api_util.price_forecast(df, record_percentage_to_predict, is_data_from_tase)


def get_json_data(name):
    return api_util.get_json_data(name)


def get_from_and_to_date(num_of_years):  # TODO FIX RETURN TUPLE
    return api_util.get_from_and_to_dates(num_of_years)


# plot functions
def plot_three_portfolios_graph(three_best_portfolios: list, three_best_sectors_weights, sectors: list,
                                pct_change_table, mode: str) -> None:
    min_variance_port = three_best_portfolios[0]
    sharpe_portfolio = three_best_portfolios[1]
    max_returns = three_best_portfolios[2]
    plt_instance = plot_functions.plot_three_portfolios_graph(min_variance_port, sharpe_portfolio, max_returns,
                                                              three_best_sectors_weights, sectors, pct_change_table)
    if mode == 'regular':
        plot_functions.save_graphs(plt_instance, STATIC_GRAPH_FILES_LOCATION + 'three_portfolios.png')
    else:
        plot_functions.save_graphs(plt_instance, '../../' + STATIC_GRAPH_FILES_LOCATION + 'three_portfolios.png')


def plot_distribution_of_stocks(stock_names, pct_change_table) -> None:
    plt_instance = plot_functions.plot_distribution_of_stocks(stock_names, pct_change_table)
    plot_functions.plot(plt_instance)  # TODO plot at site


def plot_distribution_of_portfolio(distribution_graph, mode: str) -> None:
    plt_instance = plot_functions.plot_distribution_of_portfolio(distribution_graph)
    if mode == 'regular':
        plot_functions.save_graphs(plt_instance, STATIC_GRAPH_FILES_LOCATION + 'distribution_graph.png')
    else:
        plot_functions.save_graphs(plt_instance, '../../' + STATIC_GRAPH_FILES_LOCATION + 'distribution_graph.png')


def plotbb_strategy_portfolio(pct_change_table, new_portfolio) -> None:
    plt_instance = plot_functions.plotbb_strategy_portfolio(pct_change_table, new_portfolio)
    plot_functions.plot(plt_instance)  # TODO plot at site


def plot_price_forecast(stock_symbol, df, is_data_got_from_tase) -> None:
    plt_instance = plot_functions.plot_price_forecast(stock_symbol, df, is_data_got_from_tase)
    plot_functions.plot(plt_instance)  # TODO plot at site


def get_score_by_answer_from_user(string_to_show: str) -> int:
    return console_handler.get_score_by_answer_from_user(string_to_show)


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


def get_investment_amount():
    return console_handler.get_investment_amount()
