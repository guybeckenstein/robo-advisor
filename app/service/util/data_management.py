import datetime
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from investment.models import Investment
from service.config import settings
from service.impl.portfolio import Portfolio
from service.impl.sector import Sector
from service.impl.stats_models import StatsModels
from service.impl.user import User
from service.util import helpers, console_handler, stocks_weights_table
from service.util.helpers import Analyze
from service.util.graph import image_methods as graph_image_methods
from service.util.graph import plot_methods as graph_plot_methods
from service.util.pillow import plot_methods as pillow_plot_methods
import os
# django imports
import django
from django.db.models import QuerySet

from service.util import draw_table

# Set up Django settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "robo_advisor_project.settings")
django.setup()
from accounts.models import InvestorUser


######################################################################################
# update dataset tables
def update_all_tables(num_of_years_history, is_daily_running=True):  # build DB for withdraw
    today = datetime.date.today()
    formatted_date = today.strftime("%Y-%m-%d")
    models_data: dict[dict, list, list, list, list] = helpers.get_collection_json_data()
    for i in range(1, len(models_data)):
        curr_collection = models_data[str(i)][0]
        stocks_symbols = curr_collection['stocksSymbols']
        path = settings.BASIC_STOCK_COLLECTION_REPOSITORY_DIR + str(str(i)) + '/'  # where to save the datasets
        update_closing_prices_tables(formatted_date, stocks_symbols, num_of_years_history, path, is_daily_running)
        update_data_frame_tables(formatted_date, curr_collection, path, models_data, str(i),
                                 is_daily_running)


def update_closing_prices_tables(formatted_date_today, stocks_symbols, num_of_years_history, path, is_daily_running):
    with open(path + "lastUpdatedClosingPrice.txt", "r") as file:
        last_updated_date_closing_prices = file.read().strip()

    if last_updated_date_closing_prices != formatted_date_today or not is_daily_running:
        helpers.convert_data_to_tables(path, settings.CLOSING_PRICES_FILE_NAME,
                                       stocks_symbols,
                                       num_of_years_history, save_to_csv=True)

        with open(path + "lastUpdatedClosingPrice.txt", "w") as file:
            file.write(formatted_date_today)


def update_data_frame_tables(formatted_date_today, collection_json_data, path,
                             models_data: dict[dict, list, list, list, list], collection_num,
                             is_daily_running: bool = True):
    stocks_symbols = collection_json_data['stocksSymbols']

    with open(path + "lastUpdatedDftables.txt", "r") as file:
        last_updated_df_tables = file.read().strip()
    if last_updated_df_tables != formatted_date_today or not is_daily_running:
        sectors: list[Sector] = helpers.set_sectors(stocks_symbols)
        closing_prices_table = get_closing_prices_table(path)
        pct_change_table = closing_prices_table.pct_change()

        # Without machine learning - Markowitz, Gini, With machine learning - Markowitz, Gini
        options_list: list[tuple[int, str]] = [(0, 'Markowitz'), (0, 'Gini'), (1, 'Markowitz'), (1, 'Gini')]
        for i, machine_model_tuple in enumerate(options_list):
            if i == 2:  # change pct_change_table due to machine learning
                pct_change_table, annual_return, excepted_returns = helpers.update_daily_change_with_machine_learning(
                    pct_change_table, closing_prices_table.index, models_data
                )

            update_three_level_data_frame_tables(
                machine_learning_opt=machine_model_tuple[0], model_name=machine_model_tuple[1],
                stocks_symbols=stocks_symbols,
                sectors_list=sectors, closing_prices_table=closing_prices_table, pct_change_table=pct_change_table,
                path=path, models_data=models_data, collection_num=collection_num
            )

        with open(path + "lastUpdatedDftables.txt", "w") as file:
            file.write(formatted_date_today)


def update_three_level_data_frame_tables(machine_learning_opt, model_name, stocks_symbols, sectors_list,
                                         closing_prices_table, pct_change_table, path, models_data, collection_num):
    collection_data = models_data[collection_num][0]
    limit_percent_medium_risk_stocks: float = collection_data['LIMIT_PERCENT_MEDIUM_RISK_STOCKS']
    limit_percent_medium_risk_commodity: float = collection_data['LIMIT_PERCENT_MEDIUM_RISK_COMMODITY']
    limit_percent_low_risk_stocks: float = collection_data['LIMIT_PERCENT_LOW_RISK_STOCKS']
    limit_percent_low_risk_commodity: float = collection_data['LIMIT_PERCENT_LOW_RISK_COMMODITY']

    limit_percent_stocks_list: list[float] = [limit_percent_low_risk_stocks,
                                              limit_percent_medium_risk_stocks, 1]

    limit_percent_commodity_list: list[float] = [limit_percent_low_risk_commodity,
                                                 limit_percent_medium_risk_commodity, 1]
    #  low, medium, high
    for i, level_of_risk in enumerate(settings.LEVEL_OF_RISK_LIST):
        update_specific_data_frame_table(is_machine_learning=machine_learning_opt, model_name=model_name,
                                         stocks_symbols=stocks_symbols,
                                         sectors=sectors_list, risk_level=level_of_risk,
                                         max_percent_commodity=limit_percent_commodity_list[i],
                                         max_percent_stocks=limit_percent_stocks_list[i],
                                         closing_prices_table=closing_prices_table, pct_change_table=pct_change_table,
                                         path=path, models_data=models_data)


def update_specific_data_frame_table(is_machine_learning, model_name, stocks_symbols, sectors, risk_level,
                                     max_percent_commodity, max_percent_stocks, closing_prices_table, pct_change_table,
                                     path, models_data):
    num_por_simulation: int = int(models_data["models_data"]['num_por_simulation'])
    min_num_por_simulation: int = int(models_data["models_data"]['min_num_por_simulation'])
    gini_v_value: float = float(models_data["models_data"]['gini_v_value'])

    if is_machine_learning:
        locationForSaving = path + settings.MACHINE_LEARNING_LOCATION
    else:
        locationForSaving = path + settings.NON_MACHINE_LEARNING_LOCATION

    if max_percent_commodity <= 0:
        stock_sectors = helpers.setStockSectors(stocks_symbols, sectors)
        filtered_stocks = []
        for i in range(len(stock_sectors)):
            if stock_sectors[i] != "US commodity indexes":
                filtered_stocks.append(stocks_symbols[i])
            else:
                closing_prices_table = closing_prices_table.drop(stocks_symbols[i], axis=1)
                pct_change_table = pct_change_table.drop(stocks_symbols[i], axis=1)
        stocks_symbols = filtered_stocks

    stats_models: StatsModels = StatsModels(
        _stocks_symbols=stocks_symbols,
        _sectors=sectors,
        _model_name=model_name,
        _gini_v_value=gini_v_value,
    )
    stats_models.check_model_name_and_get_optimal_portfolio_as_dataframe(
        num_por_simulation=num_por_simulation,
        min_num_por_simulation=min_num_por_simulation,
        pct_change_table=pct_change_table,
        max_percent_commodity=max_percent_commodity,
        max_percent_stocks=max_percent_stocks,
    )
    df: pd.DataFrame = stats_models.df
    df.to_csv(locationForSaving + model_name + '_df_' + risk_level + '.csv')
    print(f'Updated DataFrame -> (ML - {is_machine_learning}; Model Name - {model_name}; Risk Level - {risk_level})')


##################################################################
# operations
def create_new_user_portfolio(stocks_symbols: list, investment_amount: int, is_machine_learning: int,
                              stat_model_name: str, risk_level: int, extended_data_from_db: tuple) -> Portfolio:
    __, sectors, closing_prices_table, three_best_portfolios, _, pct_change_table, _ = extended_data_from_db

    final_portfolio = three_best_portfolios[risk_level - 1]
    if settings.LEVEL_OF_RISK_LIST[risk_level - 1] == "low":
        # drop from stocks_symbols the stocks that are in US commodity indexes sector
        stocks_symbols = helpers.drop_stocks_from_specific_sector(
            stocks_symbols, helpers.set_stock_sectors(stocks_symbols, sectors), sector_name="US commodity indexes"
        )

    portfolio = Portfolio(
        stocks_symbols, sectors, risk_level, investment_amount, stat_model_name, is_machine_learning
    )

    portfolio.update_stocks_data(
        closing_prices_table, pct_change_table, final_portfolio.iloc[0][3:],
        final_portfolio.iloc[0][0], final_portfolio.iloc[0][1], final_portfolio.iloc[0][2]
    )

    return portfolio


# investment functions
# site mode
def changing_portfolio_investments_treatment_web(investor_user: InvestorUser, portfolio: Portfolio,
                                                 investments: QuerySet[Investment]) -> None:
    try:
        if len(investments) > 0:
            total_profit: float = portfolio.calculate_total_profit_according_to_dates_dates(investments)
            capital_investments = get_total_capital_investments_web(
                investments)  # Sums all prior ACTIVE & USER investments
            investor_user.total_profit += math.floor(total_profit)
            Investment.objects.create(
                investor_user=investor_user,
                amount=investor_user.total_profit + capital_investments,
                mode=Investment.Mode.ROBOT
            )
            investor_user.save()
        else:
            raise ValueError('User does not have prior investments, therefore this action won\'t affect it')
    except ValueError:
        pass


# console mode
def get_investment_format(investment_amount, entered_as_an_automatic_investment):
    purchase_date = datetime.datetime.now().strftime("%Y-%m-%d")
    is_it_active = True
    new_investment = {
        "amount": investment_amount,
        "date": purchase_date,
        "status": is_it_active,
        "automatic_investment": entered_as_an_automatic_investment
    }

    return new_investment


def add_new_investment(user_id, investment_amount, entered_as_an_automatic_investment=False,
                       investments: list = []) -> dict:
    if investment_amount < 0:
        return None
    new_investment = get_investment_format(investment_amount, entered_as_an_automatic_investment)

    try:
        investments = get_user_investments_from_json_file(user_id)  # from json file
    except KeyError:
        investments = []
    except ValueError:
        investments = []
    except AttributeError:
        investments = []
    investments.append(new_investment)

    # save the new investment to the db  (json file)
    save_investment_to_json_File(user_id, investments)

    return new_investment, investments


def changing_portfolio_investments_treatment_console(selected_user: User, investments: list) -> None:
    user_portfolio = selected_user.portfolio
    if len(investments) > 0:
        total_profit: float = user_portfolio.calculate_total_profit_according_to_dates_dates(investments)
        capital_investments = get_total_capital_investments_console(investments=investments)
        for i, investment in enumerate(investments):
            if investment["status"]:
                investment["status"] = False
                investments[i] = investment
        add_new_investment(selected_user.name, (total_profit + capital_investments),
                           entered_as_an_automatic_investment=True, investments=investments)


############################################################################################################
# UTILITY FUNCTIONS
############################################################################################################
# database utility functions:
def get_extended_data_from_db(stocks_symbols: list, is_machine_learning: int, model_option: int,
                              stocks_collection_number):
    """
    Get extended data information from DB (CSV tables)
    """

    path = settings.BASIC_STOCK_COLLECTION_REPOSITORY_DIR + stocks_collection_number + '/'
    sectors_data = get_json_data(settings.SECTORS_JSON_NAME)
    sectors: list[Sector] = helpers.set_sectors(stocks_symbols)
    closing_prices_table: pd.DataFrame = get_closing_prices_table(path)
    df = get_three_levels_df_tables(is_machine_learning, settings.MODEL_NAME[model_option], path)
    three_best_portfolios = helpers.get_best_portfolios(df, model_name=settings.MODEL_NAME[model_option])
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
def get_closing_prices_table(path) -> pd.DataFrame:
    closing_prices_table: pd.DataFrame = pd.read_csv(filepath_or_buffer=f'{path}closing_prices.csv', index_col=0)
    # Check if there's a key with a numeric value in the table
    numeric_keys: list[str] = [key for key in closing_prices_table.keys() if key.strip().isnumeric()]
    if len(numeric_keys) > 0:
        closing_prices_table = closing_prices_table.iloc[1:]
    else:
        closing_prices_table = closing_prices_table.iloc[2:]
    closing_prices_table = closing_prices_table.apply(pd.to_numeric, errors='coerce')

    return closing_prices_table


def get_three_levels_df_tables(is_machine_learning: int, model_name, collection_path: str) -> list[pd.DataFrame]:
    """
    Get the three level df tables according to machine learning option and model name
    """
    three_levels_df_tables: list[pd.DataFrame] = [
        get_df_table(is_machine_learning, model_name, risk, collection_path) for risk in settings.LEVEL_OF_RISK_LIST
    ]

    return three_levels_df_tables


def get_df_table(is_machine_learning: int, model_name, level_of_risk: str, collection_path: str) -> pd.DataFrame:
    """
    get specific df table from csv file according to machine learning option, model name and level of risk
    """
    if is_machine_learning:
        collection_path += settings.MACHINE_LEARNING_LOCATION
    else:
        collection_path += settings.NON_MACHINE_LEARNING_LOCATION
    df: pd.DataFrame = pd.read_csv(f'{collection_path}{model_name}_df_{level_of_risk}.csv')
    df = df.iloc[:, 1:]
    df = df.apply(pd.to_numeric, errors='coerce')
    return df


def get_stocks_symbols_from_collection(stocks_collection_number) -> list:
    """
    Get all stocks symbols from json file
    """
    models_data: dict[dict, list, list, list, list] = helpers.get_collection_json_data()
    stocks_symbols = models_data[str(stocks_collection_number)][0]['stocksSymbols']
    return stocks_symbols


def get_models_data_from_collections_file():
    models_data: dict[dict, list, list, list, list] = helpers.get_collection_json_data()
    return models_data["models_data"]


def get_total_capital_investments_web(investments: QuerySet[Investment]) -> float:
    """
    Returning the investment amount that the useer invested
    """
    total_capital: int = 0
    for investment in investments:
        if investment.mode == Investment.Mode.USER:
            total_capital += investment.amount
        else:
            continue
    return total_capital


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


def get_level_of_risk_by_score(count: int) -> int:
    if count <= 4:
        return 1
    elif count <= 7:
        return 2
    elif count <= 9:
        return 3
    else:
        raise ValueError


# impl utility functions
def set_sectors(stocks_symbols: list[object]) -> list[Sector]:
    return helpers.set_sectors(stocks_symbols)


def makes_yield_column(_yield, weighted_sum_column):
    return helpers.makes_yield_column(_yield, weighted_sum_column)


def get_json_data(name):
    return helpers.get_json_data(name)


def get_from_and_to_date(num_of_years) -> tuple[str, str]:
    return helpers.get_from_and_to_dates(num_of_years)


def update_models_data_settings(
        num_por_simulation: int, min_num_por_simulation: int, record_percent_to_predict: float,
        test_size_machine_learning: float, selected_ml_model_for_build: int, gini_v_value: int
) -> None:
    # Read the JSON file
    fully_qualified_file_name: str = f'{settings.STOCKS_JSON_NAME}.json'
    with open(file=fully_qualified_file_name, mode='r') as json_file:
        data = json.load(json_file)

    # Update the nested dictionary
    data['collections']['models_data']['num_por_simulation'] = num_por_simulation
    data['collections']['models_data']['min_num_por_simulation'] = min_num_por_simulation
    data['collections']['models_data']['record_percent_to_predict'] = record_percent_to_predict
    data['collections']['models_data']['test_size_machine_learning'] = test_size_machine_learning
    data['collections']['models_data']['selected_ml_model_for_build'] = selected_ml_model_for_build
    data['collections']['models_data']['gini_v_value'] = gini_v_value

    # Write the updated data back to the JSON file
    with open(file=fully_qualified_file_name, mode='w') as json_file:
        json.dump(data, json_file, indent=4)


# graph_plot_methods functions
def plot_three_portfolios_graph(three_best_portfolios: list, three_best_sectors_weights, sectors: list,
                                pct_change_table, sub_folder: str = '00/'):
    plt_instance_three_graph = graph_plot_methods.three_portfolios_graph(
        max_returns_portfolio=three_best_portfolios[2],
        sharpe_portfolio=three_best_portfolios[1],
        min_variance_portfolio=three_best_portfolios[0],
        three_best_sectors_weights=three_best_sectors_weights,
        sectors=sectors,
        pct_change_table=pct_change_table
    )
    fully_qualified_name = settings.GRAPH_IMAGES + sub_folder + 'three_portfolios'
    graph_image_methods.save_graph(plt_instance_three_graph, fully_qualified_name)

    return plt_instance_three_graph


def plot_distribution_of_portfolio(distribution_graph, sub_folder: str = '00/') -> plt:
    plt_instance = graph_plot_methods.portfolio_distribution(distribution_graph)
    fully_qualified_name = settings.GRAPH_IMAGES + sub_folder + 'distribution_graph'
    graph_image_methods.save_graph(plt_instance, fully_qualified_name)

    return plt_instance


def plot_stat_model_graph(stocks_symbols: list[object], is_machine_learning: int, model_name: str,
                          num_of_years_history, closing_prices_table_path: str, sub_folder: str) -> None:
    sectors: list = set_sectors(stocks_symbols)

    if type(model_name) == int:
        model_name = settings.MODEL_NAME[model_name]

    three_levels_df_tables: list[pd.DataFrame] = [
        get_df_table(is_machine_learning, model_name, risk, closing_prices_table_path)
        for risk in settings.LEVEL_OF_RISK_LIST
    ]

    three_best_portfolios = helpers.get_best_portfolios(df=[three_levels_df_tables[0],
                                                            three_levels_df_tables[1],
                                                            three_levels_df_tables[2]], model_name=model_name)
    three_best_stocks_weights = helpers.get_three_best_weights(three_best_portfolios)
    three_best_sectors_weights = helpers.get_three_best_sectors_weights(sectors, three_best_stocks_weights)

    union_df = pd.concat([three_levels_df_tables[0], three_levels_df_tables[1], three_levels_df_tables[2]])

    if model_name == "Markowitz":
        plt_instance = graph_plot_methods.markowitz_graph(
            sectors=sectors, three_best_sectors_weights=three_best_sectors_weights,
            min_variance_portfolio=three_best_portfolios[0], sharpe_portfolio=three_best_portfolios[1],
            max_returns_portfolio=three_best_portfolios[2],
            df=union_df
        )
    else:
        plt_instance = graph_plot_methods.gini_graph(
            sectors=sectors, three_best_sectors_weights=three_best_sectors_weights,
            min_variance_portfolio=three_best_portfolios[0], sharpe_portfolio=three_best_portfolios[1],
            max_portfolios_annual_portfolio=three_best_portfolios[2], df=union_df
        )

    graph_image_methods.save_graph(plt_instance,
                                   f'{settings.GRAPH_IMAGES}{sub_folder}all_options')


def plot_research_graphs(data_tuple_list: list, intersection_data, sector_name: str, labels: list[str]) -> None:
    prefix_str = "Top Stocks"
    path = f'{settings.RESEARCH_IMAGES}{prefix_str} {sector_name}'
    draw_table._draw_research_table(path, data_tuple_list, intersection_data, labels)


def save_user_portfolio(user: User) -> None:
    # Creating directories
    curr_user_directory = settings.USER_IMAGES + user.id
    models_data = get_models_data_from_collections_file()
    record_percent_to_predict = models_data['record_percent_to_predict']
    test_size_machine_learning = models_data['test_size_machine_learning']

    try:
        os.mkdir(settings.USER_IMAGES)  # Creates 'static/img/user' folder
    except FileExistsError:  # Ignore the exception
        pass

    try:
        os.mkdir(curr_user_directory)  # Creates 'static/img/user/<USER_ID>' folder
    except FileExistsError:  # Ignore the exception
        pass

    # get data from user
    portfolio: Portfolio = user.portfolio
    stocks_symbols: list[str] = portfolio.stocks_symbols

    # pie chart of sectors.json.json & sectors.json.json weights
    plt_sectors_component = graph_plot_methods.sectors_component(
        weights=portfolio.get_sectors_weights(), names=portfolio.get_sectors_names()
    )

    graph_image_methods.save_graph(plt_sectors_component, file_name=f'{curr_user_directory}/sectors_weights_graph')
    plt.clf()
    plt.cla()
    plt.close()

    # Table of stocks weights
    header_text: list[str, str, str] = ['Stock', 'Weight', 'Description']
    draw_table.draw_all_and_save_as_png(
        file_name=f'{curr_user_directory}/stocks_weights_graph',
        symbols=stocks_symbols,
        values=portfolio.stocks_weights,
        descriptions=helpers.get_stocks_descriptions(stocks_symbols)[1:],
        header_text=header_text
    )

    # Total yield graph with sectors.json.json weights
    table: pd.DataFrame = portfolio.pct_change_table
    table['yield__selected_percent'] = (table["yield_selected"] - 1) * 100
    analyze: Analyze = Analyze(
        returns_stock=table['yield__selected_percent'],
        table_index=table.index,
        record_percent_to_predict=float(record_percent_to_predict),
        is_closing_prices_mode=True
    )
    df, annual_return_with_forecast, excepted_returns = analyze.linear_regression_model(test_size_machine_learning)
    df['yield__selected_percent'] = df['Col']
    df['yield__selected_percent_forecast'] = df["Forecast"]

    stats_details_tuple: tuple[float, float, float, float, float] = portfolio.get_portfolio_stats()
    annual_returns, volatility, sharpe, max_loss, total_change = stats_details_tuple
    plt_yield_graph = graph_plot_methods.investment_portfolio_estimated_yield(
        df=df, annual_returns=annual_returns, volatility=volatility, sharpe=sharpe, max_loss=max_loss,
        total_change=total_change, sectors=portfolio.sectors, excepted_returns=excepted_returns
    )
    graph_image_methods.save_graph(plt_yield_graph, file_name=f'{curr_user_directory}/estimated_yield_graph')
    plt.clf()
    plt.cla()
    plt.close()


def plot_investments_history(login_id, investments_list) -> plt:  # from json file
    file_name = f'{settings.USER_IMAGES}{login_id}/investments history'
    header_text: list[str, str, str] = ['id', 'amount', 'Description']
    descriptions: list = []
    for investment in investments_list:
        purchase_date = investment["date"]
        status = "Active" if investment["status"] else "Inactive"
        mode = "User" if investment["automatic_investment"] else "Robot"
        description = f'Date:{purchase_date}, Status:{status}, Model:{mode}'
        descriptions.append(description)

    draw_table.draw_all_and_save_as_png(
        file_name=file_name,
        symbols=[f'{i + 1}' for i in range(len(investments_list))],
        values=[investment["amount"] for investment in investments_list],
        descriptions=descriptions,
        header_text=header_text,
        percent_mode=False
    )


def view_investment_report(login_id, investment_amount, stocks_weights, stocks_symbols) -> plt:
    file_name = f'{settings.USER_IMAGES}{login_id}/investment report'
    # Table of stocks weights
    header_text: list[str, str, str] = ['Stock', 'Amount to invest', 'Description']
    descriptions: list = []
    values: list = []
    ils_to_usd: float = helpers.currency_exchange(from_currency="USD", to_currency="ILS")
    for i, stock in enumerate(stocks_symbols):
        if type(stock) == int or stock.isnumeric():
            currency = f'{settings.CURRENCY_LIST[0]}'
            values.append(stocks_weights[i] * investment_amount * ils_to_usd)
        else:
            currency = f'{settings.CURRENCY_LIST[1]}'
            values.append(stocks_weights[i] * investment_amount)
        description = f'{currency}, {helpers.get_stocks_descriptions([stock])[1:][0]}'
        descriptions.append(description)
    draw_table.draw_all_and_save_as_png(
        file_name=file_name,
        symbols=stocks_symbols,
        values=values,
        descriptions=descriptions,
        header_text=header_text,
        percent_mode=False
    )


def plot_image(file_name) -> None:
    pillow_plot_methods.plot_image(file_name)


def get_stocks_from_json_file() -> dict[list]:
    models_data: dict[dict, list, list, list, list] = helpers.get_collection_json_data()
    stocks: dict[list] = {}
    for i in range(1, len(models_data)):
        stocks_symbols_list = models_data[str(i)][0]['stocksSymbols']
        stocks_description_list = helpers.get_stocks_descriptions(stocks_symbols_list, is_reverse_mode=False)[1:]
        stocks[str(i)] = [stocks_symbols_list, stocks_description_list]
    return stocks


def get_styled_stocks_symbols_data(stocks_symbols_data) -> dict[list]:
    styled_stocks_symbols_data: dict[list] = dict()
    for key, value in stocks_symbols_data.items():
        styled_value: list[str] = list()
        for symbol, description in zip(value[0], value[1]):
            styled_value.append(f'{symbol} -> {description}')
        styled_stocks_symbols_data[key] = styled_value
    return styled_stocks_symbols_data


def get_stocks_symbols_from_json_file(collection_number: int) -> list[str]:
    models_data: dict[dict, list, list, list, list] = helpers.get_collection_json_data()
    collection: dict = models_data[str(collection_number)][0]
    stocks_symbols: list[str] = collection['stocksSymbols']
    return stocks_symbols


def is_today_date_change_from_last_updated_df(collection_number: int) -> bool:
    __path = settings.BASIC_STOCK_COLLECTION_REPOSITORY_DIR + collection_number + '/'
    today = datetime.date.today()
    formatted_date = today.strftime("%Y-%m-%d")
    with open(__path + "lastUpdatedClosingPrice.txt", "r") as file:
        lastUpdatedDateClosingPrices = file.read().strip()

    if lastUpdatedDateClosingPrices != formatted_date:
        return True


# json file DB functions:
def get_user_from_db(user_id: int, user_name: str):  # users.json file
    """
    Get specific user by his name with his portfolio details from json file
    """
    json_data = get_json_data(settings.USERS_JSON_NAME)
    if user_name not in json_data['usersList']:
        print("User not found")
        return None
    user_data = json_data['usersList'][user_name][0]
    total_investment_amount = user_data['startingInvestmentAmount']
    is_machine_learning = user_data['machineLearningOpt']
    stat_model_name = user_data['statModelName']
    risk_level = user_data['levelOfRisk']
    stocks_symbols = user_data['stocksSymbols']
    stocks_weights = user_data['stocksWeights']
    annual_returns = user_data['annualReturns']
    annual_volatility = user_data['annualVolatility']
    annual_sharpe = user_data['annualSharpe']
    try:
        stocks_collection_number = user_data['stocksCollectionNumber']
    except KeyError:  # Default value
        stocks_collection_number = "1"
    sectors: list[Sector] = helpers.set_sectors(stocks_symbols)

    path = f'{settings.BASIC_STOCK_COLLECTION_REPOSITORY_DIR}{stocks_collection_number}/'

    closing_prices_table: pd.DataFrame = get_closing_prices_table(path=path)
    portfolio: Portfolio = Portfolio(
        stocks_symbols, sectors, risk_level, total_investment_amount, stat_model_name, is_machine_learning,
    )
    pct_change_table: pd = closing_prices_table.pct_change()
    pct_change_table.dropna(inplace=True)
    weighted_sum = np.dot(stocks_weights, pct_change_table.T)
    pct_change_table["weighted_sum_" + str(risk_level)] = weighted_sum
    models_data = helpers.get_collection_json_data()
    if is_machine_learning:  # TODO maybe remove
        weighted_sum = helpers.update_daily_change_with_machine_learning(
            [weighted_sum], pct_change_table.index, models_data
        )[0][0]
    yield_column: str = "yield_" + str(risk_level)
    pct_change_table[yield_column] = weighted_sum
    pct_change_table[yield_column] = makes_yield_column(pct_change_table[yield_column], weighted_sum)
    portfolio.update_stocks_data(closing_prices_table, pct_change_table, stocks_weights, annual_returns,
                                 annual_volatility, annual_sharpe)
    user: User = User(_id=user_id, _name=user_name, _portfolio=portfolio)
    save_user_portfolio(user)

    return user


def save_investment_to_json_File(user_id, investments):
    json_data = get_json_data(settings.USERS_JSON_NAME)
    if user_id not in json_data['usersList']:
        print("User not found")
        return None
    # save in DB
    json_data['usersList'][user_id][0]['investments_list'] = investments

    # Write the updated JSON data back to the file
    with open(settings.USERS_JSON_NAME + ".json", 'w') as file:
        json.dump(json_data, file, indent=4)


def get_user_investments_from_json_file(user_id):
    json_data = get_json_data(settings.USERS_JSON_NAME)
    if user_id not in json_data['usersList']:
        # find name by id
        for user_name in json_data['usersList']:
            if json_data['usersList'][user_name][0]['id'] == user_id:
                user_id = user_name
                break
    try:
        investment_list = json_data['usersList'][user_id][0]['investments_list']
    except:
        investment_list = []
    return investment_list


def get_total_capital_investments_console(investments: list) -> float:
    """
    Returning the investment amount that the useer invested
    """
    total_capital = 0
    for investment in investments:
        if not investment["automatic_investment"]:
            total_capital += investment["amount"]
    return total_capital


def get_total_active_investments(investments):
    total_amount = 0

    for investment in investments:
        if investment["status"]:
            total_amount += investment["amount"]

    return total_amount


def get_total_investments_details(selected_user, investments) -> tuple:
    """
return:
    - The amount of capital investments
    - The amount of profit on the current portfolio only
    - The amount of profit in general
    - The sum of all investments including the profit
    """
    total_capital = get_total_capital_investments_console(investments)  # capital investments
    user_portfolio = selected_user.portfolio
    total_portfolio_profit: float = user_portfolio.calculate_total_profit_according_to_dates_dates(investments)
    total_investments_value = get_total_active_investments(investments) + total_portfolio_profit
    total_profit = total_investments_value - total_capital

    return total_capital, total_portfolio_profit, total_profit, total_investments_value


def show_investments_from_json_file(login_name: str):
    investments_list = get_user_investments_from_json_file(login_name)
    if len(investments_list) == 0:
        print("No investments yet")
        return None

    graph_plot_methods.plot_investments_graph(investments_list)
    print("Investments:")
    for investment in investments_list:
        print(investment)

    # TODO - plot graph of investments and send in email


# console functions
def show_main_menu() -> None:
    console_handler.show_main_menu()


def get_menu_choice() -> int:
    return console_handler.get_menu_choice()


def get_name() -> str:
    return console_handler.get_name()


def get_num_of_years_history() -> int:
    return console_handler.get_num_of_years_history()


def get_machine_learning_option() -> int:  # Whether to use machine learning or not
    return console_handler.get_machine_learning_option()


def get_stat_model_option() -> int:
    return console_handler.get_stat_model_option()


def get_machine_learning_model() -> str:  # for forecast graph 4 options
    option: int = console_handler.get_machine_learning_mdoel()
    return settings.MACHINE_LEARNING_MODEL[option - 1]


def get_investment_amount() -> int:
    return console_handler.get_investment_amount()


def get_collection_number() -> str:
    stocks = get_stocks_from_json_file()
    return console_handler.get_collection_number(stocks)


def get_sector_name_from_user() -> str:
    sectors_list = helpers.get_sectors_names_list()
    return sectors_list[console_handler.get_sector_name_from_user(sectors_list) - 1]


def get_basic_data_from_user() -> tuple[int, int, str]:
    is_machine_learning: int = get_machine_learning_option()
    model_option: int = get_stat_model_option()
    stocks_collection_number: str = get_collection_number()

    return is_machine_learning, model_option, stocks_collection_number


def get_level_of_risk_according_to_questionnaire_form_from_console(sub_folder, tables, is_machine_learning,
                                                                   model_option, stocks_symbols,
                                                                   stocks_collection_number) -> int:
    sectors_data, sectors, closing_prices_table, three_best_portfolios, three_best_sectors_weights, \
        pct_change_table, yield_list = tables
    closing_prices_table_path = (settings.BASIC_STOCK_COLLECTION_REPOSITORY_DIR
                                 + stocks_collection_number + '/')
    # question #1
    string_to_show = "for how many years do you want to invest?\n" + "0-1 - 1\n""1-3 - 2\n""3-100 - 3\n"
    first_question_score = get_score_by_answer_from_user(string_to_show)

    # question #2
    string_to_show = "Which distribution do you prefer?\nlow risk - 1, medium risk - 2, high risk - 3 ?\n"
    # display distribution of portfolio graph(matplotlib)
    plot_distribution_of_portfolio(yield_list, sub_folder=sub_folder)
    plot_image(settings.GRAPH_IMAGES + sub_folder + 'distribution_graph.png')
    second_question_score = get_score_by_answer_from_user(string_to_show)

    # question #3
    string_to_show = "Which graph do you prefer?\nsafest - 1, sharpe - 2, max return - 3 ?\n"

    # display 3 best portfolios graph (matplotlib)
    plot_three_portfolios_graph(
        three_best_portfolios, three_best_sectors_weights, sectors, pct_change_table, sub_folder=sub_folder
    )
    plot_image(settings.GRAPH_IMAGES + sub_folder + 'three_portfolios.png')

    # display stat model graph (matplotlib)
    plot_stat_model_graph(
        stocks_symbols=stocks_symbols, is_machine_learning=is_machine_learning,
        model_name=settings.MODEL_NAME[model_option], num_of_years_history=settings.NUM_OF_YEARS_HISTORY,
        closing_prices_table_path=closing_prices_table_path, sub_folder=sub_folder)

    # show result
    plot_image(settings.GRAPH_IMAGES + sub_folder + 'all_options' + '.png')

    third_question_score = get_score_by_answer_from_user(string_to_show)

    # calculate level of risk by sum of score
    sum_of_score = first_question_score + second_question_score + third_question_score
    level_of_risk = get_level_of_risk_by_score(sum_of_score)

    return level_of_risk


def get_score_by_answer_from_user(string_to_show: str) -> int:
    return console_handler.get_score_by_answer_from_user(string_to_show)
