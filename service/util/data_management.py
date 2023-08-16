import csv
import datetime
import json
import math

import numpy as np
import pandas as pd
from typing import Tuple, List
from accounts.models import InvestorUser
from investment.models import Investment
from ..impl.portfolio import Portfolio
from ..impl.stats_models import StatsModels
from ..impl.user import User
from ..config import settings
from . import helpers, console_handler, plot_functions
import os
# django imports
import django
from django.db.models import QuerySet
from django.conf import settings as django_settings

# Set up Django settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "robo_advisor_project.settings")
django.setup()
from accounts.models import InvestorUser


######################################################################################
# update DB tables
def update_all_tables(numOfYearsHistory, is_daily_running=True):  # build DB for withdraw
    today = datetime.date.today()
    formatted_date = today.strftime("%Y-%m-%d")
    collections_json_data = helpers.get_collection_json_data()
    for i in range(1, len(collections_json_data)):
        curr_collection = collections_json_data[str(i)][0]
        stocksSymbols = curr_collection['stocksSymbols']
        __path = settings.BASIC_STOCK_COLLECTION_REPOSITORY_DIR + str(str(i)) + '/'  # where to save the datasets
        update_closing_prices_tables(formatted_date, stocksSymbols, numOfYearsHistory, __path, is_daily_running)
        update_data_frame_tables(formatted_date, curr_collection, __path, collections_json_data, str(i),
                                 is_daily_running)


def update_closing_prices_tables(formatted_date_today, stocksSymbols, numOfYearsHistory, __path, is_daily_running):
    with open(__path + "lastUpdatedClosingPrice.txt", "r") as file:
        lastUpdatedDateClosingPrices = file.read().strip()

    if lastUpdatedDateClosingPrices != formatted_date_today or not is_daily_running:
        helpers.convert_data_to_tables(__path, settings.CLOSING_PRICES_FILE_NAME,
                                       stocksSymbols,
                                       numOfYearsHistory, save_to_csv=True)

        with open(__path + "lastUpdatedClosingPrice.txt", "w") as file:
            file.write(formatted_date_today)


def update_data_frame_tables(formatted_date_today, collection_json_data, __path, models_data, collection_num,
                             is_daily_running=True):
    stocksSymbols = collection_json_data['stocksSymbols']

    with open(__path + "lastUpdatedDftables.txt", "r") as file:
        lastUpdatedDftables = file.read().strip()
    if lastUpdatedDftables != formatted_date_today or not is_daily_running:
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
    num_por_simulation: int = int(models_data["models_data"]['num_por_simulation'])
    min_num_por_simulation: int = int(models_data["models_data"]['min_num_por_simulation'])
    gini_v_value: float = float(models_data["models_data"]['gini_v_value'])

    if is_machine_learning:
        locationForSaving = __path + settings.MACHINE_LEARNING_LOCATION
    else:
        locationForSaving = __path + settings.NON_MACHINE_LEARNING_LOCATION

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

    stats_models = StatsModels(
        stocks_symbols=stocks_symbols,
        sectors=sectors,
        closing_prices_table=closing_prices_table,
        pct_change_table=pct_change_table,
        num_por_simulation=num_por_simulation,
        min_num_por_simulation=min_num_por_simulation,
        max_percent_commodity=max_percent_commodity,
        max_percent_stocks=max_percent_stocks,
        model_name=model_name,
        gini_value=gini_v_value

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
        stocks_symbols = helpers.drop_stocks_from_specific_sector(
            stocks_symbols, helpers.set_stock_sectors(stocks_symbols, sectors), sector_name="US commodity"
        )

    portfolio = Portfolio(
        stocks_symbols=stocks_symbols,
        sectors=sectors,
        risk_level=risk_level,
        total_investment_amount=investment_amount,
        selected_model=model_option,
        is_machine_learning=is_machine_learning
    )

    portfolio.update_stocks_data(
        closing_prices_table, pct_change_table, final_portfolio.iloc[0][3:],
        final_portfolio.iloc[0][0], final_portfolio.iloc[0][1], final_portfolio.iloc[0][2]
    )
    return portfolio


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
                       db_type="django", investments: list = []) -> dict:
    if investment_amount < 0:
        return None
    new_investment = get_investment_format(investment_amount, entered_as_an_automatic_investment)

    if len(investments) > 0:
        try:
            get_investments_from_db(user_id, db_type)
        except:
            investments = []
    investments.append(new_investment)

    # save the new investment to the db

    save_new_investments_to_db(user_id, investments, db_type)

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
                           entered_as_an_automatic_investment=True, db_type="json", investments=investments)


def changing_portfolio_investments_treatment_web(investor_user: InvestorUser, portfolio: Portfolio,
                                                 investments: QuerySet[Investment]) -> None:
    try:
        if len(investments) > 0:
            total_profit: float = portfolio.calculate_total_profit_according_to_dates_dates(investments)
            capital_investments = get_total_capital_investments_web(
                investments)  # Sums all prior ACTIVE & USER investments
            Investment.objects.create(
                investor_user=investor_user,
                amount=math.floor(total_profit) + capital_investments,
                mode=Investment.Mode.ROBOT
            )
            investor_user.total_profit += math.floor(total_profit)
            investor_user.save()
        else:
            raise ValueError('User does not have prior investments, therefore this action won\'t affect it')
    except ValueError:
        pass


############################################################################################################
# UTILITY FUNCTIONS
############################################################################################################
# database utility functions:
def get_extended_data_from_db(stocks_symbols: list, is_machine_learning: int, model_option: int,
                              stocks_collection_number, mode: str):
    """
    Get extended data information from DB (CSV tables)
    """

    __path = settings.BASIC_STOCK_COLLECTION_REPOSITORY_DIR + stocks_collection_number + '/'
    if mode == 'regular':
        sectors_data = get_json_data(settings.SECTORS_JSON_NAME)
    else:
        sectors_data = get_json_data('../../' + settings.SECTORS_JSON_NAME)
    sectors: list = helpers.set_sectors(stocks_symbols, mode)
    closing_prices_table: pd.DataFrame = get_closing_prices_table(__path, mode=mode)
    df = get_three_level_df_tables(is_machine_learning, settings.MODEL_NAME[model_option], __path, mode=mode)
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
def get_closing_prices_table(__path, mode: str) -> pd.DataFrame:
    if mode == 'regular':
        closing_prices_table = pd.read_csv(
            __path + 'closing_prices.csv', index_col=0
        )
    else:
        closing_prices_table = pd.read_csv(
            '../../' + __path + 'closing_prices.csv', index_col=0
        )
        # Check if there's a key with a numeric value in the table
    numeric_keys = [key for key in closing_prices_table.keys() if key.strip().isnumeric()]
    if len(numeric_keys) > 0:
        closing_prices_table = closing_prices_table.iloc[1:]
    else:
        closing_prices_table = closing_prices_table.iloc[2:]
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
    for user_name in users_data.items():
        user_id: int = user_name['id']
        users.append(get_user_from_db(user_id=user_id, user_name=user_name))

    return users


def get_user_from_db(user_id: int, user_name: str):
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
    selected_model = user_data['selectedModel']
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
    sectors = helpers.set_sectors(stocks_symbols)

    __path = settings.BASIC_STOCK_COLLECTION_REPOSITORY_DIR + stocks_collection_number + '/'

    closing_prices_table: pd.DataFrame = get_closing_prices_table(__path=__path, mode='regular')
    portfolio: Portfolio = Portfolio(
        stocks_symbols=stocks_symbols,
        sectors=sectors,
        risk_level=risk_level,
        total_investment_amount=total_investment_amount,
        selected_model=selected_model,
        is_machine_learning=is_machine_learning,
    )
    pct_change_table: pd = closing_prices_table.pct_change()
    pct_change_table.dropna(inplace=True)
    weighted_sum = np.dot(stocks_weights, pct_change_table.T)
    pct_change_table["weighted_sum_" + str(risk_level)] = weighted_sum
    models_data = helpers.get_collection_json_data()
    if is_machine_learning:
        weighted_sum = helpers.update_daily_change_with_machine_learning([weighted_sum],
                                                                         pct_change_table.index,
                                                                         models_data)[0][0]
    yield_column: str = "yield_" + str(risk_level)
    pct_change_table[yield_column] = weighted_sum
    pct_change_table[yield_column] = makes_yield_column(pct_change_table[yield_column], weighted_sum)
    portfolio.update_stocks_data(closing_prices_table, pct_change_table, stocks_weights, annual_returns,
                                 annual_volatility, annual_sharpe)
    curr_user = User(user_id=user_id, name=user_name, portfolio=portfolio)
    save_user_portfolio(curr_user)

    return curr_user


def get_stocks_symbols_from_collection(stocks_collection_number) -> List:
    """
    Get all stocks symbols from json file
    """
    json_data = helpers.get_collection_json_data()
    stocks_symbols = json_data[str(stocks_collection_number)][0]['stocksSymbols']
    return stocks_symbols


def get_models_data_from_collections_file():  # TODO - maybe use from admin
    json_data = helpers.get_collection_json_data()
    return json_data["models_data"]


def find_user_in_list(user_name: str, users: list):
    for curr_user in users:
        if curr_user.getUserName() == user_name:
            return curr_user
    return None


def get_num_of_users_in_db() -> int:
    json_data = get_json_data(settings.USERS_JSON_NAME)
    return len(json_data['usersList'])


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


def get_total_capital_investments_console(investments: list) -> float:
    """
    Returning the investment amount that the useer invested
    """
    total_capital = 0
    for investment in investments:
        if not investment["automatic_investment"]:
            total_capital += investment["amount"]
    return total_capital


def get_total_capital_investments_web(investments: QuerySet[Investment]) -> float:
    """
    Returning the investment amount that the useer invested
    """
    total_capital: int = 0
    for investment in investments:
        if investment.mode == Investment.Mode.USER:
            total_capital += investment.amount
        else:
            break
    return total_capital


def get_total_active_investments(investments):
    total_amount = 0

    for investment in investments:
        if investment["status"]:  # TODO
            total_amount += investment["amount"]

    return total_amount


def get_total_investments_details(selected_user, investments) -> Tuple:
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


def get_user_investments_from_json_file(user_id):
    json_data = get_json_data(settings.USERS_JSON_NAME)
    if user_id not in json_data['usersList']:
        print("User not found")
        return None
    try:
        investment_list = json_data['usersList'][user_id][0]['investments_list']
    except:
        investment_list = []
    return investment_list


def get_investments_from_db(user_id, db_type):
    if db_type == "django":  # TODO
        pass
        investment_id = 1  # Replace with the actual Investment ID
        # investments_list = Investment.objects.get(pk=investment_id)

        # try:
        # Try to get the existing InvestorUser instance
        # investor_user = InvestorUser.objects.get(pk=1)
        # except InvestorUser.DoesNotExist:
        # If the user doesn't exist, create a new instance
        # investor_user = InvestorUser.objects.create()

        # investments = investor_user.investments if investor_user.investments else []


    else:
        investments_list = get_user_investments_from_json_file(user_id)  # from json file

    return investments_list


def save_new_investments_to_db(user_id, investments_list, db_type):
    if db_type == "django":
        pass  # TODO
        # Append the new investment to the list of investments
        # investor_user.investments.append(new_investment)
        # Update the investor_user's investments field
        # investor_user.investments = investments_list
        # Save the updated InvestorUser instance
        # investor_user.save()
    else:
        save_investment_to_json_File(user_id, investments_list)


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
    with open(settings.CONFIG + last_element + ".json", "w") as f:
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
    num_por_simulation: int = models_data["models_data"]['num_por_simulation']
    min_num_por_simulation: int = models_data["models_data"]['min_num_por_simulation']
    gini_v_value: int = models_data["models_data"]['gini_v_value']
    closing_prices_table: pd.DataFrame = get_closing_prices_table(closing_prices_table_path,
                                                                  mode='regular')
    pct_change_table = closing_prices_table.pct_change()
    if num_of_years_history != settings.NUM_OF_YEARS_HISTORY:
        pct_change_table = pct_change_table.tail(num_of_years_history * 252)

    if is_machine_learning == 1:
        pct_change_table, _, _ = helpers.update_daily_change_with_machine_learning(
            pct_change_table, models_data, closing_prices_table.index
        )

    if model_option == "Markowitz":
        model_name = "Markowitz"
    else:
        model_name = "Gini"

    stats_models = StatsModels(
        stocks_symbols=stocks_symbols,
        sectors=sectors,
        closing_prices_table=closing_prices_table,
        pct_change_table=pct_change_table,
        num_por_simulation=num_por_simulation,
        min_num_por_simulation=min_num_por_simulation,
        max_percent_commodity=1,
        max_percent_stocks=1,
        model_name=model_name,
        gini_value=gini_v_value,
    )
    df = stats_models.get_df()
    three_best_portfolios = helpers.get_best_portfolios(df=[df, df, df], model_name=model_option)
    three_best_stocks_weights = helpers.get_three_best_weights(three_best_portfolios)
    three_best_sectors_weights = helpers.get_three_best_sectors_weights(sectors, three_best_stocks_weights)
    min_variance_portfolio = three_best_portfolios[0]
    sharpe_portfolio = three_best_portfolios[1]
    max_returns = three_best_portfolios[2]
    max_vols_portfolio = stats_models.get_max_vols()
    df = stats_models.get_df()

    if model_option == "Markowitz":
        plt_instance = plot_functions.plot_markowitz_graph(
            sectors=sectors, three_best_sectors_weights=three_best_sectors_weights,
            min_variance_portfolio=min_variance_portfolio, sharpe_portfolio=sharpe_portfolio,
            max_returns=max_returns, max_vols_portfolio=max_vols_portfolio, df=df
        )
    else:
        plt_instance = plot_functions.plot_gini_graph(
            sectors=sectors, three_best_sectors_weights=three_best_sectors_weights,
            min_variance_portfolio=min_variance_portfolio, sharpe_portfolio=sharpe_portfolio,
            max_portfolios_annual=max_returns, max_ginis=max_vols_portfolio, df=df
        )

    plot_functions.save_graphs(plt_instance, f'{settings.GRAPH_IMAGES}{model_option}_all_option')  # TODO plot at site


def plot_research_graphs(data_tuple_list: list, intersection_data_list: list):
    path = settings.RESEARCH_IMAGES
    research_plt = plot_functions.plot_research_graphs(path, data_tuple_list, intersection_data_list)
    # plot_functions.save_graphs(research_plt, path)


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
    stocks_symbols: List[str] = portfolio.stocks_symbols

    # pie chart of sectors & sectors weights
    plt_sectors_component = plot_functions.plot_sectors_component(
        user_name=user.name, weights=portfolio.get_sectors_weights(), names=portfolio.get_sectors_names()
    )

    plot_functions.save_graphs(plt_sectors_component, file_name=f'{curr_user_directory}/sectors_component')

    # pie chart of stocks & stocks weights , TODO: show as tables instead of pie chart
    plt_stocks_component = plot_functions.plot_portfolio_component_stocks(
        user_name=user.name, stocks_weights=portfolio.stocks_weights, stocks_symbols=stocks_symbols,
        descriptions=helpers.get_stocks_descriptions(stocks_symbols)[1:]
    )
    plot_functions.save_graphs(plt_stocks_component, file_name=f'{curr_user_directory}/stocks_component')

    # Total yield graph with sectors weights
    table: pd.DataFrame = portfolio.pct_change_table
    table['yield__selected_percent'] = (table["yield_selected"] - 1) * 100
    df, _, _ = helpers.analyze_with_machine_learning_linear_regression(
        returns_stock=table['yield__selected_percent'],
        table_index=table.index,
        record_percent_to_predict=record_percent_to_predict,
        test_size_machine_learning=test_size_machine_learning,
        closing_prices_mode=True
    )
    df['yield__selected_percent'] = df['col']
    df['yield__selected_percent_forecast'] = df["Forecast"]

    stats_details_tuple = portfolio.get_portfolio_stats()
    plt_yield_graph = plot_functions.plot_investment_portfolio_yield(
        user_name=user.name, table=df, stats_details_tuple=stats_details_tuple, sectors=portfolio.sectors
    )
    plot_functions.save_graphs(plt_yield_graph, file_name=f'{curr_user_directory}/yield_graph')


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


def get_investment_amount() -> int:
    return console_handler.get_investment_amount()


def get_stocks_from_json_file():
    collections_data = helpers.get_collection_json_data()
    stocks = {}
    for i in range(1, len(collections_data)):
        stocks_symbols_list = collections_data[str(i)][0]['stocksSymbols']
        stocks_description_list = helpers.get_stocks_descriptions(stocks_symbols_list, is_reverse_mode=False)[1:]
        stocks[str(i)] = [stocks_symbols_list, stocks_description_list]
    return stocks


def get_collection_number() -> str:
    stocks = get_stocks_from_json_file()
    return console_handler.get_collection_number(stocks)


def get_stocks_symbols_from_json_file(collection_number: int) -> list[str]:
    collection: dict = helpers.get_collection_json_data()[str(collection_number)][0]
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
