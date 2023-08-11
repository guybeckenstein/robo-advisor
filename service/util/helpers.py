import codecs
import csv
import datetime
import json
import math

import boto3
from bidi import algorithm as bidi_algorithm

import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from typing import Tuple
import pmdarima as pm
from sklearn.ensemble import GradientBoostingRegressor
from prophet import Prophet

from ..config import aws_settings, settings
from ..impl.sector import Sector
from . import tase_interaction


def get_best_portfolios(df, model_name: str):
    if model_name == 'Markowitz':
        optional_portfolios = [build_return_markowitz_portfolios_dic(df[0]),
                               build_return_markowitz_portfolios_dic(df[1]),
                               build_return_markowitz_portfolios_dic(df[2])]
    else:
        optional_portfolios = [build_return_gini_portfolios_dic(df[0]),
                               build_return_gini_portfolios_dic(df[1]),
                               build_return_gini_portfolios_dic(df[2])]
    return [optional_portfolios[0]['Safest Portfolio'], optional_portfolios[1]['Sharpe Portfolio'],
            optional_portfolios[2]['Max Risk Portfolio']]


def get_best_weights_column(stocks_symbols, sectors_list, optional_portfolios, pct_change_table) -> list:
    pct_change_table.dropna(inplace=True)
    stock_sectors = set_stock_sectors(stocks_symbols, sectors_list)
    high = np.dot(optional_portfolios[2].iloc[0][3:], pct_change_table.T)
    medium = np.dot(optional_portfolios[1].iloc[0][3:], pct_change_table.T)
    pct_change_table_low = pct_change_table.copy()
    for i in range(len(stock_sectors)):
        if stock_sectors[i] == "US commodity":
            pct_change_table_low = pct_change_table_low.drop(stocks_symbols[i], axis=1)
    low = np.dot(optional_portfolios[0].iloc[0][3:], pct_change_table_low.T)
    return [low, medium, high]


def get_three_best_weights(optional_portfolios) -> list:
    weighted_low = optional_portfolios[0].iloc[0][3:]
    weighted_medium = optional_portfolios[1].iloc[0][3:]
    weighted_high = optional_portfolios[2].iloc[0][3:]

    return [weighted_low, weighted_medium, weighted_high]


def get_three_best_sectors_weights(sectors_list, three_best_stocks_weights) -> list:
    sectors_weights_list = []
    for i in range(len(three_best_stocks_weights)):
        sectors_weights_list.append(return_sectors_weights_according_to_stocks_weights(sectors_list,
                                                                                       three_best_stocks_weights[i]))

    return sectors_weights_list


def build_return_gini_portfolios_dic(df: pd.DataFrame):
    return_dic = {'Max Risk Portfolio': {}, 'Safest Portfolio': {}, 'Sharpe Portfolio': {}}
    min_gini = df['Gini'].min()
    max_sharpe = df['Sharpe Ratio'].max()
    max_portfolio_annual = df['Portfolio_annual'].max()

    # use the min, max values to locate and create the two special portfolios
    sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
    safe_portfolio = df.loc[df['Gini'] == min_gini]
    max_portfolio = df.loc[df['Portfolio_annual'] == max_portfolio_annual]

    return_dic['Max Risk Portfolio'] = max_portfolio
    return_dic['Safest Portfolio'] = safe_portfolio
    return_dic['Sharpe Portfolio'] = sharpe_portfolio

    return return_dic


def build_return_markowitz_portfolios_dic(df: pd.DataFrame):
    return_dic = {'Max Risk Portfolio': {}, 'Safest Portfolio': {}, 'Sharpe Portfolio': {}}
    min_markowitz = df['Volatility'].min()
    max_sharpe = df['Sharpe Ratio'].max()
    max_portfolio_annual = df['Returns'].max()

    # use the min, max values to locate and create the two special portfolios
    sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
    safe_portfolio = df.loc[df['Volatility'] == min_markowitz]
    max_portfolio = df.loc[df['Returns'] == max_portfolio_annual]

    return_dic['Max Risk Portfolio'] = max_portfolio
    return_dic['Safest Portfolio'] = safe_portfolio
    return_dic['Sharpe Portfolio'] = sharpe_portfolio

    return return_dic


def return_sectors_weights_according_to_stocks_weights(sectors: list, stocks_weights) -> list:
    sectors_weights = [0.0] * len(sectors)
    for i in range(len(sectors)):
        sectors_weights[i] = 0
        # get from stocks_weights each symbol name without weight

        for j in range(len(stocks_weights.index)):

            first_component = stocks_weights.index[j].split()[0]

            # Check if the first component can be converted to an integer
            try:
                first_component_int = int(first_component)
                # The first component is a valid integer, use it for integer comparison
                if first_component_int in sectors[i].stocks:
                    sectors_weights[i] += stocks_weights[j]
            except ValueError:
                # The first component is not a valid integer, use it for string comparison
                if first_component in sectors[i].stocks:
                    sectors_weights[i] += stocks_weights[j]

    return sectors_weights


def analyze_with_machine_learning_linear_regression(returns_stock, table_index, RECORD_PERCENT_TO_PREDICT, TEST_SIZE_MACHINE_LEARNING, closing_prices_mode=False):
    df_final = pd.DataFrame({})
    forecast_col = 'col'
    df_final[forecast_col] = returns_stock
    df_final.fillna(value=-0, inplace=True)
    forecast_out = int(math.ceil(RECORD_PERCENT_TO_PREDICT * len(df_final)))
    df_final['label'] = df_final[forecast_col].shift(-forecast_out)

    # Added date
    df = df_final
    df['Date'] = table_index
    # print(df)
    X = np.array(df.drop(labels=['label', 'Date'], axis=1))
    X = preprocessing.scale(X)
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]
    df.dropna(inplace=True)
    y = np.array(df['label'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE_MACHINE_LEARNING)
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    # print(confidence)
    forecast_set = clf.predict(X_lately)
    df['Forecast'] = np.nan

    last_date = df.iloc[-1]['Date']
    last_date_datetime = pd.to_datetime(last_date)
    last_unix = last_date_datetime.timestamp()
    one_day = 86400
    next_unix = last_unix + one_day

    for i in forecast_set:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += 86400
        df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

    col = df["Forecast"]
    col = col.dropna()
    df["label"] = df['col']
    df["label"].fillna(df["Forecast"], inplace=True)

    forecast_returns_annual = (((1 + df['label'].mean()) ** 254) - 1) * 100
    excepted_returns = (((1 + df['Forecast'].mean()) ** 254) - 1) * 100
    if closing_prices_mode:
        forecast_returns_annual = (((1 + df['label'].pct_change().mean()) ** 254) - 1) * 100
        excepted_returns = (((1 + df['Forecast'].pct_change().mean()) ** 254) - 1) * 100
    return df, forecast_returns_annual, excepted_returns


def analyze_with_machine_learning_arima(returns_stock, table_index, RECORD_PERCENT_TO_PREDICT = 0.05,  closing_prices_mode=False):
    df_final = pd.DataFrame({})
    forecast_col = 'col'
    df_final[forecast_col] = returns_stock
    df_final.fillna(value=-0, inplace=True)

    forecast_out = int(math.ceil(RECORD_PERCENT_TO_PREDICT * len(df_final)))
    df_final['label'] = df_final[forecast_col].shift(-forecast_out)

    # ARIMA requires datetime index for time series data
    df_final.index = pd.to_datetime(table_index)

    # Perform ARIMA forecasting
    model = pm.auto_arima(df_final[forecast_col], seasonal=False, suppress_warnings=True)
    forecast, conf_int = model.predict(n_periods=forecast_out, return_conf_int=True)

    df_final['Forecast'] = np.nan
    df_final.loc[df_final.index[-forecast_out]:, 'Forecast'] = forecast

    # forecast_returns_annual = (forecast.iloc[-1] / df_final[forecast_col].iloc[-forecast_out - 1]) ** 254 - 1

    # add dates
    last_date = df_final.iloc[-1].name
    last_unix = last_date.timestamp()
    one_day = 86400
    next_unix = last_unix + one_day

    for i in forecast:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += 86400
        df_final.loc[next_date] = [np.nan for _ in range(len(df_final.columns) - 1)] + [i]
    col = df_final["Forecast"]
    col = col.dropna()
    df_final["label"] = df_final["col"]
    df_final["label"].fillna(df_final["Forecast"], inplace=True)

    forecast_returns_annual = (((1 + df_final['label'].mean()) ** 254) - 1) * 100
    excepted_returns = (((1 + df_final['Forecast'].mean()) ** 254) - 1) * 100
    if closing_prices_mode:
        forecast_returns_annual = (((1 + df_final['label'].pct_change().mean()) ** 254) - 1) * 100
        excepted_returns = (((1 + df_final['Forecast'].pct_change().mean()) ** 254) - 1) * 100

    # Calculate annual return based on the forecast
    return df_final, forecast_returns_annual, excepted_returns


def analyze_with_machine_learning_gbm(returns_stock, table_index, RECORD_PERCENT_TO_PREDICT,  closing_prices_mode=False):
    df_final = pd.DataFrame({})
    forecast_col = 'col'
    df_final[forecast_col] = returns_stock
    df_final.fillna(value=-0, inplace=True)

    forecast_out = int(math.ceil(RECORD_PERCENT_TO_PREDICT * len(df_final)))
    df_final['label'] = df_final[forecast_col].shift(-forecast_out)

    df_final.index = pd.to_datetime(table_index)

    # Perform GBM forecasting
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    X = np.arange(len(df_final))[:, None]  # Use a simple sequence as features for demonstration
    y = df_final[forecast_col].values
    model.fit(X, y)
    forecast = model.predict(np.arange(len(df_final), len(df_final) + forecast_out)[:, None])
    df_final['Forecast'] = np.nan
    df_final.loc[df_final.index[-forecast_out]:, 'Forecast'] = forecast

    # add dates
    last_date = df_final.iloc[-1].name
    last_unix = last_date.timestamp()
    one_day = 86400
    next_unix = last_unix + one_day

    for i in forecast:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += 86400
        df_final.loc[next_date] = [np.nan for _ in range(len(df_final.columns) - 1)] + [i]
    df_final["Forecast"] = df_final["Forecast"].shift(forecast_out)
    col = df_final["Forecast"]
    col = col.dropna()
    df_final["label"] = df_final["col"]
    df_final["label"].fillna(df_final["Forecast"], inplace=True)

    # forecast_returns_annual = (forecast[-1] / df_final[forecast_col].iloc[-forecast_out - 1]) ** 254 - 1
    forecast_returns_annual = (((1 + df_final['label'].mean()) ** 254) - 1) * 100
    excepted_returns = (((1 + df_final['Forecast'].mean()) ** 254) - 1) * 100
    if closing_prices_mode:
        forecast_returns_annual = (((1 + df_final['label'].pct_change().mean()) ** 254) - 1) * 100
        excepted_returns = (((1 + df_final['Forecast'].pct_change().mean()) ** 254) - 1) * 100

    # Calculate annual return based on the forecast
    return df_final, forecast_returns_annual, excepted_returns


def analyze_with_machine_learning_prophet(returns_stock, table_index, RECORD_PERCENT_TO_PREDICT= 0.05,
                                          closing_prices_mode=False,
                                          ):
    df_final = pd.DataFrame({})
    forecast_col = 'col'
    df_final[forecast_col] = returns_stock
    df_final.fillna(value=-0, inplace=True)

    forecast_out = int(math.ceil(RECORD_PERCENT_TO_PREDICT * len(df_final)))
    df_final['label'] = df_final[forecast_col].shift(-forecast_out)

    # Prepare the data for Prophet
    df_prophet = pd.DataFrame({'ds': table_index, 'y': df_final[forecast_col]})

    # Create and fit the Prophet model
    model = Prophet()
    model.fit(df_prophet)

    # Generate future dates for forecasting
    future = model.make_future_dataframe(periods=forecast_out, freq='D')

    # Perform the forecast
    forecast = model.predict(future)

    # Extract the forecasted values for future dates
    forecast_for_future = forecast[forecast['ds'].isin(table_index[-forecast_out:])]['yhat']

    # Assign the forecasted values to the 'Forecast' column for future dates
    df_final.loc[table_index[-forecast_out:], 'Forecast'] = forecast_for_future.values

    # add dates
    last_date = df_final.iloc[-1].name
    try:
        last_unix = last_date.timestamp()
    except Exception:  # TODO: write relevant and specific Exception here
        # convert str to datetime
        last_date = datetime.datetime.strptime(last_date, '%Y-%m-%d')
        last_unix = last_date.timestamp()

    one_day = 86400
    next_unix = last_unix + one_day

    for i in forecast_for_future:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += 86400
        df_final.loc[next_date] = [np.nan for _ in range(len(df_final.columns) - 1)] + [i]
    df_final["label"] = forecast['yhat'].values
    df_final["Forecast"][len(df_final) - forecast_out:] = forecast['yhat'][len(df_final) - forecast_out:].values
    # df_final["label"].fillna(df_final["Forecast"], inplace=True)

    forecast_returns_annual = (((1 + forecast['yhat'][len(df_final) - forecast_out:].mean()) ** 254) - 1) * 100
    excepted_returns = (((1 + forecast['yhat'].mean()) ** 254) - 1) * 100
    if closing_prices_mode:
        # Plot the forecast
        model.plot(forecast, xlabel='Date', ylabel='Stock Price', figsize=(12, 6))
        plt.title('Stock Price Forecast using Prophet')
        # plt.show()
        forecast_returns_annual = (((1 + forecast['yhat'][
                                         len(df_final) - forecast_out:].pct_change().mean()) ** 254) - 1) * 100
        excepted_returns = (((1 + forecast['yhat'].pct_change().mean()) ** 254) - 1) * 100

    return df_final, forecast_returns_annual, excepted_returns, plt


def update_daily_change_with_machine_learning(returns_stock, table_index, models_data,  closing_prices_mode=False):
    RECORD_PERCENT_TO_PREDICT = models_data["models_data"]["RECORD_PERCENT_TO_PREDICT"]
    SELECTED_ML_MODEL_FOR_BUILD = models_data["models_data"]["SELECTED_ML_MODEL_FOR_BUILD"]
    TEST_SIZE_MACHINE_LEARNING = models_data["models_data"]["TEST_SIZE_MACHINE_LEARNING"]
    SELECTED_ML_MODEL_FOR_BUILD = settings.MACHINE_LEARNING_MODEL[SELECTED_ML_MODEL_FOR_BUILD]
    
    num_of_rows = len(table_index)
    prefix_row = int(math.ceil(RECORD_PERCENT_TO_PREDICT * num_of_rows))
    for i, stock in enumerate(returns_stock.columns):
        df = None
        stock_name = str(stock)
        if SELECTED_ML_MODEL_FOR_BUILD == settings.MACHINE_LEARNING_MODEL[0]:
            df, annual_return, excepted_returns = analyze_with_machine_learning_linear_regression(
                returns_stock[stock_name], table_index, RECORD_PERCENT_TO_PREDICT, TEST_SIZE_MACHINE_LEARNING,
                closing_prices_mode
            )
        elif SELECTED_ML_MODEL_FOR_BUILD == settings.MACHINE_LEARNING_MODEL[1]:
            df, annual_return, excepted_returns = analyze_with_machine_learning_arima(
                returns_stock[stock_name], table_index, RECORD_PERCENT_TO_PREDICT, closing_prices_mode
            )

        elif SELECTED_ML_MODEL_FOR_BUILD == settings.MACHINE_LEARNING_MODEL[2]:
            df, annual_return, excepted_returns = analyze_with_machine_learning_gbm(
                returns_stock[stock_name], table_index, RECORD_PERCENT_TO_PREDICT, closing_prices_mode
            )


        elif SELECTED_ML_MODEL_FOR_BUILD == settings.MACHINE_LEARNING_MODEL[3]:
            df, annual_return, excepted_returns, plt = analyze_with_machine_learning_prophet(
                returns_stock[stock_name], table_index, RECORD_PERCENT_TO_PREDICT, closing_prices_mode
            )
        else:
            raise ValueError('Invalid machine model')
        returns_stock[stock_name] = df['label'][prefix_row:].values
    # TODO: (annual_return, excepted_returns) not initialized
    return returns_stock, annual_return, excepted_returns


def convert_data_to_tables(location_saving, file_name, stocksNames, numOfYearsHistory, saveToCsv):
    frame = {}
    yf.pdr_override()
    start_date, end_date = get_from_and_to_dates(numOfYearsHistory)
    file_url = location_saving + file_name + ".csv"

    for i, stock in enumerate(stocksNames):

        if type(stock) == int or stock.isnumeric():
            num_of_digits = len(str(stock))
            if num_of_digits > 3:
                is_index_type = False
            else:
                is_index_type = True
            df = get_israeli_symbol_data("get_past_10_years_history",
                                         start_date, end_date, stock, is_index_type)
            # list to dateframe
            df = pd.DataFrame(df)
            df["tradeDate"] = pd.to_datetime(df["tradeDate"])
            df.set_index("tradeDate", inplace=True)
            if is_index_type:
                price = df[["closingIndexPrice"]]
            else:
                price = df[["closingPrice"]]
            frame[stocksNames[i]] = price
        else:
            df = yf.download(stock, start=start_date, end=end_date)
            price = df[["Adj Close"]]
            frame[stock] = price

    closingPricesTable = pd.concat(frame.values(), axis=1, keys=frame.keys())

    if saveToCsv:
        # convert to csv
        closingPricesTable.to_csv(file_url, index=True, header=True)

    return closingPricesTable


def choose_portfolio_by_risk_score(optional_portfolios_list, risk_score):
    if 0 < risk_score <= 4:
        return optional_portfolios_list[0]
    elif 5 < risk_score <= 7:
        return optional_portfolios_list[1]
    elif risk_score > 7:
        return optional_portfolios_list[2]
    else:
        raise ValueError


def get_json_data(name: str):
    with codecs.open(name + ".json", "r", encoding="utf-8") as file:
        json_data = json.load(file)
    return json_data


def get_sectors_data_from_file(mode: str = "regular"):
    if mode == 'regular':
        sectors_data = get_json_data(settings.SECTORS_JSON_NAME)
    else:
        sectors_data = get_json_data('../../' + settings.SECTORS_JSON_NAME)
    return sectors_data['sectorsList']['result']


def set_sectors(stocks_symbols: list, mode: str = 'regular') -> list:  # TODO - make more efficient
    """
    For each stock symbol, it checks for which sector does it belong.
    :return: It returns a list of sectors with the relevant stocks within each sector. Subset of the stock symbol
    """
    sectors: list = []
    sectors_data = get_sectors_data_from_file(mode)

    for i in range(len(sectors_data)):
        curr_sector = Sector(sectors_data[i]['name'])
        for j in range(len(stocks_symbols)):
            if stocks_symbols[j] in sectors_data[i]['stocks']:
                curr_sector.add_stock(stocks_symbols[j])
        if len(curr_sector.stocks) > 0:
            sectors.append(curr_sector)

    return sectors


def set_stock_sectors(stocks_symbols, sectors: list) -> list:
    stock_sectors = []  # TODO - FIX ORDER
    for symbol in stocks_symbols:
        found_sector = False
        for curr_sector in sectors:
            if symbol in curr_sector.stocks:
                stock_sectors.append(curr_sector.name)
                found_sector = True
                break
        if not found_sector:
            stock_sectors.append(None)

    return stock_sectors


def drop_stocks_from_us_commodity_sector(stocks_symbols, stock_sectors):  # TODO - MAKE DYNAMIC
    new_stocks_symbols = []
    for i in range(len(stock_sectors)):
        if stock_sectors[i] != "US commodity":
            new_stocks_symbols.append(stocks_symbols[i])

    return new_stocks_symbols


def get_from_and_to_dates(num_of_years) -> Tuple[str, str]:
    today = datetime.datetime.now()
    start_year = today.year - num_of_years
    start_month = today.month
    start_day = today.day
    end_year = today.year
    end_month = today.month
    end_day = today.day
    from_date = str(start_year) + "-" + str(start_month) + "-" + str(start_day)
    to_date = str(end_year) + "-" + str(end_month) + "-" + str(end_day)
    return from_date, to_date


def setStockSectors(stocksSymbols, sectorList) -> list:  # TODO
    stock_sectors = []  # TODO - FIX ORDER
    for symbol in stocksSymbols:
        found_sector = False
        for sector in sectorList:
            if symbol in sector.stocks:
                stock_sectors.append(sector.name)
                found_sector = True
                break
        if not found_sector:
            stock_sectors.append(None)

    return stock_sectors


def makes_yield_column(_yield, weighted_sum_column):
    _yield.iloc[0] = 1
    for i in range(1, weighted_sum_column.size):
        change = weighted_sum_column.item(i) + 1
        last_value = _yield.iloc[i - 1]
        new_value = last_value * change
        _yield.iloc[i] = new_value
    return _yield


# yfinance and israel tase impl:
def get_israeli_symbol_data(command, start_date, end_date, israeli_symbol_name, is_index_type):
    if is_index_type:
        data = \
        tase_interaction.get_israeli_index_data(command, start_date, end_date, israeli_symbol_name)["indexEndOfDay"][
            "result"]
    else:
        data = tase_interaction.get_israeli_security_data(
            command, start_date, end_date, israeli_symbol_name)["securitiesEndOfDayTradingData"]["result"]
    return data


def get_stocks_descriptions(stocks_symbols, is_reverse_mode=True):  # TODO - ALSO FOR STOCKS
    stocks_descriptions = [len(stocks_symbols)]
    usa_stocks_table = get_usa_stocks_table()
    usa_indexes_table = get_usa_indexes_table()
    for i, stock in enumerate(stocks_symbols):
        if type(stock) == int:
            num_of_digits = len(str(stock))
            if num_of_digits > 3:
                is_index_type = False
            else:
                is_index_type = True
            stocks_descriptions.append(convert_israeli_symbol_number_to_name(stock, is_index_type=is_index_type,
                                                                             is_reverse_mode=is_reverse_mode))
        else:

            try:
                description = usa_indexes_table.loc[usa_indexes_table['symbol'] == stock, 'shortName'].item()
                stocks_descriptions.append(description)
            except:
                try:
                    description = usa_stocks_table.loc[usa_stocks_table['Symbol'] == stock, 'Name'].item()
                    stocks_descriptions.append(description)
                except:
                    description = yf.Ticker(stock).info['shortName']
                    stocks_descriptions.append(description)

    return stocks_descriptions


def convert_israeli_symbol_number_to_name(symbol_number: int, is_index_type: bool, is_reverse_mode: bool = True) -> str:
    if is_index_type:
        json_data = get_json_data(settings.INDICES_LIST_JSON_NAME)
        hebrew_text = [item['name'] for item in json_data['indicesList']['result'] if item['id'] == symbol_number][0]
    else:
        json_data = get_json_data(settings.SECURITIES_LIST_JSON_NAME)
        hebrew_text = [item['securityName'] for item in json_data['tradeSecuritiesList']['result'] if
                       item['securityId'] == symbol_number][0]
    if is_reverse_mode:
        hebrew_text = bidi_algorithm.get_display(u'' + hebrew_text)

    return hebrew_text


def convert_israeli_security_number_to_company_name(israeli_security_number: str) -> str:
    securities_list = get_json_data(settings.SECURITIES_LIST_JSON_NAME)["tradeSecuritiesList"]["result"]
    result = [item['companyName'] for item in securities_list if
              item['securityId'] == israeli_security_number]

    return result[0]


def convert_company_name_to_israeli_security_number(companyName: str) -> str:
    securities_list = get_json_data(settings.SECURITIES_LIST_JSON_NAME)["tradeSecuritiesList"]["result"]
    result = [item['securityId'] for item in securities_list if
              item['companyName'] == companyName]

    return result[0]


# get directly from tase api instead of json file config
def get_israeli_companies_list():
    json_data = tase_interaction.get_israeli_companies_list()
    securities_list = json_data["tradeSecuritiesList"]["result"]
    return securities_list


def get_israeli_indexes_list():
    json_data = tase_interaction.get_israeli_indexes_list()
    indexes_list = json_data['indicesList']['result']
    return indexes_list


def get_usa_stocks_table():
    return pd.read_csv(settings.CONFIG_RESOURCE_LOCATION + "nasdaq_all_stocks.csv")


def get_usa_indexes_table():
    return pd.read_csv(settings.CONFIG_RESOURCE_LOCATION + "usa_indexes.csv")




def collect_all_stocks():
    path = settings.CONFIG_RESOURCE_LOCATION + "all_stocks_basic_data.csv"
    sectors_data = get_sectors_data_from_file(mode="regular")
    sectors_list = []
    stocks_symbols_list = []
    description_list = []

    for i in range(len(sectors_data)):
        data = sectors_data[i]["stocks"]
        # append sector name to sectors list len(data) times
        for j in range(len(data)):
            stocks_symbols_list.append(data[j])
            sectors_list.append(sectors_data[i]["name"])

            # maybe add more details later
            if i < 3:  # israeli indexes
                pass
            elif i < 5:  # usa indexes
                pass
            elif i == 6:  # israeli stocks
                pass
            else:  # usa stocks
                # TODO - MAYBE ADD HERE DESCRITON FOR USA STOCKS
                pass

    description_list.append(get_stocks_descriptions(stocks_symbols_list))

    data = [
        sectors_list,
        stocks_symbols_list,
        description_list
    ]
    with open(path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # TODO - maybe add more details later
        csv_writer.writerow(['Symbol', 'sector', 'description'])

        # Write data rows
        for row in data:
            csv_writer.writerow(row)

    print("CSV file created successfully!")

def save_usa_indexes_table(): # dont delete it, use for admin
    sectors_data = get_sectors_data_from_file(mode="regular")
    stock_data_list = []
    # create table
    for i in range(3,6):
        stocks = sectors_data[i]["stocks"]
        for j in range(len(stocks)):
            stock_info = yf.Ticker(stocks[j]).info
            stock_data_list.append(stock_info)

    # Create a set of all keys present in the stock data dictionaries
    all_keys = set()
    for stock_data in stock_data_list:
        all_keys.update(stock_data.keys())


    # Define the CSV file path
    csv_file_path = settings.CONFIG_RESOURCE_LOCATION + 'usa_indexes.csv'

    # Write the data to a CSV file
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=all_keys)

        writer.writeheader()
        for stock_data in stock_data_list:
            writer.writerow(stock_data)


# AWS , TODO
def connect_to_s3() -> boto3.client:
    s3 = boto3.resource(service_name='s3',
                        region_name=aws_settings.REGION_NAME,
                        aws_secret_access_key=aws_settings.AWS_SECRET_ACCESS_KEY,
                        aws_access_key_id=aws_settings.AWS_ACCESS_KEY_ID)

    s3_client = boto3.client('s3', aws_access_key_id=aws_settings.AWS_ACCESS_KEY_ID,
                             aws_secret_access_key=aws_settings.AWS_SECRET_ACCESS_KEY,
                             region_name=aws_settings.REGION_NAME)
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
