import codecs
import csv
import datetime
from datetime import datetime as data_time, timedelta
import json
import math

import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pmdarima as pm
from prophet import Prophet

from service.config import settings
from service.impl.sector import Sector
from service.util import tase_interaction
from service.util.deep_learining import models

SINGLE_DAY: int = 86400


def get_best_portfolios(df, model_name: str) -> list:
    if model_name == 'Markowitz':
        optional_portfolios: list = [
            build_return_model_portfolios_dict(df=df[i], max_val='Returns', min_val='Volatility') for i in range(3)
        ]
    else:
        optional_portfolios: list = [
            build_return_model_portfolios_dict(df=df[i], max_val='Portfolio Annual', min_val='Gini') for i in range(3)
        ]
    return [optional_portfolios[0]['Safest Portfolio'], optional_portfolios[1]['Sharpe Portfolio'],
            optional_portfolios[2]['Max Risk Portfolio']]


def get_best_weights_column(stocks_symbols, sectors_list, optional_portfolios, pct_change_table) -> list:
    pct_change_table.dropna(inplace=True)
    stock_sectors = set_stock_sectors(stocks_symbols, sectors_list)
    high = np.dot(optional_portfolios[2].iloc[0][3:], pct_change_table.T)
    medium = np.dot(optional_portfolios[1].iloc[0][3:], pct_change_table.T)
    pct_change_table_low = pct_change_table.copy()
    for i in range(len(stock_sectors)):
        if stock_sectors[i] == "US Commodity Indexes":
            pct_change_table_low = pct_change_table_low.drop(stocks_symbols[i], axis=1)
    low = np.dot(optional_portfolios[0].iloc[0][3:], pct_change_table_low.T)
    return [low, medium, high]


def get_three_best_weights(optional_portfolios) -> list:
    return [optional_portfolios[i].iloc[0][3:] for i in range(3)]


def get_three_best_sectors_weights(sectors_list, three_best_stocks_weights) -> list:
    return [return_sectors_weights_according_to_stocks_weights(sectors_list, three_best_stocks_weights[i])
            for i in range(len(three_best_stocks_weights))]


def build_return_model_portfolios_dict(df: pd.DataFrame, max_val: str,
                                       min_val: str) -> dict[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    max_portfolio_annual = df[max_val].max()
    max_sharpe = df['Sharpe Ratio'].max()
    min_markowitz = df[min_val].min()

    # use the min, max values to locate and create the two special portfolios
    return_dic: dict[pd.DataFrame] = dict()
    return_dic['Max Risk Portfolio'] = df.loc[df[max_val] == max_portfolio_annual]
    return_dic['Sharpe Portfolio'] = df.loc[df['Sharpe Ratio'] == max_sharpe]
    return_dic['Safest Portfolio'] = df.loc[df[min_val] == min_markowitz]

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
                    sectors_weights[i] += stocks_weights.iloc[j]
            except ValueError:
                # The first component is not a valid integer, use it for string comparison
                if first_component in sectors[i].stocks:
                    sectors_weights[i] += stocks_weights.iloc[j]

    return sectors_weights


class Analyze:
    def __init__(
            self, returns_stock: pd.DataFrame,
            table_index: pd.Index,
            record_percent_to_predict: float,
            is_closing_prices_mode: bool = False
    ):
        self._returns_stock: pd.DataFrame = returns_stock
        self._table_index: pd.Index = table_index
        self._record_percent_to_predict: float = record_percent_to_predict
        self._is_closing_prices_mode: bool = is_closing_prices_mode

    def linear_regression_model(
            self, test_size_machine_learning: str
    ) -> tuple[pd.DataFrame, np.longdouble, np.longdouble]:
        df, forecast_out = self.get_final_dataframe()

        # Added date
        df['Date'] = pd.to_datetime(self._table_index)

        X = np.array(df.drop(labels=['Label', 'Date'], axis=1))
        X = preprocessing.scale(X)
        X_lately = X[-forecast_out:]
        X = X[:-forecast_out]
        df.dropna(inplace=True)
        y = np.array(df['Label'])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=float(test_size_machine_learning)
        )

        clf = LinearRegression()
        clf.fit(X_train, y_train)

        forecast = clf.predict(X_lately)

        forecast_dates = pd.date_range(start=df['Date'].iloc[-1], periods=forecast_out + 1, freq='D')[1:]
        forecast_df = pd.DataFrame(index=forecast_dates, columns=['Forecast'])
        forecast_df['Forecast'] = forecast

        # Combine the original DataFrame and the forecast DataFrame
        combined_df = pd.concat([df, forecast_df])

        forecast_with_historical_returns_annual, expected_returns = self.calculate_returns(combined_df)
        return combined_df, forecast_with_historical_returns_annual, expected_returns

    def arima_model(self) -> tuple[pd.DataFrame, np.longdouble, np.longdouble]:
        df, forecast_out = self.get_final_dataframe()

        # ARIMA requires datetime index for time series data
        df.index = pd.to_datetime(self._table_index, format='%Y-%m-%d')

        # Perform ARIMA forecasting
        model: pm.ARIMA = pm.auto_arima(df['Col'], seasonal=False, suppress_warnings=True, stepwise=False)
        forecast, conf_int = model.predict(n_periods=forecast_out, return_conf_int=True)

        df['Forecast'] = np.nan
        df.loc[df.index[-forecast_out]:, 'Forecast'] = forecast

        # Add dates
        last_date = df.iloc[-1].name
        last_unix: pd.Timestamp = last_date.timestamp()
        next_unix: float = last_unix + SINGLE_DAY

        for i in forecast:
            next_date: datetime.datetime = datetime.datetime.fromtimestamp(next_unix)
            next_unix += SINGLE_DAY
            df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

        forecast_with_historical_returns_annual, expected_returns = self.calculate_returns(df)
        return df, forecast_with_historical_returns_annual, expected_returns

    def lstm_model(self) -> tuple[pd.DataFrame, np.longdouble, np.longdouble]:
        df, forecast_out = self.get_final_dataframe()
        df.index = pd.to_datetime(self._table_index)
        df = models.lstm_model(df=df, forecast_out=forecast_out, use_features=True)
        forecast_with_historical_returns_annual, expected_returns = self.calculate_returns(df)

        return df, forecast_with_historical_returns_annual, expected_returns

    def prophet_model(self) -> tuple[pd.DataFrame, np.longdouble, np.longdouble, plt]:
        df, forecast_out = self.get_final_dataframe()
        df.index = pd.to_datetime(self._table_index)

        # Prepare the data for Prophet
        df_prophet: pd.DataFrame = pd.DataFrame({'ds': self._table_index, 'y': df['Col']})

        # Create and fit the Prophet model
        model: Prophet = Prophet()
        model.fit(df_prophet)

        # Generate future dates for forecasting
        future: pd.DataFrame = model.make_future_dataframe(periods=forecast_out, freq='D')

        # Perform the forecast
        forecast: pd.DataFrame = model.predict(future)

        # Extract the forecasted values for future dates
        forecast_for_future: pd.Series = forecast[forecast['ds'].isin(self._table_index[-forecast_out:])]['yhat']

        # Assign the forecasted values to the 'Forecast' column for future dates
        df.loc[self._table_index[-forecast_out:], 'Forecast'] = forecast_for_future.values

        # add dates
        last_date: pd.Timestamp = df.iloc[-1].name
        try:
            last_unix = last_date.timestamp()
        except AttributeError:
            # convert str to datetime
            last_date: datetime.datetime = datetime.datetime.strptime(last_date, '%Y-%m-%d')
            last_unix: float = last_date.timestamp()

        next_unix: float = last_unix + SINGLE_DAY
        for i in forecast_for_future:
            next_date: datetime.datetime = datetime.datetime.fromtimestamp(next_unix)
            next_unix += SINGLE_DAY
            df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

        col_offset: int = len(df) - forecast_out
        yhat: pd.Series = forecast['yhat']
        df['Label'] = yhat.values
        df['Forecast'][col_offset:] = yhat[col_offset:]

        longdouble: np.longdouble = np.longdouble(254)
        excepted_returns: np.longdouble = ((np.exp(longdouble * np.log1p(yhat[col_offset:].mean()))) - 1) * 100
        forecast_with_historical_returns_annual: np.longdouble = ((np.exp(
            longdouble * np.log1p(yhat.mean()))) - 1) * 100
        if self._is_closing_prices_mode:
            # Plot the forecast
            model.plot(forecast, xlabel='Date', ylabel='Stock Price', figsize=(12, 6))
            plt.title('Stock Price Forecast using Prophet')
            excepted_returns: np.longdouble = ((np.exp(longdouble * np.log1p(
                yhat[col_offset:].pct_change().mean()))) - 1) * 100
            forecast_with_historical_returns_annual: np.longdouble = ((np.exp(longdouble * np.log1p(
                yhat.pct_change().mean()))) - 1) * 100

        return df, forecast_with_historical_returns_annual, excepted_returns, plt

    def get_final_dataframe(self) -> tuple[pd.DataFrame, int]:
        df: pd.DataFrame = pd.DataFrame({})
        df['Col'] = self._returns_stock
        df.fillna(value=-0, inplace=True)
        forecast_out = int(math.ceil(self._record_percent_to_predict * len(df)))
        df['Label'] = df['Col'].shift(-forecast_out)
        return df, forecast_out

    def calculate_returns(self, df: pd.DataFrame, forecast_out=None) -> tuple[np.longdouble, np.longdouble]:
        if forecast_out is not None:
            df["Forecast"] = df["Forecast"].shift(forecast_out)
        df['Label'] = df['Col']
        df['Label'].fillna(df['Forecast'], inplace=True)

        longdouble: np.longdouble = np.longdouble(254)

        logged_label_mean: np.ndarray = np.log1p(df['Label'].mean())
        np_exp_res1: np.ndarray = np.exp(
            longdouble * logged_label_mean)  # Generates RuntimeWarning: overflow encountered in exp
        forecast_with_historical_returns_annual: np.longdouble = (np_exp_res1 - 1) * 100

        logged_forecast_mean: np.ndarray = np.log1p(df['Forecast'].mean())
        np_exp_res2: np.ndarray = np.exp(
            longdouble * logged_forecast_mean)  # Generates RuntimeWarning: overflow encountered in exp
        expected_returns: np.longdouble = (np_exp_res2 - 1) * 100

        if self._is_closing_prices_mode:
            logged_label_mean: np.ndarray = np.log1p(
                df['Label'].ffill().pct_change().mean()
            )  # Generates RuntimeWarning: invalid value encountered in log1p
            forecast_with_historical_returns_annual: np.longdouble = (np.exp(longdouble * logged_label_mean) - 1) * 100
            logged_forecast_mean: np.ndarray = np.log1p(
                df['Forecast'].ffill().pct_change().mean()
            )  # Generates RuntimeWarning: invalid value encountered in log1p
            expected_returns: np.longdouble = (np.exp(longdouble * logged_forecast_mean) - 1) * 100
        return forecast_with_historical_returns_annual, expected_returns

    @property
    def returns_stock(self):
        return self._returns_stock


def update_daily_change_with_machine_learning(returns_stock, table_index: pd.Index, models_data: dict,
                                              closing_prices_mode: bool = False) -> tuple:
    # Calculate offset of the table (get sub-table)
    offset_row, record_percent_to_predict = get_daily_change_sub_table_offset(models_data, table_index)

    selected_ml_model_for_build: int = int(models_data["models_data"]["selected_ml_model_for_build"])
    test_size_machine_learning: float = float(models_data["models_data"]["test_size_machine_learning"])
    selected_ml_model_for_build: str = settings.MACHINE_LEARNING_MODEL[selected_ml_model_for_build]
    is_ndarray_mode = False
    try:
        columns = returns_stock.columns
    except AttributeError:
        columns = returns_stock
        is_ndarray_mode = True
    if len(columns) == 0:
        raise AttributeError('columns length is invalid - 0. Should be at least 1')
    else:
        excepted_returns = None
        annual_return_with_forecast = None

        for i, stock in enumerate(columns):
            if is_ndarray_mode:
                stock_name = 0
            else:
                stock_name = str(stock)

            analyze: Analyze = Analyze(
                returns_stock=returns_stock[stock_name],
                table_index=table_index,
                record_percent_to_predict=float(record_percent_to_predict),
                is_closing_prices_mode=closing_prices_mode
            )
            if selected_ml_model_for_build == settings.MACHINE_LEARNING_MODEL[0]:  # Linear Regression
                df, annual_return_with_forecast, excepted_returns = \
                    analyze.linear_regression_model(test_size_machine_learning)
            elif selected_ml_model_for_build == settings.MACHINE_LEARNING_MODEL[1]:  # Arima
                df, annual_return_with_forecast, excepted_returns = analyze.arima_model()
            elif selected_ml_model_for_build == settings.MACHINE_LEARNING_MODEL[2]:  # Gradient Boosting Regressor
                df, annual_return_with_forecast, excepted_returns = analyze.lstm_model()
            elif selected_ml_model_for_build == settings.MACHINE_LEARNING_MODEL[3]:  # Prophet
                df, annual_return_with_forecast, excepted_returns, plt = analyze.prophet_model()
            else:
                raise ValueError('Invalid machine model')
            if df['Label'][offset_row:].values.size == returns_stock[stock_name].size:
                returns_stock[stock_name] = df['Label'][offset_row:].values
            else:
                returns_stock[stock_name] = df['Label'].values

        return returns_stock, annual_return_with_forecast, excepted_returns


def get_daily_change_sub_table_offset(models_data, table_index) -> tuple[int, float]:
    record_percent_to_predict: float = float(models_data["models_data"]["record_percent_to_predict"])
    num_of_rows: int = len(table_index)
    offset_row: int = int(math.ceil(record_percent_to_predict * num_of_rows))
    return offset_row, record_percent_to_predict


def convert_data_to_tables(location_saving, file_name, stocks_names, num_of_years_history, save_to_csv,
                           start_date: str = None, end_date: str = None):
    df: pd.DataFrame = pd.DataFrame({})

    # for israeli stocks
    today = datetime.datetime.now()
    min_start_year = today.year - 10
    min_start_month = today.month
    min_start_day = today.day
    min_date: str = f"{min_start_year}-{min_start_month}-{min_start_day}"

    frame = {}
    yf.pdr_override()
    if start_date is None or end_date is None:
        start_date, end_date = get_from_and_to_dates(num_of_years_history)
    file_url: str = f'{location_saving}{file_name}.csv'

    for i, stock in enumerate(stocks_names):
        if isinstance(stock, float):
            continue
        if isinstance(stock, int) or stock.isnumeric():  # Israeli stock
            num_of_digits = len(str(stock))
            if num_of_digits > 3:
                is_index_type = False
            else:
                is_index_type = True
            try:
                if start_date < min_date:
                    start_date = min_date
                df: pd.DataFrame = get_israeli_symbol_data(
                    'get_past_10_years_history', start_date, end_date, stock, is_index_type
                )

            except ValueError:
                print('Invalid start_date or end_date format, should be %Y-%m-%d')
            except (AttributeError, IndexError):
                print(f"Error in stock: {stock}")
            finally:
                # list to DateFrame
                df = pd.DataFrame(df)
                df["tradeDate"] = pd.to_datetime(df["tradeDate"])
                df.set_index("tradeDate", inplace=True)
                if is_index_type:
                    price = df[["closingIndexPrice"]]
                else:
                    price = df[["closingPrice"]]
                frame[stocks_names[i]] = price
        else:  # US stock
            try:
                df: pd.DataFrame = yf.download(stock, start=start_date, end=end_date)
            except ValueError:
                print('Invalid start_date or end_date format, should be %Y-%m-%d')
                continue
            except AttributeError:
                print(f"Error in stock: {stock}")
                continue
            except IndexError:
                print(f"Error in stock: {stock}")
                continue
            price = df[["Adj Close"]]
            frame[stock] = price

    closing_prices_table = pd.concat(frame.values(), axis=1, keys=frame.keys())

    if save_to_csv:
        # convert to csv
        closing_prices_table.to_csv(file_url, index=True, header=True)

    return closing_prices_table


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
    with codecs.open(f"{name}.json", "r", encoding="utf-8") as file:
        json_data = json.load(file)
    return json_data


def get_sectors_data_from_file():
    sectors_data = get_json_data(settings.SECTORS_JSON_NAME)
    return sectors_data['sectorsList']['result']


def set_sectors(stocks_symbols: list[object]) -> list[Sector]:
    """
    For each stock symbol, it checks for which sector does it belong.
    :return: It returns a list of sectors.json.json with the relevant stocks within each sector.
    Subset of the stock symbol
    """
    sectors: list = []
    sectors_data: [list[dict[str, str, list[object]]]] = get_sectors_data_from_file()

    for i in range(len(sectors_data)):
        curr_sector: Sector = Sector(_name=sectors_data[i]['name'])
        for j in range(len(stocks_symbols)):
            if stocks_symbols[j] in sectors_data[i]['stocks']:
                curr_sector.add_stock(stocks_symbols[j])
        if len(curr_sector.stocks) > 0:
            sectors.append(curr_sector)

    return sectors


def set_stock_sectors(stocks_symbols, sectors: list) -> list:
    stock_sectors = []
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


def drop_stocks_from_specific_sector(stocks_symbols, stock_sectors, sector_name):
    new_stocks_symbols = []
    for i in range(len(stock_sectors)):
        if stock_sectors[i] != sector_name:
            new_stocks_symbols.append(stocks_symbols[i])

    return new_stocks_symbols


def get_from_and_to_dates(num_of_years) -> tuple[str, str]:
    today = datetime.datetime.now()
    start_year = today.year - num_of_years
    start_month = today.month
    start_day = today.day
    end_year = today.year
    end_month = today.month
    end_day = today.day
    from_date: str = f"{start_year}-{start_month}-{start_day}"
    to_date: str = f"{end_year}-{end_month}-{end_day}"
    return from_date, to_date


def setStockSectors(stocksSymbols, sectorList) -> list:
    stock_sectors = []
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
def get_israeli_symbol_data(command: str, start_date, end_date, israeli_symbol_name: int, is_index_type: bool):
    if is_index_type:
        data = tase_interaction.get_israeli_index_data(
            command, start_date, end_date, israeli_symbol_name
        )["indexEndOfDay"]["result"]
    else:
        data = tase_interaction.get_israeli_security_data(
            command, start_date, end_date, israeli_symbol_name)["securitiesEndOfDayTradingData"]["result"]
    return data


def get_stocks_descriptions(stocks_symbols: list, is_reverse_mode: bool = True):
    stocks_descriptions = [len(stocks_symbols)]
    usa_stocks_table: pd.DataFrame = get_usa_stocks_table()
    usa_indexes_table: pd.DataFrame = get_usa_indexes_table()
    for i, stock in enumerate(stocks_symbols):
        try:
            if isinstance(stock, int) or stock.isnumeric():
                num_of_digits = len(str(stock))
                if num_of_digits > 3:
                    is_index_type = False
                else:  # israeli index name always has maximum of 3 digits
                    is_index_type = True
                if isinstance(stock, str):
                    stock = int(stock)
                stocks_descriptions.append(convert_israeli_symbol_number_to_name(stock, is_index_type=is_index_type,
                                                                                 is_reverse_mode=is_reverse_mode))
            else:
                try:
                    description = usa_indexes_table.loc[usa_indexes_table['symbol'] == stock, 'shortName'].item()
                    stocks_descriptions.append(description)
                except (KeyError, ValueError):
                    try:
                        description = usa_stocks_table.loc[usa_stocks_table['Symbol'] == stock, 'Name'].item()
                        stocks_descriptions.append(description)
                    except (KeyError, ValueError):
                        description = yf.Ticker(stock).info['shortName']
                        stocks_descriptions.append(description)
        except AttributeError:
            raise AttributeError('Invalid `stocks_symbols`')

    return stocks_descriptions


def convert_israeli_symbol_number_to_name(symbol_number: int, is_index_type: bool, is_reverse_mode: bool = True) -> str:
    from bidi import algorithm as bidi_algorithm

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


# get directly from tase api instead of json file config
def get_israeli_companies_list():
    json_data = tase_interaction.get_israeli_companies_list()
    securities_list = json_data["tradeSecuritiesList"]["result"]
    return securities_list


def get_israeli_indexes_list():
    json_data = tase_interaction.get_israeli_indexes_list()
    indexes_list = json_data['indicesList']['result']
    return indexes_list


def get_usa_stocks_table() -> pd.DataFrame:
    return pd.read_csv(f"{settings.CONFIG_RESOURCE_LOCATION}nasdaq_all_stocks.csv")


def get_usa_indexes_table() -> pd.DataFrame:
    return pd.read_csv(f"{settings.CONFIG_RESOURCE_LOCATION}usa_indexes.csv")


def get_all_stocks_table():
    return pd.read_csv(f"{settings.CONFIG_RESOURCE_LOCATION}all_stocks_basic_data.csv")


def get_sector_by_symbol(symbol):
    all_stocks_Data = get_all_stocks_table()
    sector = all_stocks_Data.loc[all_stocks_Data['Symbol'] == str(symbol), 'sector'].item()
    return sector


def get_description_by_symbol(symbol):
    all_stocks_Data = get_all_stocks_table()
    try:
        description = all_stocks_Data.loc[all_stocks_Data['Symbol'] == str(symbol), 'description'].item()
    except KeyError:
        description = yf.Ticker(symbol).info['shortName']
    except ValueError:
        description = yf.Ticker(symbol).info['shortName']
    except AttributeError:
        description = yf.Ticker(symbol).info['shortName']
    return description


def get_symbol_by_description(description: str) -> pd.Series:
    all_stocks_data = get_all_stocks_table()
    try:
        symbol = all_stocks_data.loc[all_stocks_data['description'] == str(description), 'Symbol'].item()
    except ValueError:
        description: str = get_description_by_symbol(description)
        symbol = all_stocks_data.loc[all_stocks_data['description'] == str(description), 'Symbol'].item()
    return symbol


def get_stocks_symbols_list_by_sector(sector):
    all_stocks_Data = get_all_stocks_table()
    stocks_list = all_stocks_Data.loc[all_stocks_Data['sector'] == sector, 'Symbol'].tolist()
    return stocks_list


def get_sectors_names_list() -> list[str]:
    all_stocks_data: pd.DataFrame = get_all_stocks_table()
    sectors_list: list[str] = all_stocks_data['sector'].unique().tolist()
    return sectors_list


def get_collection_json_data() -> dict[
    dict[str, str, str, str, str, str],
    list[dict[list[object], float, float, int]],
    list[dict[list[object], float, float, int]],
    list[dict[list[object], float, float, int]],
    list[dict[list[object], float, float, int]]
]:
    if settings.FILE_ACCESS_SELECTED == settings.FILE_ACCESS_TYPE[0]:
        return convert_data_stream_to_json(file_stream=None)['collections']  # TODO: add parameter to method call
    else:
        return get_json_data(settings.STOCKS_JSON_NAME)['collections']


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


def get_symbols_names_list() -> list[str]:
    all_stocks_Data: pd.DataFrame = get_all_stocks_table()
    symbols_list: list[str] = all_stocks_Data['Symbol'].unique().tolist()
    return symbols_list


def get_descriptions_list() -> list[str]:
    all_stocks_Data: pd.DataFrame = get_all_stocks_table()
    descriptions_list: list[str] = all_stocks_Data['description'].unique().tolist()
    return descriptions_list


def create_graphs_folders_locally() -> None:
    import os
    try:
        os.mkdir(f'{settings.GRAPH_IMAGES}')
    except FileExistsError:
        pass
    for i in range(1, 4 + 1):
        try:
            os.mkdir(f'{settings.GRAPH_IMAGES}{i}/')
        except FileExistsError:
            pass
        try:
            os.mkdir(f'{settings.GRAPH_IMAGES}{i}/00/')
        except FileExistsError:
            pass
        try:
            os.mkdir(f'{settings.GRAPH_IMAGES}{i}/01/')
        except FileExistsError:
            pass
        try:
            os.mkdir(f'{settings.GRAPH_IMAGES}{i}/10/')
        except FileExistsError:
            pass
        try:
            os.mkdir(f'{settings.GRAPH_IMAGES}{i}/11/')
        except FileExistsError:
            pass


def create_graphs_folders_aws_s3(bucket_name: str) -> None:
    import boto3

    # Create an S3 client
    s3 = boto3.client('s3')
    # Define the folder structure you want to create
    folders = [
        '1/',
        '1/00/',
        '1/01/',
        '1/10/',
        '1/11/',
        '2/',
        '2/00/',
        '2/01/',
        '2/10/',
        '2/11/',
        '3/',
        '3/00/',
        '3/01/',
        '3/10/',
        '3/11/',
        '4/',
        '4/00/',
        '4/01/',
        '4/10/',
        '4/11/',
    ]
    # Create empty objects with the folder prefixes
    for folder in folders:
        s3.put_object(Bucket=bucket_name, Key=f'{settings.GRAPH_IMAGES}{folder}')


def currency_exchange(from_currency="USD", to_currency="ILS"):
    start_date = (data_time.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    end_date = (data_time.now() + timedelta(days=1)).strftime('%Y-%m-%d')  # To ensure we get today's data

    ticker = f'{from_currency}{to_currency}=X'  # Yahoo Finance symbol
    data = yf.download(ticker, start=start_date, end=end_date)

    if not data.empty:
        latest_exchange_rate = data['Close'].iloc[-1]
        return latest_exchange_rate
    else:
        raise ValueError("No exchange rate data available for the given date range.")


def save_all_stocks():  # dont delete it
    path: str = f"{settings.CONFIG_RESOURCE_LOCATION}all_stocks_basic_data.csv"
    sectors_data = get_sectors_data_from_file()
    # Assuming you have lists named list_symbol, list_sector, and list_description
    list_symbol = []
    list_sector = []
    list_description = []

    # Assuming you have lists named list_symbol, list_sector, and list_description

    for i in range(len(sectors_data)):
        sector_name = sectors_data[i]["name"]
        stocks_data = sectors_data[i]["stocks"]

        for stock_symbol in stocks_data:
            list_symbol.append(stock_symbol)
            list_sector.append(sector_name)
            description = get_stocks_descriptions([stock_symbol], is_reverse_mode=False)[1]
            list_description.append(description)

    data = list(zip(list_symbol, list_sector, list_description))

    with open(path, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Symbol', 'sector', 'description'])

        # Write data rows
        for row in data:
            csv_writer.writerow(row)

    print("CSV file created successfully!")


def save_usa_indexes_table():  # dont delete it
    sectors_data = get_sectors_data_from_file()
    stock_data_list = []
    # create table
    for i in range(3, 6):
        stocks = sectors_data[i]["stocks"]
        for j in range(len(stocks)):
            stock_info = yf.Ticker(stocks[j]).info
            stock_data_list.append(stock_info)

    # Create a set of all keys present in the stock data dictionaries
    all_keys = set()
    for stock_data in stock_data_list:
        all_keys.update(stock_data.keys())

    # Define the CSV file path
    csv_file_path: str = f'{settings.CONFIG_RESOURCE_LOCATION}usa_indexes.csv'

    # Write the data to a CSV file
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=all_keys)

        writer.writeheader()
        for stock_data in stock_data_list:
            writer.writerow(stock_data)


def save_json_data(path: str, sectors_json_file) -> None:
    with open(f"{path}.json", 'w', encoding='utf-8') as f:
        json.dump(sectors_json_file, f, ensure_ascii=False, indent=4)


def convert_data_stream_to_pd(file_stream):
    # Convert the binary content directly to a Pandas DataFrame
    return pd.read_csv(file_stream, encoding='utf-8', index_col=0)


def convert_data_stream_to_png(file_stream):
    import io
    from PIL import Image
    # Create an Image object from the binary stream
    return Image.open(io.BytesIO(file_stream.read()))


def convert_data_stream_to_json(file_stream):
    # Convert the binary content directly to a Pandas DataFrame
    return json.load(file_stream)


def get_sorted_path(full_path, num_of_last_elements):
    split_path = full_path.split('/')
    return '/'.join(split_path[-num_of_last_elements:])
