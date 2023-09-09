import codecs
import csv
import datetime
import io
import requests
from datetime import datetime as data_time, timedelta
import json
import math
import os

from bidi import algorithm as bidi_algorithm

import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt, rcParams
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pmdarima as pm
from prophet import Prophet

from service.config import settings
from service.impl.sector import Sector
from service.util import tase_interaction

from PIL import Image

# lstm imports
import matplotlib.dates as mdates
import seaborn as sns
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from keras import regularizers
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam
import shap

# Global variables
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
                    sectors_weights[i] += stocks_weights[j]
            except ValueError:
                # The first component is not a valid integer, use it for string comparison
                if first_component in sectors[i].stocks:
                    sectors_weights[i] += stocks_weights[j]

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

    """def lstm_model(self) -> tuple[pd.DataFrame, np.longdouble, np.longdouble]:
        df, forecast_out = self.get_final_dataframe()

        df.index = pd.to_datetime(self._table_index)

        # Perform GBM forecasting
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        X = np.arange(len(df))[:, None]  # Use a simple sequence as features for demonstration
        y = df['Col'].values
        model.fit(X, y)
        forecast = model.predict(np.arange(len(df), len(df) + forecast_out)[:, None])
        df['Forecast'] = np.nan
        df.loc[df.index[-forecast_out]:, 'Forecast'] = forecast

        # add dates
        last_date = df.iloc[-1].name
        last_unix: pd.Timestamp = last_date.timestamp()
        next_unix: float = last_unix + SINGLE_DAY

        for i in forecast:
            next_date: datetime.datetime = datetime.datetime.fromtimestamp(next_unix)
            next_unix += SINGLE_DAY
            df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

        forecast_with_historical_returns_annual, expected_returns = self.calculate_returns(df,
                                                                                           forecast_out=forecast_out)
        return df, forecast_with_historical_returns_annual, expected_returns"""

    def lstm_model(self, pct_change_mode=False, use_features=True) -> tuple[pd.DataFrame, np.longdouble, np.longdouble]:
        df_final, forecast_out = self.get_final_dataframe()
        start_date = self._table_index[0].strftime("%Y-%m-%d")
        end_date = self._table_index[-1].strftime("%Y-%m-%d")
        days = forecast_out
        df_final.index = pd.to_datetime(self._table_index)
        if not pct_change_mode:
            closing_price_df = df_final
            df_final = df_final.pct_change()
        else:
            closing_price_df = None
        forecast_col = "Forecast"
        df_final[forecast_col] = df_final['Col']

        df_final.fillna(value=-0, inplace=True)
        df_final['Label'] = df_final[forecast_col].shift(-forecast_out)

        df_final = df_final.dropna(subset=['Label'])
        df_final = df_final[df_final['Label'] != 0.0]
        df_final['Date'] = df_final.index

        # lstm_show_plt_graph(df_final, mode='M')
        # lstm_show_plt_graph(df_final, mode='Y')

        # Count the number of zeroes in each row
        zero_counts = (df_final == 0.0).sum(axis=1)
        # df_final = df_final.drop(['Month', 'Year'], axis=1)
        # Filter rows with fewer or equal to 5 zeroes
        df_final = df_final[zero_counts <= 5]

        # drop weekends
        df_final = drop_weekends(df_final)

        if use_features:
            lstm_show_data_plot_wth_labels(df_final, forecast_col)
            # feature for scaling
            df_final = lstm_add_unemployment_rate_and_cpi(df_final=df_final, start_date=start_date, end_date=end_date)
            df_final = lstm_add_interest_rate(df_final=df_final, start_date=start_date, end_date=end_date)
            df_final = lstm_add_gdp_growth_rate(
                merged_df=df_final, start_date=start_date, end_date=end_date, api_key=None  # TODO: add API Key
            )
            lstm_confusion_matrix(df_final, forecast_col)

            # Scaling
            cols_to_scale = ['CPI', 'unemployment_rate', 'Interest Rate', 'GDP Growth Rate']
            df_final[cols_to_scale] /= 100
            df_final[['CPI', 'unemployment_rate']] /= 30
            df_final[['GDP Growth Rate']] /= 90
            scaled_data = df_final
            # all tickers columns
            tickers_cols_to_scale = scaled_data.columns.drop(['Label', 'Date'] + cols_to_scale)
            scaled_data[tickers_cols_to_scale] *= 100
            # Sliding Window
            scaled_data = scaled_data.dropna(thresh=(scaled_data.shape[1] - 5))
            scaled_data = scaled_data[scaled_data['Label'] != 0]
            scaled_data.to_csv(f'{settings.RESEARCH_LOCATION}LSTM_Final-Runung.csv')
        else:
            # tickers_df = df_final.drop(['Date', forecast_col, 'Label'], axis=1)  # TODO: unused
            scaled_data = df_final
            tickers_cols_to_scale = scaled_data.columns.drop(['Date', 'Label'])
            scaled_data[tickers_cols_to_scale] *= 100
            # Sliding Window
            scaled_data = scaled_data.dropna(thresh=(scaled_data.shape[1] - 5))
            scaled_data = scaled_data[scaled_data['Label'] != 0]

        seq_len = days

        # train , takes time
        # Convert dataframe into array of sequences for LSTM
        X, y = [], []
        scaled_data = scaled_data.set_index('Date')
        input_features = list(scaled_data.columns)
        input_features.remove('Label')

        output_feature = ['Label']
        for i in range(seq_len, len(scaled_data) - seq_len):
            X.append(scaled_data[input_features].iloc[i - seq_len:i].values)
            y.append(scaled_data[output_feature].iloc[i])
        scaled_data = scaled_data.dropna(thresh=(scaled_data.shape[1] - 5))
        X, y = np.array(X), np.array(y)

        # Split data into train and test sets
        X_train, X_test = X[:int(X.shape[0] * 0.85)], X[int(X.shape[0] * 0.85):]
        y_train, y_test = y[:int(y.shape[0] * 0.85)], y[int(y.shape[0] * 0.85):]

        # Reshape input to be 3D [samples, timesteps, features]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], len(input_features)))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], len(input_features)))

        # Model
        tf.compat.v1.disable_v2_behavior()
        dropout = 0.2
        model = Sequential()

        model.add(
            tf.keras.layers.LSTM(units=days, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(rate=dropout))
        model.add(tf.keras.layers.LSTM(units=days, return_sequences=True))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(rate=dropout))
        model.add(tf.keras.layers.LSTM(units=days, return_sequences=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(rate=dropout))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        model.add(tf.keras.layers.Dense(units=1))

        # takes long time
        # create an early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

        # opt = Adam(lr=0.001, clipvalue=1.0)

        optimizer = Adam(learning_rate=0.001)

        # model.compile(loss='mse', optimizer=opt)
        model.compile(loss='mse', optimizer=optimizer)

        # Train the model
        # add the early stopping callback here
        model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.15, callbacks=[early_stopping])

        # Predict on the test data
        predictions = model.predict(X_test)
        predictions = [x for x in predictions]
        # predictions['Date'] = date_column
        # Evaluate model performance
        mse = mean_squared_error(y_test, predictions)
        print('Test MSE error: ', mse)

        # Results
        # Adjusted percentage change prediction
        # date_values = scaled_data.index  # TODO: unused

        scaled_data['Forecast'] = np.nan

        last_date = scaled_data.iloc[-1].name
        last_unix = last_date.timestamp()
        one_day = 86400
        next_unix = last_unix + one_day

        predictions = np.concatenate(predictions)

        for i in predictions:
            next_date = datetime.datetime.fromtimestamp(next_unix)
            next_unix += 86400
            scaled_data.loc[next_date] = [np.nan for _ in range(len(scaled_data.columns) - 1)] + [i]

        # daily change and forecast
        scaled_data['Label'].plot()
        scaled_data['Forecast'].plot()
        plt.legend(loc=4)
        plt.xlabel('Date')
        plt.ylabel('Precentage Change')
        plt.show()

        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv(f'{settings.RESEARCH_LOCATION}lstm_predictions.csv')

        if not pct_change_mode and closing_price_df:
            # Closing Price Prediction
            closing_price_df = closing_price_df.dropna()
            last_price = closing_price_df.iloc[-1]

            forecast_price = []
            for i in list(predictions):
                current_val = (1 + i) * last_price

                last_price = current_val
                # print(last_price)
                forecast_price.append(last_price)
            print(((1 + df_final['ADJ_PCT_change_SPY'].mean()) ** 254 - 1) * 100)
            # print(((1+df_final['ADJ_PCT_change_SPY'].std())*(254**0.5)))
            realchancg = scaled_data['Forecast'].mean()
            print((1 + realchancg) ** 254 - 1)

            # plot only forecast closing price graph
            new_dates = scaled_data.dropna(subset='Forecast').index

            new_spy_forecast_price = pd.Series(closing_price_df, index=new_dates)

            fig, ax = plt.subplots(figsize=(14, 5))

            # Plot the actual and predicted values
            ax.plot(new_spy_forecast_price, label='Predictions')

            # Format the x-axis to display only month and year
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

            ax.set_title('S&P 500 Closing Price Prediction')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')

        if use_features:
            lstm_show_snap_graph(seq_len, input_features, X_test, shap_days=1, model=model)

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
                df['Label'].pct_change().mean())  # Generates RuntimeWarning: invalid value encountered in log1p
            forecast_with_historical_returns_annual: np.longdouble = (np.exp(longdouble * logged_label_mean) - 1) * 100
            logged_forecast_mean: np.ndarray = np.log1p(
                df['Forecast'].pct_change().mean())  # Generates RuntimeWarning: invalid value encountered in log1p
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
                # list to DateFrame
                df["tradeDate"] = pd.to_datetime(df["tradeDate"])
                df.set_index("tradeDate", inplace=True)
                if is_index_type:
                    price = df[["closingIndexPrice"]]
                else:
                    price = df[["closingPrice"]]
                frame[stocks_names[i]] = price
            except ValueError:
                print('Invalid start_date or end_date format, should be %Y-%m-%d')
            except (AttributeError, IndexError):
                print(f"Error in stock: {stock}")
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


def create_graphs_folders() -> None:
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


# Drop Weekends and Fill Nan
def drop_weekends(df):
    for col in df.columns:
        not_nan_indices = df[col].dropna().index
        for i in df[col].index:
            if pd.isna(df.at[i, col]):
                previous_indices = not_nan_indices[not_nan_indices < i]
                next_indices = not_nan_indices[not_nan_indices > i]
                if previous_indices.empty:
                    df.at[i, col] = df.at[next_indices[0], col]
                elif next_indices.empty:
                    df.at[i, col] = df.at[previous_indices[-1], col]
                else:
                    df.at[i, col] = (df.at[previous_indices[-1], col] + df.at[next_indices[0], col]) / 2
    return df


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
    # Create an Image object from the binary stream
    return Image.open(io.BytesIO(file_stream.read()))


def convert_data_stream_to_json(file_stream):
    # Convert the binary content directly to a Pandas DataFrame
    return json.load(file_stream)


def get_sorted_path(full_path, num_of_last_elements):
    split_path = full_path.split('/')
    return '/'.join(split_path[-num_of_last_elements:])


# lstm helpers functions
def lstm_add_unemployment_rate_and_cpi(df_final, start_date, end_date):
    # Add unemployment rate and Consumer Price Index maybe not relevant
    # Your API key from the BLS website goes here
    api_key = "9b57d771cf414aa49a022707be95e269"

    # Series ID for the Consumer Price Index for All Urban Consumers: All Items
    cpi_series_id = "CUUR0000SA0"

    # Series ID for the Unemployment Rate
    unemployment_series_id = "LNS14000000"

    headers = {"Content-type": "application/json"}

    data = json.dumps({
        "seriesid": [cpi_series_id, unemployment_series_id],
        "startyear": start_date[:4],
        "endyear": end_date[:4],
        "registrationKey": api_key
    })
    # U.S. BUREAU OF LABOR STATISTICS website
    response = requests.post("https://api.bls.gov/publicAPI/v2/timeseries/data/", data=data, headers=headers)
    response_data = json.loads(response.text)
    # Extract the series data
    series_data = response_data['Results']['series']

    # Initialize empty lists to store the extracted data
    cpi_data = []
    unemployment_data = []

    # Extract data for CPI and unemployment rate separately
    for series in series_data:
        series_id = series['seriesID']
        series_data = series['data']
        for data in series_data:
            year = int(data['year'])
            period = data['period']
            period_name = data['periodName']
            value = float(data['value'])
            if series_id == 'CUUR0000SA0':
                cpi_data.append({'year': year, 'period': period, 'periodName': period_name, 'CPI': value})
            elif series_id == 'LNS14000000':
                unemployment_data.append(
                    {'year': year, 'period': period, 'periodName': period_name, 'unemployment_rate': value})

    # Create DataFrames from the extracted data
    df_cpi = pd.DataFrame(cpi_data)
    df_unemployment = pd.DataFrame(unemployment_data)

    # Merge the two DataFrames based on the 'year' and 'period' columns
    inflation_and_unemployment_data = pd.merge(df_cpi, df_unemployment, on=['year', 'period', 'periodName'],
                                               how='outer').rename(columns={'period': 'month'})
    inflation_and_unemployment_data['month'] = inflation_and_unemployment_data['month'].str.replace('M', '').astype(
        int)

    inflation_and_unemployment_data['date'] = pd.to_datetime(
        inflation_and_unemployment_data[['year', 'month']].assign(day=1))
    df_final['temp_date'] = df_final['Date'].apply(lambda dt: dt.replace(day=1))

    merged_df = pd.merge(df_final, inflation_and_unemployment_data, left_on='temp_date', right_on='date',
                         how='left').drop(['temp_date', 'date', 'year', 'month',
                                           'periodName'], axis=1)

    return merged_df


def lstm_add_interest_rate(df_final, start_date, end_date):
    # Add interest rate
    # Set your FRED API key
    api_key = "5391d650f6bc47fe6d288fd9b7b7b366"

    # Define the series ID for interest rates
    interest_rate_series_id = "DGS10"  # Example: 10-year Treasury constant maturity rate

    # The Economic Research Division of the Federal Reserve Bank of St. Louis website
    interest_rate_url = (
        f"https://api.stlouisfed.org/fred/series/observations?series_id={interest_rate_series_id}"
        f"&api_key={api_key}&file_type=json&observation_start={start_date}&observation_end={end_date}"
    )

    try:
        # Send a GET request to the FRED API for interest rates
        interest_rate_response = requests.get(interest_rate_url)

        # Check if the request was successful
        if interest_rate_response.status_code == 200:
            interest_rate_data = interest_rate_response.json()

            # Extract historical interest rate observations
            interest_rate_observations = interest_rate_data["observations"]

            # Create an empty DataFrame
            df_interest_rates = pd.DataFrame(columns=["Date", "Interest Rate"])

            # Populate the DataFrame with the interest rate data
            for observation in interest_rate_observations:
                date = observation["date"]
                value = observation["value"]
                df_interest_rates = df_interest_rates._append({"Date": date, "Interest Rate": value}, ignore_index=True)

        else:
            df_interest_rates = None
            print("Error occurred while fetching interest rate data from the API.")
    except requests.exceptions.RequestException as e:
        df_interest_rates = None
        print("An error occurred:", e)

    if df_interest_rates:
        df_interest_rates['Date'] = pd.to_datetime(df_interest_rates["Date"])

    merged_df = pd.merge(df_final, df_interest_rates, left_on='Date', right_on='Date', how='left')
    merged_df = lstm_fill_na_values(merged_df)
    return merged_df


def lstm_add_gdp_growth_rate(merged_df, start_date, end_date, api_key) -> pd.DataFrame | None:
    # Add GDP growth
    # Define the series ID for GDP growth rate
    gdp_growth_rate_series_id: str = "A191RL1Q225SBEA"

    # The Economic Research Division of the Federal Reserve Bank of St. Louis website
    gdp_growth_rate_url: str = (
        f"https://api.stlouisfed.org/fred/series/observations?series_id={gdp_growth_rate_series_id}"
        f"&api_key={api_key}&file_type=json&observation_start={start_date}&observation_end={end_date}"
    )

    try:
        # Send a GET request to the FRED API for GDP growth rate
        gdp_growth_rate_response = requests.get(gdp_growth_rate_url)
        # Check if the request was successful
        if gdp_growth_rate_response.status_code == 200:
            gdp_growth_rate_data = gdp_growth_rate_response.json()
            # Extract historical GDP growth rate observations
            gdp_growth_rate_observations = gdp_growth_rate_data["observations"]
            # Create an empty DataFrame
            df_gdp_growth_rate = pd.DataFrame(columns=["Date", "GDP Growth Rate"])
            # Populate the DataFrame with the GDP growth rate data
            for observation in gdp_growth_rate_observations:
                date = observation["date"]
                value = float(observation["value"])
                df_gdp_growth_rate = df_gdp_growth_rate._append(
                    {"Date": date, "GDP Growth Rate": value}, ignore_index=True
                )
            # Convert "Date" column to datetime format
            df_gdp_growth_rate["Date"] = pd.to_datetime(df_gdp_growth_rate["Date"])
        else:
            df_gdp_growth_rate = None
            print("Error occurred while fetching GDP growth rate data from the API.")
    except requests.exceptions.RequestException as e:
        df_gdp_growth_rate = None
        print("An error occurred:", e)
    # Join the data using the available GDP growth observations
    if df_gdp_growth_rate:
        joined_df = pd.merge_asof(merged_df, df_gdp_growth_rate, on="Date", direction="backward")
        # Perform data alignment and fill missing values
        joined_df["GDP Growth Rate"] = joined_df["GDP Growth Rate"].ffill()
        return joined_df
    else:
        return None


def lstm_fill_na_values(df_final):
    # Fill NA Values
    # TODO: unused
    # zero_mask_columns = df_final.eq('.').any(axis=0)

    # Convert the column to numeric, treating '.' as NaN
    df_final['Interest Rate'] = pd.to_numeric(df_final['Interest Rate'], errors='coerce')

    # Find the indices of '.' values
    dot_indices = df_final.index[df_final['Interest Rate'].isna()]

    # Replace '.' values with the average of the previous and next rows
    for idx in dot_indices:
        prev_value = df_final.at[idx - 1, 'Interest Rate']
        average = prev_value
        df_final.at[idx, 'Interest Rate'] = average

    return df_final


def lstm_confusion_matrix(merged_df, forecast_col):
    # Confusion Matrix
    sns.set(font_scale=0.7)  # Decrease font size
    plt.figure(figsize=(20, 12))

    # Calculate correlation matrix and round to 3 decimal places
    correlation_matrix = merged_df.rename(columns={'Label': forecast_col}).corr().round(3)

    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.show()


def lstm_show_plt_graph(df_final, mode):
    # montly graph
    df_final['Year'] = df_final['Date'].dt.year
    df_final['Month'] = df_final['Date'].dt.month
    if mode == 'M':
        df_monthly = df_final.groupby(['Year', 'Month']).apply(lambda x: ((1 + x['Label'].mean()) ** 21 - 1) * 100)
        df_monthly.plot(figsize=(10, 6))
        plt.title('Monthly Return')
        plt.xlabel('Year, Month')
    else:
        df_annualized = df_final.groupby('Year').apply(lambda x: ((1 + x['Label'].mean()) ** 254 - 1) * 100)
        df_annualized.plot(figsize=(10, 6))
        plt.title('Annualized Return')
        plt.xlabel('Year')

    plt.ylabel('Return (%)')
    plt.show()


def lstm_show_data_plot_wth_labels(df_final, forecast_col):
    rcParams['figure.figsize'] = 14, 8
    sns.set(style='whitegrid', palette='muted', font_scale=1.5)

    # data plot
    ax = df_final.plot(x='Date', y='Label')
    ax.set_xlabel('Year')

    ax.set_ylabel('Price')
    ax.set_title('Price Over Time')

    # price by years
    # Extract the year from the 'Date' column and create a new 'Year' column
    df_final['Year'] = df_final['Date'].dt.year

    # Create boxplot
    plt.figure(figsize=(20, 10))  # Optional, for adjusting figure size
    sns.boxplot(x='Year', y=forecast_col, data=df_final)
    plt.title('Price by Year')
    plt.show()
    # price by years with lables
    tickers_df = df_final.drop(['Date', 'Year', forecast_col], axis=1)

    # Selecting only columns that start with 'ADJ_PCT_change_'
    # columns_to_plot = [col for col in tickers_df.columns if col.startswith('ADJ_PCT_change_')]
    columns_to_plot = tickers_df.columns
    # Melting the dataframe to have a format suitable for boxplots for multiple columns
    df_melted = pd.melt(tickers_df, value_vars=columns_to_plot)

    # Plotting boxplot using seaborn
    plt.figure(figsize=(20, 10))
    sns.boxplot(x="variable", y="value", data=df_melted)
    plt.xlabel('ticker')
    plt.title('Adjusted Percentage Change by Ticker')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability if they're long
    plt.show()


def lstm_show_snap_graph(seq_len, input_features, X_test, shap_days, model):
    # shap          takes too long time
    # Initialize JS visualization code
    shap.initjs()

    # TODO unused
    # feature_names = [f"{feature}_{t}" for t in range(seq_len) for feature in input_features]

    # Define a predict function wrapper to handle 3D input
    def lstm_predict_wrapper(x):
        x = x.reshape((x.shape[0], seq_len, len(input_features)))  # Reshape to [samples, timesteps, features]
        return model.predict(x)

    # Create a 2D version of your test data for SHAP
    X_test_2d = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2]))

    # Compute SHAP values for the first 100 test samples
    # Note: SHAP can be slow, so you might want to compute it for a subset of your data first
    explainer = shap.KernelExplainer(lstm_predict_wrapper, shap.sample(X_test_2d, shap_days))
    shap_values = explainer.shap_values(X_test_2d[:shap_days])

    # Convert shap_values to numpy array
    shap_values_array = np.array([np.array(vals) for vals in shap_values])

    # Reshape shap_values_array into (n_samples, seq_len, n_features)
    shap_values_reshaped = shap_values_array.reshape(-1, seq_len, len(input_features))

    # Sum shap_values over the seq_len dimension
    aggregated_shap_values = shap_values_reshaped.sum(axis=1)

    # Create a summary plot of the aggregated SHAP values
    shap.summary_plot(aggregated_shap_values, feature_names=input_features)
