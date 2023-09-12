import requests
import json

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from service.config import settings


def lstm_model(df=None, forecast_out: int = 20,
               use_features: bool = True) -> tuple[pd.DataFrame, np.longdouble, np.longdouble]:
    import datetime
    import tensorflow as tf
    from keras.models import Sequential
    from keras import regularizers
    from tensorflow.python.keras.callbacks import EarlyStopping
    from sklearn.metrics import mean_squared_error

    np.random.seed(1)
    start_date = df.index[0].strftime("%Y-%m-%d")
    end_date = df.index[-1].strftime("%Y-%m-%d")
    forecast_col = "Forecast"
    df[forecast_col] = df['Col']
    days = 20
    df['Label'] = df[forecast_col].pct_change().shift(-forecast_out)
    df = df.dropna(subset=['Label'])
    df = df[df['Label'] != 0.0]

    df_final = df.copy()
    df_final = df_final.drop(['Forecast'], axis=1)
    df_final["Col"] = df_final["Col"].pct_change()

    # drop weekends
    df_final['Date'] = pd.to_datetime(df_final.index)
    df_final['Year'] = df_final['Date'].dt.year
    df_final['Month'] = df_final['Date'].dt.month
    df_final = drop_weekends(df_final)

    if use_features:
        # feature for scaling
        df_final = lstm_add_unemployment_rate_and_cpi(df_final, start_date, end_date)
        df_final, api_key = lstm_add_interest_rate(df_final, start_date, end_date)
        df_final = lstm_add_gdp_growth_rate(df_final, start_date, end_date, api_key)
        # lstm_confusion_matrix(df_final, forecast_col)

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
        scaled_data.to_csv(f'{settings.RESEARCH_LOCATION}LSTM_Final-Running.csv')
    else:
        scaled_data = df_final
        # Sliding Window
        scaled_data = scaled_data.dropna(thresh=(scaled_data.shape[1] - 5))
        scaled_data = scaled_data[scaled_data['Label'] != 0]
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

    # create an early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

    opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.001, clipvalue=1.0)
    # Compile model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )
    # model.compile(loss='mse', optimizer=opt)

    # Train the model
    model.fit(X_train, y_train, epochs=4, batch_size=64, validation_split=0.15,
              callbacks=[early_stopping])  # add the early stopping callback here

    # Predict on the test data
    predictions = model.predict(X_test)
    predictions = [x for x in predictions]
    # predictions['Date'] = date_column
    # Evaluate model performance
    mse = mean_squared_error(y_test, predictions)
    print('Test MSE error: ', mse)

    # Results
    # Adjusted percentage change prediction
    scaled_data['Forecast'] = np.nan
    df[forecast_col] = np.nan

    last_date = scaled_data.iloc[-1].name
    last_unix = last_date.timestamp()
    one_day = 86400
    next_unix = last_unix + one_day

    predictions = np.concatenate(predictions)
    last_price = df["Col"].iloc[-1]
    for i in predictions:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += 86400
        scaled_data.loc[next_date] = [np.nan for _ in range(len(scaled_data.columns) - 1)] + [i]
        df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

        current_val = (1 + i / 20) * last_price

        last_price = current_val
        df[forecast_col].loc[next_date] = current_val

    return df


# LSTM helpers functions
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

    if df_interest_rates is not None and not df_interest_rates.empty:
        df_interest_rates['Date'] = pd.to_datetime(df_interest_rates["Date"])

    merged_df = pd.merge(df_final, df_interest_rates, left_on='Date', right_on='Date', how='left')
    merged_df = lstm_fill_na_values(merged_df)
    return merged_df, api_key


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
    if df_gdp_growth_rate is not None and not df_gdp_growth_rate.empty:
        joined_df = pd.merge_asof(merged_df, df_gdp_growth_rate, on="Date", direction="backward")
        # Perform data alignment and fill missing values
        joined_df["GDP Growth Rate"] = joined_df["GDP Growth Rate"].ffill()
        return joined_df
    else:
        return None


def drop_weekends(df_final: pd.DataFrame):
    # Count the number of zeroes in each row
    zero_counts = (df_final == 0.0).sum(axis=1)

    df_final = df_final.drop(['Month', 'Year'], axis=1)
    # Filter rows with fewer or equal to 5 zeroes
    df_final = df_final[zero_counts <= 5]
    df_final = fillna_custom(df_final)
    return df_final


def fillna_custom(df: pd.DataFrame):
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


def lstm_fill_na_values(df_final):
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
    plt.clf()
    plt.cla()
    plt.close()


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
    plt.clf()
    plt.cla()
    plt.close()


def lstm_show_data_plot_wth_labels(df_final: pd.DataFrame, tickers_df: pd.DataFrame, forecast_col):
    from matplotlib import rcParams

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
    plt.clf()
    plt.cla()
    plt.close()
    # price by years with labels
    tickers_df = tickers_df.drop(['Date', 'Year', forecast_col], axis=1)

    # Selecting only columns that start with 'ADJ_PCT_change_'
    columns_to_plot = [col for col in tickers_df.columns if col.startswith('ADJ_PCT_change_')]
    # Melting the dataframe to have a format suitable for boxplots for multiple columns
    df_melted = pd.melt(tickers_df, value_vars=columns_to_plot)

    # Plotting boxplot using seaborn
    plt.figure(figsize=(20, 10))
    sns.boxplot(x="variable", y="value", data=df_melted)
    plt.xlabel('ticker')
    plt.title('Adjusted Percentage Change by Ticker')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability if they're long
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()


def lstm_show_snap_graph(seq_len, input_features, X_test, shap_days, model):
    import shap

    # Initialize JS visualization code
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

    # Save the plot to a file
    plt.savefig(f'{settings.RESEARCH_LOCATION}LSTM_Final-Running.png')
