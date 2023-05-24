import yfinance as yf
import numpy as np
import pandas as pd
import datetime
import math
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import json
import codecs
from api import Sector as Sector
from util import taseUtil


def getBestPortfolios(optionalPortfolios):
    return [optionalPortfolios['Safest Portfolio'],optionalPortfolios['Sharpe Portfolio'], optionalPortfolios['Max Risk Porfolio']]


def getBestWeightsColumn(optionalPortfolios, pctChangeTable):
    pctChangeTable.dropna(inplace=True)
    low = np.dot(optionalPortfolios[0].iloc[0][3:], pctChangeTable.T)
    medium = np.dot(optionalPortfolios[1].iloc[0][3:], pctChangeTable.T)
    high = np.dot(optionalPortfolios[2].iloc[0][3:], pctChangeTable.T)
    return [low, medium, high]

def getThreeBestweights(optionalPortfolios):
    weighted_low = optionalPortfolios[0].iloc[0][3:]
    weighted_medium = optionalPortfolios[1].iloc[0][3:]
    weighted_high = optionalPortfolios[2].iloc[0][3:]
    return [weighted_low, weighted_medium, weighted_high]


def choosePortfolioByRiskScore(optionalPortfoliosList, riskScore):
        if 0 < riskScore <= 4:
            return optionalPortfoliosList[0]
        if 5 < riskScore <= 7:
            return optionalPortfoliosList[1]
        if riskScore > 7:
            return optionalPortfoliosList[2]


def buildReturnGiniPortfoliosDic(df):
        returnDic = {'Max Risk Porfolio': {}, 'Safest Portfolio': {}, 'Sharpe Portfolio': {}}
        min_gini = df['Gini'].min()
        max_sharpe = df['Sharpe Ratio'].max()
        max_profolio_annual = df['Profolio_annual'].max()

        # use the min, max values to locate and create the two special portfolios
        sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
        safe_portfolio = df.loc[df['Gini'] == min_gini]
        max_portfolio = df.loc[df['Profolio_annual'] == max_profolio_annual]

        returnDic['Max Risk Porfolio'] = max_portfolio
        returnDic['Safest Portfolio'] = safe_portfolio
        returnDic['Sharpe Portfolio'] = sharpe_portfolio
        return returnDic


def buildReturnMarkowitzPortfoliosDic(df):
    returnDic = {'Max Risk Porfolio': {}, 'Safest Portfolio': {}, 'Sharpe Portfolio': {}}
    min_Markowiz = df['Volatility'].min()
    max_sharpe = df['Sharpe Ratio'].max()
    max_profolio_annual = df['Returns'].max()

    # use the min, max values to locate and create the two special portfolios
    sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
    safe_portfolio = df.loc[df['Volatility'] == min_Markowiz]
    max_portfolio = df.loc[df['Returns'] == max_profolio_annual]

    returnDic['Max Risk Porfolio'] = max_portfolio
    returnDic['Safest Portfolio'] = safe_portfolio
    returnDic['Sharpe Portfolio'] = sharpe_portfolio
    return returnDic


def convertDataToTables(stocksNames, record_percentage_to_predict,
        numOfYearsHistory, machineLearningOpt):
    frame = {}
    yf.pdr_override()
    start_date, end_date = getfromAndToDate(numOfYearsHistory)

    for i, stock in enumerate(stocksNames):
        if type(stock) == int:
            IsraeliStockData = getIsraeliIndexesData("get_past_10_years_history", stock)
            df = pd.DataFrame(IsraeliStockData["indexEndOfDay"]["result"])
            df["tradeDate"] = pd.to_datetime(df["tradeDate"])
            df.set_index("tradeDate", inplace=True)
            if machineLearningOpt == 1:
                DF, price = price_forecast(df, record_percentage_to_predict, 1)  # 1-Data From Tase
            else:
                price = df[["closingIndexPrice"]]
            frame[stocksNames[i]] = price
        else:
            df = yf.download(stock, start=start_date, end=end_date)
            if machineLearningOpt == 1:
                DF, price = price_forecast(df, record_percentage_to_predict, 0)  # 0 -Data from Yahoo
            else:
                price = df[["Adj Close"]]
            frame[stock] = price

    closingPricesTable = pd.concat(frame.values(), axis=1, keys=frame.keys())

    return closingPricesTable


def getJsonData(name):
    with codecs.open(name + ".json", "r", encoding="utf-8") as file:
        json_data = json.load(file)
    return json_data


def getSectorsDataFromFile():
    sectorsData = getJsonData("api/resources/sectors")
    return sectorsData['sectorsList']['result']


def setSectors(stocksSymbols):
    sectorsData = getSectorsDataFromFile()
    sectorsList = []
    if(len(sectorsData))>0:
        for i in range(len(sectorsData)):
            sector = Sector.Sector(sectorsData[i]['sectorName'])
            for j in range (len(stocksSymbols)):
                if stocksSymbols[j] in sectorsData[i]['stocks']:
                    sector.addStock(stocksSymbols[j])
            if(len(sector.getStocks())>0):
                sectorsList.append(sector)
    return sectorsList


def returnSectorsWeightsAccordingToStocksWeights(sectorsList, stockSymbols, stocksWeights):
    sectorsWeights = [0.0] * len(sectorsList)
    for i in range(len(sectorsList)):
        sectorsWeights[i] = 0
        for j in range(len(stockSymbols)):
            if stockSymbols[j] in sectorsList[i].getStocks():
                sectorsWeights[i] += stocksWeights[j]
    return sectorsWeights


def getThreeBestSectorsWeights(sectorsList, stocksSymbols, threeBestStocksWeights):
    sectorsWeightsList = []
    for i in range(len(threeBestStocksWeights)):
        sectorsWeightsList.append(returnSectorsWeightsAccordingToStocksWeights(sectorsList, stocksSymbols, threeBestStocksWeights[i]))
    return sectorsWeightsList


def getfromAndToDate(numOfYears):
    today = datetime.datetime.now()
    startYear = today.year - numOfYears
    startMounth = today.month
    startDay = today.day
    endYear = today.year
    endMonth = today.month
    endDay = today.day
    fromDate = str(startYear) + "-" + str(startMounth) + "-" + str(startDay)
    toDate = str(endYear) + "-" + str(endMonth) + "-" + str(endDay)
    return fromDate, toDate


def getIsraeliIndexesData(command, israeliIndexes):# get israli indexes data from tase
    return taseUtil.getIsraeliIndexesData(command, israeliIndexes)


def getJsonData(name):
    with codecs.open(name + ".json", "r", encoding="utf-8") as file:
        json_data = json.load(file)
    return json_data


def convertUsaIndexToName(UsaIndexes):
    UsaIndexesNames = {}
    for i, index in enumerate(UsaIndexes):
        ticker = yf.Ticker(index)
        UsaIndexesNames[i] = ticker.info["longName"]

    return UsaIndexesNames.values()


def price_forecast(df, record_percentage_to_predict, isDataFromTase):

    if isDataFromTase == 1:
        df["HL_PCT"] = (df["high"] - df["low"]) / df["low"] * 100.0
        df["PCT_change"] = (
            (df["closingIndexPrice"] - df["indexOpeningPrice"])
            / df["indexOpeningPrice"]
            * 100.0
        )

        df = df[["closingIndexPrice", "HL_PCT", "PCT_change"]]

        forecast_col = "closingIndexPrice"

    else:
        df["HL_PCT"] = (df["High"] - df["Low"]) / df["Low"] * 100.0
        df["PCT_change"] = (df["Close"] - df["Open"]) / df["Open"] * 100.0

        df = df[["Adj Close", "HL_PCT", "PCT_change", "Volume"]]

        forecast_col = "Adj Close"

    df.fillna(value=-99999, inplace=True)
    forecast_out = int(math.ceil(record_percentage_to_predict * len(df)))
    df["label"] = df[forecast_col].shift(-forecast_out)
    print(df.head())

    X = np.array(df.drop(["label"], 1))
    X = preprocessing.scale(X)
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]
    df.dropna(inplace=True)

    y = np.array(df["label"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=record_percentage_to_predict
    )
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print(confidence)
    forecast_set = clf.predict(X_lately)
    df["Forecast"] = np.nan

    last_date = df.iloc[-1].name
    last_unix = last_date.timestamp()
    one_day = 86400
    next_unix = last_unix + one_day

    for i in forecast_set:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += 86400
        df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

    col = df["Forecast"]
    col = col.dropna()
    return df, col


def implement_bb_strategy(data, lower_bb, upper_bb):
    buy_price = []
    sell_price = []
    bb_signal = []
    signal = 0

    for i in range(len(data)):
        if data[i - 1] > lower_bb[i - 1] and data[i] < lower_bb[i]:
            if signal != 1:
                buy_price.append(data[i])
                sell_price.append(np.nan)
                signal = 1
                bb_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                bb_signal.append(0)
        elif data[i - 1] < upper_bb[i - 1] and data[i] > upper_bb[i]:
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(data[i])
                signal = -1
                bb_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                bb_signal.append(0)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            bb_signal.append(0)

    return buy_price, sell_price, bb_signal


def makesYieldColumn(_yield, weighted_sum_column, investment):
    _yield.iloc[0] = investment
    for i in range(1, weighted_sum_column.size):
        change = weighted_sum_column.item(i) + 1
        lastValue = _yield.iloc[i - 1]
        newValue = lastValue * change
        _yield.iloc[i] = newValue
    return _yield


def scanGoodStocks():
    # Define the parameters for the scanner
    min_avg_volume = 1000000  # minimum average daily volume
    min_rsi = 50  # minimum RSI value
    min_price = 0  # minimum price (in dollars)
    max_price = 1000  # maximum price (in dollars)

    # Download the list of all tickers from Yahoo Finance
    tickers = yf.Tickers("") # not work
    # Fetch the list of top 2000 stocks listed on NASDAQ
    nasdaq_2000 = pd.read_csv('https://www.nasdaq.com/api/v1/screener?page=1&pageSize=2000')
    # Get the ticker symbols for the top 2000 stocks
    tickers = nasdaq_2000['symbol'].to_list()


    # Create an empty DataFrame to store the results
    results = pd.DataFrame(columns=["Ticker", "Price", "50-Day MA", "200-Day MA", "52-Week High", "RSI", "Avg Volume"])

    # Loop through the tickers and scan for the best stocks
    for ticker in tickers.tickers:
        try:
            # Download the historical prices for the stock
            history = ticker.history(period="max")

            # Compute the 50-day and 200-day moving averages
            ma50 = ta.trend.SMAIndicator(history["Close"], window=50).sma_indicator()
            ma200 = ta.trend.SMAIndicator(history["Close"], window=200).sma_indicator()

            # Check if the stock is in an uptrend
            if ma50.iloc[-1] > ma200.iloc[-1]:
                # Compute the 52-week high
                high52 = history["High"].rolling(window=252).max().iloc[-1]

                # Check if the stock has broken out to a new high
                if history["Close"].iloc[-1] > high52:
                    # Compute the RSI
                    rsi = ta.momentum.RSIIndicator(history["Close"]).rsi()

                    # Check if the RSI is above the minimum value
                    if rsi.iloc[-1] > min_rsi:
                        # Compute the average daily volume
                        avg_volume = history["Volume"].rolling(window=20).mean().iloc[-1]

                        # Check if the average daily volume is above the minimum value
                        if avg_volume > min_avg_volume:
                            # Check if the price is within the specified range
                            price = history["Close"].iloc[-1]
                            if min_price <= price <= max_price:
                                # Add the result to the DataFrame
                                results = results.append({"Ticker": ticker.ticker, "Price": price,
                                                          "50-Day MA": ma50.iloc[-1], "200-Day MA": ma200.iloc[-1],
                                                          "52-Week High": high52, "RSI": rsi.iloc[-1],
                                                          "Avg Volume": avg_volume}, ignore_index=True)
        except:
            pass

    # Sort the results by RSI in descending order
    results = results.sort_values(by="RSI", ascending=False)
    return results.head(10)


def findBestStocks():
    """"# Fetch the list of top 2000 stocks listed on NASDAQ
    nasdaq_2000 = pd.read_csv('https://www.nasdaq.com/api/v1/screener?page=1&pageSize=2000')

    # Get the ticker symbols for the top 2000 stocks
    tickers = nasdaq_2000['symbol'].to_list()

    # Fetch the historical data for each stock
    data = yf.download(tickers, start="2022-05-15", end="2023-05-15")

    # Calculate the annual returns and volatility for each stock
    returns = data['Adj Close'].pct_change().groupby([pd.Grouper(freq='Y'), data.index.get_level_values(1)]).apply(
        lambda x: (1 + x).prod() - 1).reset_index(level=1, drop=True)
    volatility = data['Adj Close'].pct_change().groupby(
        [pd.Grouper(freq='Y'), data.index.get_level_values(1)]).std().reset_index(level=1, drop=True)

    # Calculate the Sharpe ratio for each stock
    sharpe = returns / volatility

    # Sort the stocks based on their Sharpe ratio
    sharpe_sorted = sharpe.sort_values(ascending=False)

    # Select the top 5 stocks with the highest Sharpe ratio
    top_5_stocks = sharpe_sorted.head(5)

    # Print the top 5 stocks with the highest Sharpe ratio
    print(top_5_stocks)"""

    """ Filter the stocks based on a set of criteria
     filtered_stocks = nasdaq_100[
         (nasdaq_100['lastsale'] >= 10) &  # Stocks with a last sale price of at least $10
         (nasdaq_100['marketCap'] >= 1e9) &  # Stocks with a market capitalization of at least $1 billion
         (nasdaq_100['ipoyear'] <= 2020)  # Stocks that went public before or in 2020
         ]

     # Get the ticker symbols for the filtered stocks
     tickers = filtered_stocks['symbol'].to_list()"""
    # Fetch the list of NASDAQ tickers from Yahoo Finance
    #nasdaq_tickers = yf.Tickers('^IXIC').tickers
    # Convert the dictionary of Ticker objects to a list of Ticker objects
    #ticker_list = list(nasdaq_tickers.values())
    #tickers = [ticker.ticker for ticker in ticker_list]
    tickers = [ "TA35.TA",
    "TA90.TA",
    'SPY',
    'QQQ',
    'rut',
    'IEI',
    'LQD',
    'Gsg',
    'GLD',
    'OIL']



    # Fetch the historical data for the filtered stocks
    data = yf.download(tickers, start="2022-05-15", end="2023-05-15")
    data = data.set_index(pd.to_datetime(data.index))
    # Calculate the annual returns and volatility for each stock
    returns = data['Adj Close'].pct_change().groupby(pd.Grouper(freq='Y')).apply(
        lambda x: (1 + x).prod() - 1).reset_index(level=0, drop=True)
    volatility = data['Adj Close'].pct_change().groupby(pd.Grouper(freq='Y')).std().reset_index(level=0, drop=True)

    # Calculate the Sharpe ratio for each stock
    sharpe = returns / volatility

    # Sort the stocks based on their Sharpe ratio
    sharpe_sorted = sharpe.sort_values()

    # Select the top 5 stocks with the highest Sharpe ratio
    top_5_stocks = sharpe_sorted.head(5)

    return top_5_stocks
