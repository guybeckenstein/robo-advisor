import http.client
import json
import codecs
import re
import requests
import base64
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import yfinance as yf
import seaborn as sns
import setting


def buildingPortfolio(pctChangeTable, sharpe_portfolio, min_variance_port, max_returns, level, investment):

    stock_weights = {}
    annualReturns = 0
    volatility = 0
    sharpe = 0

    if level == 1:
        stock_weights = min_variance_port.iloc[0][3:]
        annualReturns = min_variance_port.iloc[0][0]
        volatility = min_variance_port.iloc[0][1]
        sharpe = min_variance_port.iloc[0][2]

    elif level == 2:
        stock_weights = sharpe_portfolio.iloc[0][3:]
        annualReturns = sharpe_portfolio.iloc[0][0]
        volatility = sharpe_portfolio.iloc[0][1]
        sharpe = sharpe_portfolio.iloc[0][2]

    elif level == 3:
        stock_weights = max_returns.iloc[0][3:]
        annualReturns = max_returns.iloc[0][0]
        volatility = max_returns.iloc[0][1]
        sharpe = max_returns.iloc[0][2]

    pctChangeTable["weighted_sum_selected"] = np.dot(stock_weights, pctChangeTable.T)
    pctChangeTable["yield"] = pctChangeTable["weighted_sum_selected"]
    pctChangeTable["yield"].iloc[0] = investment
    for i in range(1, len(pctChangeTable["weighted_sum_selected"].values)):
        change = pctChangeTable["weighted_sum_selected"].values.item(i) + 1
        lastValue = pctChangeTable["yield"].iloc[i - 1]
        newValue = lastValue * change
        pctChangeTable["yield"].iloc[i] = newValue

    return pctChangeTable, stock_weights, annualReturns, volatility, sharpe


def markovichModel(Num_porSimulation, pctChangeTable, stocksSymbols):

    stocksNames = []
    for symbol in stocksSymbols:
        if type(symbol) == int:
            stocksNames.append(str(symbol))
        else:
            stocksNames.append(symbol)

    returns_daily = pctChangeTable
    returns_annual = ((1 + returns_daily.mean()) ** 254) - 1
    cov_daily = returns_daily.cov()
    cov_annual = cov_daily * 254

    # empty lists to store returns, volatility and weights of imiginary portfolios
    port_returns = []
    port_volatility = []
    sharpe_ratio = []
    stock_weights = []

    # set the number of combinations for imaginary portfolios
    num_assets = len(stocksSymbols)
    num_portfolios = Num_porSimulation

    # set random seed for reproduction's sake
    np.random.seed(101)

    # populate the empty lists with each portfolios returns,risk and weights
    for single_portfolio in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        returns = np.dot(weights, returns_annual)
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
        sharpe = returns / volatility
        sharpe_ratio.append(sharpe)
        port_returns.append(returns * 100)
        port_volatility.append(volatility * 100)
        stock_weights.append(weights)

    # a dictionary for Returns and Risk values of each portfolio
    portfolio = {
        "Returns": port_returns,
        "Volatility": port_volatility,
        "Sharpe Ratio": sharpe_ratio,
    }

    # extend original dictionary to accomodate each ticker and weight in the portfolio
    for counter, symbol in enumerate(stocksNames):
        portfolio[symbol + " Weight"] = [Weight[counter] for Weight in stock_weights]

    # make a nice dataframe of the extended dictionary
    df = pd.DataFrame(portfolio)

    # get better labels for desired arrangement of columns
    column_order = ["Returns", "Volatility", "Sharpe Ratio"] + [
        stock + " Weight" for stock in stocksNames
    ]
    # reorder dataframe columns
    df = df[column_order]

    sharpe_portfolio = df.loc[df["Sharpe Ratio"] == df["Sharpe Ratio"].max()]  # medium risk
    min_variance_port = df.loc[df["Volatility"] == df["Volatility"].min()]  # low risk
    max_returns = df.loc[df["Returns"] == df["Returns"].max()]  # high level
    max_vols = df.loc[df["Volatility"] == df["Volatility"].max()]

    # get 3 portfolios weights
    weighted_low = np.dot(min_variance_port.iloc[0][3:], pctChangeTable.T)
    weighted_medium = np.dot(sharpe_portfolio.iloc[0][3:], pctChangeTable.T)
    weighted_high = np.dot(max_returns.iloc[0][3:], pctChangeTable.T)

    return df, sharpe_portfolio, min_variance_port, max_returns, max_vols, weighted_low, weighted_medium, weighted_high


def convertDataToTables(newPortfolio, israeliIndexesChoice, usaIndexesChoice, record_percentage_to_predict,
                        numOfYearsHistory, machineLearningOpt):
    frame = {}
    stocksNames = newPortfolio.getStocksSymbols()
    israeliIndexes = newPortfolio.getIsraeliStocksIndexes()
    usaIndexes = newPortfolio.getUsaStocksIndexes()

    # ISRAEL STOCKS

    if israeliIndexesChoice == 1:
        # israelIndexesNames = {}
        # israelIndexesNames = list(convertIsraeliIndexToName(israeliIndexes))
        # stocksNames = list(stocksNames) + israelIndexesNames
        IsraeliIndexesData = getIsraeliIndexesData("get_past_10_years_history", israeliIndexes)
        for i, stock in enumerate(IsraeliIndexesData):
            df = pd.DataFrame(stock["indexEndOfDay"]["result"])
            df["tradeDate"] = pd.to_datetime(df["tradeDate"])
            df.set_index("tradeDate", inplace=True)
            if machineLearningOpt == 1:
                DF, price = price_forecast(df, record_percentage_to_predict, 1)  # 1-ISRAEL
            else:
                price = df[["closingIndexPrice"]]
            frame[stocksNames[i]] = price

    # USA STOCKS

    if usaIndexesChoice == 1:
        yf.pdr_override()
        start_date, end_date = getfromAndToDate(numOfYearsHistory)
        for stock in usaIndexes:
            df = yf.download(stock, start=start_date, end=end_date)
            if machineLearningOpt == 1:
                DF, price = price_forecast(df, record_percentage_to_predict, 0)  # 0 -USA
            else:
                price = df[["Adj Close"]]
            frame[stock] = price

    closingPricesTable = pd.concat(frame.values(), axis=1, keys=frame.keys())
    pctChangeTable = closingPricesTable.pct_change()
    pctChangeTable.fillna(0, inplace=True)

    return closingPricesTable, pctChangeTable


def price_forecast(df, record_percentage_to_predict, isIsraeliStock):

    if isIsraeliStock == 1:
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


def forcastSpecificStock(stock, isIsraeliStock, numOfYearsHistory):
    if isIsraeliStock:
        # GET INDEXES DATA AUTOMATICALLY FROM TASE-API
        # indexesData=manageData.setPortfolioData("get_past_10_years_history",sybmolIndexs)
        df = pd.DataFrame(stock["indexEndOfDay"]["result"])
        df["tradeDate"] = pd.to_datetime(df["tradeDate"])
        df.set_index("tradeDate", inplace=True)
        df, col = price_forecast(df, setting.record_percentage_to_predict, 1)
        plotpriceForcast(stock, df, 1)

    else:
        yf.pdr_override()
        start_date, end_date = getfromAndToDate(numOfYearsHistory)
        df = yf.download(stock, start=start_date, end=end_date)
        df, col = price_forecast(df, setting.record_percentage_to_predict, 0)
        plotpriceForcast(stock, df, 0)


# plot graph
def plotMarkovichPortfolioGraph(df, sharpe_portfolio, min_variance_port, max_returns, max_vols, newPortfolio):

    # plot frontier, max sharpe & min Volatility values with a scatterplot
    figSize_X = 10
    figSize_Y = 8
    figSize = (figSize_X, figSize_Y)
    plt.style.use("seaborn-dark")
    df.plot.scatter(
        x="Volatility",
        y="Returns",
        c="Sharpe Ratio",
        cmap="RdYlGn",
        edgecolors="black",
        figsize=figSize,
        grid=True,
    )
    plt.scatter(
        x=sharpe_portfolio["Volatility"],
        y=sharpe_portfolio["Returns"],
        c="green",
        marker="D",
        s=200,
    )
    plt.scatter(
        x=min_variance_port["Volatility"],
        y=min_variance_port["Returns"],
        c="orange",
        marker="D",
        s=200,
    )
    plt.scatter(
        x=max_vols["Volatility"], y=max_returns["Returns"], c="red", marker="D", s=200
    )
    plt.style.use("seaborn-dark")

    plt.xlabel("Volatility (Std. Deviation) Percentage %")
    plt.ylabel("Expected Returns Percentage %")
    plt.title("Efficient Frontier")
    plt.subplots_adjust(bottom=0.4)

    # ------------------ Pritning 3 optimal Protfolios -----------------------
    # Setting max_X, max_Y to act as relative border for window size
    sectorsNames = newPortfolio.getSectorsNames()
    lowWeight = min_variance_port.iloc[0][3:]
    mediumWeight = sharpe_portfolio.iloc[0][3:]
    highWeight = max_returns.iloc[0][3:]
    stocksStrHigh = ""
    stocksStrLow = ""
    stocksStrMedium = ""
    highSectorsWeight = newPortfolio.returnSectorsWeightsAccordingToStocksWeights(highWeight)
    lowSectorsWeight = newPortfolio.returnSectorsWeightsAccordingToStocksWeights(lowWeight)
    mediumSectorsWeight = newPortfolio.returnSectorsWeightsAccordingToStocksWeights(mediumWeight)

    # stocksStrHigh
    for i in range(len(sectorsNames)):
        weight = highSectorsWeight[i] * 100
        stocksStrHigh += sectorsNames[i] + "(" + str("{:.2f}".format(weight)) + "%),\n "
    # stocksStrMedium
    for i in range(len(sectorsNames)):
        weight = mediumSectorsWeight[i] * 100
        stocksStrMedium += sectorsNames[i] + "(" + str("{:.2f}".format(weight)) + "%),\n "
    # stocksStrLow
    for i in range(len(sectorsNames)):
        weight = lowSectorsWeight[i] * 100
        stocksStrLow += sectorsNames[i] + "(" + str("{:.2f}".format(weight)) + "%),\n "

    with pd.option_context("display.float_format", "%{:,.2f}".format):
        plt.figtext(
            0.2,
            0.15,
            "Max returns Porfolio: \n"
            + "Returns: " + str(round(max_returns.iloc[0][0], 2)) + "%\n"
            + "Volatility: " + str(round(max_returns.iloc[0][1], 2)) + "%\n"
            + "Sharpe Ratio: " + str(round(max_returns.iloc[0][2], 2)) + "\n"
            + stocksStrHigh,
            bbox=dict(facecolor="red", alpha=0.5),
            fontsize=11,
            style="oblique",
            ha="center",
            va="center",
            fontname="Arial",
            wrap=True,
        )
        plt.figtext(
            0.45,
            0.15,
            "Safest Portfolio: \n"
            + "Returns: " + str(round(min_variance_port.iloc[0][0], 2)) + "%\n"
            + "Volatility: " + str(round(min_variance_port.iloc[0][1], 2)) + "%\n"
            + "Sharpe Ratio: " + str(round(min_variance_port.iloc[0][2], 2)) + "\n"
            + stocksStrLow,
            bbox=dict(facecolor="yellow", alpha=0.5),
            fontsize=11,
            style="oblique",
            ha="center",
            va="center",
            fontname="Arial",
            wrap=True,
        )
        plt.figtext(
            0.7,
            0.15,
            "Sharpe  Portfolio: \n"
            + "Returns: " + str(round(sharpe_portfolio.iloc[0][0], 2)) + "%\n"
            + "Volatility: " + str(round(sharpe_portfolio.iloc[0][1], 2)) + "%\n"
            + "Sharpe Ratio: " + str(round(sharpe_portfolio.iloc[0][2], 2)) + "\n"
            + stocksStrMedium,
            bbox=dict(facecolor="green", alpha=0.5),
            fontsize=11,
            style="oblique",
            ha="center",
            va="center",
            fontname="Arial",
            wrap=True,
        )
    plt.show()


def plotpriceForcast(stockSymbol, df, isIsraeliStock):

    if isIsraeliStock:
        df['closingIndexPrice'].plot()
    else:
        df['Adj Close'].plot()
    df['Forecast'].plot()
    plt.title(stockSymbol + " Stock Price Forecast")
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()


def plotDistributionOfStocks(stockNames, pctChangeTable):

    plt.subplots(figsize=(8, 8))
    plt.legend()
    plt.xlabel('Return', fontsize=12)
    plt.ylabel('Distribution', fontsize=12)
    for i in range(len(stockNames)):
        sns.distplot(pctChangeTable[stockNames[i]], kde=True, hist=False, rug=False, label=stockNames[i])
    plt.grid(True)
    plt.legend()
    plt.show()


def plotDistributionOfPortfolio(weighted_low, weighted_medium, weighted_high):
    plt.subplots(figsize=(8, 8))
    plt.legend()
    plt.xlabel('Return', fontsize=12)
    plt.ylabel('Distribution', fontsize=12)
    sns.distplot(weighted_low, kde=True, hist=False, rug=False, label="low risk")
    sns.distplot(weighted_medium, kde=True, hist=False, rug=False, label="high risk")
    sns.distplot(weighted_high, kde=True, hist=False, rug=False, label="high risk")
    plt.grid(True)
    plt.legend()
    plt.show()


def getIsraeliIndexesData(command, israeliIndexes):
    JsonDataList = [0] * len(israeliIndexes)
    if command == "get_past_10_years_history":
        for i in range(len(israeliIndexes)):
            appUrl = getIndexHistory(
                setting.indexEndOfDayHistoryTenYearsUpToday, israeliIndexes[i], 10
            )
            JsonDataList[i] = getSymbolInfo(appUrl)
        # return JsonDataList - TODO - fix , makes unlimited requests
        return getIndexesDataManuallyFromJSON(israeliIndexes)
    # TOOD: add more commands

    # return JsonDataList


def createJsonDataFromTase(indexId, nameFile):
    appUrl = getIndexHistory(setting.indexEndOfDayHistoryTenYearsUpToday, indexId, 10)
    jsonData = getSymbolInfo(appUrl)
    createsJsonFile(jsonData, nameFile)


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


def plotbb_strategy_stock(stockName, start="2009-12-01", end="2021-01-01"):
    stockprices = yf.download(stockName, start, end)
    stockprices['MA50'] = stockprices['Adj Close'].rolling(window=50).mean()
    stockprices['50dSTD'] = stockprices['Adj Close'].rolling(window=50).std()
    stockprices['Upper'] = stockprices['MA50'] + (stockprices['50dSTD'] * 2)
    stockprices['Lower'] = stockprices['MA50'] - (stockprices['50dSTD'] * 2)

    stockprices = stockprices.dropna()
    stockprices = stockprices.iloc[51:]

    buy_price, sell_price, bb_signal = implement_bb_strategy(stockprices['Adj Close'], stockprices['Lower'],
                                                             stockprices['Upper'])

    stockprices[['Adj Close', 'Lower', 'Upper']].plot(figsize=(10, 4))
    plt.scatter(stockprices.index, buy_price, marker='^', color='green', label='BUY', s=200)
    plt.scatter(stockprices.index, sell_price, marker='v', color='red', label='SELL', s=200)
    plt.show()

    print("number of green :")
    print(np.count_nonzero(~np.isnan(buy_price)))
    print("number of red :")
    print(np.count_nonzero(~np.isnan(sell_price)))

    print("ID : 208604694")


def plotbb_strategy_Portfolio(pctChangeTable):
    stockprices = pctChangeTable
    stockprices['Adj Close'] = pctChangeTable['yield']
    stockprices['MA50'] = stockprices['Adj Close'].rolling(window=50).mean()
    stockprices['50dSTD'] = stockprices['Adj Close'].rolling(window=50).std()
    stockprices['Upper'] = stockprices['MA50'] + (stockprices['50dSTD'] * 2)
    stockprices['Lower'] = stockprices['MA50'] - (stockprices['50dSTD'] * 2)

    stockprices = stockprices.dropna()
    stockprices = stockprices.iloc[51:]

    buy_price, sell_price, bb_signal = implement_bb_strategy(stockprices['Adj Close'], stockprices['Lower'],
                                                             stockprices['Upper'])

    stockprices[['Adj Close', 'Lower', 'Upper']].plot(figsize=(10, 4))
    plt.scatter(stockprices.index, buy_price, marker='^', color='green', label='BUY', s=200)
    plt.scatter(stockprices.index, sell_price, marker='v', color='red', label='SELL', s=200)
    plt.show()

    print("number of green :")
    print(np.count_nonzero(~np.isnan(buy_price)))
    print("number of red :")
    print(np.count_nonzero(~np.isnan(sell_price)))


# Connect to TASE API and get json data from appUrl
def getSymbolInfo(appUrl):
    conn = http.client.HTTPSConnection("openapigw.tase.co.il")
    payload = ""
    headers = {
        "Authorization": "Bearer " + get_tase_access_token(),
        "Accept-Language": "he-IL",
        "Content-Type": "application/json",
    }
    conn.request("GET", appUrl, payload, headers)
    res = conn.getresponse()
    data = res.read()
    # Decode the bytes object to a string
    json_string = data.decode("utf-8")
    json_obj = json.loads(json_string)
    return json_obj


# Auth
def get_base_64_token():
    # key = '7e247414e7047349d83b7b8a427c529c'
    # secret = '7a809c498662c88a0054b767a92f0399'
    token = setting.key + ":" + setting.secret
    base_64_token = base64.b64encode(token.encode("ascii")).decode("ascii")
    return base_64_token


def get_tase_access_token():
    base_64_token = get_base_64_token()
    # tokenUrl = "https://openapigw.tase.co.il/tase/prod/oauth/oauth2/token"
    payload = "grant_type=client_credentials&scope=tase"
    headers = {
        "Authorization": "Basic " + base_64_token,
        "Content-Type": "application/x-www-form-urlencoded",
    }
    response = requests.request("POST", setting.tokenUrl, headers=headers, data=payload)
    return json.loads(response.text)["access_token"]


# BUILD URL FOR REQUEST
def getIndexHistory(appName, indexId, numOfYears):
    today = datetime.datetime.now()
    startyear = today.year - numOfYears
    startmonth = today.month
    startday = today.day
    endyear = today.year
    endmonth = today.month
    endday = today.day
    return getAppUrlWithDateAndIndex(
        appName, startyear, startmonth, startday, endyear, endmonth, endday, indexId
    )


def getAppUrlWithDateAndIndex(
    appName, startYear, startMounth, startDay, endYear, endMonth, endDay, indexName
):
    return (
        getAppUrlWithoutDate(appName)
        + str(indexName)
        + "&fromDate="
        + str(startYear)
        + "-"
        + str(startMounth)
        + "-"
        + str(startDay)
        + "&toDate="
        + str(endYear)
        + "-"
        + str(endMonth)
        + "-"
        + str(endDay)
    )


def getAppUrlWithoutDate(appName):  # /tase/prod/api/v1/short-sales/weekly-balance
    return setting.prefixUrl + "/" + appName


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


# Other functions
def getNameByIndexNumber(indexNumber):
    # Load the JSON data from the file
    jsonData = getJsonData("DB/indicesList")
    result = [
        item["indexName"]
        for item in jsonData["indicesList"]["result"]
        if item["indexId"] == indexNumber
    ]
    # makes it from right to left
    name = result[0]
    return name


def getIndexesDataManuallyFromJSON(sybmolIndexs):  # FOR ISRAELI STOCKS
    portfolio = [0] * len(sybmolIndexs)
    for i in range(len(sybmolIndexs)):
        portfolio[i] = getJsonData("DB/History" + str(sybmolIndexs[i]))
    return portfolio


def createsJsonFile(json_obj, nameProduct):
    # Open a file in write mode
    parts = nameProduct.split("/")
    last_element = parts[-1]
    with open(
        "DB/"+last_element + ".json", "w"
    ) as f:  # Use the `dump()` function to write the JSON data to the file
        json.dump(json_obj, f)


def getJsonData(name):
    with codecs.open(name + ".json", "r", encoding="utf-8") as file:
        json_data = json.load(file)
    return json_data


def getDataFromForm():  # not it from terminal
    name = getName()
    level = getLevelOfRisk()
    amount = getInverstmentAmount()
    machineLearningOpt = getMachineLearningOption()
    # israeliIndexesChoice, usaIndexesChoice=getTypeOFindexes()  -TODO - DELETE OR USE MANUALLY
    israeliIndexesChoice = 1
    usaIndexesChoice = 1
    return level, amount, machineLearningOpt, israeliIndexesChoice, usaIndexesChoice, name


def getName():
    print("enter name")
    name = input()
    return name


def getLevelOfRisk():
    print("enter Level of risk\n1-LOW\n2-MEDIUM\n3-HIGH")
    level = int(input())
    while level < 1 or level > 3:
        print("enter Level of risk\n1-LOW\n2-MEDIUM\n3-HIGH")
        level = int(input())
    return level


def getInverstmentAmount():
    print("enter amount of money to invest")
    amount = int(input())
    while amount < 1:
        print("enter amount of money to invest")
        amount = int(input())
    return amount


def getMachineLearningOption():
    print("Interested in using machine learning? 0-no, 1-yes")
    machineLearningOpt = int(input())
    while machineLearningOpt != 0 and machineLearningOpt != 1:
        print("Please enter 0 or 1")
        machineLearningOpt = int(input())
    return machineLearningOpt


def getTypeOFindexes():
    print("do you want to include israel indexes? 0-no, 1-yes")
    israeliIndexesChoice = int(input())
    while israeliIndexesChoice != 0 and israeliIndexesChoice != 1:
        print("Please enter 0 or 1")
        israeliIndexesChoice = int(input())

    print("do you want to include usa indexes? 0-no, 1-yes")
    usaIndexesChoice = int(input())
    while usaIndexesChoice != 0 and usaIndexesChoice != 1:
        print("Please enter 0 or 1")
        usaIndexesChoice = int(input())

    return israeliIndexesChoice, usaIndexesChoice


def convertIsraeliIndexToName(IsraliIndexes):
    hebrew_pattern = r"[\u0590-\u05FF\s]+"
    stocksNames = {}
    for i, index in enumerate(IsraliIndexes):
        text = getNameByIndexNumber(index)
        hebrew_parts = re.findall(
            hebrew_pattern, text
        )  # find all the Hebrew parts in the text
        for hebrew_part in hebrew_parts:
            hebrew_part_reversed = "".join(
                reversed(hebrew_part)
            )  # reverse the order of the Hebrew characters in the part
            text = text.replace(
                hebrew_part, hebrew_part_reversed
            )  # replace the original part with the reversed part in the text
        stocksNames[i] = text
    return stocksNames.values()


def convertUsaIndexToName(UsaIndexes):
    UsaIndexesNames = {}
    for i, index in enumerate(UsaIndexes):
        ticker = yf.Ticker(index)
        UsaIndexesNames[i] = ticker.info["longName"]

    return UsaIndexesNames.values()


# GET OTHER DATA- NOT RELEVANIC BY NOW

"""def getFundHistoryById(
    appName, fundId, startyear, startmonth, startday, endyear, endmonth, endday
):  # /tase/prod/api/v1/index_end_of_day_data/2022/11/22
    return "/tase/prod/api/v1/mutual-fund/history-data/5100474?startDate=2015-12-31&endDate=2020-12-31"""
