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

# from py_vollib.black_scholes import black_scholes as bs
# from py_vollib.black_scholes.greeks.analytical import delta, gamma, vega, theta, rho

import setting


def buildingPortfolio(IsraliIndexes, UsaIndexes, weights, sLossR, level):

    # TODO- GET DIRECTLY FROM API
    # GET INDEXES DATA MANUALLY FROM JSON FILES THAT DOWNLOADED FROM TASE-API
    IsraeliIndexesData = getIndexesDataManuallyFromJSON(IsraliIndexes)

    frame = {}
    for i, stock in enumerate(IsraeliIndexesData):
        df = pd.DataFrame(stock["indexEndOfDay"]["result"])
        df["tradeDate"] = pd.to_datetime(df["tradeDate"])
        df.set_index("tradeDate", inplace=True)
        data_var = df[["closingIndexPrice"]].pct_change().copy()
        frame.update({str(IsraliIndexes[i]): data_var})

    table = pd.concat(frame.values(), axis=1, keys=frame.keys())
    # sum the column and multiply by the weight
    weighted_sum = (table * weights[level - 1]).sum(axis=1)
    table["weighted_sum"] = weighted_sum

    returns_daily = weighted_sum.mean()  # averge change
    returns_annual = ((1 + returns_daily.mean()) ** 254) - 1
    sLoss = returns_annual - sLossR * returns_annual
    cov_daily = weighted_sum.std()
    cov_annual = cov_daily * 254

    return table, sLoss, returns_annual, cov_annual


def plotPctChange(table, sLoss, returns_annual, name):
    plt.title("Hello, " + name + "! This is your yield portfolio")
    table.plot(figsize=(10, 8))
    plt.axhline(y=sLoss, color="r", linestyle="-")
    plt.axhline(y=returns_annual, color="g", linestyle="-")
    plt.show()


def plotPortfolioComponent(indexes, weights, name):  # TODO-FIX HEBREW TEXT
    plt.title("Hello, " + name + "! This is your portfolio")
    stocksNames = {}
    for i, index in enumerate(indexes):
        stocksNames[i] = getNameByIndexNumber(index)
    plt.pie(
        weights,
        labels=stocksNames.values(),
        autopct="%1.1f%%",
        shadow=True,
        startangle=140,
    )
    plt.axis("equal")
    plt.show()


def plotYieldChange(table, name, investment):
    plt.title("Hello, " + name + "! This is your yield portfolio")
    # creates a new column called yield, and sets in the row 4 the value of the inverstment
    table["yield"] = table["weighted_sum"].copy()
    table["yield"].iloc[0] = investment
    for i in range(1, len(table["weighted_sum"])):
        lastValue = table["yield"].iloc[i - 1]
        table["yield"].iloc[i] = lastValue * (table["weighted_sum"].iloc[i] + 1)
    table["yield"].plot(figsize=(10, 8))
    table.to_csv("table.csv")
    plt.show()


def markovich(
    Num_porSimulation, IsraliIndexes, UsaIndexes, record_percentage_to_predict, name
):
    frame = {}
    stocksNames = {}
    israelIndexesNames = {}
    UsaIndexesNames = {}

    # TODO- GET DIRECTLY FROM FORM
    print("Interested in using machine learning? 0-no, 1-yes")
    machineLearning = int(input())
    while machineLearning != 0 and machineLearning != 1:
        print("Please enter 0 or 1")
        machineLearning = int(input())

    print("include israel indexes? 0-no, 1-yes")
    israeliIndexesChoice = int(input())
    while israeliIndexesChoice != 0 and israeliIndexesChoice != 1:
        print("Please enter 0 or 1")
        israeliIndexesChoice = int(input())

    print("include usa indexes? 0-no, 1-yes")
    usaIndexesChoice = int(input())
    while usaIndexesChoice != 0 and usaIndexesChoice != 1:
        print("Please enter 0 or 1")
        usaIndexesChoice = int(input())

    if israeliIndexesChoice == 1:
        israelIndexesNames = list(convertIsraeliIndexToName(IsraliIndexes))
        stocksNames = list(stocksNames) + israelIndexesNames

        # TODO- FIX LIMIT NUM OF REQUESTS TO TASE-API
        # GET INDEXES DATA AUTOMATICALLY FROM TASE-API
        # indexesData=manageData.setPortfolioData("get_past_10_years_history",sybmolIndexs)

        # ISRAEL STOCKS
        # GET INDEXES DATA MANUALLY FROM JSON FILES THAT DOWNLOADED FROM TASE-API
        IsraeliIndexesData = getIndexesDataManuallyFromJSON(IsraliIndexes)

        # create table

        for i, stock in enumerate(IsraeliIndexesData):
            df = pd.DataFrame(stock["indexEndOfDay"]["result"])
            df["tradeDate"] = pd.to_datetime(df["tradeDate"])
            df.set_index("tradeDate", inplace=True)
            if machineLearning == 1:
                price = price_forecast(df, record_percentage_to_predict, 1)  # 1-ISRAEL
            else:
                price = df[["closingIndexPrice"]]
            #data_var = price
            frame[stocksNames[i]] = price
            # frame.update({str(IsraliIndexes[i]):data_var})

    if usaIndexesChoice == 1:
        # USA STOCKS
        #UsaIndexesNames = convertUsaIndexToName(UsaIndexes)
        stocksNames = UsaIndexes
        yf.pdr_override()
        start_date, end_date = getfromAndToDate(2)
        for stock in UsaIndexes:
            df = yf.download(stock, start=start_date, end=end_date)
            if machineLearning == 1:
                price = price_forecast(df, record_percentage_to_predict, 0)  # 0 -USA
            else:
                price = df[["Adj Close"]]
            frame[stock] = price

    if israeliIndexesChoice == 1 and usaIndexesChoice == 1:
        stocksNames = list(israelIndexesNames) + UsaIndexes

    table = pd.concat(frame.values(), axis=1, keys=frame.keys())
    table.to_csv("Out123.csv")
    # plt.plot(pd.DataFrame(frame))
    # pd.DataFrame(frame).to_csv('Out.csv')

    returns_daily = table.pct_change()
    returns_daily.to_csv("Out1.csv")
    returns_annual = ((1 + returns_daily.mean()) ** 254) - 1
    returns_annual.to_csv("Out2.csv")

    # get daily and covariance of returns of the stock
    cov_daily = returns_daily.cov()
    cov_annual = cov_daily * 254

    # empty lists to store returns, volatility and weights of imiginary portfolios
    port_returns = []
    port_volatility = []
    sharpe_ratio = []
    stock_weights = []

    # set the number of combinations for imaginary portfolios
    num_assets = len(stocksNames)
    num_portfolios = Num_porSimulation  # Change porfolio numbers here

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

    # plot frontier, max sharpe & min Volatility values with a scatterplot
    # find min Volatility & max sharpe values in the dataframe (df)
    min_volatility = df["Volatility"].min()
    # min_volatility1 = df['Volatility'].min()+1
    max_sharpe = df["Sharpe Ratio"].max()
    max_return = df["Returns"].max()
    max_vol = df["Volatility"].max()
    # use the min, max values to locate and create the two special portfolios
    sharpe_portfolio = df.loc[df["Sharpe Ratio"] == max_sharpe]
    min_variance_port = df.loc[df["Volatility"] == min_volatility]
    max_returns = df.loc[df["Returns"] == max_return]
    max_vols = df.loc[df["Volatility"] == max_vol]

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

    red_num = df.index[df["Returns"] == max_return]
    yellow_num = df.index[df["Volatility"] == min_volatility]
    green_num = df.index[df["Sharpe Ratio"] == max_sharpe]
    multseries = pd.Series(
        [1, 1, 1] + [100 for stock in stocksNames],
        index=["Returns", "Volatility", "Sharpe Ratio"]
        + [stock + " Weight" for stock in stocksNames],
    )
    with pd.option_context("display.float_format", "%{:,.2f}".format):
        plt.figtext(
            0.2,
            0.15,
            "Max returns Porfolio: \n"
            + df.loc[red_num[0]].multiply(multseries).to_string(),
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
            + df.loc[yellow_num[0]].multiply(multseries).to_string(),
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
            + df.loc[green_num[0]].multiply(multseries).to_string(),
            bbox=dict(facecolor="green", alpha=0.5),
            fontsize=11,
            style="oblique",
            ha="center",
            va="center",
            fontname="Arial",
            wrap=True,
        )
    plt.show()


def price_forecast(df, record_percentage_to_predict, isIsrael):

    if isIsrael == 1:
        df["HL_PCT"] = (df["high"] - df["low"]) / df["low"] * 100.0
        df["PCT_change"] = (
            (df["closingIndexPrice"] - df["indexOpeningPrice"])
            / df["indexOpeningPrice"]
            * 100.0
        )

        # df = df[['Adj Close', 'HL_PCT', 'PCT_change', 'Volume']]
        df = df[["closingIndexPrice", "HL_PCT", "PCT_change"]]

        forecast_col = "closingIndexPrice"
        # forecast_col = 'Adj. Close'
        df.fillna(value=-99999, inplace=True)
        forecast_out = int(math.ceil(record_percentage_to_predict * len(df)))
        df["label"] = df[forecast_col].shift(-forecast_out)
        print(df.head())
        # df['Adj Close'].plot()

        X = np.array(df.drop(["label"], 1))
        X = preprocessing.scale(X)
        X_lately = X[-forecast_out:]
        X = X[:-forecast_out]
        df.dropna(inplace=True)

        y = np.array(df["label"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=record_percentage_to_predict
        )
        # clf =  svm.SVR() # #LinearRegression()
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

        # df['closingIndexPrice'].plot()
    # df['Forecast'].plot()
    # plt.legend(loc=4)
    # plt.xlabel('Date')
    # plt.ylabel('Price')
    # plt.show()
    else:
        df["HL_PCT"] = (df["High"] - df["Low"]) / df["Low"] * 100.0
        df["PCT_change"] = (df["Close"] - df["Open"]) / df["Open"] * 100.0

        df = df[["Adj Close", "HL_PCT", "PCT_change", "Volume"]]

        forecast_col = "Adj Close"
        # forecast_col = 'Adj. Close'
        df.fillna(value=-99999, inplace=True)
        forecast_out = int(math.ceil(record_percentage_to_predict * len(df)))
        df["label"] = df[forecast_col].shift(-forecast_out)
        print(df.head())
        # df['Adj Close'].plot()

        X = np.array(df.drop(["label"], 1))
        X = preprocessing.scale(X)
        X_lately = X[-forecast_out:]
        X = X[:-forecast_out]
        df.dropna(inplace=True)

        y = np.array(df["label"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=record_percentage_to_predict
        )
        # clf =  svm.SVR() # #LinearRegression()
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

        # df['Adj Close'].plot()
    # df['Forecast'].plot()
    # plt.legend(loc=4)
    # plt.xlabel('Date')
    # plt.ylabel('Price')
    # plt.show()

    col = df["Forecast"]
    col = col.dropna()
    return col


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


def setPortfolioData(command, sybmolIndex):

    portfolio = []
    if command == "get_past_10_years_history":
        for i in range(len(sybmolIndex)):
            appUrl = getIndexHistory(
                setting.indexEndOfDayHistoryTenYearsUpToday, sybmolIndex[i], 10
            )
            portfolio[i] = getSymbolInfo(
                appUrl, "pastTenYearsHistory" + str(sybmolIndex[i])
            )
    # TOOD: add more commands

    return portfolio


def getIndexesDataManuallyFromJSON(sybmolIndexs):  # FOR ISRAELI STOCKS
    portfolio = [0] * len(sybmolIndexs)
    for i in range(len(sybmolIndexs)):
        portfolio[i] = getJsonData("DB/History" + str(sybmolIndexs[i]))
    return portfolio


def plotPortfolioGraph(json_data):

    df = pd.DataFrame(json_data["indexEndOfDay"]["result"])
    df["tradeDate"] = pd.to_datetime(df["tradeDate"])
    df.set_index("tradeDate", inplace=True)
    df["closingIndexPrice"].pct_change().plot()
    plt.grid(True)
    plt.legend()
    plt.show()


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


# Connect to TASE API and get json data
def getSymbolInfo(appUrl, nameProduct):
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
    # createsJsonFile(json_obj, nameProduct)


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


def getAppUrlWithDate(
    appName, year, month, day
):  # /tase/prod/api/v1/index_end_of_day_data/2022/11/22
    return (
        getAppUrlWithoutDate(appName)
        + "/"
        + str(year)
        + "/"
        + str(month)
        + "/"
        + str(day)
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


def getAppUrlWithDate(
    appName, year, month, day, indexId
):  # /tase/prod/api/v1/index_end_of_day_data/2022/11/22?indexId=142
    return (
        getAppUrlWithoutDate(appName)
        + "/"
        + str(year)
        + "/"
        + str(month)
        + "/"
        + str(day)
        + "?indexId="
        + str(indexId)
    )


def getAppUrlWithoutDate(appName):  # /tase/prod/api/v1/short-sales/weekly-balance
    return setting.prefixUrl + "/" + appName


def getFundHistoryById(
    appName, fundId, startyear, startmonth, startday, endyear, endmonth, endday
):  # /tase/prod/api/v1/index_end_of_day_data/2022/11/22
    return "/tase/prod/api/v1/mutual-fund/history-data/5100474?startDate=2015-12-31&endDate=2020-12-31"


# Other functions
def createsJsonFile(json_obj, nameProduct):
    # Open a file in write mode
    parts = nameProduct.split("/")
    last_element = parts[-1]
    with open(
        last_element + ".json", "w"
    ) as f:  # Use the `dump()` function to write the JSON data to the file
        json.dump(json_obj, f)


def getJsonData(name):
    with codecs.open(name + ".json", "r", encoding="utf-8") as file:
        json_data = json.load(file)
    return json_data


def getLevelFromTerminal():
    print("enter Level of risk\n1-LOW\n2-MEDIUM\n3-HIGH")
    level = int(input())
    while level < 1 or level > 3:
        print("enter Level of risk\n1-LOW\n2-MEDIUM\n3-HIGH")
        level = int(input())
    return level


def getInverstmentFromTerminal():
    print("enter amount of money to invest")
    amount = int(input())
    while amount < 1:
        print("enter amount of money to invest")
        amount = int(input())
    return amount


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

# already have:
# get_trade_info(getAppUrlWithoutDate(basicIndexList),setting.basicIndexList)# a list of all basic indices in tase- indices-list
# get_trade_info(getAppUrlWithDate(basicIndexComponents, year, month, day),basicIndexComponents)- components of each index
# get_trade_info(getAppUrlWithoutDate(basicSecuritiesList),basocSecirityList)# a list of all basic securities in tase- securities-types- not use
# get_trade_info(getAppUrlWithoutDate(basicSecuritiesCompnayList),basicSecuritiesCompnayList)# a list of all basic securities companies in tase- companies-list- not use
# get_trade_info(indexEndOfDay,indexEndOfDay)// specific date
""" get_trade_info(indexEndOfDayHistoryTenYearsUpToday
,"gsfgsfgfds/indexEndOfDayHistoryTenYearsUpTodaytelBondMaagar") """


""" get_trade_info(getAppUrlWithDate(indexEndOfDayName
, year, month, day),indexEndOfDayName): """
# index end of day data
""" get_trade_info(getAppUrlWithDate(OTC_transaction_name
, year, month, day),OTC_transaction_name): """
# OTC transaction
""" get_trade_info(getAppUrlWithoutDate(shortSalesWeeklyBalanceName)
,shortSalesWeeklyBalanceName): """
# short sales weekly balance
# get_trade_info(getAppUrlWithDate(mayaNoticeByDay, year, month, day),mayaNoticeByDay):
# maya notice by day
# new:
# get_trade_info(getAppUrlWithoutDate(fundListName),fundListName):
# fund list, fund history data
""" get_trade_info(getFundHistoryById(fundHistoryDataName, 1143718
, 2015, 12, 31, 2022, 12, 28), fundHistoryDataName) """


# not working
# get_trade_info(getAppUrlWithDate(endOfDayTransactionName, year, month, day),endOfDayTransactionName):
# end of day transaction
# get_trade_info(getAppUrlWithoutDate(shortSalesHistoricalData),shortSalesHistoricalData):
# short sales historical data

# more relevant apps:
