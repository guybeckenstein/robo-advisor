import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date
from RoboAdvisorDataScience.api import Portfolio, StatsModels, User
from RoboAdvisorDataScience.util import apiUtil, consoleHandler, plotFunctions, setting, taseUtil


######################################################################################
# update all tables
def updateAllTables(stocksSymbols, numOfYearsHistory):  # build DB for withdraw
    today = date.today()
    formatted_date = today.strftime("%Y-%m-%d")
    updateClosingPricesTables(formatted_date, stocksSymbols, numOfYearsHistory)
    updateDfTables(formatted_date, stocksSymbols)


def updateClosingPricesTables(formatted_date_today, stocksSymbols, numOfYearsHistory):
    with open(setting.bucketRepositoriy + "lastUpdatedClosingPrice.txt", "r") as file:
        lastUpdatedDateClosingPrices = file.read().strip()
    if lastUpdatedDateClosingPrices != formatted_date_today:
        apiUtil.convertDataToTables(stocksSymbols, setting.record_percentage_to_predict,
                                    numOfYearsHistory, machineLearningOpt=0)
        print("updateClosingPricesTables: without maching learning")
        apiUtil.convertDataToTables(stocksSymbols, setting.record_percentage_to_predict,
                                    numOfYearsHistory, machineLearningOpt=1)
        print("updateClosingPricesTables: including maching learning")
    with open(setting.bucketRepositoriy + "lastUpdatedClosingPrice.txt", "w") as file:
        file.write(formatted_date_today)


def updateDfTables(formatted_date_today, stocksSymbols):
    with open(setting.bucketRepositoriy + "lastUpdatedDftables.txt", "r") as file:
        lastUpdatedDftables = file.read().strip()
    if lastUpdatedDftables != formatted_date_today:
        sectorsList = apiUtil.setSectors(stocksSymbols)

        # without maching learning
        closingPricesTable = getClosingPricesTable(machineLearningOpt=0)
        # Markowitz
        updateThreeLevelDfTables(machingLearningOpt=0, modelName="Markowitz",
                                 stocksSymbols=stocksSymbols, sectorsList=sectorsList,
                                 closingPricesTable=closingPricesTable)
        # Gini
        updateThreeLevelDfTables(machingLearningOpt=0, modelName="Gini",
                                 stocksSymbols=stocksSymbols, sectorsList=sectorsList,
                                 closingPricesTable=closingPricesTable)

        # Including maching learning
        closingPricesTable = getClosingPricesTable(machineLearningOpt=0)
        # Markowitz
        updateThreeLevelDfTables(machingLearningOpt=1, modelName="Markowitz",
                                 stocksSymbols=stocksSymbols, sectorsList=sectorsList,
                                 closingPricesTable=closingPricesTable)
        # Gini
        updateThreeLevelDfTables(machingLearningOpt=1, modelName="Gini",
                                 stocksSymbols=stocksSymbols, sectorsList=sectorsList,
                                 closingPricesTable=closingPricesTable)

        with open(setting.bucketRepositoriy + "lastUpdatedDftables.txt", "w") as file:
            file.write(formatted_date_today)


def updateThreeLevelDfTables(machingLearningOpt, modelName, stocksSymbols, sectorsList,
                             closingPricesTable):
    # high risk
    updateSpecificDfTable(machingLearningOpt=machingLearningOpt, modelName=modelName, stocksSymbols=stocksSymbols,
                          sectorsList=sectorsList, levelOfRisk="high", maxPercentCommodity=1,
                          maxPercentStocks=1, closingPricesTable=closingPricesTable)
    # medium risk
    updateSpecificDfTable(machingLearningOpt=machingLearningOpt, modelName=modelName, stocksSymbols=stocksSymbols,
                          sectorsList=sectorsList, levelOfRisk="medium",
                          maxPercentCommodity=setting.limitPercentMediumRiskCommodity,
                          maxPercentStocks=setting.limitPercentMediumRiskStocks, closingPricesTable=closingPricesTable)
    # low risk
    updateSpecificDfTable(machingLearningOpt=machingLearningOpt, modelName=modelName, stocksSymbols=stocksSymbols,
                          sectorsList=sectorsList, levelOfRisk="low",
                          maxPercentCommodity=setting.limitPercenLowRiskCommodity,
                          maxPercentStocks=setting.limitPercenLowRiskStocks, closingPricesTable=closingPricesTable)


def updateSpecificDfTable(machingLearningOpt, modelName, stocksSymbols, sectorsList, levelOfRisk,
                          maxPercentCommodity, maxPercentStocks, closingPricesTable):
    if machingLearningOpt:
        locationForSaving = setting.machineLearningLocation
    else:
        locationForSaving = setting.nonMachineLearningLocation

    if maxPercentCommodity <= 0:
        stock_sectors = apiUtil.setStockSectors(stocksSymbols, sectorsList)
        filtered_stocks = []
        for i in range(len(stock_sectors)):
            if stock_sectors[i] != "US commodity":
                filtered_stocks.append(stocksSymbols[i])
            else:
                closingPricesTable = closingPricesTable.drop(stocksSymbols[i], axis=1)
        stocksSymbols = filtered_stocks

    statsModels = StatsModels.StatsModels(stocksSymbols, sectorsList, closingPricesTable,
                                          setting.Num_porSimulation, setting.min_Num_porSimulation,
                                          maxPercentCommodity,
                                          maxPercentStocks, modelName)
    df = statsModels.getDf()
    df.to_csv(locationForSaving + modelName + '_df_' + levelOfRisk + '.csv')
    # TODO - Maybe save the protfolio table


def getClosingPricesTable(machineLearningOpt):
    if machineLearningOpt:
        closingPricesTable = pd.read_csv(setting.machineLearningLocation + 'closing_prices.csv', index_col=0)
    else:
        closingPricesTable = pd.read_csv(setting.nonMachineLearningLocation + 'closing_prices.csv', index_col=0)
    closingPricesTable = closingPricesTable.iloc[1:]
    closingPricesTable = closingPricesTable.apply(pd.to_numeric, errors='coerce')
    return closingPricesTable


def getThreeLevelDfTables(machineLearningOpt, modelName):
    lowRiskDfTable = getDfTable(machineLearningOpt, modelName, "low")
    mediumRiskDfTable = getDfTable(machineLearningOpt, modelName, "medium")
    highRiskDfTable = getDfTable(machineLearningOpt, modelName, "high")

    return [lowRiskDfTable, mediumRiskDfTable, highRiskDfTable]


def getDfTable(machineLearningOpt, modelName, levelOfRisk):
    if machineLearningOpt:
        dfTable = pd.read_csv(setting.machineLearningLocation + modelName + '_df_' + levelOfRisk + '.csv')
    else:
        dfTable = pd.read_csv(setting.nonMachineLearningLocation + modelName + '_df_' + levelOfRisk + '.csv')
    dfTable = dfTable.iloc[:, 1:]
    dfTable = dfTable.apply(pd.to_numeric, errors='coerce')
    return dfTable

#############################################################
# manual commands
# 1
def createsNewUser(name, stocksSymbols, investmentAmount, machineLearningOpt, modelOption):
    sectorsData = taseUtil.getJsonData("api/resources/sectors")
    sectorsList = apiUtil.setSectors(stocksSymbols)
    closingPricesTable = getClosingPricesTable(machineLearningOpt=machineLearningOpt)
    df = getThreeLevelDfTables(machineLearningOpt, setting.modelName[modelOption - 1])
    threeBestPortfolios = apiUtil.getBestPortfolios(df, modelName=setting.modelName[modelOption - 1])
    bestStocksWeightsColumn = apiUtil.getBestWeightsColumn(stocksSymbols, sectorsList, threeBestPortfolios,
                                                           closingPricesTable.pct_change())
    threeBestStocksWeights = apiUtil.getThreeBestWeights(threeBestPortfolios)
    threeBestSectorsWeights = apiUtil.getThreeBestSectorsWeights(sectorsList, setting.stocksSymbols,
                                                                 threeBestStocksWeights)
    pctChangeTable = closingPricesTable.pct_change()
    yieldList = updatePctChangeTable(bestStocksWeightsColumn, pctChangeTable, investmentAmount)

    # TODO - MOVE TO SITE
    # TODO - change yield list to percent
    levelOfRisk = getDataFromForm(threeBestPortfolios, threeBestSectorsWeights, sectorsList, yieldList, pctChangeTable)

    finalPortfolio = threeBestPortfolios[levelOfRisk - 1]
    if levelOfRisk == 1:
        # drop from stocksSymbols the stocks that are in Us COmmodity SECTOR
        stocksSymbols = apiUtil.dropStocksFromUsCommoditySector(stocksSymbols,
                                                                apiUtil.setStockSectors(stocksSymbols, sectorsList))
    newPortfolio = Portfolio.Portfolio(levelOfRisk, investmentAmount, stocksSymbols, sectorsData, modelOption,
                                       machineLearningOpt)
    newPortfolio.updateStocksData(closingPricesTable, pctChangeTable,
                                  finalPortfolio.iloc[0][3:], finalPortfolio.iloc[0][0],
                                  finalPortfolio.iloc[0][1], finalPortfolio.iloc[0][2])
    user = User.User(name, newPortfolio)
    user.updateJsonFile("DB/users")

    return user


#############################################################################################################
# 2 refresh user data
def refreshUserData(user) -> None:

    portfolio = user.getPortfolio()
    pctChangeTable = portfolio.getPctChangeTable()
    returns_daily = pctChangeTable
    weights = portfolio.getStocksWeights()
    returns_annual = ((1 + returns_daily.mean()) ** 254) - 1
    cov_daily = returns_daily.cov()
    cov_annual = cov_daily * 254
    returns = np.dot(weights, returns_annual[:-4])
    volatility = np.sqrt(np.dot(weights, np.dot(cov_annual[:-4], weights))) # TODO - FIX PROBLEM
    sharpe = returns / volatility
    portfolio.updateStocksData(portfolio.getclosingPricesTable(), pctChangeTable, weights,
                               returns_annual, volatility, sharpe)
    user.updatePortfolio(portfolio)


"""def updateTableOfPortfolios():
    sectorsData = taseUtil.getJsonData("api/resources/sectors")  # TODO REMOVE
    sectorsList = taseUtil.setSectors(setting.stocksSymbols)

    # GET BASIC DATA FROM TERMINAL- TODO- get the data from site-form
    investmentAmount, machineLearningOpt, modelOption = taseUtil.getUserBasicDataFromForm()
    newPortfolio = Portfolio.Portfolio(1, investmentAmount, taseUtil.stocksSymbols, sectorsData, modelOption,
                                       machineLearningOpt)

    # use models to find the best portfolios
    if modelOption == 1:
        # use markovich model to find the best portfolios
        statsModels = StatsModels.StatsModels(setting.stocksSymbols, taseUtil.sectorsList, setting.Num_porSimulation,
                                              setting.record_percentage_to_predict,
                                              setting.numOfYearsHistory, 0, setting.limitPercenLowRiskCommodity
                                              , setting.limitPercentMediumRiskCommodity,
                                              setting.limitPercenLowRiskStocks,
                                              setting.limitPercentMediumRiskStocks, "Markowitz")

    else:
        # use gini model to find the best portfolios
        statsModels = StatsModels.StatsModels(setting.stocksSymbols, sectorsList, setting.Num_porSimulation,
                                              setting.record_percentage_to_predict,
                                              setting.numOfYearsHistory, 0, setting.limitPercenLowRiskCommodity
                                              , setting.limitPercentMediumRiskCommodity,
                                              setting.limitPercenLowRiskStocks,
                                              setting.limitPercentMediumRiskStocks, "Gini")

    closingPricesTable = statsModels.getClosingPricesTable()
    pctChangeTable = statsModels.getPctChangeTable()
    yieldList = updatePctChangeTable(statsModels, pctChangeTable, investmentAmount)
    levelOfRisk = getDataFromForm(statsModels, sectorsList, yieldList, pctChangeTable)
    finalPortfolio = statsModels.getThreeBestPortfolios()[levelOfRisk - 1]

    newPortfolio.updateLevelOfRisk(levelOfRisk)
    # build the portfolio according to the level of risk
    newPortfolio.updateStocksData(closingPricesTable, pctChangeTable,
                                  finalPortfolio.iloc[0][3:], finalPortfolio.iloc[0][0],
                                  finalPortfolio.iloc[0][1], finalPortfolio.iloc[0][2])
    user = User.User("yarden", newPortfolio)
    user.updateJsonFile("DB/users")

    return user"""


#############################################################################################################
# 3 - plot user portfolio - TODO- PLOT FROM SITE

def plotUserPortfolio(user) -> None:
    plt = user.plotPortfolioComponent()
    plotFunctions.plot(plt)
    plt = user.plotInvestmentPortfolioYield()
    plotFunctions.plot(plt)


#############################################################################################################
# 4- EXPERT OPTIONS:
#############################################################################################################
# EXPERT - 1


def forcastSpecificStock(stock, isDataComeFromTase, numOfYearsHistory) -> None:
    if isDataComeFromTase:
        df = pd.DataFrame(stock["indexEndOfDay"]["result"])
        df["tradeDate"] = pd.to_datetime(df["tradeDate"])
        df.set_index("tradeDate", inplace=True)
        df, col = price_forecast(df, setting.record_percentage_to_predict, 1)
        plotPriceForcast(stock, df, 1)

    else:
        yf.pdr_override()
        start_date, end_date = getFromAndToDate(numOfYearsHistory)
        df = yf.download(stock, start=start_date, end=end_date)
        df, col = price_forecast(df, setting.record_percentage_to_predict, 0)
        plotPriceForcast(stock, df, 0)


#############################################################################################################
# EXPERT -2


def plotbb_strategy_stock(stockName, start="2009-01-01", end="2023-01-01") -> None:
    stock_prices = yf.download(stockName, start, end)
    stock_prices['MA50'] = stock_prices['Adj Close'].rolling(window=50).mean()
    stock_prices['50dSTD'] = stock_prices['Adj Close'].rolling(window=50).std()
    stock_prices['Upper'] = stock_prices['MA50'] + (stock_prices['50dSTD'] * 2)
    stock_prices['Lower'] = stock_prices['MA50'] - (stock_prices['50dSTD'] * 2)

    stock_prices = stock_prices.dropna()
    stock_prices = stock_prices.iloc[51:]

    buy_price, sell_price, bb_signal = apiUtil.implement_bb_strategy(stock_prices['Adj Close'],
                                                                     stock_prices['Lower'], stock_prices['Upper'])
    plotFunctions.plotbb_strategy_stock(stock_prices, buy_price, sell_price)


#############################################################################################################
# 4 -3


def createJsonDataFromTase(indexId, nameFile) -> None:
    jsonData = taseUtil.getJsonDataFromTase(indexId, nameFile)
    createsJsonFile(jsonData, nameFile)


#############################################################################################################
# EXPERT -4


def findBestStocks() -> None:
    plotFunctions.plotTopStocks(apiUtil.findBestStocks())


def scanGoodStocks() -> None:
    plotFunctions.plotTopStocks(apiUtil.scanGoodStocks())


############################################################################################################
# EXPERT- 5&6


def plotStatModelGraph(stocksSymbols, machineLearningOpt, modelOption) -> None:
    sectorsList = setSectors(stocksSymbols)

    closingPricesTable = getClosingPricesTable(machineLearningOpt)

    if modelOption == "Markowitz":
        statsModels = StatsModels.StatsModels(stocksSymbols, sectorsList, closingPricesTable,
                                              setting.Num_porSimulation,
                                              setting.min_Num_porSimulation,
                                              1, 1, "Markowitz")
    else:
        statsModels = StatsModels.StatsModels(stocksSymbols, sectorsList, closingPricesTable,
                                              setting.Num_porSimulation,
                                              setting.min_Num_porSimulation,
                                              1, 1, "Gini")
    df = statsModels.getDf()
    ThreePortfoliosList = apiUtil.getBestPortfolios(df, modelName=setting.modelName[modelOption - 1])
    threeBestStocksWeights = apiUtil.getThreeBestWeights(ThreePortfoliosList)
    threeBestSectorsWeights = apiUtil.getThreeBestSectorsWeights(sectorsList, setting.stocksSymbols,
                                                                 threeBestStocksWeights)

    min_variance_port = ThreePortfoliosList[0]
    sharpe_portfolio = ThreePortfoliosList[1]
    max_returns = ThreePortfoliosList[2]
    max_vols = statsModels.getMaxVols()
    df = statsModels.getDf()

    if modelOption == "Markowitz":
        plotFunctions.plotMarkowitzGraph(sectorsList, threeBestSectorsWeights, min_variance_port,
                                         sharpe_portfolio, max_returns, max_vols, df)
    else:
        plotFunctions.plotGiniGraph(sectorsList, threeBestSectorsWeights, min_variance_port,
                                    sharpe_portfolio, max_returns, max_vols, df)


############################################################################################################
# UTILITY FUNCTIONS
############################################################################################################
# database utility functions:


def getAllUsers():
    jsonData = getJsonData(setting.usersJsonName)
    numOfUser = len(jsonData['usersList'])
    usersData = jsonData['usersList']
    _usersList = [] * numOfUser
    for name in usersData.items():
        _usersList.append(getUserFromDB(name))

    return _usersList


def getUserFromDB(userName):
    jsonData = getJsonData(setting.usersJsonName)
    if userName not in jsonData['usersList']:
        print("User not found")
        return None
    userData = jsonData['usersList'][userName][0]
    startingInvestmentAmount = userData['startingInvestmentAmount']
    machineLearningOpt = userData['machineLearningOpt']
    selectedModel = userData['selectedModel']
    levelOfRisk = userData['levelOfRisk']
    stocksSymbols = userData['stocksSymbols']
    stocksWeights = userData['stocksWeights']
    annualReturns = userData['annualReturns']
    annualVolatility = userData['annualVolatility']
    annualSharpe = userData['annualSharpe']

    # TODO - if each user has different sectors make sector data using sectorsNames and sectorsWeights
    sectorsData = getJsonData("api/resources/sectors")  # universal from file

    # get data from api and convert it to tables
    closingPricesTable = getClosingPricesTable(int(machineLearningOpt))
    userPortfolio = Portfolio.Portfolio(levelOfRisk, startingInvestmentAmount, stocksSymbols, sectorsData,
                                        selectedModel, machineLearningOpt)
    pctChangeTable = closingPricesTable.pct_change()
    pctChangeTable.dropna(inplace=True)
    weighted_sum = np.dot(stocksWeights, pctChangeTable.T)
    pctChangeTable["weighted_sum_" + str(levelOfRisk)] = weighted_sum
    pctChangeTable["yield_" + str(levelOfRisk)] = weighted_sum
    pctChangeTable["yield_" + str(levelOfRisk)] = makesYieldColumn(pctChangeTable["yield_" + str(levelOfRisk)],
                                                                   weighted_sum, startingInvestmentAmount)
    userPortfolio.updateStocksData(closingPricesTable, pctChangeTable, stocksWeights, annualReturns,
                                   annualVolatility, annualSharpe)
    user = User.User(userName, userPortfolio)

    return user


def findUserInList(userName, usersList):
    for user in usersList:
        if user.getUserName() == userName:
            return user

    return None


def getNumOfUsersInDB() -> int:
    jsonData = getJsonData(setting.usersJsonName)

    return len(jsonData['usersList'])


def updatePctChangeTable(bestStocksWeightsColumn, pctChangeTable, investment):
    [weighted_low, weighted_medium, weighted_high] = bestStocksWeightsColumn
    pctChangeTable.dropna(inplace=True)
    pctChangeTable["weighted_sum_1"] = weighted_low
    pctChangeTable["weighted_sum_2"] = weighted_medium
    pctChangeTable["weighted_sum_3"] = weighted_high
    pctChangeTable["yield_1"] = weighted_low
    pctChangeTable["yield_2"] = weighted_medium
    pctChangeTable["yield_3"] = weighted_high
    yield_low = makesYieldColumn(pctChangeTable["yield_1"], weighted_low, investment)
    yield_medium = makesYieldColumn(pctChangeTable["yield_2"], weighted_medium, investment)
    yield_high = makesYieldColumn(pctChangeTable["yield_3"], weighted_high, investment)
    pctChangeTable["yield_1"] = yield_low
    pctChangeTable["yield_2"] = yield_medium
    pctChangeTable["yield_3"] = yield_high

    return [yield_low, yield_medium, yield_high]


def getDataFromForm(threeBestPortfolosList, threeBestSectorsWeights, sectorsList, yieldsList, pctChangeTable) -> int:
    # TODO - get from form instead of console
    count = 0

    # question 1
    stringToShow = "for how many years do you want to invest?\n" + "0-1 - 1\n""1-3 - 2\n""3-100 - 3\n"
    count += getScoreByAnswerFromUser(stringToShow)

    # question 2
    stringToShow = "Which distribution do you prefer?\nlow risk - 1, medium risk - 2, high risk - 3 ?\n"
    plotDistributionOfPortfolio(yieldsList)
    count += getScoreByAnswerFromUser(stringToShow)

    # question 3
    stringToShow = "Which graph do you prefer?\nsaftest - 1, sharpest - 2, max return - 3 ?\n"
    plotThreePortfoliosGraph(threeBestPortfolosList, threeBestSectorsWeights, sectorsList, pctChangeTable)
    count += getScoreByAnswerFromUser(stringToShow)

    return getLevelOfRiskByForm(count)


def getLevelOfRiskByForm(count: int) -> int:
    if count <= 4:
        return 1
    elif count <= 7:
        return 2
    else:
        return 3


def createsJsonFile(json_obj, nameProduct) -> None:
    # Open a file in write mode
    parts = nameProduct.split("/")
    last_element = parts[-1]
    with open(
            "api/resources/" + last_element + ".json", "w"
    ) as f:  # Use the `dump()` function to write the JSON data to the file
        json.dump(json_obj, f)


# api utility functions
def convertDataToTables(stocksNames, record_percentage_to_predict, numOfYearsHistory, machineLearningOpt):
    return apiUtil.convertDataToTables(stocksNames, record_percentage_to_predict, numOfYearsHistory, machineLearningOpt)


def getSectorsDataFromFile():
    return apiUtil.getSectorsDataFromFile()


def setSectors(stocksSymbols):
    return apiUtil.setSectors(stocksSymbols)


def makesYieldColumn(_yield, weighted_sum_column, investment):
    return apiUtil.makesYieldColumn(_yield, weighted_sum_column, investment)


def price_forecast(df, record_percentage_to_predict, isDataFromTase):
    return apiUtil.price_forecast(df, record_percentage_to_predict, isDataFromTase)


def getJsonData(name):
    return apiUtil.getJsonData(name)


def getFromAndToDate(numOfYears) -> tuple[str, str]:
    return apiUtil.getFromAndToDates(numOfYears)


# plot functions
def plotThreePortfoliosGraph(threeBestPortfolosList, threeBestSectorsWeights, sectorsList, pctChangeTable) -> None:
    min_variance_port = threeBestPortfolosList[0]
    sharpe_portfolio = threeBestPortfolosList[1]
    max_returns = threeBestPortfolosList[2]
    plotFunctions.plotThreePortfoliosGraph(min_variance_port, sharpe_portfolio, max_returns,
                                           threeBestSectorsWeights, sectorsList, pctChangeTable)


def plotDistributionOfStocks(stockNames, pctChangeTable) -> None:
    plotFunctions.plotDistributionOfStocks(stockNames, pctChangeTable)


def plotDistributionOfPortfolio(yieldsList) -> None:
    plotFunctions.plotDistributionOfPortfolio(yieldsList)


def plotbb_strategy_Portfolio(pctChangeTable, newPortfolio) -> None:
    plotFunctions.plotbb_strategy_Portfolio(pctChangeTable, newPortfolio)


def plotPriceForcast(stockSymbol, df, isDataGotFromTase) -> None:
    plotFunctions.plotpriceForcast(stockSymbol, df, isDataGotFromTase)


def getScoreByAnswerFromUser(stringToShow):
    return consoleHandler.getScoreByAnswerFromUser(stringToShow)


# input from user functions(currently from console)


def getUserBasicDataFromForm():
    return consoleHandler.getUserBasicDataFromForm()


def mainMenu() -> None:
    consoleHandler.mainMenu()


def expertMenu() -> None:
    consoleHandler.expertMenu()


def selectedMenuOption():
    return consoleHandler.selectedMenuOption()


def getName():
    return consoleHandler.getName()


def getNumOfYearsHistory():
    return consoleHandler.getNumOfYearsHistory()


def getMachineLearningOption():
    return consoleHandler.getMachineLearningOption()
