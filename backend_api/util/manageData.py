import json
import numpy as np
import pandas as pd
import yfinance as yf
from backend_api.api import Portfolio, StatsModels, User
from backend_api.util import apiUtil, consoleHandler, plotFunctions, setting

STATIC_FILES_LOCATION = 'static/img/graphs/'


######################################################################################
# 1
def createsNewUser(name: str, stocksSymbols: list, investmentAmount: int,
                   machineLearningOpt: bool, modelOption: bool, levelOfRisk: int, sectorsData, sectorsList,
                   closingPricesTable, threeBestPortfolios, pctChangeTable)-> User:

    finalPortfolio = threeBestPortfolios[levelOfRisk - 1]
    if levelOfRisk == 1:
        # drop from stocksSymbols the stocks that are in Us Commodity sector
        stocksSymbols = apiUtil.dropStocksFromUsCommoditySector(stocksSymbols,
        apiUtil.setStockSectors(stocksSymbols, sectorsList))

    newPortfolio = Portfolio.Portfolio(levelOfRisk, investmentAmount, stocksSymbols,
                                       sectorsData, modelOption,
                                       machineLearningOpt)

    newPortfolio.updateStocksData(closingPricesTable, pctChangeTable,
                                  finalPortfolio.iloc[0][3:], finalPortfolio.iloc[0][0],
                                  finalPortfolio.iloc[0][1], finalPortfolio.iloc[0][2])
    user = User.User(name, newPortfolio)

    return user


#############################################################################################################
# 3 - plot user portfolio -# TODO plot at site

def plotUserPortfolio(user) -> None:
    # pie chart of sectors weights
    plt_instance = user.plotPortfolioComponent()
    plotFunctions.plot(plt_instance)# TODO plot at site
    # chart of stocks weights
    plt_instance = user.plotPortfolioComponentStocks()
    plotFunctions.plot(plt_instance)  # TODO plot at site
    plt_instance = user.plotInvestmentPortfolioYield()
    plotFunctions.plot(plt_instance)# TODO plot at site


#############################################################################################################
# 4- EXPERT OPTIONS:
#############################################################################################################
# EXPERT - 1


def forcastSpecificStock(stock:str, isDataComeFromTase:bool, numOfYearsHistory:int) -> None:
    if isDataComeFromTase:
        df = pd.DataFrame(stock["indexEndOfDay"]["result"])
        df["tradeDate"] = pd.to_datetime(df["tradeDate"])
        df.set_index("tradeDate", inplace=True)
        df, col = price_forecast(df, setting.record_percentage_to_predict, 1)
        plt_instance = plotPriceForcast(stock, df, 1)

    else:
        yf.pdr_override()
        start_date, end_date = getFromAndToDate(numOfYearsHistory)
        df = yf.download(stock, start=start_date, end=end_date)
        df, col = price_forecast(df, setting.record_percentage_to_predict, 0)
        plt_instance = plotPriceForcast(stock, df, 0)
    plotFunctions.plot(plt_instance)  # TODO plot at site


#############################################################################################################
# EXPERT -2


def plotbb_strategy_stock(stockName:str, start="2009-01-01", end="2023-01-01") -> None:
    stock_prices = yf.download(stockName, start, end)
    stock_prices['MA50'] = stock_prices['Adj Close'].rolling(window=50).mean()
    stock_prices['50dSTD'] = stock_prices['Adj Close'].rolling(window=50).std()
    stock_prices['Upper'] = stock_prices['MA50'] + (stock_prices['50dSTD'] * 2)
    stock_prices['Lower'] = stock_prices['MA50'] - (stock_prices['50dSTD'] * 2)

    stock_prices = stock_prices.dropna()
    stock_prices = stock_prices.iloc[51:]

    buy_price, sell_price, bb_signal = apiUtil.implement_bb_strategy(stock_prices['Adj Close'],
                                                                     stock_prices['Lower'], stock_prices['Upper'])
    plt_instance = plotFunctions.plotbb_strategy_stock(stock_prices, buy_price, sell_price)
    plotFunctions.plot(plt_instance)  # TODO plot at site

#############################################################################################################
# EXPERT -4


def findBestStocks() -> None:
    plt_instance = plotFunctions.plotTopStocks(apiUtil.findBestStocks())
    plotFunctions.plot(plt_instance)  # TODO plot at site


def scanGoodStocks() -> None:
    plt_instance = plotFunctions.plotTopStocks(apiUtil.scanGoodStocks())
    plotFunctions.plot(plt_instance)  # TODO plot at site


############################################################################################################
# EXPERT- 5&6


def plotStatModelGraph(stocksSymbols: list, machineLearningOpt: int, modelOption: int) -> None:
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
        plt_instance = plotFunctions.plotMarkowitzGraph(sectorsList, threeBestSectorsWeights, min_variance_port,
                                         sharpe_portfolio, max_returns, max_vols, df)
    else:
        plt_instance = plotFunctions.plotGiniGraph(sectorsList, threeBestSectorsWeights, min_variance_port,
                                    sharpe_portfolio, max_returns, max_vols, df)

    plotFunctions.plot(plt_instance)# TODO plot at site


############################################################################################################
# UTILITY FUNCTIONS
############################################################################################################
# database utility functions:

# Get extended data information from Db (csv tables)
def getExtendedDataFromDB(stocksSymbols: list, machineLearningOpt: int, modelOption: int):
    sectorsData = apiUtil.getJsonData("backend_api/api/resources/sectors")
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
    yieldList = updatePctChangeTable(bestStocksWeightsColumn, pctChangeTable)

    return (sectorsData, sectorsList, closingPricesTable, threeBestPortfolios,
            threeBestSectorsWeights, pctChangeTable, yieldList)


# Tables according to stocks symbols
def getClosingPricesTable(machineLearningOpt: bool)-> pd.DataFrame:
    if machineLearningOpt:
        closingPricesTable = pd.read_csv(setting.machineLearningLocation + 'closing_prices.csv', index_col=0)
    else:
        closingPricesTable = pd.read_csv(setting.nonMachineLearningLocation + 'closing_prices.csv', index_col=0)
    closingPricesTable = closingPricesTable.iloc[1:]
    closingPricesTable = closingPricesTable.apply(pd.to_numeric, errors='coerce')
    return closingPricesTable

# get the three level df tables according to machine learning option and model name
def getThreeLevelDfTables(machineLearningOpt: bool, modelName: bool)-> list:
    lowRiskDfTable = getDfTable(machineLearningOpt, modelName, "low")
    mediumRiskDfTable = getDfTable(machineLearningOpt, modelName, "medium")
    highRiskDfTable = getDfTable(machineLearningOpt, modelName, "high")

    return [lowRiskDfTable, mediumRiskDfTable, highRiskDfTable]

# get specific df table from csv file according to machine learning option, model name and level of risk
def getDfTable(machineLearningOpt: bool, modelName: bool, levelOfRisk: str)-> pd.DataFrame:
    if machineLearningOpt:
        dfTable = pd.read_csv(setting.machineLearningLocation + modelName + '_df_' + levelOfRisk + '.csv')
    else:
        dfTable = pd.read_csv(setting.nonMachineLearningLocation + modelName + '_df_' + levelOfRisk + '.csv')
    dfTable = dfTable.iloc[:, 1:]
    dfTable = dfTable.apply(pd.to_numeric, errors='coerce')
    return dfTable

# Get all users with their portfolios details from json file
def getAllUsers()-> list:
    jsonData = getJsonData(setting.usersJsonName)
    numOfUser = len(jsonData['usersList'])
    usersData = jsonData['usersList']
    _usersList = [] * numOfUser
    for name in usersData.items():
        _usersList.append(getUserFromDB(name))

    return _usersList

# Get specific user by his name with his portfolio details from json file
def getUserFromDB(userName: str)-> User.User:
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
    sectorsData = getJsonData("backend_api/api/resources/sectors")  # universal from file

    closingPricesTable = getClosingPricesTable(int(machineLearningOpt))
    userPortfolio = Portfolio.Portfolio(levelOfRisk, startingInvestmentAmount, stocksSymbols, sectorsData,
                                        selectedModel, machineLearningOpt)
    pctChangeTable = closingPricesTable.pct_change()
    pctChangeTable.dropna(inplace=True)
    weighted_sum = np.dot(stocksWeights, pctChangeTable.T)
    pctChangeTable["weighted_sum_" + str(levelOfRisk)] = weighted_sum
    pctChangeTable["yield_" + str(levelOfRisk)] = weighted_sum
    pctChangeTable["yield_" + str(levelOfRisk)] = makesYieldColumn(pctChangeTable["yield_" + str(levelOfRisk)],
                                                                   weighted_sum)
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


def updatePctChangeTable(bestStocksWeightsColumn, pctChangeTable):
    [weighted_low, weighted_medium, weighted_high] = bestStocksWeightsColumn
    pctChangeTable.dropna(inplace=True)
    pctChangeTable["weighted_sum_1"] = weighted_low
    pctChangeTable["weighted_sum_2"] = weighted_medium
    pctChangeTable["weighted_sum_3"] = weighted_high
    pctChangeTable["yield_1"] = weighted_low
    pctChangeTable["yield_2"] = weighted_medium
    pctChangeTable["yield_3"] = weighted_high
    yield_low = makesYieldColumn(pctChangeTable["yield_1"], weighted_low)
    yield_medium = makesYieldColumn(pctChangeTable["yield_2"], weighted_medium)
    yield_high = makesYieldColumn(pctChangeTable["yield_3"], weighted_high)
    pctChangeTable["yield_1"] = yield_low
    pctChangeTable["yield_2"] = yield_medium
    pctChangeTable["yield_3"] = yield_high

    return [yield_low, yield_medium, yield_high]


def getDataFromForm(threeBestPortfolosList, threeBestSectorsWeights, sectorsList, yieldsList, pctChangeTable) -> int:
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

    return getLevelOfRiskByScore(count)


def getLevelOfRiskByScore(count: int) -> int:
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

def setSectors(stocksSymbols):
    return apiUtil.setSectors(stocksSymbols)


def makesYieldColumn(_yield, weighted_sum_column):
    return apiUtil.makesYieldColumn(_yield, weighted_sum_column)


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
    plt_instance = plotFunctions.plotThreePortfoliosGraph(min_variance_port, sharpe_portfolio, max_returns,
                                           threeBestSectorsWeights, sectorsList, pctChangeTable)
    plotFunctions.save_graphs(plt_instance, STATIC_FILES_LOCATION + 'three_portfolios.png')


def plotDistributionOfStocks(stockNames, pctChangeTable) -> None:
    plt_instance = plotFunctions.plotDistributionOfStocks(stockNames, pctChangeTable)
    plotFunctions.plot(plt_instance)# TODO plot at site


def plotDistributionOfPortfolio(distribution_graph) -> None:
    plt_instance = plotFunctions.plotDistributionOfPortfolio(distribution_graph)
    plotFunctions.save_graphs(plt_instance, STATIC_FILES_LOCATION + 'distribution_graph.png')


def plotbb_strategy_Portfolio(pctChangeTable, newPortfolio) -> None:
    plt_instance = plotFunctions.plotbb_strategy_Portfolio(pctChangeTable, newPortfolio)
    plotFunctions.plot(plt_instance)# TODO plot at site


def plotPriceForcast(stockSymbol, df, isDataGotFromTase) -> None:
    plt_instance = plotFunctions.plotpriceForcast(stockSymbol, df, isDataGotFromTase)
    plotFunctions.plot(plt_instance)# TODO plot at site


def getScoreByAnswerFromUser(stringToShow: str)-> int:
    return consoleHandler.getScoreByAnswerFromUser(stringToShow)


# console functions

def mainMenu() -> None:
    consoleHandler.mainMenu()


def expertMenu() -> None:
    consoleHandler.expertMenu()


def selectedMenuOption()->int:
    return consoleHandler.selectedMenuOption()


def getName()-> str:
    return consoleHandler.getName()


def getNumOfYearsHistory()->int:
    return consoleHandler.getNumOfYearsHistory()


def getMachineLearningOption()->int:
    return consoleHandler.getMachineLearningOption()

def getModelOption()->int:
    return consoleHandler.getModelOption()

def getInvestmentAmount():
    return consoleHandler.getInvestmentAmount()
