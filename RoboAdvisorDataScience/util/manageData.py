import json
import pandas as pd
import numpy as np
import yfinance as yf
from RoboAdvisorDataScience.util import setting
from RoboAdvisorDataScience.util import taseUtil
from RoboAdvisorDataScience.util import consoleHandler
from RoboAdvisorDataScience.util import apiUtil
from RoboAdvisorDataScience.util import plotFunctions
from RoboAdvisorDataScience.api import Portfolio as Portfolio
from RoboAdvisorDataScience.api import User as User
from RoboAdvisorDataScience.api import StatsModels as StatsModels

######################################################################################
# 1 - create new user by form


def createsNewUser(name, stocksSymbols, numOfYearsHistory):
    setting
    sectorsData = getJsonData("api/resources/sectors")  # TODO REMOVE
    sectorsList = setSectors(stocksSymbols)

    # GET BASIC DATA FROM TERMINAL- TODO- get the data from site-form
    investmentAmount, machineLearningOpt, modelOption = getUserBasicDataFromForm()
    newPortfolio = Portfolio.Portfolio(1, investmentAmount, stocksSymbols, sectorsData, modelOption, machineLearningOpt)

    # use models to find the best portfolios
    if modelOption == 1:
        # use markovich model to find the best portfolios
        statsModels = StatsModels.StatsModels(stocksSymbols, sectorsList, setting.Num_porSimulation,
                                              setting.record_percentage_to_predict,
                                              numOfYearsHistory, machineLearningOpt, "Markowitz")

    else:
        # use gini model to find the best portfolios
        statsModels = StatsModels.StatsModels(stocksSymbols, sectorsList, setting.Num_porSimulation,
                                              setting.record_percentage_to_predict,
                                              numOfYearsHistory, machineLearningOpt, "Gini")

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
    user = User.User(name, newPortfolio)
    user.updateJsonFile("DB/users")

    return user


#############################################################################################################
# 2 refresh user data

def refreshUserData(user):

    portfolio = user.getPortfolio()
    pctChangeTable = portfolio.getPctChangeTable()
    returns_daily = pctChangeTable
    weights = portfolio.getStocksWeights()
    returns_annual = ((1 + returns_daily.mean()) ** 254) - 1
    cov_daily = returns_daily.cov()
    cov_annual = cov_daily * 254
    returns = np.dot(weights, returns_annual[:-4])
    volatility = np.sqrt(np.dot(weights, np.dot(cov_annual[:-4], weights)))
    sharpe = returns / volatility
    # find the level of risk according to the user's choice
    # build the portfolio according to the level of risk
    portfolio.updateStocksData(portfolio.getclosingPricesTable(), pctChangeTable, weights,
                               returns_annual, volatility, sharpe)
    user.updatePortfolio(portfolio)

#############################################################################################################
# 3 - plot user portfolio - TODO- PLOT FROM SITE


def plotUserPortfolio(user):
    plt = user.plotPortfolioComponent()
    plotFunctions.plot(plt)
    plt = user.plotInvestmentPortfolioYield()
    plotFunctions.plot(plt)

#############################################################################################################
# 4- EXPERT OPTIONS:
#############################################################################################################
# EXPERT - 1


def forcastSpecificStock(stock, isDataComeFromTase, numOfYearsHistory):
    if isDataComeFromTase:
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

#############################################################################################################
# EXPERT -2


def plotbb_strategy_stock(stockName, start="2009-01-01", end="2023-01-01"):
    stockprices = yf.download(stockName, start, end)
    stockprices['MA50'] = stockprices['Adj Close'].rolling(window=50).mean()
    stockprices['50dSTD'] = stockprices['Adj Close'].rolling(window=50).std()
    stockprices['Upper'] = stockprices['MA50'] + (stockprices['50dSTD'] * 2)
    stockprices['Lower'] = stockprices['MA50'] - (stockprices['50dSTD'] * 2)

    stockprices = stockprices.dropna()
    stockprices = stockprices.iloc[51:]

    buy_price, sell_price, bb_signal = apiUtil.implement_bb_strategy(stockprices['Adj Close'],
                                                                     stockprices['Lower'], stockprices['Upper'])
    plotFunctions.plotbb_strategy_stock(stockprices, buy_price, sell_price)

#############################################################################################################
# 4 -3


def createJsonDataFromTase(indexId, nameFile):
    jsonData = taseUtil.getJsonDataFromTase(indexId, nameFile)
    createsJsonFile(jsonData, nameFile)

#############################################################################################################
# EXPERT -4


def findBestStocks():
    plotFunctions.plotTopStocks(apiUtil.findBestStocks())


def scanGoodStocks():
    plotFunctions.plotTopStocks(apiUtil.scanGoodStocks())
############################################################################################################
# EXPERT- 5&6


def plotStatModelGraph(stocksSymbols, numOfYearsHistory, machineLearningOpt, modelOption):
    sectorsList = setSectors(stocksSymbols)

    if modelOption == "Markowitz":
        statsModels = StatsModels.StatsModels(stocksSymbols, sectorsList, setting.Num_porSimulation,
                                              setting.record_percentage_to_predict, numOfYearsHistory,
                                              machineLearningOpt, "Markowitz")
    else:
        statsModels = StatsModels.StatsModels(stocksSymbols, sectorsList, setting.Num_porSimulation,
                                              setting.record_percentage_to_predict,
                                              numOfYearsHistory, machineLearningOpt, "Gini")

    ThreePortfoliosList = statsModels.getThreeBestPortfolios()
    threeBestSectorsWeights = statsModels.getThreeBestSectorsWeights()
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
    closingPricesTable = convertDataToTables(stocksSymbols, setting.record_percentage_to_predict,
                                             setting.numOfYearsHistory, machineLearningOpt)
    userPortfolio = Portfolio.Portfolio(levelOfRisk, startingInvestmentAmount, stocksSymbols, sectorsData,
                                        selectedModel, machineLearningOpt)
    pctChangeTable = closingPricesTable.pct_change()

    pctChangeTable.dropna(inplace=True)
    weighted_sum = np.dot(stocksWeights, pctChangeTable.T)
    pctChangeTable["weighted_sum_"+str(levelOfRisk)] = weighted_sum
    pctChangeTable["yield_"+str(levelOfRisk)] = weighted_sum
    makesYieldColumn(pctChangeTable["yield_"+str(levelOfRisk)], weighted_sum, startingInvestmentAmount)
    userPortfolio.updateStocksData(closingPricesTable, pctChangeTable, stocksWeights, annualReturns,
                                   annualVolatility, annualSharpe)
    user = User.User(userName, userPortfolio)

    return user


def findUserInList(userName, usersList):
    for user in usersList:
        if user.getUserName() == userName:
            return user

    return None


def getNumOfUsersInDB():
    jsonData = getJsonData(setting.usersJsonName)

    return len(jsonData['usersList'])


def updatePctChangeTable(statModel, pctChangeTable, investment):
    [weighted_low, weighted_medium, weighted_high] = statModel.getBestStocksWeightsColumn()
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

    return [yield_low, yield_medium, yield_high]


def getDataFromForm(statModel, sectorsList, yieldsList, pctChangeTable):  # TODO - get from form instead of console

    threeBestPortfolosList = statModel.getThreeBestPortfolios()
    threeBestSectorsWeights = statModel.getThreeBestSectorsWeights()
    count = 0

    # question 1
    stringToShow = "for how many years do you want to invest?\n" + "0-1 - 1\n""1-3 - 2\n""3-100 - 3\n"
    count += getScoreByAnswerFromUser(stringToShow)

    # question 2
    stringToShow = "Which distribution do you prefer?\nlow risk - 1, medium risk - 2, high risk - 3 ?\n"
    plotDistributionOfPortfolio(yieldsList)
    count += getScoreByAnswerFromUser(stringToShow)

    # question 3
    stringToShow = "Which graph do you prefer?\nsaftest - 1, sharpe - 2, max return - 3 ?\n"
    plotThreePortfoliosGraph(threeBestPortfolosList, threeBestSectorsWeights, sectorsList, pctChangeTable)
    count += getScoreByAnswerFromUser(stringToShow)

    return getLevelOfRiskByForm(count)


def getLevelOfRiskByForm(count):
    if count <= 4:
        return 1
    elif count <= 7:
        return 2
    else:
        return 3


def createsJsonFile(json_obj, nameProduct):
    # Open a file in write mode
    parts = nameProduct.split("/")
    last_element = parts[-1]
    with open(
        "RoboAdvisorDataScience/api/resources/"+last_element + ".json", "w"
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


def getfromAndToDate(numOfYears):
    return apiUtil.getfromAndToDate(numOfYears)


# plot functions
def plotThreePortfoliosGraph(threeBestPortfolosList, threeBestSectorsWeights, sectorsList, pctChangeTable):
    min_variance_port = threeBestPortfolosList[0]
    sharpe_portfolio = threeBestPortfolosList[1]
    max_returns = threeBestPortfolosList[2]
    plotFunctions.plotThreePortfoliosGraph(min_variance_port, sharpe_portfolio, max_returns,
                                           threeBestSectorsWeights, sectorsList, pctChangeTable)


def plotDistributionOfStocks(stockNames, pctChangeTable):
    plotFunctions.plotDistributionOfStocks(stockNames, pctChangeTable)


def plotDistributionOfPortfolio(yieldsList):
    plotFunctions.plotDistributionOfPortfolio(yieldsList)


def plotbb_strategy_Portfolio(pctChangeTable, newPortfolio):
    plotFunctions.plotbb_strategy_Portfolio(pctChangeTable, newPortfolio)


def plotpriceForcast(stockSymbol, df, isDataGotFromTase):
    plotFunctions.plotpriceForcast(stockSymbol, df, isDataGotFromTase)


def getScoreByAnswerFromUser(stringToShow):
    return consoleHandler.getScoreByAnswerFromUser(stringToShow)

# input from user functions(currently from console)


def getUserBasicDataFromForm():
    return consoleHandler.getUserBasicDataFromForm()


def mainMenu():
    consoleHandler.mainMenu()


def expertMenu():
    consoleHandler.expertMenu()


def selectedMenuOption():
    return consoleHandler.selectedMenuOption()


def getName():
    return consoleHandler.getName()


def getNumOfYearsHistory():
    return consoleHandler.getNumOfYearsHistory()


def getMachineLearningOption():
    return consoleHandler.getMachineLearningOption()
