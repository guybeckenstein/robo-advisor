#!/usr/bin/python
# -*- coding: utf-8 -*-
import setting
import manageData
import Portfolio as Portfolio
import User as User


######################################################################################

def init(_sectorsNames, _stocksSymbols):

    # GET BASIC DATA FROM TERMINAL- TODO- get the data from site-form

    (
        levelOfRisk,
        investmentAmount,
        machineLearningOpt,
        israeliIndexesChoice,
        usaIndexesChoice,
        name,
        ) = manageData.getDataFromForm()

    newPortfolio = Portfolio.Portfolio(levelOfRisk, investmentAmount, _stocksSymbols, _sectorsNames)

    # get data from api and convert it to tables

    (closingPricesTable, pctChangeTable) = \
        manageData.convertDataToTables(
        newPortfolio,
        israeliIndexesChoice,
        usaIndexesChoice,
        setting.record_percentage_to_predict,
        numOfYearsHistory,
        machineLearningOpt,
        )

    # use markovich model to find the best portfolios

    (
        df,
        sharpe_portfolio,
        min_variance_port,
        max_returns,
        max_vols,
        weighted_low,
        weighted_medium,
        weighted_high,
        ) = manageData.markovichModel(setting.Num_porSimulation, pctChangeTable, _stocksSymbols)

    # plot all portfolios options by markovich model

    manageData.plotMarkovichPortfolioGraph(
        df,
        sharpe_portfolio,
        min_variance_port,
        max_returns,
        max_vols,
        newPortfolio,
        )

    # plot distributions of the 3 best portfolios that returned from markovich model

    manageData.plotDistributionOfPortfolio(weighted_low, weighted_medium, weighted_high)

    # TODO -add option- use jini model to find the best portfolios

    # building portfolio according to the level of risk

    (pctChangeTable, stock_weights, annualReturns, volatility,
     sharpe) = manageData.buildingPortfolio(
        pctChangeTable,
        sharpe_portfolio,
        min_variance_port,
        max_returns,
        levelOfRisk,
        investmentAmount,
        )

    # creates new user

    newPortfolio.updateStocksData(
        closingPricesTable,
        pctChangeTable,
        stock_weights,
        annualReturns,
        volatility,
        sharpe,
        )
    user = User.User(name, newPortfolio)

    # PLOT the selected portfolio

    manageData.plotbb_strategy_Portfolio(pctChangeTable)
    print(pctChangeTable.describe())
    user.plotPortfolioComponent()  # DIAGRAM
    user.plotInvestmentPortfolioYield()  # plot portfolio yield


# manageData.price_forecast(pctChangeTable, name, stocksNames, annualReturns, volatility, sharpe, stocks_weights)
######################################################################################

# set manually

stocksSymbols = [  # TODO -NEW FEATURES- creates features to scan good stocks and indexes and fit them to the portfolio
    142,
    601,
    602,
    700,
    701,
    702,
    'SPY',
    'IEI',
    'LQD',
    'Gsg',
    ]
numOfYearsHistory = 10
sectorsNames = setting.sectorsNames  # TODO - manage sectors names (add or remove) and fit them to the portfolio
command = {
    'createNewUser': 1,
    'forcastSpecificStock': 2,
    'updateUserData': 3,
    'plotbb_strategy_stock': 4,
    }

# operations- TODO- operates from the site

selection = command['createNewUser']
if selection == 1:

    # creates new user and build portfolio

    init(sectorsNames, stocksSymbols)
elif selection == 2:

    # forcast specific stock

    manageData.forcastSpecificStock(142, 1, 10)  # 1 if is israeli stock, 0 if usa stock
elif selection == 3:

    # update user data
    # manageData.updateUserData()

    pass
elif selection == 4:

    # plot specific stock

    stock = 'APPL'
    manageData.plotbb_strategy_stock('stock')

######################################################################################
