#!/usr/bin/python
# -*- coding: utf-8 -*-
from util import manageData
from api import User as User
from api import StatsModels as StatsModels
import os

######################################################################################
clear = lambda: os.system('cls')
#israel bonds stocks: ( not updated because of limit requests in tase
"""
    601,
    602,
    700,
    701,
    702,
"""
stocksSymbols = [  # TODO -NEW FEATURES- creates features to scan good stocks and indexes and fit them to the portfolio
    "TA35.TA",
    "TA90.TA",
    'SPY',
    'QQQ',
    '^RUT',
    'IEI',
    'LQD',
    'Gsg',
    'GLD',
    'OIL'
    ]

numOfYearsHistory = 10

######################################################################################

if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=8000, debug=True)
    manageData.mainMenu()
    selection = manageData.selectedMenuOption()
    exitLoopOperation = 8

    while selection != exitLoopOperation:

        if selection == 1:
            name = manageData.getName()
            user = manageData.createsNewUser(name, stocksSymbols, numOfYearsHistory) # plot results
            manageData.plotUserPortfolio(user)

        elif selection == 2:# TODO GET USER FROM LIST IN DJANGO PROJECT
            # update exist user data using model markovich or gini
            name = manageData.getName()
            selectedUser = manageData.getUserFromDB(name)
            manageData.refreshUserData(selectedUser)# TODO - FIX

        elif selection == 3:# TODO GET USER FROM LIST IN DJANGO PROJECT
            # plot user portfolio's data
            name = manageData.getName()
            selectedUser = manageData.getUserFromDB(name)
            if selectedUser is not None:
                manageData.plotUserPortfolio(selectedUser)

        elif selection == 4:
            manageData.expertMenu()
            # clear screen - TODO
            selection = manageData.selectedMenuOption()
            while selection != exitLoopOperation:
                if selection == 1:
                    # forcast specific stock using machine learning
                    stockName = manageData.getName()
                    numOfYearsHistory = manageData.getNumOfYearsHistory()
                    manageData.forcastSpecificStock(str(stockName), 0, numOfYearsHistory)

                elif selection == 2:
                    # plotbb_strategy_stock for specific stock
                    stockName = manageData.getName()
                    numOfYearsHistory = manageData.getNumOfYearsHistory()
                    staringDate, todayDate = manageData.getfromAndToDate(numOfYearsHistory)
                    manageData.plotbb_strategy_stock(str(stockName), staringDate, todayDate)

                elif selection == 3:
                    # TODO- IN PROGRESS
                    # add history of index's data from taske(json file)
                    indexid = 143
                    manageData.createJsonDataFromTase(143, "History" + str(indexid))

                elif selection == 4:
                    # TODO- IN PROGRESS
                    # manageData.scanGoodStocks()
                    manageData.findBestStocks()

                elif selection == 5:
                    # plot markovich graph
                    numOfYearsHistory = manageData.getNumOfYearsHistory()
                    machineLearningOpt = manageData.getMachineLearningOption()
                    manageData.plotStatModelGraph(stocksSymbols, numOfYearsHistory, machineLearningOpt, "Markowitz")

                elif selection == 6:
                    # plot gini graph
                    numOfYearsHistory = manageData.getNumOfYearsHistory()
                    machineLearningOpt = manageData.getMachineLearningOption()
                    manageData.plotStatModelGraph(stocksSymbols, numOfYearsHistory, machineLearningOpt, "Gini")

                else:
                    break
                manageData.expertMenu()
                selection = manageData.selectedMenuOption()

            else:
                break
        manageData.mainMenu()
        selection = manageData.selectedMenuOption()

######################################################################################
