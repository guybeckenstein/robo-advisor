#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd

from RoboAdvisorDataScience.util import manageData, apiUtil
from RoboAdvisorDataScience.util import setting

if __name__ == '__main__':
    manageData.updateAllTables(setting.stocksSymbols, setting.numOfYearsHistory)  # TODO -MAKE DAILY RUNNING
if __name__ == '__main__':
    manageData.mainMenu()
    selection = manageData.selectedMenuOption()
    exitLoopOperation = 8

    while selection != exitLoopOperation:

        if selection == 1:  # TODO MAKES AUTOMACTIC WITH DAILY RUNNING

            name = manageData.getName()  # TODO- NAME OF USER
            # GET BASIC DATA FROM TERMINAL- TODO- get the data from site-form
            investmentAmount, machineLearningOpt, modelOption = manageData.getUserBasicDataFromForm()
            manageData.createsNewUser(name, setting.stocksSymbols, investmentAmount, machineLearningOpt,
                                      modelOption)
        elif selection == 2:
            name = manageData.getName()
            electedUser = manageData.getUserFromDB(name)
            manageData.refreshUserData(selectedUser)


        elif selection == 3:

            # plot user portfolio's data
            name = manageData.getName()
            selectedUser = manageData.getUserFromDB(name)
            if selectedUser is not None:
                manageData.plotUserPortfolio(selectedUser)

        elif selection == 4:

            manageData.expertMenu()
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
                    (staringDate, todayDate) = manageData.getFromAndToDate(numOfYearsHistory)
                    manageData.plotbb_strategy_stock(str(stockName), staringDate, todayDate)

                elif selection == 3:

                    # TODO- IN PROGRESS
                    # add history of index's data from taske(json file)
                    indexid = input("enter index:")
                    manageData.createJsonDataFromTase(indexid, 'History' + str(indexid))

                elif selection == 4:

                    # TODO- IN PROGRESS
                    # manageData.scanGoodStocks()
                    manageData.findBestStocks()

                elif selection == 5:

                    # plot markovich graph
                    numOfYearsHistory = manageData.getNumOfYearsHistory()
                    machineLearningOpt = manageData.getMachineLearningOption()
                    manageData.plotStatModelGraph(setting.stocksSymbols, machineLearningOpt,
                                                  'Markowitz')

                elif selection == 6:

                    # plot gini graph
                    numOfYearsHistory = manageData.getNumOfYearsHistory()
                    machineLearningOpt = manageData.getMachineLearningOption()
                    manageData.plotStatModelGraph(setting.stocksSymbols, machineLearningOpt, 'Gini')

                else:
                    break
                manageData.expertMenu()
                selection = manageData.selectedMenuOption()
            else:

                break
        manageData.mainMenu()
        selection = manageData.selectedMenuOption()
