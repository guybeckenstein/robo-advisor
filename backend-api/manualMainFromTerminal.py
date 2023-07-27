from util import manageData, setting

if __name__ == '__main__':
    manageData.mainMenu()
    selection = manageData.selectedMenuOption() # TODO get selection from page in site
    exitLoopOperation = 8

    while selection != exitLoopOperation:

        if selection == 1:
            # get basic data from user
            loginName: str = manageData.getName()# TODO - get login name instead of terminal
            machineLearningOpt: int = manageData.getMachineLearningOption()# TODO - get int value from radio button (0\1)
            modelOption: int = manageData.getModelOption()# TODO - get int value from radio button (0\1)
            investmentAmount: int = 1000 # manageData.getInvestmentAmount()# TODO REMOVE AND ADD OPTION TO ADD MONEY(YARDEN)

            # get extended data from DB(csv Tables)
            (sectorsData, sectorsList, closingPricesTable, threeBestPortfolios, threeBestSectorsWeights,
             pctChangeTable, yieldList) = (manageData.getExtendedDataFromDB(setting.stocksSymbols, machineLearningOpt, modelOption))

            # get data from risk questionnaire form
            # question 1
            stringToShow = "for how many years do you want to invest?\n" + "0-1 - 1\n""1-3 - 2\n""3-100 - 3\n"
            firstQuestionScore = manageData.getScoreByAnswerFromUser(stringToShow) # TODO - get the value from selected radio button(1\2\3)

            # question 2
            stringToShow = "Which distribution do you prefer?\nlow risk - 1, medium risk - 2, high risk - 3 ?\n"
            # display distribution of portfolio graph(matpolotlib)
            manageData.plotDistributionOfPortfolio(yieldList)# TODO - find the way to display the graph (image or data frame)
            secondQuestionScore = manageData.getScoreByAnswerFromUser(stringToShow) # TODO - get the value from selected radio button(1\2\3)

            # question 3
            stringToShow = "Which graph do you prefer?\nsaftest - 1, sharpest - 2, max return - 3 ?\n"
            # display 3 best portfolios graph (matpolotlib)
            manageData.plotThreePortfoliosGraph(threeBestPortfolios, threeBestSectorsWeights, sectorsList, pctChangeTable)# TODO - find the way to display the graph (image or data frame)
            thirdQuestionScore = manageData.getScoreByAnswerFromUser(stringToShow) # TODO - get the value from selected radio button(1\2\3)

            # calculate level of risk by sum of score
            sumOfScore = firstQuestionScore + secondQuestionScore + thirdQuestionScore
            levelOfRisk = manageData.getLevelOfRiskByScore(sumOfScore)

            #creates new user with portfolio details
            newUser = manageData.createsNewUser(loginName, setting.stocksSymbols,
            investmentAmount, machineLearningOpt, modelOption, levelOfRisk , sectorsData, sectorsList,
            closingPricesTable, threeBestPortfolios, pctChangeTable)

            # add user to DB(json file)
            newUser.updateJsonFile("DB/users")
            # TODO use sqlite instead of json file or makes conversion from json to sqlite


        elif selection == 2:
            pass


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
                    pass

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
