
def getName() -> str:
    print("enter name")
    name = input()

    return name


def getInvestmentAmount() -> int:
    print("enter amount of money to invest")
    amount = int(input())
    while amount < 1:
        print("enter amount of money to invest")
        amount = int(input())

    return amount


def getMachineLearningOption() -> int:
    print("Interested in using machine learning? 0-no, 1-yes")
    machineLearningOpt = int(input())
    while machineLearningOpt != 0 and machineLearningOpt != 1:
        print("Please enter 0 or 1")
        machineLearningOpt = int(input())

    return machineLearningOpt


def getNumOfYearsHistory() -> int:
    print("enter number of years for history")
    numOfYears = int(input())
    while numOfYears < 1:
        print("enter number of years for history")
        numOfYears = int(input())

    return numOfYears


def getModelOption() -> int:
    print("choose model: 1 - markovich, 2 - gini\n")
    modelOption = int(input())
    while modelOption != 1 and modelOption != 2:
        print("Please enter 1 or 2")
        modelOption = int(input())

    return modelOption


def getScoreByAnswerFromUser(stringToShow) -> int:
    count = 0

    print(stringToShow)
    answer = int(input())
    validAnswerInputForm(answer)
    count += answer

    return count


def validAnswerInputForm(answer) -> None:
    while answer < 1 or answer > 3:
        print("Please enter 1,2 or 3")
        answer = int(input())


def selectedMenuOption() -> int:
    return int(input("Enter your selection: "))  # TODO -GET FROM SITE


def mainMenu() -> None:
    # operations- TODO- operates from the site
    print("Welcome to the stock market simulator")
    print("Please select one of the following options:")
    print("1. FORM - for new user")
    print("2. Refresh user data")
    print("3. Plot user's portfolio data")
    print("4. operations for experts:")
    print("8. Exit")


def expertMenu() -> None:
    print("Please select one of the following options:")
    print("1. Forcast specific stock")
    print("2. Plot specific stock")
    print("3. Add new history of data's index from tase(json)")
    print("4. makes scanner and find stocks with good result")
    print("5. plot markovich graph")
    print("6. plot gini graph")
    print("8. Exit")
