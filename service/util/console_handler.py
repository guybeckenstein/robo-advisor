from typing import List

from pip._internal.utils.misc import tabulate


def get_name() -> str:
    print("enter name")
    name = input()

    return name


def get_investment_amount() -> int:
    print("enter amount of money to invest")
    amount = int(input())
    while amount < 1:
        print("enter amount of money to invest")
        amount = int(input())

    return amount


def get_machine_learning_option() -> int:
    print("Interested in using machine learning? 0-no, 1-yes")
    is_machine_learning = int(input())
    while is_machine_learning != 0 and is_machine_learning != 1:
        print("Please enter 0 or 1")
        is_machine_learning = int(input())

    return is_machine_learning


def get_machine_learning_mdoel() -> int:
    print("choose machine learning model:\n 1 - LinearRegression\n"
          " 2 - ARIMA\n 3 - GradientBoostingRegressor\n 4- Prophet\n")
    model_option = int(input())
    while model_option <= 0 or model_option >= 5:
        print("Please enter option between 1 to 4")
        model_option = int(input())

    return model_option


def get_num_of_years_history() -> int:
    print("enter number of years for history")
    num_of_years = int(input())
    while num_of_years < 1:
        print("enter number of years for history")
        num_of_years = int(input())

    return num_of_years


def get_model_option() -> int:
    print("choose model: 1 - Markowitz, 2 - Gini\n")
    model_option = int(input())
    while model_option != 1 and model_option != 2:
        print("Please enter 1 or 2")
        model_option = int(input())

    return model_option


def get_group_of_stocks_option() -> int:
    print("choose group of stocks for research:\n 1 - usa stocks\n"
          " 2 - usa bonds\n 3 - israel indexes\n 4 - nasdaq stocks\n"
          " 5 - s&p500 stocks\n 6 - dowjones stocks\n 7- tel aviv 35 stocks\n"
          " 8 - tel aviv 90 stocks\n 9 - tel aviv 125 stocks\n 10- all stocks\n")
    model_option = int(input())
    while model_option <= 0 or model_option >= 11:
        print("Please enter option between 1 to 10")
        model_option = int(input())

    return model_option


def get_collection_number(stocks: List) -> str: # TODO -display in site
    collection_description = ["indexes(recommended)", "mostly indexes and stocks", "yarden portfolio",
                          "Combined from everything"]
    print("Choose a collection of stocks:")
    for i, collection in enumerate(stocks):
        print(f"{i + 1} - {collection_description[i]}")
        print_collection_table(stocks[collection][0],
                               stocks[collection][1])
        print()
    collection_number = int(input("Enter the collection number: "))

    while collection_number < 1 or collection_number > len(stocks):
        print("Please enter a valid option.")
        collection_number = int(input("Enter the collection number: "))

    return str(collection_number)


def print_collection_table(stocks_symbols_list: List, stocks_description_list: List):
    table_data = []
    for symbol, description in zip(stocks_symbols_list, stocks_description_list):
        print(f"{symbol} - {description}")

def get_score_by_answer_from_user(string_to_show) -> int:
    count = 0

    print(string_to_show)
    answer = int(input())
    valid_answer_input_form(answer)
    count += answer

    return count


def valid_answer_input_form(answer) -> None:
    while answer < 1 or answer > 3:
        print("Please enter 1, 2 or 3")
        answer = int(input())


def selected_menu_option() -> int:
    return int(input("Enter your selection: "))  # TODO -GET FROM SITE


def main_menu() -> None:
    # operations- TODO- operates from the site
    print("Welcome to the stock market simulator")
    print("Please select one of the following options:")
    print("1. FORM - for new user")
    print("2. Refresh user data")
    print("3. Plot user's portfolio data")
    print("4. operations for experts:")
    print("8. Exit")


def expert_menu() -> None:
    print("Please select one of the following options:")
    print("1. Forecast specific stock")
    print("2. Plot specific stock")
    print("3. Add new history of data's index from tase(json)")
    print("4. makes scanner and find stocks with good result")
    print("5. plot Markowitz graph")
    print("6. plot Gini graph")
    print("8. Exit")
