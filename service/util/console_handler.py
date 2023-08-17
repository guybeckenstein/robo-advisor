from typing import List


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
    is_machine_learning: int = None
    invalid: bool = True
    while invalid:
        try:
            is_machine_learning = int(input())
            invalid = False
        except ValueError:
            print('You can only choose numbers! Try again')
            invalid = True
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
    print("choose model: 0 - Markowitz, 1 - Gini\n")
    model_option = int(input())
    while model_option != 0 and model_option != 1:
        print("Please enter 0 or 1")
        model_option = int(input())

    return model_option


def get_collection_number(stocks: List) -> str: # TODO -display in site
    collection_description = ["indexes(recommended)", "top indexes", "indexes and stocks",
                          "top stocks"]
    print("Choose a collection of stocks:")
    for i, collection in enumerate(stocks):
        print(f"{i + 1} - {collection_description[i]}")
        print_collection_table(stocks[collection][0],
                               stocks[collection][1])
        print()

    collection_number: int = None
    invalid: bool = True
    while invalid:
        try:
            collection_number = int(input("Enter the collection number: "))
        except ValueError:
            print('Invalid input! Please choose only numbers')
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
    print("1. build a new portfolio")
    print("2. add investment")
    print("3. Plot portfolio's data")
    print("4. operations for experts:")
    print("8. Exit")


def expert_menu() -> None:
    print("Please select one of the following options:")
    print("1. Forecast specific stock")
    print("2. Plot specific stock with bb_strategy")
    print("3. makes scanner and find stocks with good result")
    print("4. plot Markowitz graph")
    print("5. plot Gini graph")
    print("8. Exit")
