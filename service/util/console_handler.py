menu_operations: list[str] = ["build a new portfolio", "add investment", "Plot portfolio's data",
                              "forecast specific stock", "Plot specific stock with bb_strategy",
                              "makes scanner and find stocks with good result", "plot stat model graph(scatter points)",
                              "Exit"]


def get_name() -> str:
    print("enter name")
    name: str = input()
    return name


def get_investment_amount() -> int:
    print("enter amount of money to invest")
    amount: int = int(input())
    while amount < 1:
        print("Enter an amount of money to invest. Must be at least 1")
        amount = int(input())

    return amount


def get_machine_learning_option() -> int:
    print("Interested in using machine learning? 0 - no, 1 - yes")
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


def get_machine_learning_model() -> int:
    print(
        "Choose machine learning model:\n"
        "\t1 - Linear Regression\n"
        "\t2 - ARIMA\n"
        "\t3 - LSTM\n"
        "\t4- Prophet\n"
    )
    model_option = int(input())
    while model_option <= 0 or model_option >= 5:
        print("Please enter an option between 1 to 4")
        model_option = int(input())

    return model_option


def get_num_of_years_history() -> int:
    print("Enter number of years for history")
    num_of_years = int(input())
    while num_of_years < 1:
        print("Enter number of years for history")
        num_of_years = int(input())

    return num_of_years


def get_stat_model_option() -> int:
    print("Choose model: 0 - Markowitz, 1 - Gini\n")
    model_option = int(input())
    while model_option != 0 and model_option != 1:
        print("Please enter 0 or 1")
        model_option = int(input())

    return model_option


def get_collection_number(stocks: list[object]) -> str:
    collection_description: list[str] = ["indexes(recommended)", "top indexes", "indexes and stocks", "top stocks"]
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
            invalid = False
        except ValueError:
            print('Invalid input! Please choose only numbers')
            invalid = True
            collection_number = int(input("Enter the collection number: "))

    while collection_number < 1 or collection_number > len(stocks):
        print("Please enter a valid option.")
        collection_number = int(input("Enter the collection number: "))

    return str(collection_number)


def get_sector_name_from_user(sectors: list) -> int:
    print("Please select one of the following options:")
    for i, option in enumerate(sectors):
        print(f"{i + 1} - {option}")
        print()

    choice: int = int(input("Enter your selection: "))
    while choice < 1 or choice > len(sectors):
        print("Please enter a valid option.")
        choice: int = int(input("Enter your selection: "))

    return choice


def print_collection_table(stocks_symbols_list: list, stocks_description_list: list):
    # table_data = []
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


def get_menu_choice() -> int:
    choice = int(input("Enter your selection: "))
    while choice < 1 or choice > (len(menu_operations) + 1):
        print("Please enter a valid option.")
        choice = int(input("Enter your selection: "))

    return choice


def show_main_menu() -> None:
    print("Welcome to the stock market simulator")
    print("Please select one of the following options:")
    for i, option in enumerate(menu_operations):
        print(f"{i + 1} - {option}\n")
