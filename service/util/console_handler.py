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
