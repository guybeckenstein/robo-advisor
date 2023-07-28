from util import manage_data, settings

if __name__ == '__main__':
    manage_data.main_menu()
    selection = manage_data.selected_menu_option()  # TODO get selection from page in site
    exit_loop_operation = 8

    while selection != exit_loop_operation:
        if selection == 1:  # Basic data from user
            # TODO - get login name instead of terminal
            login_name: str = manage_data.get_name()
            # TODO - get int value from radio button (0\1)
            machine_learning_opt: int = manage_data.get_machine_learning_option()
            # TODO - get int value from radio button (0\1)
            model_option: int = manage_data.get_model_option()
            # TODO: remove and add option to add money (yarden)
            investment_amount: int = 1000  # manage_data.get_investment_amount()

            # Extended data from DB (CSV Tables)
            tables = manage_data.get_extended_data_from_db(settings.STOCKS_SYMBOLS, machine_learning_opt, model_option)
            sectors_data, sectors_list, closing_prices_table, three_best_portfolios, three_best_sectors_weights, \
                pct_change_table, yield_list = tables

            # get data from risk questionnaire form
            # question #1
            string_to_show = "for how many years do you want to invest?\n" + "0-1 - 1\n""1-3 - 2\n""3-100 - 3\n"
            first_question_score = manage_data.get_score_by_answer_from_user(string_to_show)

            # question #2
            string_to_show = "Which distribution do you prefer?\nlow risk - 1, medium risk - 2, high risk - 3 ?\n"
            # display distribution of portfolio graph(matplotlib)
            # TODO - find the way to display the graph (image or data frame)
            manage_data.plot_distribution_of_portfolio(yield_list)
            second_question_score = manage_data.get_score_by_answer_from_user(string_to_show)

            # question #3
            string_to_show = "Which graph do you prefer?\nsafest - 1, sharpest - 2, max return - 3 ?\n"
            # display 3 best portfolios graph (matplotlib)
            # TODO - find the way to display the graph (image or data frame)
            manage_data.plot_three_portfolios_graph(three_best_portfolios, three_best_sectors_weights, sectors_list,
                                                    pct_change_table)
            third_question_score = manage_data.get_score_by_answer_from_user(string_to_show)

            # calculate level of risk by sum of score
            sum_of_score = first_question_score + second_question_score + third_question_score
            level_of_risk = manage_data.get_level_of_risk_by_score(sum_of_score)

            # creates new user with portfolio details
            new_user = manage_data.create_new_user(
                login_name, settings.STOCKS_SYMBOLS, investment_amount, machine_learning_opt, model_option,
                level_of_risk, sectors_data, sectors_list, closing_prices_table, three_best_portfolios, pct_change_table
            )

            # add user to DB(json file)
            new_user.update_json_file("backend_api/DB/users")
            # TODO use sqlite instead of json file or makes conversion from json to sqlite

        elif selection == 2:
            pass

        elif selection == 3:
            # plot user portfolio's data
            name = manage_data.get_name()
            selected_user = manage_data.get_user_from_db(name)
            if selected_user is not None:
                manage_data.plot_user_portfolio(selected_user)

        elif selection == 4:

            manage_data.expert_menu()
            selection = manage_data.selected_menu_option()
            while selection != exit_loop_operation:
                if selection == 1:
                    # forecast specific stock using machine learning
                    stock_name = manage_data.get_name()
                    num_of_years_history = manage_data.get_num_of_years_history()
                    manage_data.forecast_specific_stock(str(stock_name), False, num_of_years_history)

                elif selection == 2:
                    # plotbb_strategy_stock for specific stock
                    stock_name = manage_data.get_name()
                    num_of_years_history = manage_data.get_num_of_years_history()
                    staring_date, today_date = manage_data.get_from_and_to_date(num_of_years_history)
                    manage_data.plotbb_strategy_stock(str(stock_name), staring_date, today_date)

                elif selection == 3:
                    pass

                elif selection == 4:
                    # TODO: in progress
                    # manage_data.scan_good_stocks()
                    manage_data.find_best_stocks()

                elif selection == 5:
                    # plot Markowitz graph
                    num_of_years_history = manage_data.get_num_of_years_history()
                    machine_learning_opt = manage_data.get_machine_learning_option()
                    manage_data.plot_stat_model_graph(settings.STOCKS_SYMBOLS, machine_learning_opt, 'Markowitz')

                elif selection == 6:
                    # plot Gini graph
                    num_of_years_history = manage_data.get_num_of_years_history()
                    machine_learning_opt = manage_data.get_machine_learning_option()
                    manage_data.plot_stat_model_graph(settings.STOCKS_SYMBOLS, machine_learning_opt, 'Gini')

                else:
                    break
                manage_data.expert_menu()
                selection = manage_data.selected_menu_option()
            else:
                break
        manage_data.main_menu()
        selection = manage_data.selected_menu_option()
