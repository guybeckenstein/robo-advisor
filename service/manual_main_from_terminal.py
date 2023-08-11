from impl.user import User
from service.util import data_management, research, helpers
from service.config import settings

if __name__ == '__main__':
    data_management.main_menu()
    selection = data_management.selected_menu_option()  # TODO get selection from page in site
    exit_loop_operation = 8
    login_name: str = 'yarden'  # data_management.get_name()

    while selection != exit_loop_operation:
        if selection == 1:  # Basic data from user
            is_machine_learning: int = 0  # data_management.get_machine_learning_option()
            model_option: int = 0  # data_management.get_model_option()
            investment_amount: int = 1000  # data_management.get_investment_amount() # TODO
            stocks_collection_number = 1  # default
            # TODO in site(GUY)
            # 1 default collection, Option for the customer to choose (recommended)
            stocks_collection_number: str = data_management.get_collection_number()  # TODO: user investor
            stocks_symbols = data_management.get_stocks_symbols_from_collection(stocks_collection_number)
            sub_folder = str(stocks_collection_number) + '/' + str(is_machine_learning) + str(model_option) + "/"
            # Extended data from datasets (CSV Tables)
            tables = data_management.get_extended_data_from_db(
                stocks_symbols, is_machine_learning, model_option, stocks_collection_number,
                mode='regular'
            )
            sectors_data, sectors, closing_prices_table, three_best_portfolios, three_best_sectors_weights, \
                pct_change_table, yield_list = tables

            # get data from risk questionnaire form
            # question #1
            string_to_show = "for how many years do you want to invest?\n" + "0-1 - 1\n""1-3 - 2\n""3-100 - 3\n"
            first_question_score = data_management.get_score_by_answer_from_user(string_to_show)

            # question #2
            string_to_show = "Which distribution do you prefer?\nlow risk - 1, medium risk - 2, high risk - 3 ?\n"
            # display distribution of portfolio graph(matplotlib)
            data_management.plot_distribution_of_portfolio(yield_list, mode='regular', sub_folder=sub_folder)
            data_management.plot_image(settings.GRAPH_IMAGES + sub_folder + 'distribution_graph.png')
            second_question_score = data_management.get_score_by_answer_from_user(string_to_show)

            # question #3
            string_to_show = "Which graph do you prefer?\nsafest - 1, sharpest - 2, max return - 3 ?\n"
            # display 3 best portfolios graph (matplotlib)
            data_management.plot_three_portfolios_graph(three_best_portfolios, three_best_sectors_weights,
                                                    sectors, pct_change_table, mode='regular', sub_folder=sub_folder)
            # data_management.plot_functions.plot(plt_instance)
            data_management.plot_image(settings.GRAPH_IMAGES + sub_folder + 'three_portfolios.png')
            third_question_score = data_management.get_score_by_answer_from_user(string_to_show)

            # calculate level of risk by sum of score
            sum_of_score = first_question_score + second_question_score + third_question_score
            level_of_risk = data_management.get_level_of_risk_by_score(sum_of_score)

            # creates new user with portfolio details
            new_portfolio = data_management.create_new_user_portfolio(
                stocks_symbols=stocks_symbols,
                investment_amount=investment_amount,
                is_machine_learning=is_machine_learning,
                model_option=model_option,
                risk_level=level_of_risk,
                extended_data_from_db=tables,
            )

            user_portfolio = User(login_name, new_portfolio, stocks_collection_number)
            # add user to datasets (json file)
            user_portfolio.update_json_file(settings.USERS_JSON_NAME)

        elif selection == 2:
            helpers.collect_all_stocks()
            print("yarden")

        elif selection == 3:
            # plot user portfolio's data
            selected_user = data_management.get_user_from_db(login_name)
            if selected_user is not None:
                data_management.save_user_portfolio(selected_user)
                data_management.plot_image(settings.USER_IMAGES + login_name + '/sectors_component.png')
                data_management.plot_image(settings.USER_IMAGES + login_name + '/stocks_component.png')
                data_management.plot_image(settings.USER_IMAGES + login_name + '/yield_graph.png')

        elif selection == 4:

            data_management.expert_menu()
            selection = data_management.selected_menu_option()
            while selection != exit_loop_operation:
                if selection == 1: # TODO : add to research page
                    # forecast specific stock using machine learning
                    stock_name = data_management.get_name()
                    num_of_years_history = data_management.get_num_of_years_history()
                    machine_learning_model = data_management.get_machine_learning_model()
                    models_data = data_management.get_models_data_from_collections_file()
                    plt_instance = research.forecast_specific_stock(str(stock_name), machine_learning_model,
                                                                       models_data, num_of_years_history)
                    operation = '_forecast'
                    research.save_user_specific_stock(stock_name, operation, plt_instance)
                    data_management.plot_image(settings.RESEARCH_RESULTS_LOCATION
                                               + stock_name + operation + '.png')
                elif selection == 2: # TODO : add to research page
                    # plotbb_strategy_stock for specific stock
                    stock_name = data_management.get_name()
                    num_of_years_history = data_management.get_num_of_years_history()
                    staring_date, today_date = data_management.get_from_and_to_date(num_of_years_history)
                    plt_instance = research.plotbb_strategy_stock(str(stock_name), staring_date, today_date)
                    operation = '_bb_strategy'
                    research.save_user_specific_stock(stock_name, operation, plt_instance)
                    data_management.plot_image(
                        settings.RESEARCH_RESULTS_LOCATION +
                        stock_name + operation + '.png')
                elif selection == 3:
                    pass

                elif selection == 4:
                    # TODO - get group of stocks
                    selected_option = data_management.get_group_of_stocks_option()
                    models_data = data_management.get_models_data_from_collections_file()
                    (max_returns_stocks_list,
                     min_volatility_stocks_list,
                     max_sharpest_stocks_list) = research.find_good_stocks(
                        settings.GROUP_OF_STOCKS[selected_option - 1])
                    # TODO - SHOW IMAGES

                # plot 3 best portfolios graph
                elif selection == 5: # TODO : maybe show in the site
                    # plot Markowitz graph
                    num_of_years_history = data_management.get_num_of_years_history()
                    is_machine_learning = data_management.get_machine_learning_option()
                    stocks_collection_number: str = data_management.get_collection_number()
                    stocks_symbols = data_management.get_stocks_symbols_from_collection(stocks_collection_number)
                    models_data = data_management.get_models_data_from_collections_file()
                    closing_prices_table_path = (settings.BASIC_STOCK_COLLECTION_REPOSITORY_DIR
                                                 + stocks_collection_number + '/')
                    data_management.plot_stat_model_graph(stocks_symbols, is_machine_learning,
                                                          settings.MODEL_NAME[0], num_of_years_history,
                                                          models_data, closing_prices_table_path)
                    data_management.plot_image(
                        settings.GRAPH_IMAGES + settings.MODEL_NAME[0] + '_all_option' + '.png')

                elif selection == 6:  # TODO : maybe show in the site
                    # plot Gini graph
                    num_of_years_history = data_management.get_num_of_years_history()
                    is_machine_learning = data_management.get_machine_learning_option()
                    stocks_collection_number: str = data_management.get_collection_number()  # 1 default, Option for the customer to choose TODO -user investor
                    stocks_symbols = data_management.get_stocks_symbols_from_collection(stocks_collection_number)
                    models_data = data_management.get_models_data_from_collections_file()
                    closing_prices_table_path = (settings.BASIC_STOCK_COLLECTION_REPOSITORY_DIR
                                                 + stocks_collection_number + '/')
                    data_management.plot_stat_model_graph(stocks_symbols, is_machine_learning,
                                                          settings.MODEL_NAME[1], num_of_years_history,
                                                          models_data, closing_prices_table_path
                                                          )
                    data_management.plot_image(
                        settings.GRAPH_IMAGES + settings.MODEL_NAME[1] + '_all_option' + '.png')

                else:
                    break
                data_management.expert_menu()
                selection = data_management.selected_menu_option()
            else:
                break
        data_management.main_menu()
        selection = data_management.selected_menu_option()
