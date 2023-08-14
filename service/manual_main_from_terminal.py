from impl.user import User
from service.util import data_management, research, helpers
from service.config import settings

if __name__ == '__main__':
    data_management.main_menu()
    selection = data_management.selected_menu_option()  # TODO get selection from page in site
    exit_loop_operation = 8
    login_id: int = 1
    login_name: str = 'yarden'  # data_management.get_name()

    while selection != exit_loop_operation:
        if selection == 1:  # Basic data from user
            is_machine_learning: int = 1  # data_management.get_machine_learning_option()
            model_option: int = 1  # data_management.get_model_option()
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
                                                        sectors, pct_change_table, mode='regular',
                                                        sub_folder=sub_folder)
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

            user_portfolio = User(user_id=login_id,
                                  name=login_name,
                                  portfolio=new_portfolio,
                                  stocks_collection_number=stocks_collection_number)
            try:
                investments_list = data_management.get_user_investments_from_json_file(login_name)
                data_management.changing_portfolio_investments_treatment(user_portfolio, investments_list)
            except Exception as e:
                print(e)
            # add user to datasets (json file)
            user_portfolio.update_json_file(settings.USERS_JSON_NAME)
            data_management.save_user_portfolio(user_portfolio) # TODO - separate thread

        elif selection == 2:
            # add new investment to user
            investment_amount: int = data_management.get_investment_amount()  # get from terminal
            if investment_amount is not None:
                data_management.add_new_investment(login_name, investment_amount, db_type="json") # TODO

        elif selection == 3:

            data_management.plot_image(f'{settings.USER_IMAGES}{login_id}/sectors_component.png')
            data_management.plot_image(f'{settings.USER_IMAGES}{login_id}/stocks_component.png')
            data_management.plot_image(f'{settings.USER_IMAGES}{login_id}/yield_graph.png')

        elif selection == 4:

            data_management.expert_menu()
            selection = data_management.selected_menu_option()
            while selection != exit_loop_operation:

                # Forecast specific stock using machine learning
                if selection == 1:  # TODO : add to research page
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

                # plotbb_strategy_stock for specific stock
                elif selection == 2:  # TODO : add to research page
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
                    # research good stocks
                    sector = "US stocks"  # TODO, in site select sector or "ALL"
                    # TODO - SHOW IMAGES




                # plot 3 best portfolios graph
                elif selection == 4:  # TODO : maybe show in the site
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

                elif selection == 5:  # TODO : maybe show in the site
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


                elif selection == 6:
                    # dynamic commands
                    # save tables
                    # helpers.save_usa_indexes_table()
                    # helpers.save_all_stocks()

                    sector = "US stocks"
                    num_of_best_stocks = 100  # how many best stocks to show
                    minCap = 0
                    maxCap = 1000000000000
                    minAnnualReturns = 5
                    maxVolatility = 15
                    minSharpe = 0.6
                    data_tuple = research.find_good_stocks(sector=sector,
                                                           num_of_best_stocks=num_of_best_stocks,
                                                           minCap=minCap,
                                                           maxCap=maxCap,
                                                           minAnnualReturns=minAnnualReturns,
                                                           maxVolatility=maxVolatility, minSharpe=minSharpe
                                                           )

                    # save images TODO
                    path = settings.RESEARCH_RESULTS_TOP_STOCKS
                    data_management.plot_research_graphs(data_tuple)

                    pass

                else:
                    break
                data_management.expert_menu()
                selection = data_management.selected_menu_option()
            else:
                break
        data_management.main_menu()
        selection = data_management.selected_menu_option()
