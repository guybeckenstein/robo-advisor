# Add the root directory of your project to the sys.path list
import sys
import os
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

from impl.user import User
from util import data_management, research
from config import settings





if __name__ == '__main__':
    data_management.show_main_menu()
    selection = data_management.get_menu_choice()
    exit_loop_operation = 8
    user_id: int = 1
    user_name: int = 'yarden'  # data_management.get_name()
    data_changed = False

    while selection != exit_loop_operation:

        if selection == 1:  # create new user and build portfolio

            # get choices from user
            is_machine_learning, model_option, stocks_collection_number = data_management.get_basic_data_from_user()

            # get stocks symbols from db and create sub folder for user
            stocks_symbols = data_management.get_stocks_symbols_from_collection(stocks_collection_number)
            sub_folder = str(stocks_collection_number) + '/' + str(is_machine_learning) + str(model_option) + "/"

            # Extended data from datasets (CSV Tables)
            tables = data_management.get_extended_data_from_db(
                stocks_symbols, is_machine_learning, model_option, stocks_collection_number,
            )
            sectors_data, sectors, closing_prices_table, three_best_portfolios, three_best_sectors_weights, \
                pct_change_table, yield_list = tables

            # get data according to risk questionnaire form
            level_of_risk: int = (data_management.get_level_of_risk_according_to_questionnaire_form_from_console
                                  (sub_folder, tables, is_machine_learning, model_option, stocks_symbols, stocks_collection_number))

            # creates new user with portfolio details
            portfolio = data_management.create_new_user_portfolio(
                stocks_symbols=stocks_symbols,
                investment_amount=0,
                is_machine_learning=is_machine_learning,
                stat_model_name=settings.MODEL_NAME[model_option],
                risk_level=level_of_risk,
                extended_data_from_db=tables,
            )

            user: User = User(
                _id=user_id, _name=user_name, _portfolio=portfolio, _stocks_collection_number=stocks_collection_number
            )
            try:
                investments_list = data_management.get_user_investments_from_json_file(user_name)
                data_management.changing_portfolio_investments_treatment_console(user.portfolio, investments_list)
            except Exception as e:
                print(e)
            # add user to datasets (json file)
            user.update_json_file(settings.USERS_JSON_NAME)
            data_management.save_user_portfolio(user)

        elif selection == 2:  # add new investment to user

            investment_amount: int = data_management.get_investment_amount()  # get from terminal
            if investment_amount is not None:
                __, investments_list = data_management.add_new_investment(user_name, investment_amount)

                # save investments history
                data_management.plot_investments_history(user_id, investments_list)
                # show result
                data_management.plot_image(f'{settings.USER_IMAGES}{user_id}/investments history.png')

                # save report according to a new investment
                # get stocks weights and stocks symbols from db
                json_data = data_management.get_json_data(settings.USERS_JSON_NAME)
                stocks_weights = json_data['usersList'][user_name][0]['stocksWeights']
                stocks_symbols = json_data['usersList'][user_name][0]['stocksSymbols']
                data_management.view_investment_report(user_id, investment_amount,
                                                       stocks_weights, stocks_symbols)
                # show result
                data_management.plot_image(f'{settings.USER_IMAGES}{user_id}/investment report.png')

        elif selection == 3:  # show user portfolio graphs

            json_data = data_management.get_json_data(settings.USERS_JSON_NAME)
            collection_number = json_data['usersList'][user_name][0]['stocksCollectionNumber']
            if data_management.is_today_date_change_from_last_updated_df(collection_number) or data_changed:
                data_management.get_user_from_db(user_id, user_name)

            # show results
            data_management.plot_image(f'{settings.USER_IMAGES}{user_id}/sectors_weights_graph.png')
            data_management.plot_image(f'{settings.USER_IMAGES}{user_id}/stocks_weights_graph.png')
            data_management.plot_image(f'{settings.USER_IMAGES}{user_id}/estimated_yield_graph.png')

        elif selection == 4:  # forecast specific stock using machine learning

            stock_name = data_management.get_name()
            num_of_years_history = data_management.get_num_of_years_history()
            machine_learning_model = data_management.get_machine_learning_model()
            models_data: dict = data_management.get_models_data_from_collections_file()
            plt_instance = research.forecast_specific_stock(str(stock_name), machine_learning_model,
                                                            models_data, num_of_years_history)
            operation = '_forecast'
            research.save_user_specific_stock(stock_name, operation, plt_instance)

            # show result
            data_management.plot_image(
                settings.RESEARCH_IMAGES + stock_name + operation + '.png'
            )

        elif selection == 5:  # plotbb_strategy_stock for specific stock

            stock_name = data_management.get_name()
            num_of_years_history = data_management.get_num_of_years_history()
            staring_date, today_date = data_management.get_from_and_to_date(num_of_years_history)
            plt_instance = research.plot_bb_strategy_stock(str(stock_name), staring_date, today_date)
            operation = '_bb_strategy'
            research.save_user_specific_stock(stock_name, operation, plt_instance)

            # show result
            data_management.plot_image(
                settings.RESEARCH_IMAGES +
                stock_name + operation + '.png')

        elif selection == 6:  # discover good stocks
            filters = [0, 1000000000000, 4, 30, 0.5, 1500, 0.0]
            sector_name: str = data_management.get_sector_name_from_user()
            intersection = research.get_stocks_stats(sector_name)
            sorted_data_tuple, intersection_with_filters, intersection_without_filters = research.sort_good_stocks(intersection, filters)
            data_management.plot_research_graphs(sorted_data_tuple, intersection_with_filters, sector_name, research.labels)
            prefix_str = 'top_stocks_'

            # show result
            data_management.plot_image(f'{settings.RESEARCH_IMAGES}{prefix_str}{sector_name}.png')

        elif selection == 7:  # plot stat model graph(scatter points)

            # get choices from user
            is_machine_learning, model_option, stocks_collection_number = \
                data_management.get_basic_data_from_user()
            num_of_years_history = settings.NUM_OF_YEARS_HISTORY
            sub_folder = str(stocks_collection_number) + '/' + str(is_machine_learning) + str(model_option) + "/"

            stocks_symbols = data_management.get_stocks_symbols_from_collection(stocks_collection_number)
            closing_prices_table_path = (settings.BASIC_STOCK_COLLECTION_REPOSITORY_DIR
                                         + stocks_collection_number + '/')
            data_management.plot_stat_model_graph(
                stocks_symbols=stocks_symbols, is_machine_learning=is_machine_learning,
                model_name=settings.MODEL_NAME[model_option], num_of_years_history=num_of_years_history,
                closing_prices_table_path=closing_prices_table_path, sub_folder=sub_folder)

            # show result
            data_management.plot_image(
                settings.GRAPH_IMAGES + sub_folder + 'all_options' + '.png')

        elif selection == 9:  # dynamic commands for programmers
            all_data_tuple, intersection = research.get_all_best_stocks(settings.RESEARCH_FILTERS)


        else:
            break

        data_management.show_main_menu()
        selection = data_management.get_menu_choice()
