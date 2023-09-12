# Add the root directory of your project to the sys.path list
import sys
import os
from impl.user import User
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)
download_data: bool = False

if __name__ == '__main__':
    from util import research
    from config import settings

    if download_data:
        # download 10 years history data for all stocks
        research.download_data_for_research(settings.NUM_OF_YEARS_HISTORY)
        research.save_stocks_intersection_to_csv()  # save stocks intersection to csv file
        research.update_stocks_names_tables()  # update stocks names tables
    # filters stocks and save top stocks images
    all_stats_data_list_of_lists, unified_table_data, unified_table_data_list = research.get_all_best_stocks()
    # update stocks.json file according to the new stocks intersection
    research.update_collections_file(
        all_stats_data_list_of_lists, unified_table_data,
    )
    # upload top stocks images to google drive
    research.upload_top_stocks_to_google_drive()
