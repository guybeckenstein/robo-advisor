# Add the root directory of your project to the sys.path list
import sys
import os
from impl.user import User
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

from util import research
from config import settings
download_data = False

if __name__ == '__main__':
    if download_data:
        # download 10 years history data for all stocks
        research.download_data_for_research(settings.NUM_OF_YEARS_HISTORY)
        research.save_stocks_intersection_to_csv()  # save stocks intersection to csv file
        research.update_stocks_names_tables()  # update stocks names tables
    all_stats_data_list_of_lists, unified_intersection_data, unified_intersection_data_tuple = research.get_all_best_stocks(settings.RESEARCH_FILTERS)  # filters stocks and save top stocks images
    research.update_collections_file(all_stats_data_list_of_lists, unified_intersection_data, unified_intersection_data_tuple)  # update stocks.json file according to the new stocks intersection
    research.upload_top_stocks_to_google_drive()  # upload top stocks images to google drive