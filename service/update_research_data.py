# Add the root directory of your project to the sys.path list
import sys
import os
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

from util import research
from config import settings

if __name__ == '__main__':
    research.download_data_for_research(settings.NUM_OF_YEARS_HISTORY)  # download 10 years history data for all stocks
    # research.update_volume_and_cap()  # get stocks intersection
    research.save_stocks_intersection_to_csv()  # save stocks intersection to csv file
    research.get_all_best_stocks(settings.RESEARCH_FILTERS)  # filters stocks and save top stocks images
    research.update_stocks_names_tables()  # update stocks names tables
    research.update_collections_file()  # update stocks.json file according to the new stocks intersection
