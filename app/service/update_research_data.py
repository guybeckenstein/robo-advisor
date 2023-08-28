from util import research
from config import settings

if __name__ == '__main__':
    research.download_data_for_research(settings.NUM_OF_YEARS_HISTORY)  # download 10 years history data for all stocks
    research.save_stocks_intersection_to_csv()  # save stocks intersection to csv file
    research.get_all_best_stocks(settings.RESEARCH_FILTERS)  # filters stocks and save top stocks images
    research.update_collections_file()  # update stocks.json file according to the new stocks intersection
