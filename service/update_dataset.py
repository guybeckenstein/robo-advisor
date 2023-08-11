from service.util import data_management
from service.config import settings

if __name__ == '__main__':

    # update stocks.json file according to the stocks in research_data
    # keep collection 1 as it is
    # TODO

    # TODO -MAKE DAILY RUNNING
    # udpate collections
    data_management.update_all_tables(settings.NUM_OF_YEARS_HISTORY)

