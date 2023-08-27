from util import research
from config import settings


if __name__ == '__main__':
    # TODO - automatic update of the data every few days
    research.download_data_for_research(settings.NUM_OF_YEARS_HISTORY)
    # TODO - update stocks collection 2,3,4 according to the new data
