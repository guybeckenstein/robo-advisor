from util import data_management
from service.config import settings

if __name__ == '__main__':
    # TODO filter with volume
    data_management.download_data_for_research(settings.NUM_OF_YEARS_HISTORY)

