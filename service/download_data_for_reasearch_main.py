from util import manage_data, settings

if __name__ == '__main__':
    # TODO filter with volume
    manage_data.download_data_for_research(settings.NUM_OF_YEARS_HISTORY)

