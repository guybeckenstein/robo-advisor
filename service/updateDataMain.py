from util import manage_data, settings

if __name__ == '__main__':
    #TODO -MAKE DAILY RUNNING
    manage_data.update_all_tables(settings.STOCKS_SYMBOLS, settings.NUM_OF_YEARS_HISTORY)

