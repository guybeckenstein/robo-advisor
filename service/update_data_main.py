from service.util import data_management, settings

if __name__ == '__main__':
    # TODO -MAKE DAILY RUNNING
    data_management.update_all_tables(settings.NUM_OF_YEARS_HISTORY)

