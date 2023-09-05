# Add the root directory of your project to the sys.path list
import sys
import os
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

from impl.user import User
from util import data_management, research
from config import settings

if __name__ == '__main__':
    # TODO update stocks.json file according to the stocks in research_data
    # keep collection 1 as it is

    # TODO -MAKE DAILY RUNNING OF THIS SCRIPT
    # daily update  TODO upload to google drive
    data_management.update_all_tables(settings.NUM_OF_YEARS_HISTORY, is_daily_running=True)

    # update from other reasons like chainging the number of years in history or changing the stocks collections
    # data_management.update_all_tables(settings.NUM_OF_YEARS_HISTORY, is_daily_running=False)

    # TODO  update users.json file according according to stocks collection
    # only the stocks weight and stocks symbols
    # call save graphs or save directly or add txt file with last updated images

    # TODO update users graphs in db according to users.json file

    # TODO upload stocks.json to google drive


