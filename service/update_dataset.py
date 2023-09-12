# Add the root directory of your project to the sys.path list
import sys
import os

from service.config import settings

project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)
# flake8 --disable-noqa
from impl.user import User
from util import data_management
upload_to_google_drive: bool = False
if __name__ == '__main__':
    if settings.UPLOAD_TO_GOOGLE_DRIVE:
        data_management.upload_file_to_google_drive(file_path=f'{settings.USERS_JSON_NAME}.json', num_of_elements=2)
    data_management.update_all_tables(is_daily_running=True)
