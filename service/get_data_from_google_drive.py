# Add the root directory of your project to the sys.path list
import sys
import os

project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)
# flake8 --disable-noqa
from impl.user import User

if __name__ == '__main__':
    from util import data_management

    data_management.update_files_from_google_drive()
