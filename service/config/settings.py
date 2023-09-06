############################################
# Global constants for Stats Models
import os

MODEL_NAME: list = ['Markowitz', 'Gini']
MACHINE_LEARNING_MODEL: list = ['Linear Regression', 'Arima', 'Lstm', 'Prophet']
LEVEL_OF_RISK_LIST: list = ["low", "medium", "high"]
CURRENCY_LIST = ['₪', '$', '€']
FILE_ACCESS_TYPE = ["google_drive", "local"]
FILE_ACCESS_SELECTED = FILE_ACCESS_TYPE[1]
GOOGLE_DRIVE_DAILY_DOWNLOAD = False
NUM_OF_YEARS_HISTORY = 10  # the number of history years data - default
BASE_SERVICE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
DJANGO_DIR = BASE_SERVICE_DIR
if 'service/' in DJANGO_DIR:
    DJANGO_DIR = DJANGO_DIR[:(0 - len('service/'))]
CONFIG = 'config/'
CONFIG_RESOURCE_LOCATION = BASE_SERVICE_DIR + CONFIG
SECTORS_JSON_NAME = CONFIG_RESOURCE_LOCATION + 'sectors'  # sectors.json.json json file
INDICES_LIST_JSON_NAME = CONFIG_RESOURCE_LOCATION + 'indicesList'  # indices list json file
SECURITIES_LIST_JSON_NAME = CONFIG_RESOURCE_LOCATION + 'securitiesList'  # indices list json file
# Datasets
DATASET_LOCATION = BASE_SERVICE_DIR + 'dataset/'
BASIC_STOCK_COLLECTION_REPOSITORY_DIR = DATASET_LOCATION + 'collection'
STOCKS_JSON_NAME = DATASET_LOCATION + 'stocks'  # collection json file
USERS_JSON_NAME = DATASET_LOCATION + 'users'  # where to save the users details

MACHINE_LEARNING_LOCATION = 'includingMachineLearning/'  # directory with 6 tables
NON_MACHINE_LEARNING_LOCATION = 'withoutMachineLearning/'  # directory with 6 tables
CLOSING_PRICES_FILE_NAME = "closing_prices"
PCT_CHANGE_FILE_NAME = "pct_change"

# STATIC FILES
GRAPH_IMAGES = DJANGO_DIR + 'static/img/graphs/'
USER_IMAGES = DJANGO_DIR + 'static/img/user/'
RESEARCH_IMAGES = DJANGO_DIR + 'static/img/research/'
RESEARCH_TOP_STOCKS_IMAGES = DJANGO_DIR + 'static/img/research/'

# RESEARCH
RESEARCH_LOCATION = BASE_SERVICE_DIR + 'research/'
RESEARCH_RESULTS_LOCATION = RESEARCH_LOCATION + 'img/'
RESEARCH_RESULTS_TOP_STOCKS = RESEARCH_RESULTS_LOCATION + 'top_stocks/'
RESEARCH_FILTERS = [0, 1000000000000, 4, 30, 0.5, 1500, 0.25]
