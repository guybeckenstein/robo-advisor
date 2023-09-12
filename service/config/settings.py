############################################
# Global constants for Stats Models
import os

MODEL_NAME: list[str, str] = ['Markowitz', 'Gini']
MACHINE_LEARNING_MODEL: list[str, str, str, str] = ['Linear Regression', 'ARIMA', 'LSTM', 'Prophet']
LEVEL_OF_RISK_LIST: list[str, str, str] = ["low", "medium", "high"]
CURRENCY_LIST: list[str, str, str] = ['₪', '$', '€']
FILE_ACCESS_TYPE: list[str, str] = ["google_drive", "local"]
FILE_ACCESS_SELECTED: str = FILE_ACCESS_TYPE[1]
GOOGLE_DRIVE_DAILY_DOWNLOAD: bool = False
UPLOAD_TO_GOOGLE_DRIVE: bool = False
NUM_OF_YEARS_HISTORY: int = 10  # the number of history years data - default
BASE_SERVICE_DIR: str = f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/'
DJANGO_DIR: str = BASE_SERVICE_DIR
if 'service/' in DJANGO_DIR:
    DJANGO_DIR = DJANGO_DIR[:(0 - len('service/'))]
CONFIG: str = 'config/'
CONFIG_RESOURCE_LOCATION: str = f'{BASE_SERVICE_DIR}{CONFIG}'
SECTORS_JSON_NAME: str = f'{CONFIG_RESOURCE_LOCATION}sectors'  # sectors.json.json json file
INDICES_LIST_JSON_NAME: str = f'{CONFIG_RESOURCE_LOCATION}indicesList'  # indices list json file
SECURITIES_LIST_JSON_NAME: str = f'{CONFIG_RESOURCE_LOCATION}securitiesList'  # indices list json file
# Datasets
DATASET_LOCATION: str = f'{BASE_SERVICE_DIR}dataset/'
BASIC_STOCK_COLLECTION_REPOSITORY_DIR: str = f'{DATASET_LOCATION}collection'
STOCKS_JSON_NAME: str = f'{DATASET_LOCATION}stocks'  # collection json file
USERS_JSON_NAME: str = f'{DATASET_LOCATION}users'  # where to save the users details

MACHINE_LEARNING_LOCATION: str = 'includingMachineLearning/'  # directory with 6 tables
NON_MACHINE_LEARNING_LOCATION: str = 'withoutMachineLearning/'  # directory with 6 tables
CLOSING_PRICES_FILE_NAME: str = "closing_prices"
PCT_CHANGE_FILE_NAME: str = "pct_change"

# STATIC FILES
GRAPH_IMAGES: str = f'{DJANGO_DIR}static/img/graphs/'
USER_IMAGES: str = f'{DJANGO_DIR}static/img/user/'
RESEARCH_IMAGES: str = f'{DJANGO_DIR}static/img/research/'
RESEARCH_TOP_STOCKS_IMAGES: str = f'{DJANGO_DIR}static/img/research/'

# RESEARCH
RESEARCH_LOCATION: str = f'{BASE_SERVICE_DIR}research/'
RESEARCH_RESULTS_LOCATION: str = f'{RESEARCH_LOCATION}img/'
RESEARCH_RESULTS_TOP_STOCKS: str = f'{RESEARCH_RESULTS_LOCATION}top_stocks/'
RESEARCH_FILTERS: list[int, int, int, int, float, int, float] = [0, 1000000000000, 2, 30, 0.5, 2500, 0.25]
