############################################
# Global constants for Stats Models
import os

# Models & Algorithms
MODEL_NAME: list[str, str] = ['Markowitz', 'Gini']
MACHINE_LEARNING_MODEL: list[str, str, str, str] = ['Linear Regression', 'ARIMA', 'LSTM', 'Prophet']
LEVEL_OF_RISK_LIST: list[str, str, str] = ["low", "medium", "high"]
CURRENCY_LIST: dict[str, str, str] = {
    'SHEKEL': '₪',
    'DOLLAR': '$',
    'EURO': '€'
}
# Google Drive
FILE_ACCESS_TYPE: dict[str, str] = {"google_drive": "google_drive", "local": "local"}
FILE_ACCESS_SELECTED: str = "local"
GOOGLE_DRIVE_DAILY_DOWNLOAD: bool = False   # Guy: It is always False in the current code!
UPLOAD_TO_GOOGLE_DRIVE: bool = False        # Guy: It is always False in the current code!

NUM_OF_YEARS_HISTORY: int = 10  # the number of history years data - default
# Directories & Files
BASE_SERVICE_DIR: str = f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/'
DJANGO_DIR: str = BASE_SERVICE_DIR
if 'service/' in DJANGO_DIR:
    DJANGO_DIR = DJANGO_DIR[:(0 - len('service/'))]
CONFIG: str = 'config/'
CONFIG_RESOURCE_LOCATION: str = f'{BASE_SERVICE_DIR}{CONFIG}'
SECTORS_JSON_NAME: str = f'{CONFIG_RESOURCE_LOCATION}sectors'  # sectors.json
INDICES_LIST_JSON_NAME: str = f'{CONFIG_RESOURCE_LOCATION}indicesList'  # indicesList.json
SECURITIES_LIST_JSON_NAME: str = f'{CONFIG_RESOURCE_LOCATION}securitiesList'  # securitiesList.json

# Datasets
DATASET_LOCATION: str = f'{BASE_SERVICE_DIR}dataset/'
BASIC_STOCK_COLLECTION_REPOSITORY_DIR: str = f'{DATASET_LOCATION}collection'
STOCKS_JSON_NAME: str = f'{DATASET_LOCATION}stocks'  # collection json file
USERS_JSON_NAME: str = f'{DATASET_LOCATION}users'  # where to save the users details

MACHINE_LEARNING_LOCATION: str = 'includingMachineLearning/'  # directory with 6 tables
NON_MACHINE_LEARNING_LOCATION: str = 'withoutMachineLearning/'  # directory with 6 tables
CLOSING_PRICES_FILE_NAME: str = "closing_prices"
PCT_CHANGE_FILE_NAME: str = "pct_change"

# Static files
GRAPH_IMAGES: str = f'{DJANGO_DIR}static/img/graphs/'
USER_IMAGES: str = f'{DJANGO_DIR}static/img/user/'
RESEARCH_IMAGES: str = f'{DJANGO_DIR}static/img/research/'
RESEARCH_TOP_STOCKS_IMAGES: str = f'{DJANGO_DIR}static/img/research/'

# RESEARCH
RESEARCH_LOCATION: str = f'{BASE_SERVICE_DIR}research/'
RESEARCH_RESULTS_LOCATION: str = f'{RESEARCH_LOCATION}img/'
RESEARCH_RESULTS_TOP_STOCKS: str = f'{RESEARCH_RESULTS_LOCATION}top_stocks/'
RESEARCH_FILTERS: list[int, int, int, int, float, int, float] = [0, 1000000000000, 2, 30, 0.5, 2500, 0.25]
