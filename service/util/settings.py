############################################
# TODO - order this file and makes multiple settings - class
# setting for stats Models
import os

NUM_POR_SIMULATION = 2000  # number of portfolios to simulate, the more, the better
MIN_NUM_POR_SIMULATION = 30  # min number of final portfolios result
RECORD_PERCENT_TO_PREDICT = 0.05  # the part of years to predict of the total years
TEST_SIZE_MACHINE_LEARNING = 0.1
MODEL_NAME = ['Markowitz', 'Gini']
MACHINE_LEARNING_MODEL = ['LinearRegression', 'ARIMA', 'GradientBoostingRegressor', 'Prophet']
SELECTED_ML_MODEL_FOR_BUILD = MACHINE_LEARNING_MODEL[1]  # TODO
NUM_OF_YEARS_HISTORY = 10  # the number of history years data
# TODO order min max percent later
LIMIT_PERCENT_MEDIUM_RISK_STOCKS = 0.3  # TODO
LIMIT_PERCENT_MEDIUM_RISK_COMMODITY = 0.1
LIMIT_PERCENT_LOW_RISK_STOCKS = 0.15
LIMIT_PERCENT_LOW_RISK_COMMODITY = 0
# TODO , MAYBE ADD LIMIT FOR ALL SECTORS
GINI_V_VALUE = 1.5
#   'TA35.TA',
#     'TA90.TA',
# TODO
STOCKS_SYMBOLS = [
    601,
    602,
    700,
    701,
    702,
    142,
    143,
    'SPY',
    'QQQ',
    '^RUT',
    'IEI',
    'LQD',
    'GSG',
    'GLD',
    'OIL',
]
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
PACKAGE_DIR = BASE_DIR
RESOURCE_LOCATION = PACKAGE_DIR + 'impl/config/'
SECTORS_JSON_NAME = RESOURCE_LOCATION + 'sectors'  # sectors json file
INDICES_LIST_JSON_NAME = RESOURCE_LOCATION + 'indicesList'  # indices list json file
SECURITIES_LIST_JSON_NAME = RESOURCE_LOCATION + 'securitiesList'  # indices list json file
# DB
DB_LOCATION = PACKAGE_DIR + 'DB/'
# NAME_OF_STOCKS_COLLECTION = '1'# TODO
BASIC_STOCK_COLLECTION_REPOSITORY = DB_LOCATION + 'collection'
COLLECTION_JSON_NAME = DB_LOCATION + 'stocks_collections'  # collection json file
MACHINE_LEARNING_LOCATION = 'includingMachineLearning/'  # directory with 6 tables
NON_MACHINE_LEARNING_LOCATION = 'withoutMachineLearning/'  # directory with 6 tables
USERS_JSON_NAME = DB_LOCATION + 'users'  # where to save the users details
CLOSING_PRICES_FILE_NAME = "closing_prices"

# STATIC FILES
GRAPH_IMAGES = BASE_DIR + '../static/img/graphs/'
USER_IMAGES = BASE_DIR + '../static/img/user/'

# RESEARCH
RESEARCH_LOCATION = PACKAGE_DIR + 'research/'
RESEARCH_RESULTS_LOCATION = BASE_DIR + '../static/img/research/'
GROUP_OF_STOCKS = [
    "usa_stocks", "usa_bonds", "israel_indexes", "nasdaq", "sp500", "dowjones", "TA35", "TA90", "TA125", "TA125", "all"
]