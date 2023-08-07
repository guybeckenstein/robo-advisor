############################################
# TODO - order this file and makes multiple settings - class
# setting for stats Models
import os

NUM_POR_SIMULATION = 2000 # number of portfolios to simulate, the more, the better
MIN_NUM_POR_SIMULATION = 30 # min number of final portfolios result
RECORD_PERCENT_TO_PREDICT = 0.05 # the part of years to predict of the total years
TEST_SIZE_MACHINE_LEARNING = 0.1
MODEL_NAME = ['Markowitz', 'Gini']
MACHINE_LEARNING_MODEL = ['LinearRegression', 'ARIMA', 'GradientBoostingRegressor', 'Prophet']
SELECTED_ML_MODEL_FOR_BUILD = MACHINE_LEARNING_MODEL[1]
NUM_OF_YEARS_HISTORY = 10  # the number of history years data
# TODO order min max percent later
LIMIT_PERCENT_MEDIUM_RISK_STOCKS = 0.3
LIMIT_PERCENT_MEDIUM_RISK_COMMODITY = 0.1
LIMIT_PERCENT_LOW_RISK_STOCKS = 0.15
LIMIT_PERCENT_LOW_RISK_COMMODITY = 0
gini_v_value = 4
STOCKS_SYMBOLS = [
    601,
    602,
    700,
    701,
    702,
    'TA35.TA',
    'TA90.TA',
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
PACKAGE_DIR = BASE_DIR# + 'service/'
RESOURCE_LOCATION = PACKAGE_DIR + 'api/resources/'
SECTORS_JSON_NAME = RESOURCE_LOCATION + 'sectors'  # sectors json file
INDICES_LIST_JSON_NAME = RESOURCE_LOCATION + 'indicesList'  # indices list json file
# DB
DB_LOCATION = PACKAGE_DIR + 'DB/'
NAME_OF_BUCKET = '1'
BUCKET_REPOSITORY = DB_LOCATION + 'Bucket' + NAME_OF_BUCKET + '/'  # directory of the bucket (12 tables + 2 closing prices)
MACHINE_LEARNING_LOCATION = BUCKET_REPOSITORY + 'includingMachineLearning/'  # directory with 6 tables
NON_MACHINE_LEARNING_LOCATION = BUCKET_REPOSITORY + 'withoutMachineLearning/' # directory with 6 tables
USERS_JSON_NAME = DB_LOCATION + 'users'  # where to save the users details
CLOSING_PRICES_FILE_NAME = "closing_prices"

# STATIC FILES
STATIC_IMAGES = BASE_DIR + '../static/img/graphs/'
USER_IMAGES = STATIC_IMAGES + '../user/'


# RESEARCH
RESEARCH_LOCATION = PACKAGE_DIR + 'research/'
GROUP_OF_STOCKS = [
    "usa_stocks", "usa_bonds", "israel_indexes", "nasdaq", "sp500", "dowjones", "TA35", "TA90", "TA125", "TA125", "all"
]