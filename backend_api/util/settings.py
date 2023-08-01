############################################
STATIC_GRAPH_FILES_LOCATION = 'static/img/graphs/'
STATIC_USER_FILES_LOCATION = 'static/img/user/'
# setting for stats Models
NUM_POR_SIMULATION = 2000
MIN_NUM_POR_SIMULATION = 30
RECORD_PERCENT_TO_PREDICT = 0.1  # only for Gini model
MODEL_NAME = ['Markowitz', 'Gini']
NUM_OF_YEARS_HISTORY = 10  # the number of history years data
# TODO order min max percent later
LIMIT_PERCENT_MEDIUM_RISK_STOCKS = 0.3
LIMIT_PERCENT_MEDIUM_RISK_COMMODITY = 0.05
LIMIT_PERCENT_LOW_RISK_STOCKS = 0.1
LIMIT_PERCENT_LOW_RISK_COMMODITY = 0
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

# DB
NAME_OF_BUCKET = '1'
BASE_DIR = 'backend_api/'
BUCKET_REPOSITORY = BASE_DIR + 'DB/Bucket' + NAME_OF_BUCKET + '/'
MACHINE_LEARNING_LOCATION = BUCKET_REPOSITORY + 'includingMachineLearning/'     # directory with 6 tables
NON_MACHINE_LEARNING_LOCATION = BUCKET_REPOSITORY + 'withoutMachineLearning/'   # directory with 6 tables
SECTORS_LOCATION = BASE_DIR + 'api/resources/sectors'  # sectors json file
USERS_JSON_NAME = BASE_DIR + 'DB/users'  # where to save the users details
############################################
# setting for api-tase connection and apps:
URL_TOKEN = "https://openapigw.tase.co.il/tase/prod/oauth/oauth2/token"

# key and secret of ofer
# KEY = '7e247414e7047349d83b7b8a427c529c'
# SECRET = '7a809c498662c88a0054b767a92f0399'

# "key and secret of yardet1998 - "robo-advisor"
KEY = 'fb4577af7a020ad0e443632a102003d4'
SECRET = '038af8b13e67a0fd0cbdbb1cdcb973c6'

# key and secret of yarden - "roboadvisor-mta"
# KEY = "9791b4a859c4650efe0d77c2ed9d6919"
# SECRET = "e8d0264a8472c411443a6dfbcdf0992f"

BASE_URL = "https://openapigw.tase.co.il/tase/prod/api/v1"
PREFIX_URL = "/tase/prod/api/v1"
PAYLOAD = "grant_type=client_credentials&scope=tase"
LANGUAGE = "he-IL"


# apps names with date
BASIC_SECURITIES_COMPANY_TRADE = "basic-securities/trade-securities-list"  # not working

# past 10 years
INDEX_END_OF_DAY_HISTORY_TEN_YEARS_UP_TODAY = "indices/eod/history/ten-years/by-index?indexId="
# past 5 years
INDEX_END_OF_DAY_HISTORY_FIVE_YEARS_UP_TODAY = "indices/eod/history/five-years/by-index?indexId="

# other api-tase url's apps:
OTC_TRANSACTION_NAME = "transactions/otc-transactions"
END_OF_DAY_TRANSACTION_NAME = "transactions/transactions-end-of-day"
MAYA_NOTICE_BY_DAY = "maya-reports-online/tase-messages-by-date"
FUND_HISTORY_DATA_NAME = "mutual-fund/history-data"

# apps names without date
BASIC_INDEX_LIST = "basic-indices/indices-list"  # a list of all basic indices in tase-already have-not need to do again
BASIC_SECURITIES_LIST = "basic-securities/securities-types"  # not need to do again
BASIC_SECURITIES_COMPANY_LIST = "basic-securities/companies-list"
SHORT_SALES_WEEKLY_BALANCE_NAME = "short-sales/weekly-balance"
SHORT_SALES_HISTORICAL_DATA = "short-sales/history"
FUND_LIST_NAME = "fund/fund-list"
# INDEX_END_OF_DAY_HISTORY_FIVE_YEARS_SPECIFIC_DATE = PREFIX_URL + "/indices/eod/history/five-years/by-date?date
# = "+str(year)+"-"+str(month)+"-"+str(day)+"&indexId="+str(indexName) """
# past 7 days
# INDEX_END_OF_DAY_HISTORY_SEVEN_DAYS_UP_TODAY = PREFIX_URL +
# "/indices/eod/seven-days/by-index?indexId=" + str(indexName)
# INDEX_END_OF_DAY_HISTORY_SEVEN_DAYS_SPECIFIC_DATE
# PREFIX_URL + "/indices/eod//seven-days/by-date?date
# ="+str(year)+"-"+str(month)+"-"+str(day)+"&indexId="+str(indexName) """
# past day
# INDEX_END_OF_DAY = PREFIX_URL + "/indices/eod/history/ten-years/by-date?date
# ="+str(year)+"-"+str(month)+"-"+str(day)+"&indexId="+str(indexName)"""
