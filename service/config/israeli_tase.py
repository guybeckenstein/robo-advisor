# "key and secret of yardet1998 - "robo-advisor"
KEY = 'fb4577af7a020ad0e443632a102003d4'
SECRET = '038af8b13e67a0fd0cbdbb1cdcb973c6'

# key and secret of yarden - "roboadvisor-mta"
# key = "9791b4a859c4650efe0d77c2ed9d6919"
# secret = "e8d0264a8472c411443a6dfbcdf0992f"

# key and secret of ofer
# key = '7e247414e7047349d83b7b8a427c529c'
# secret = '7a809c498662c88a0054b767a92f0399'

TOKEN_URL = "https://openapigw.tase.co.il/tase/prod/oauth/oauth2/token"

BASE_URL = "https://openapigw.tase.co.il/tase/prod/api/v1"
PREFIX_URL = "/tase/prod/api/v1"
PAYLOAD = "grant_type=client_credentials&scope=tase"
LANGUAGE = "he-IL"


# INDEXES:
INDEX_EOD_HISTORY_TEN_YEARS = "indices/eod/history/ten-years/by-index?indexId="  # past 10 years
INDEX_END_OF_DAY_HISTORY_FIVE_YEARS_UP_TO_TODAY = "indices/eod/history/five-years/by-index?id="  # past 5 years

# SECURITIES:
SECURITY_END_OF_DAY_HISTORY_TEN_YEARS = "securities/trading/eod/history/ten-years/by-security?securityId="

# other impl-tase url's apps:
OTC_TRANSACTION_NAME = "transactions/otc-transactions"
END_OF_DAY_TRANSACTION_NAME = "transactions/transactions-end-of-day"
MAYA_NOTICE_BY_DAY = "maya-reports-online/tase-messages-by-date"
FUND_HISTORY_DATA_NAME = "mutual-fund/history-data"

# apps names without date
BASIC_INDEX_LIST = "basic-indices/indices-list"  # a list of all basic indices in tase-already have-not need to do again
BASIC_SECURITIES_LIST = "basic-securities/securitiesTypes"  # not need to do again
BASIC_DELISTED_SECURITIES_LIST = "basic-securities/delisted-securities-list/2023/1"
BASIC_SECURITIES_LIST_BY_TYPE = "basic-securities/trade-securities-list/2023/08/08"
BASIC_SECURITIES_COMPANY_LIST = "basic-securities/companies-list"
SHORT_SALES_WEEKLY_BALANCE_NAME = "short-sales/weekly-balance"
SHORT_SALES_HISTORICAL_DATA = "short-sales/history"
FUND_LIST_NAME = "fund/fund-list"
