############################################
# setting for stats Models
num_por_simulation = 2000
min_num_por_simulation = 30
record_percentage_to_predict = 0.1  # only for Gini model
model_name = ['Markowitz', 'Gini']
num_of_years_history = 10  # the number of history years data
# TODO order min max percent later
limit_percent_medium_risk_stocks = 0.3
limit_percent_medium_risk_commodity = 0.05
limit_percen_low_risk_stocks = 0.1
limit_percen_low_risk_commodity = 0
stocks_symbols = [
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
name_of_bucket = '1'
base_dir = 'backend_api/'
bucket_repository = 'DB/Bucket' + name_of_bucket + '/'  # directory of the bucket(12 tables + 2 closing prices)
machine_learning_location = bucket_repository + 'includingMachineLearning/'  # directory with 6 tables
non_machine_learning_location = bucket_repository + 'withoutMachineLearning/' # directory with 6 tables
sectors_location = 'api/resources/sectors'  # sectors json file
users_json_name ='DB/users'  # where to save the users details
############################################
# setting for api-tase connection and apps:
token_url = "https://openapigw.tase.co.il/tase/prod/oauth/oauth2/token"

# key and secret of ofer
# key = '7e247414e7047349d83b7b8a427c529c'
# secret = '7a809c498662c88a0054b767a92f0399'

# "key and secret of yardet1998 - "robo-advisor"
key = 'fb4577af7a020ad0e443632a102003d4'
secret = '038af8b13e67a0fd0cbdbb1cdcb973c6'

# key and secret of yarden - "roboadvisor-mta"
# key = "9791b4a859c4650efe0d77c2ed9d6919"
# secret = "e8d0264a8472c411443a6dfbcdf0992f"

base_url = "https://openapigw.tase.co.il/tase/prod/api/v1"
prefix_url = "/tase/prod/api/v1"
payload = "grant_type=client_credentials&scope=tase"
language = "he-IL"


# apps names with date
basic_securities_company_trade = "basic-securities/trade-securities-list"  # not working

# past 10 years
index_end_of_day_history_ten_years_up_today = "indices/eod/history/ten-years/by-index?indexId="
# past 5 years
index_end_of_day_history_five_years_up_today = "indices/eod/history/five-years/by-index?indexId="

# other api-tase url's apps:
otc_transaction_name = "transactions/otc-transactions"
end_of_day_transaction_name = "transactions/transactions-end-of-day"
maya_notice_by_day = "maya-reports-online/tase-messages-by-date"
fund_history_data_name = "mutual-fund/history-data"

# apps names without date
basic_index_list = "basic-indices/indices-list"  # a list of all basic indices in tase-already have-not need to do again
basic_securities_list = "basic-securities/securities-types"  # not need to do again
basic_securities_compnay_list = "basic-securities/companies-list"
short_sales_weekly_balance_name = "short-sales/weekly-balance"
short_sales_historical_data = "short-sales/history"
fund_list_name = "fund/fund-list"
# indexEndOfDayHistoryFiveYearsSpecificDate
# =prefixUrl+"/indices/eod/history/five-years/by-date?date
# ="+str(year)+"-"+str(month)+"-"+str(day)+"&indexId="+str(indexName) """
# past 7 days
# indexEndOfDayHistorySevenDaysUpToday = (prefixUrl + "/indices/eod/seven-days/by-index?indexId=" + str(indexName))
# indexEndOfDayHistorySevenDaySpecificDate
# prefixUrl+"/indices/eod//seven-days/by-date?date
# ="+str(year)+"-"+str(month)+"-"+str(day)+"&indexId="+str(indexName) """
# past day
# indexEndOfDay=prefixUrl+"/indices/eod/history/ten-years/by-date?date
# ="+str(year)+"-"+str(month)+"-"+str(day)+"&indexId="+str(indexName) """
