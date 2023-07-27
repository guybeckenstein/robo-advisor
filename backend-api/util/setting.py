############################################
# setting for stats Models
Num_porSimulation = 2000
min_Num_porSimulation = 30
record_percentage_to_predict = 0.1  # only for Gini model
modelName = ['Markowitz', 'Gini']
numOfYearsHistory = 10  # the number of history years data
# TODO order min max percent later
limitPercentMediumRiskStocks = 0.3
limitPercentMediumRiskCommodity = 0.05
limitPercenLowRiskStocks = 0.1
limitPercenLowRiskCommodity = 0
stocksSymbols = [
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
nameOfBucket = '1'
bucketRepositoriy = 'DB/Bucket'+nameOfBucket+'/'
machineLearningLocation = bucketRepositoriy + 'includingMachineLearning/'
nonMachineLearningLocation = bucketRepositoriy + 'withoutMachineLearning/'

usersJsonName = "DB/users"  # where to save the users details
############################################
# setting for api-tase connection and apps:
tokenUrl = "https://openapigw.tase.co.il/tase/prod/oauth/oauth2/token"

# key and secret of ofer
# key = '7e247414e7047349d83b7b8a427c529c'
# secret = '7a809c498662c88a0054b767a92f0399'

# "key and secret of yardet1998 - "robo-advisor"
key = 'fb4577af7a020ad0e443632a102003d4'
secret = '038af8b13e67a0fd0cbdbb1cdcb973c6'

# key and secret of yarden - "roboadvisor-mta"
# key = "9791b4a859c4650efe0d77c2ed9d6919"
# secret = "e8d0264a8472c411443a6dfbcdf0992f"

baseUrl = "https://openapigw.tase.co.il/tase/prod/api/v1"
prefixUrl = "/tase/prod/api/v1"
payload = "grant_type=client_credentials&scope=tase"
language = "he-IL"


# apps names with date
basicSecuritiesCompnayTrade = "basic-securities/trade-securities-list"  # not working

# past 10 years
indexEndOfDayHistoryTenYearsUpToday = "indices/eod/history/ten-years/by-index?indexId="
# past 5 years
indexEndOfDayHistoryFiveYearsUpToday = "indices/eod/history/five-years/by-index?indexId="

# other api-tase url's apps:
OTC_transaction_name = "transactions/otc-transactions"
endOfDayTransactionName = "transactions/transactions-end-of-day"
mayaNoticeByDay = "maya-reports-online/tase-messages-by-date"
fundHistoryDataName = "mutual-fund/history-data"

# apps names without date
basicIndexList = "basic-indices/indices-list"  # a list of all basic indices in tase-already have-not need to do again
basicSecuritiesList = "basic-securities/securities-types"  # not need to do again
basicSecuritiesCompnayList = "basic-securities/companies-list"
shortSalesWeeklyBalanceName = "short-sales/weekly-balance"
shortSalesHistoricalData = "short-sales/history"
fundListName = "fund/fund-list"
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
