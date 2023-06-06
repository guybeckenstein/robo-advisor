#!/usr/bin/python
# -*- coding: utf-8 -*-
import http.client
import json
import requests
import base64

# for TOKEN - POST METHOD

tokenUrl = 'https://openapigw.tase.co.il/tase/prod/oauth/oauth2/token'

# key and secret of ofer
# key = '7e247414e7047349d83b7b8a427c529c'
# secret = '7a809c498662c88a0054b767a92f0399'

# "key and secret of yarden - "tel_aviv_academic"
# key = '42599cb8bae3df12b563c85e4fb5a208'
# secret = 'cf7f46fcd91865d0f9c457e37bd7e726'

# key and secret of yarden - "roboadvisor-mta"

key = '9791b4a859c4650efe0d77c2ed9d6919'
secret = 'e8d0264a8472c411443a6dfbcdf0992f'

# for getting data - GET METHOD

baseUrl = 'https://openapigw.tase.co.il/tase/prod/api/v1'
prefixUrl = 'tase/prod/api/v1'
payload = 'grant_type=client_credentials&scope=tase'
language = 'he-IL'

# TODO LATER - get date from SYSTEM

year = 2022
month = 12
day = 27

# apps names with date

indexEndOfDayName = 'indices-end-of-day-data/index-end-of-day-data'
OTC_transaction_name = 'transactions/otc-transactions'
endOfDayTransactionName = 'transactions/transactions-end-of-day'
mayaNoticeByDay = 'maya-reports-online/tase-messages-by-date'
fundHistoryDataName = 'mutual-fund/history-data'

# apps names without date

shortSalesWeeklyBalanceName = 'short-sales/weekly-balance'
shortSalesHistoricalData = 'short-sales/history'
fundListName = 'fund/fund-list'


def get_base_64_token():

    # key = '7e247414e7047349d83b7b8a427c529c'
    # secret = '7a809c498662c88a0054b767a92f0399'

    token = key + ':' + secret
    base_64_token = base64.b64encode(token.encode('ascii')).decode('ascii')
    return base_64_token


def get_tase_access_tokenn():
    base_64_token = get_base_64_token()

    # tokenUrl = "https://openapigw.tase.co.il/tase/prod/oauth/oauth2/token"

    payload = 'grant_type=client_credentials&scope=tase'
    headers = {'Authorization': 'Basic ' + base_64_token,
               'Content-Type': 'application/x-www-form-urlencoded'}
    response = requests.request('POST', tokenUrl, headers=headers,
                                data=payload)
    return json.loads(response.text)['access_token']


access_token = get_tase_access_tokenn()


def get_fund_history_by_id(appName: str, fundId: int, startyear: int, startmonth: int,
                           startday: int, endyear: int, endmonth: int, endday: int, ) -> str:

    # /tase/prod/api/v1/index_end_of_day_data/2022/11/22
    return '/tase/prod/api/v1/mutual-fund/history-data/5100474?startDate=2015-12-31&endDate=2020-12-31'


def get_app_url_with_date(appName: str, year: int, month: int, day: int) -> str:
    return '/' + prefixUrl + '/' + appName + '/' + str(year) + '/' \
        + str(month) + '/' + str(day)


def get_app_url_without_date(appName: str) -> str:  # /tase/prod/api/v1/short-sales/weekly-balance
    return '/' + prefixUrl + '/' + appName


def get_trade_info(appUrl: str, nameProduct: str) -> None:
    conn = http.client.HTTPSConnection('openapigw.tase.co.il')
    payload = ''
    headers = {'Authorization': 'Bearer ' + access_token,
               'Accept-Language': 'he-IL',
               'Content-Type': 'application/json'}
    conn.request('GET', appUrl, payload, headers)
    res = conn.getresponse()
    data = res.read()

    # Decode the bytes object to a string

    json_string = data.decode('utf-8')
    json_obj = json.loads(json_string)

    # Open a file in write mode

    parts = nameProduct.split('/')
    last_element = parts[-1]
    with open(last_element + '.json', 'w') as f:
        # Use the `dump()` function to write the JSON data to the file
        json.dump(json_obj, f)


# GET THE DATA FROM PROJECTS

# already have:
# get_trade_info(get_app_url_with_date(indexEndOfDayName, year, month, day),indexEndOfDayName):
# index end of day data
# get_trade_info(get_app_url_with_date(OTC_transaction_name, year, month, day),OTC_transaction_name):
# OTC transaction
# get_trade_info(get_app_url_without_date(shortSalesWeeklyBalanceName),shortSalesWeeklyBalanceName):
# short sales weekly balance
# get_trade_info(get_app_url_with_date(mayaNoticeByDay, year, month, day),mayaNoticeByDay):
# maya notice by day
# new:
# get_trade_info(get_app_url_without_date(fundListName),fundListName):
# fund list, fund history data

get_trade_info(get_fund_history_by_id(
    fundHistoryDataName,
    1143718,
    2015,
    12,
    31,
    2022,
    12,
    28,
    ), fundHistoryDataName)


# not working
# get_trade_info(get_app_url_with_date(endOfDayTransactionName, year, month, day),endOfDayTransactionName):
# end of day transaction
# get_trade_info(get_app_url_without_date(shortSalesHistoricalData),shortSalesHistoricalData):
# short sales historical data

# more relevant apps:


"""old version"""
""""
def get_base_64_token(key='7e247414e7047349d83b7b8a427c529c', secret='7a809c498662c88a0054b767a92f0399'):
    token = key + ":" + secret
    base_64_token = base64.b64encode(token.encode("ascii")).decode("ascii")
    return base_64_token


def get_tase_access_tokenn():
    base_64_token = get_base_64_token()
    url = "https://openapigw.tase.co.il/tase/prod/oauth/oauth2/token"
    payload = 'grant_type=client_credentials&scope=tase'
    headers = {
        'Authorization': 'Basic ' + base_64_token,
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    return json.loads(response.text)['access_token']


access_token = get_tase_access_tokenn()


def get_trade_index_info():
    conn = http.client.HTTPSConnection("openapigw.tase.co.il")
    payload = ''
    headers = {
        'Authorization': 'Bearer ' + access_token,
        'Accept-Language': 'he-IL',
        'Content-Type': 'application/json'
    }
    conn.request("GET", "/tase/prod/api/v1/indices-end-of-day-data/index-end-of-day-data/2022/12/11", payload, headers)
    res = conn.getresponse()
    data = res.read()
    print(data.decode("utf-8"))


# get_trade_index_info()
"""
