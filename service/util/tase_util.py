#!/usr/bin/python
# -*- coding: utf-8 -*-
import http.client
import json
import codecs
import requests
import base64
import datetime
from ..api.resources import israeli_tase_settings as settings


# UTILITY FOR TASE- tel aviv stock exchange

def get_israeli_indexes_data(command, start_date, end_date, israeliIndexes):  # get israeli indexes data from tase
    JsonDataList = [israeliIndexes]
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    start_year = start_date.year
    start_month = start_date.month
    start_day = start_date.day
    end_year = end_date.year
    end_month = end_date.month
    end_day = end_date.day

    if command == "get_past_10_years_history":
        for i in range(len(JsonDataList)):
            appUrl = get_app_url_with_date_and_index(
                settings.INDEX_EOD_HISTORY_TEN_YEARS,
                startYear=start_year,
                startMonth=start_month,
                startDay=start_day,
                endYear=end_year,
                endMonth=end_month,
                endDay=end_day,
                indexName=JsonDataList[i]
            )
            JsonDataList[i] = get_symbol_info(appUrl)
    # TODO: add more command later
    else:
        pass


    return JsonDataList[0]

def get_symbol_info(appUrl):
    conn = http.client.HTTPSConnection('openapigw.tase.co.il')
    payload = ''
    headers = {'Authorization': 'Bearer ' + get_tase_access_token(),
               'Accept-Language': 'he-IL',
               'Content-Type': 'application/json'}
    conn.request('GET', appUrl, payload, headers)
    res = conn.getresponse()
    data = res.read()

    # Decode the bytes object to a string

    json_string = data.decode('utf-8')
    json_obj = json.loads(json_string)

    return json_obj


# Auth

def get_base_64_token():

    # key = '7e247414e7047349d83b7b8a427c529c'
    # secret = '7a809c498662c88a0054b767a92f0399'

    token = settings.key + ':' + settings.secret
    base_64_token = base64.b64encode(token.encode('ascii')).decode('ascii')
    return base_64_token


def get_tase_access_token():
    base_64_token = get_base_64_token()

    # tokenUrl = "https://openapigw.tase.co.il/tase/prod/oauth/oauth2/token"

    payload = 'grant_type=client_credentials&scope=tase'
    headers = {'Authorization': 'Basic ' + base_64_token,
               'Content-Type': 'application/x-www-form-urlencoded'}
    response = requests.request('POST', settings.tokenUrl,
                                headers=headers, data=payload)

    return json.loads(response.text)['access_token']


# BUILD URL FOR REQUEST

def get_index_history(appName, indexId, numOfYears):
    today = datetime.datetime.now()
    start_year = today.year - numOfYears
    start_month = today.month
    start_day = today.day
    end_year = today.year
    end_month = today.month
    end_day = today.day

    return get_app_url_with_date_and_index(
        appName,
        start_year,
        start_month,
        start_day,
        end_year,
        end_month,
        end_day,
        indexId,
        )


def get_app_url_without_date(appName):  # /tase/prod/api/v1/short-sales/weekly-balance
    return settings.prefixUrl + '/' + appName


def get_app_url_with_date_and_index(appName, startYear, startMonth, startDay,
                                    endYear, endMonth, endDay, indexName):

    return get_app_url_without_date(appName) + str(indexName) \
        + '&fromDate=' + str(startYear) + '-' + str(startMonth) + '-' \
        + str(startDay) + '&toDate=' + str(endYear) + '-' \
        + str(endMonth) + '-' + str(endDay)


def get_indexes_data_manually_from_json(sybmolIndexs):  # FOR ISRAELI STOCKS
    if type(sybmolIndexs) == int:
        return get_json_data('api/resources/History' + str(sybmolIndexs))
    else:
        portfolio = [0] * len(sybmolIndexs)
        for i in range(len(sybmolIndexs)):
            portfolio[i] = get_json_data('api/resources/History' + str(sybmolIndexs[i]))
        return portfolio


def get_json_data(name):
    with codecs.open(name + '.json', 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    return json_data


def get_json_data_from_tase(indexId, nameFile):
    appUrl = \
        get_index_history(settings.indexEndOfDayHistoryTenYearsUpToday,
                        indexId, 10)
    jsonData = get_symbol_info(appUrl)
    return jsonData
