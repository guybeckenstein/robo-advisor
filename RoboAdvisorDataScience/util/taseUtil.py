#!/usr/bin/python
# -*- coding: utf-8 -*-
import http.client
import json
import codecs
import re
import requests
import base64
import datetime
from util import setting


# UTILITY FOR TASE- tel aviv stock exchange

def getIsraeliIndexesData(command, israeliIndexes):  # get israli indexes data from tase
    """JsonDataList = [0] * len(israeliIndexes)
    if command == "get_past_10_years_history":
        for i in range(len(israeliIndexes)):
            appUrl = getIndexHistory(
                setting.indexEndOfDayHistoryTenYearsUpToday, israeliIndexes[i], 10
            )
            JsonDataList[i] = getSymbolInfo(appUrl)
        # return JsonDataList - TODO - fix , makes unlimited requests"""

    return getIndexesDataManuallyFromJSON(israeliIndexes)


def getSymbolInfo(appUrl):
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

    token = setting.key + ':' + setting.secret
    base_64_token = base64.b64encode(token.encode('ascii')).decode('ascii')
    return base_64_token


def get_tase_access_token():
    base_64_token = get_base_64_token()

    # tokenUrl = "https://openapigw.tase.co.il/tase/prod/oauth/oauth2/token"

    payload = 'grant_type=client_credentials&scope=tase'
    headers = {'Authorization': 'Basic ' + base_64_token,
               'Content-Type': 'application/x-www-form-urlencoded'}
    response = requests.request('POST', setting.tokenUrl,
                                headers=headers, data=payload)

    return json.loads(response.text)['access_token']


# BUILD URL FOR REQUEST

def getIndexHistory(appName, indexId, numOfYears):
    today = datetime.datetime.now()
    startyear = today.year - numOfYears
    startmonth = today.month
    startday = today.day
    endyear = today.year
    endmonth = today.month
    endday = today.day

    return getAppUrlWithDateAndIndex(
        appName,
        startyear,
        startmonth,
        startday,
        endyear,
        endmonth,
        endday,
        indexId,
        )


def getAppUrlWithoutDate(appName):  # /tase/prod/api/v1/short-sales/weekly-balance
    return setting.prefixUrl + '/' + appName


def getAppUrlWithDateAndIndex(appName, startYear, startMounth, startDay, endYear, endMonth, endDay, indexName):

    return getAppUrlWithoutDate(appName) + str(indexName) \
        + '&fromDate=' + str(startYear) + '-' + str(startMounth) + '-' \
        + str(startDay) + '&toDate=' + str(endYear) + '-' \
        + str(endMonth) + '-' + str(endDay)


def getNameByIndexNumber(indexNumber):

    # Load the JSON data from the file

    jsonData = getJsonData('DB/indicesList')
    result = [item['indexName'] for item in jsonData['indicesList']['result'] if item['indexId'] == indexNumber]

    # makes it from right to left

    name = result[0]
    return name


def getIndexesDataManuallyFromJSON(sybmolIndexs):  # FOR ISRAELI STOCKS
    if type(sybmolIndexs) == int:
        return getJsonData('api/resources/History' + str(sybmolIndexs))
    else:
        portfolio = [0] * len(sybmolIndexs)
        for i in range(len(sybmolIndexs)):
            portfolio[i] = getJsonData('api/resources/History' + str(sybmolIndexs[i]))
        return portfolio


def convertIsraeliIndexToName(IsraliIndexes):
    hebrew_pattern = r"[\u0590-\u05FF\s]+"
    stocksNames = {}
    for (i, index) in enumerate(IsraliIndexes):
        text = getNameByIndexNumber(index)
        hebrew_parts = re.findall(hebrew_pattern, text)

        for hebrew_part in hebrew_parts:  # find all the Hebrew parts in the text
            hebrew_part_reversed = ''.join(reversed(hebrew_part))
            text = text.replace(hebrew_part, hebrew_part_reversed)

        stocksNames[i] = text
    return stocksNames.values()


def getJsonData(name):
    with codecs.open(name + '.json', 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    return json_data


def getJsonDataFromTase(indexId, nameFile):
    appUrl = \
        getIndexHistory(setting.indexEndOfDayHistoryTenYearsUpToday,
                        indexId, 10)
    jsonData = getSymbolInfo(appUrl)
    return jsonData
