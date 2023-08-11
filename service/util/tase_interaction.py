#!/usr/bin/python
# -*- coding: utf-8 -*-
import http.client
import json
import codecs
import requests
import base64
import datetime
from ..impl.config import israeli_tase_settings as settings


def get_israeli_symbol_data(command, start_date, end_date, israeli_stock_symbol, used_app_url):
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    start_year = start_date.year
    start_month = start_date.month
    start_day = start_date.day
    end_year = end_date.year
    end_month = end_date.month
    end_day = end_date.day

    if command == "get_past_10_years_history":
        appUrl = get_app_url_with_date_and_index(
            used_app_url,
            start_year=start_year,
            start_month=start_month,
            start_day=start_day,
            end_year=end_year,
            end_month=end_month,
            end_day=end_day,
            index_name=israeli_stock_symbol
        )
    # TODO: maybe add more command later
    else:
        pass

    return get_json_data_of_symbol(appUrl)


def get_israeli_index_data(command, start_date, end_date, israeli_index_name):  # get israeli indexes data from tase
    used_app_url = settings.INDEX_EOD_HISTORY_TEN_YEARS
    return get_israeli_symbol_data(command, start_date, end_date, israeli_index_name, used_app_url)


def get_israeli_security_data(command, start_date, end_date, israeli_security_name):
    used_app_url = settings.SECURITY_END_OF_DAY_HISTORY_TEN_YEARS
    return get_israeli_symbol_data(command, start_date, end_date, israeli_security_name, used_app_url)


def get_israeli_companies_list():
    appUrl = get_app_url_without_date(settings.BASIC_SECURITIES_LIST_BY_TYPE)
    json_data = get_json_data_of_symbol(appUrl)
    return json_data


def get_israeli_indexes_list():
    appUrl = get_app_url_without_date(settings.BASIC_INDEX_LIST)
    json_data = get_json_data_of_symbol(appUrl)
    return json_data


def get_json_data_of_symbol(app_url):
    conn = http.client.HTTPSConnection('openapigw.tase.co.il')
    payload = ''
    headers = {'Authorization': 'Bearer ' + get_tase_access_token(),
               'Accept-Language': 'he-IL',
               'Content-Type': 'application/json'}
    conn.request('GET', app_url, payload, headers)
    res = conn.getresponse()
    data = res.read()

    # Decode the bytes object to a string

    json_string = data.decode('utf-8')
    json_obj = json.loads(json_string)

    return json_obj


def get_indexes_data_manually_from_json(symbol_indexs):  # FOR ISRAELI STOCKS
    folder_prefix = 'impl/config/History'
    if type(symbol_indexs) == int:
        return get_json_data(folder_prefix + str(symbol_indexs))
    else:
        portfolio = [0] * len(symbol_indexs)
        for i in range(len(symbol_indexs)):
            portfolio[i] = get_json_data(folder_prefix + str(symbol_indexs[i]))
        return portfolio


def get_json_data(name):
    with codecs.open(name + '.json', 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    return json_data


def get_json_data_from_tase(index_id, name_file):  # INDEX
    appUrl = \
        get_index_history(settings.indexEndOfDayHistoryTenYearsUpToday,
                          index_id, 10)
    jsonData = get_json_data_of_symbol(appUrl)
    return jsonData


# Auth
def get_base_64_token():
    # key = '7e247414e7047349d83b7b8a427c529c'
    # secret = '7a809c498662c88a0054b767a92f0399'

    token = settings.KEY + ':' + settings.SECRET
    base_64_token = base64.b64encode(token.encode('ascii')).decode('ascii')
    return base_64_token


def get_tase_access_token():
    base_64_token = get_base_64_token()

    # tokenUrl = "https://openapigw.tase.co.il/tase/prod/oauth/oauth2/token"

    payload = 'grant_type=client_credentials&scope=tase'
    headers = {'Authorization': 'Basic ' + base_64_token,
               'Content-Type': 'application/x-www-form-urlencoded'}
    response = requests.request('POST', settings.TOKEN_URL,
                                headers=headers, data=payload)

    return json.loads(response.text)['access_token']


# BUILD URL FOR REQUEST
def get_index_history(app_name, index_id, num_of_years):
    today = datetime.datetime.now()
    start_year = today.year - num_of_years
    start_month = today.month
    start_day = today.day
    end_year = today.year
    end_month = today.month
    end_day = today.day

    return get_app_url_with_date_and_index(
        app_name,
        start_year,
        start_month,
        start_day,
        end_year,
        end_month,
        end_day,
        index_id,
    )


def get_app_url_without_date(app_name):  # /tase/prod/impl/v1/short-sales/weekly-balance
    return settings.PREFIX_URL + '/' + app_name


def get_app_url_with_date_and_index(app_name, start_year, start_month, start_day,
                                    end_year, end_month, end_day, index_name):
    return get_app_url_without_date(app_name) + str(index_name) \
        + '&fromDate=' + str(start_year) + '-' + str(start_month) + '-' \
        + str(start_day) + '&toDate=' + str(end_year) + '-' \
        + str(end_month) + '-' + str(end_day)
