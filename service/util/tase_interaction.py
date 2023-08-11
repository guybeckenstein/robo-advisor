#!/usr/bin/python
# -*- coding: utf-8 -*-
import http.client
import json
import codecs
import requests
import base64
import datetime
from ..config import settings, israeli_tase_settings


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
        app_url = get_app_url_with_date_and_index(
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
        raise NotImplementedError

    return get_json_data_of_symbol(app_url)


def get_israeli_index_data(command, start_date, end_date, israeli_index_name):  # get israeli indexes data from tase
    used_app_url = israeli_tase_settings.INDEX_EOD_HISTORY_TEN_YEARS
    return get_israeli_symbol_data(command, start_date, end_date, israeli_index_name, used_app_url)


def get_israeli_security_data(command, start_date, end_date, israeli_security_name):
    used_app_url = israeli_tase_settings.SECURITY_END_OF_DAY_HISTORY_TEN_YEARS
    return get_israeli_symbol_data(command, start_date, end_date, israeli_security_name, used_app_url)


def get_israeli_companies_list():
    app_url = get_app_url_without_date(israeli_tase_settings.BASIC_SECURITIES_LIST_BY_TYPE)
    json_data = get_json_data_of_symbol(app_url)
    return json_data


def get_israeli_indexes_list():
    app_url = get_app_url_without_date(israeli_tase_settings.BASIC_INDEX_LIST)
    json_data = get_json_data_of_symbol(app_url)
    return json_data


def get_json_data_of_symbol(app_url):
    conn = http.client.HTTPSConnection('openapigw.tase.co.il')
    payload = ''
    headers = {'Authorization': 'Bearer ' + get_tase_access_token(),
               'Accept-Language': israeli_tase_settings.LANGUAGE,
               'Content-Type': 'application/json'}
    conn.request('GET', app_url, payload, headers)
    res = conn.getresponse()
    data = res.read()

    # Decode the bytes object to a string

    json_string = data.decode('utf-8')
    json_obj = json.loads(json_string)

    return json_obj


def get_indexes_data_manually_from_json(symbol_indexes):  # FOR ISRAELI STOCKS
    folder_prefix = f'{settings.CONFIG}/history'
    if type(symbol_indexes) == int:
        return get_json_data(folder_prefix + str(symbol_indexes))
    else:
        portfolio = [0] * len(symbol_indexes)
        for i in range(len(symbol_indexes)):
            portfolio[i] = get_json_data(folder_prefix + str(symbol_indexes[i]))
        return portfolio


def get_json_data(name):
    with codecs.open(name + '.json', 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    return json_data


def get_json_data_from_tase(index_id, file_name):  # INDEX
    appUrl = get_index_history(israeli_tase_settings.INDEX_END_OF_DAY_HISTORY_FIVE_YEARS_UP_TO_TODAY, index_id, 10)
    jsonData = get_json_data_of_symbol(appUrl)
    return jsonData


# Auth
def get_base_64_token():
    # key = '7e247414e7047349d83b7b8a427c529c'
    # secret = '7a809c498662c88a0054b767a92f0399'

    token: str = israeli_tase_settings.KEY + ':' + israeli_tase_settings.SECRET
    base_64_token = base64.b64encode(token.encode('ascii')).decode('ascii')
    return base_64_token


def get_tase_access_token():
    base_64_token = get_base_64_token()

    # tokenUrl = "https://openapigw.tase.co.il/tase/prod/oauth/oauth2/token"

    payload = 'grant_type=client_credentials&scope=tase'
    headers = {'Authorization': 'Basic ' + base_64_token, 'Content-Type': 'application/x-www-form-urlencoded'}
    response = requests.request('POST', israeli_tase_settings.TOKEN_URL, headers=headers, data=payload)

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
    return israeli_tase_settings.PREFIX_URL + '/' + app_name


def get_app_url_with_date_and_index(app_name, start_year, start_month, start_day,
                                    end_year, end_month, end_day, index_name):
    app_without_date = get_app_url_without_date(app_name)
    str_index = str(index_name)
    str_s_y = str(start_year)
    str_s_m = str(start_month)
    str_s_d = str(start_day)
    str_e_y = str(end_year)
    str_e_m = str(end_month)
    str_e_d = str(end_day)
    res = f'{app_without_date}{str_index}&fromDate={str_s_y}{str_s_m}{str_s_d}&toDate={str_e_y}{str_e_m}{str_e_d}'
    return res
