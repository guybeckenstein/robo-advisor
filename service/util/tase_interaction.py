import http.client
import json
import datetime
from service.config import israeli_tase


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
    used_app_url = israeli_tase.INDEX_EOD_HISTORY_TEN_YEARS
    return get_israeli_symbol_data(command, start_date, end_date, israeli_index_name, used_app_url)


def get_israeli_security_data(command, start_date, end_date, israeli_security_name):
    used_app_url = israeli_tase.SECURITY_END_OF_DAY_HISTORY_TEN_YEARS
    return get_israeli_symbol_data(command, start_date, end_date, israeli_security_name, used_app_url)


def get_israeli_companies_list():
    app_url = get_app_url_without_date(israeli_tase.BASIC_SECURITIES_LIST_BY_TYPE)
    json_data = get_json_data_of_symbol(app_url)
    return json_data


def get_israeli_indexes_list():
    app_url = get_app_url_without_date(israeli_tase.BASIC_INDEX_LIST)
    json_data = get_json_data_of_symbol(app_url)
    return json_data


def get_json_data_of_symbol(app_url):
    conn = http.client.HTTPSConnection('openapigw.tase.co.il')
    payload = ''
    headers = {'Authorization': 'Bearer ' + get_tase_access_token(),
               'Accept-Language': israeli_tase.LANGUAGE,
               'Content-Type': 'application/json'}
    conn.request('GET', app_url, payload, headers)
    res = conn.getresponse()
    data = res.read()

    # Decode the bytes object to a string

    json_string = data.decode('utf-8')
    json_obj = json.loads(json_string)

    return json_obj


# Auth
def get_base_64_token():
    import base64

    token: str = f'{israeli_tase.KEY}:{israeli_tase.SECRET}'
    base_64_token = base64.b64encode(token.encode('ascii')).decode('ascii')
    return base_64_token


def get_tase_access_token():
    import requests

    base_64_token = get_base_64_token()
    payload = 'grant_type=client_credentials&scope=tase'
    headers = {'Authorization': 'Basic ' + base_64_token, 'Content-Type': 'application/x-www-form-urlencoded'}
    response = requests.request('POST', israeli_tase.TOKEN_URL, headers=headers, data=payload)

    return json.loads(response.text)['access_token']


# BUILD URL FOR REQUEST
def get_app_url_without_date(app_name: str) -> str:
    return f'{israeli_tase.PREFIX_URL}/{app_name}'


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
    hyphen = '-'
    res = (f'{app_without_date}{str_index}&fromDate={str_s_y}{hyphen}{str_s_m}{hyphen}{str_s_d}'
           f'&toDate={str_e_y}{hyphen}{str_e_m}{hyphen}{str_e_d}')

    return res
