import http.client
import json
import requests
import base64


def get_base_64_token(key='7e247414e7047349d83b7b8a427c529c', secret='7a809c498662c88a0054b767a92f0399'):
    token = key + ":" + secret
    base_64_token = base64.b64encode(token.encode("ascii")).decode("ascii")
    return base_64_token


def get_tase_access_token():
    base_64_token = get_base_64_token()
    url = "https://openapigw.tase.co.il/tase/prod/oauth/oauth2/token"
    payload = 'grant_type=client_credentials&scope=tase'
    headers = {
        'Authorization': 'Basic ' + base_64_token,
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    return json.loads(response.text)['access_token']


access_token = get_tase_access_token()


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

