import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

def getJsonData(name):
    with open(name+".json") as file:
        json_data = json.load(file)
    return json_data

#json_data = getJsonData('weekly-balance')
#json_data = getJsonData('index_end_of_day')
#json_data = getJsonData('otc-transactions')
json_data = getJsonData('history-data')
#df = pd.DataFrame.from_dict(json_data['indexEndOfDay'])  # Total rows: 98
#print(df)
# For data science: https://colab.research.google.com/drive/1meOMfgel3py4MuPHaauvVBH7ec26-Iun#scrollTo=7wEMe9LyvJLm
