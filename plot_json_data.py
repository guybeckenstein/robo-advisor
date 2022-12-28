import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

with open('index_end_of_day_data_2022_22_11.json') as file:
    json_data = json.load(file)

df = pd.DataFrame.from_dict(json_data['indexEndOfDay'])  # Total rows: 98
# For data science: https://colab.research.google.com/drive/1meOMfgel3py4MuPHaauvVBH7ec26-Iun#scrollTo=7wEMe9LyvJLm
