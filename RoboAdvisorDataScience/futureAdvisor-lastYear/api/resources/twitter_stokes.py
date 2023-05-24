def gen_sent_graph(stock_symbol,start_date,end_date):
  import json
  from twython import TwythonStreamer, Twython
  import csv
  import pandas as pd
  import matplotlib.pyplot as plt
  import nltk
  dler = nltk.downloader.Downloader()
  dler._update_index()
  dler.download('popular', quiet = True)
  from nltk.corpus import wordnet as wn

  #Credentials we've got through twitter
  credentials = {}
  credentials['CONSUMER_KEY'] = 'rlqZBXDffdBlyhBf1Kj9fVYe4'
  credentials['CONSUMER_SECRET'] = 'xhw2FitCMZbVx0XxBN2fFZkZoGv1C5QEkOaUbOqk631LBJAxMk'
  credentials['ACCESS_TOKEN'] = '3017621772-G0jXLYKF88JAr17JK4sUYcTYCGOfc0LRTqr6CML'
  credentials['ACCESS_SECRET'] = 'P82g6pUTh9EUVL8lXAFLtzknTKwrKctd40BdIcogLycRr'

  # Save the credentials object to file
  with open("twitter_credentials.json", "w") as file:
      json.dump(credentials, file)

  # Load credentials from json file
  with open("twitter_credentials.json", "r") as file:
      creds = json.load(file)

  # Instantiate an object
  python_tweets = Twython(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'],creds['ACCESS_TOKEN'],creds['ACCESS_SECRET'])

  # Set up our query parameters based on user input
  search1 = stock_symbol
  slang = 'en'
  scount = 10000
  TimeSearch = start_date
  TimeSearchEnd = end_date


  # Create our query
  query = {'q': search1,
          # 'result_type': 'popular',
          'count': scount,
          'lang': slang,
          'since': TimeSearch,
          'until': TimeSearchEnd,
          }

  # Search tweets for tweets mentioning stock symbol
  dict_ = {'user': [], 'date': [], 'text': [], 'favorite_count': []}
  for status in python_tweets.search(**query)['statuses']:
      dict_['user'].append(status['user']['screen_name'])
      dict_['date'].append(status['created_at'])
      dict_['text'].append(status['text'])
      dict_['favorite_count'].append(status['favorite_count'])

  # Structure data in a pandas DataFrame for easier manipulation
  # And sorts it by favorite count of each Tweet
  df = pd.DataFrame(dict_)
  df.sort_values(by='favorite_count', inplace=True, ascending=False)
  df.sort_values(by='favorite_count')



  # Create list of words associated with buy/sell/hold
  buy_words = ["outperform", "buy", "sector perform", "hot", "bulles", "overweight", "positive", "strong buy","great","awesome","recommended"]
  sell_words = ["sell", "underperform", "underweight", "underwt/in-Line", "frozen", "bleeding", "reduce","sucks","throw","trash"]
  holding_words = ["hold", "neutral", "market perform"]

  # For each word in the buy words list - add it's synonyms to the list
  # Repet for sell and hold words
  arr = []
  for word in buy_words:
    for syn in wn.synsets(word):
      arr.append(syn.name().split('.')[0])
  buy_words+=arr
  arr = []
  for word in sell_words:
    for syn in wn.synsets(word):
      arr.append(syn.name().split('.')[0])
  sell_words+=arr
  arr = []
  for word in holding_words:
    for syn in wn.synsets(word):
      arr.append(syn.name().split('.')[0])
  holding_words+=arr
  arr = []

  # Erase duplicates from list
  buy_words = list(set(buy_words))
  sell_words = list(set(sell_words))
  holding_words = list(set(holding_words))

  # (Weighted) counters for each operation
  c_buy = 0
  c_sell = 0
  c_hold = 0

  # The tweet with the most likes is fetched
  # Will be used later to normalize the weight of each tweet
  max_fav = df.iloc[0]['favorite_count']
  df = df.reset_index()

  # Calculate stock operations by their weights
  for index, row in df.iterrows():
    #the tweet with the most likes gets a weight of 1,
    #the rest are weighed in relation to that
    weight = row['favorite_count']/max_fav
    #put the value of a tweet in [1,2] range
    val = 1+weight
    #the tweet text itself
    text = row['text']
    #split into words
    txt_arr = text.split(' ')
    for word in txt_arr:
      word = word.lower()
      if word in buy_words:
        # if a tweet contains a words associated with buying,
        # we count it as a "buy tweet", same for sell and hold
        c_buy+=val
        break
      if word in sell_words:
        c_sell += val
        break
      if word in holding_words:
        c_hold += val
        break

  # Set up and plot the results as a bar graph
  y_temp = [c_buy,c_sell,c_hold]
  y = [round(num,2) for num in y_temp]
  x = ['Buy','Sell','Hold']
  plt.bar(x, y)
  plt.title(f'Sentiment around the stock {search1}')
  plt.xlabel('Sentiment')
  plt.ylabel('Positive/Neutral/Negative mentions')
  for i in range(len(x)):
    shift = len(str(y[i]))*0.02
    # Add labels to bars
    plt.text(i-shift,y[i]+0.5,y[i])
  plt.ylim([0,max(y)+5])
  plt.show()
  max_val = max(y)
  return x,y


import datetime as DT

# Set up default dates to be fron 90 days ago till today
today = DT.date.today()
two_week_ago = str(today - DT.timedelta(days=90))
today = str(today)


def run_stock(stock_symbol, start_date=two_week_ago, end_date=today):
  try:
    return gen_sent_graph(stock_symbol, start_date, end_date)
  except Exception as e:
    return "an Error has occured, please make sure you've entered the Symbol correctly and with valid dates", None


# running example:
run_stock("SPY")
run_stock("QQQ")