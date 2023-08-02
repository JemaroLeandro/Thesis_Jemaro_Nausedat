import tweepy
import pandas as pd
import Config


from datetime import datetime

today = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")


# read configuration files

#config = configparser.ConfigParser()
#config.read('config.ini')

api_key = Config.API_Key
api_key_secret = Config.API_Key_Secret

access_token = Config.Access_Token
access_token_secret = Config.Access_Token_Secret
Bearer_Token = Config.Bearer_Token

#authentication with Twitter

client = tweepy.Client(Bearer_Token)

#Query for firm 1 here

query1 = '@' +Config.namer1+ ' -has:media -is:retweet lang:en -has:links'

#Query for firm 2 here

query2 = '@' +'ATVI_AB'+ ' -has:media -is:retweet lang:en -has:links'

#gemerate files


#Collect 1250 tweets from the last 7 days for firm 1


tweets1 = tweepy.Paginator(client.search_recent_tweets, query=query1, max_results=100, tweet_fields=['created_at']).flatten(limit=1250)

Columns = ['ID','Text', 'Created_At']

data = []

for tweet in tweets1:
    data.append([tweet.id, tweet.text, tweet.created_at])

tweets1_sentiment = pd.DataFrame(data, columns=Columns)

tweets1_sentiment.to_csv(Config.file_name1)

#Collect 1250 tweets from the last 7 days for firm 2

tweets2 = tweepy.Paginator(client.search_recent_tweets, query=query2, max_results=100, tweet_fields=['created_at']).flatten(limit=1250)

Columns = ['ID','Text', 'Created_At']

data = []

for tweet in tweets2:
    data.append([tweet.id, tweet.text, tweet.created_at])

tweets2_sentiment = pd.DataFrame(data, columns=Columns)

tweets2_sentiment.to_csv(Config.file_name2)

