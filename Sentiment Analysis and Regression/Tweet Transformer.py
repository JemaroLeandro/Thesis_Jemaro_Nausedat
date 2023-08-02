import pandas as pd
import numpy as np
import Config

from datetime import datetime

today = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")


#transform sentiment analysis results of firm 1

to_transform = pd.read_csv(Config.processed_file_name1)

# to_transform = pd.read_csv('Test.csv')

to_transform = to_transform.drop(columns= ['Unnamed: 0.1', 'ID', 'Unnamed: 0', 'Created_At', 'Text'])

to_transform = to_transform.describe()

#Save results table for firm 1 for weekly documentation

to_transform.to_csv(Config.Week_signifier+ ' - descriptive statistics for sentiment ' + Config.namer1 +  '.csv')

#Add results for current week to firm 1 weekly sentiment series table

tweet_dataset_1 = pd.read_csv(Config.transformed_file_name1)

tweet_dataset_1 = tweet_dataset_1.set_index('Statistics')

tweet_dataset_1[Config.Week_signifier] = ""

tweet_dataset_1.loc['roberta_neg',Config.Week_signifier] = to_transform.loc['mean', 'roberta_neg']
tweet_dataset_1.loc['roberta_neu',Config.Week_signifier] = to_transform.loc['mean', 'roberta_neu']
tweet_dataset_1.loc['roberta_pos',Config.Week_signifier] = to_transform.loc['mean', 'roberta_pos']
tweet_dataset_1.loc['roberta_sum',Config.Week_signifier] = to_transform.loc['mean', 'roberta_sum']

if Config.i > 1:

    tweet_dataset_1.loc['sensitivity',Config.Week_signifier] = tweet_dataset_1.loc['roberta_sum', Config.Week_signifier] - tweet_dataset_1.loc['roberta_sum', Config.Week_signifier_previous]


# tweet_dataset_1 = tweet_dataset_1.drop(columns = 'Unnamed: 0')

#Overwrite firm 1 sentiment series table with new results

tweet_dataset_1.to_csv(Config.transformed_file_name1)


#####


#transform sentiment analysis results of firm 2

to_transform = pd.read_csv(Config.processed_file_name2)

# to_transform = pd.read_csv('Test.csv')

to_transform = to_transform.drop(columns= ['Unnamed: 0.1', 'ID', 'Unnamed: 0', 'Created_At', 'Text'])

to_transform = to_transform.describe()

#Save results table for firm 2 for weekly documentation

to_transform.to_csv(Config.Week_signifier + ' - descriptive statistics for sentiment ' + Config.namer2 + '.csv')

#Add results for current week to firm 2 weekly sentiment series table

tweet_dataset_2 = pd.read_csv(Config.transformed_file_name2)

tweet_dataset_2 = tweet_dataset_2.set_index('Statistics')

tweet_dataset_2[Config.Week_signifier] = ""

tweet_dataset_2.loc['roberta_neg',Config.Week_signifier] = to_transform.loc['mean', 'roberta_neg']
tweet_dataset_2.loc['roberta_neu',Config.Week_signifier] = to_transform.loc['mean', 'roberta_neu']
tweet_dataset_2.loc['roberta_pos',Config.Week_signifier] = to_transform.loc['mean', 'roberta_pos']
tweet_dataset_2.loc['roberta_sum',Config.Week_signifier] = to_transform.loc['mean', 'roberta_sum']

if Config.i > 1:

    tweet_dataset_2.loc['sensitivity',Config.Week_signifier] = tweet_dataset_2.loc['roberta_sum', Config.Week_signifier] - tweet_dataset_2.loc['roberta_sum', Config.Week_signifier_previous]


# tweet_dataset_2 = tweet_dataset_2.drop(columns = 'Unnamed: 0')

#Overwrite firm 2 sentiment series table with new results

tweet_dataset_2.to_csv(Config.transformed_file_name2)


