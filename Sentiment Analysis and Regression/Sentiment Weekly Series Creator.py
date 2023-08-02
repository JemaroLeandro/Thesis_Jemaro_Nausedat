import pandas as pd
import numpy as np
import Config

tweet_weekly_series_creator = pd.DataFrame({'Statistics': ['roberta_neg', 'roberta_neu', 'roberta_pos', 'roberta_sum']})

tweet_weekly_series_creator = tweet_weekly_series_creator.set_index('Statistics')

tweet_weekly_series_creator = tweet_weekly_series_creator

# tweet_weekly_series_creator.drop(index='Unnamed: 0')

tweet_weekly_series_creator.to_csv(Config.transformed_file_name1)

tweet_weekly_series_creator.to_csv(Config.transformed_file_name2)