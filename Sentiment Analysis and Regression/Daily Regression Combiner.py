import pandas as pd
import numpy as np
import Config

from datetime import datetime


Weeks = 9



#transform date data attribute for uniform processing

for i in range(Weeks):
    weeknumeral = str(i+1)
    weeklydata = pd.read_csv('Week ' + weeknumeral + ' - Processed tweets addressed toward ' + Config.namer1 +'.csv')
    weeklydata['Created_At'] = weeklydata['Created_At'].str[:10]
    weeklydata = weeklydata[['Created_At','roberta_neg', 'roberta_neu','roberta_pos','roberta_sum']]
    weeklydata = weeklydata.set_index('Created_At')
    weeklydata.to_csv('Week ' + weeknumeral + ' - Processed tweets addressed toward ' + Config.namer1 + ' date cleansed.csv')

# get daily averages for roBERTa model variables

# for i in range(Weeks):
#     weeknumeral = str(i+1)
#     for index , row in financialdata.iterrows():
#         weeklydatafixed = pd.read_csv('Week ' + weeknumeral + ' - Processed tweets addressed toward ' + Config.namer1 + ' date cleansed.csv')
#         weeklydatafixed = weeklydatafixed.where(financialdata.loc[index, 'Date']==weeklydatafixed['Created_At'])
#         weeklydatafixed = weeklydatafixed.dropna()
#         roberta_neg = weeklydatafixed['roberta_neg'].mean()
#         roberta_neu = weeklydatafixed['roberta_neu'].mean()
#         roberta_pos = weeklydatafixed['roberta_pos'].mean()
#         roberta_sum = weeklydatafixed['roberta_sum'].mean()
#         financialdata.loc[index, 'daily_sentiment'] = roberta_sum
#         apple_daily_series.loc[index, 'apple_abnormal_returns'] = financialdata.loc[index, 'AAPL abnormal returns']
#         apple_daily_series.loc[index, 'Date'] = financialdata.loc[index, 'Date']
#         if weeklydatafixed.empty:
#             dummy = 1
#         else:
#             apple_daily_series.loc[index, 'daily_positivity'] = roberta_pos
#             apple_daily_series.loc[index, 'daily_negativity'] = roberta_neg
#             apple_daily_series.loc[index, 'daily_neutrality'] = roberta_neu
#             apple_daily_series.loc[index, 'daily_sentiment'] = roberta_sum




#transform date data attribute for uniform processing

for i in range(Weeks):
    weeknumeral = str(i+1)
    weeklydata = pd.read_csv('Week ' + weeknumeral + ' - Processed tweets addressed toward ' + Config.namer2 +'.csv')
    weeklydata['Created_At'] = weeklydata['Created_At'].str[:10]
    weeklydata = weeklydata[['Created_At','roberta_neg', 'roberta_neu','roberta_pos','roberta_sum']]
    weeklydata = weeklydata.set_index('Created_At')
    weeklydata.to_csv('Week ' + weeknumeral + ' - Processed tweets addressed toward ' + Config.namer2 + ' date cleansed.csv')

# get daily averages for roBERTa model variables

# for i in range(Weeks):
#     weeknumeral = str(i+1)
#     for index , row in financialdata.iterrows():
#         weeklydatafixed = pd.read_csv('Week ' + weeknumeral + ' - Processed tweets addressed toward ' + Config.namer2 + ' date cleansed.csv')
#         weeklydatafixed = weeklydatafixed.where(financialdata.loc[index, 'Date']==weeklydatafixed['Created_At'])
#         weeklydatafixed = weeklydatafixed.dropna()
#         roberta_neg = weeklydatafixed['roberta_neg'].mean()
#         roberta_neu = weeklydatafixed['roberta_neu'].mean()
#         roberta_pos = weeklydatafixed['roberta_pos'].mean()
#         roberta_sum = weeklydatafixed['roberta_sum'].mean()
#         financialdata.loc[index, 'daily_sentiment'] = roberta_sum
#         blizzard_ent_daily_series.loc[index, 'blizzard_ent_abnormal_returns'] = financialdata.loc[index, 'ATVI abnormal returns']
#         blizzard_ent_daily_series.loc[index, 'Date'] = financialdata.loc[index, 'Date']
#         if weeklydatafixed.empty:
#             dummy = 1
#         else:
#             blizzard_ent_daily_series.loc[index, 'daily_positivity'] = roberta_pos
#             blizzard_ent_daily_series.loc[index, 'daily_negativity'] = roberta_neg
#             blizzard_ent_daily_series.loc[index, 'daily_neutrality'] = roberta_neu
#             blizzard_ent_daily_series.loc[index, 'daily_sentiment'] = roberta_sum


# Split into chunks




def split_dataframe(df, chunk_size = 10000): 
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

#Load Apple daily data

financialdata = pd.read_csv('Apple Daily abnormal return series.csv')
#financialdata = financialdata.dropna()
financialdata.rename(columns={'AAPL abnormal returns' : 'Apple_abnormal_returns'})

apple_daily_series = pd.DataFrame(columns=['Date','apple_daily_abnormal_return','daily_roberta_sum'])
apple_daily_series.reset_index()

for i in range(Weeks):
    weeknumeral = str(i+1)
    day1 = (i * 5) + 1
    day2 = (i * 5) + 2
    day3 = (i * 5) + 3
    day4 = (i * 5) + 4
    day5 = (i * 5) + 5
    weeklydatafixed = pd.read_csv('Week ' + weeknumeral + ' - Processed tweets addressed toward ' + Config.namer1 + ' date cleansed.csv')
    weeklydatachunked = split_dataframe(weeklydatafixed, chunk_size=250)
    #sentiment
    apple_daily_series.loc[day1, 'daily_roberta_sum'] = weeklydatachunked[0]['roberta_sum'].mean()
    apple_daily_series.loc[day2, 'daily_roberta_sum'] = weeklydatachunked[1]['roberta_sum'].mean()
    apple_daily_series.loc[day3, 'daily_roberta_sum'] = weeklydatachunked[2]['roberta_sum'].mean()
    apple_daily_series.loc[day4, 'daily_roberta_sum'] = weeklydatachunked[3]['roberta_sum'].mean()
    apple_daily_series.loc[day5, 'daily_roberta_sum'] = weeklydatachunked[4]['roberta_sum'].mean()
    #negativity
    apple_daily_series.loc[day1, 'daily_roberta_neg'] = weeklydatachunked[0]['roberta_neg'].mean()
    apple_daily_series.loc[day2, 'daily_roberta_neg'] = weeklydatachunked[1]['roberta_neg'].mean()
    apple_daily_series.loc[day3, 'daily_roberta_neg'] = weeklydatachunked[2]['roberta_neg'].mean()
    apple_daily_series.loc[day4, 'daily_roberta_neg'] = weeklydatachunked[3]['roberta_neg'].mean()
    apple_daily_series.loc[day5, 'daily_roberta_neg'] = weeklydatachunked[4]['roberta_neg'].mean()
    #positivity
    apple_daily_series.loc[day1, 'daily_roberta_pos'] = weeklydatachunked[0]['roberta_pos'].mean()
    apple_daily_series.loc[day2, 'daily_roberta_pos'] = weeklydatachunked[1]['roberta_pos'].mean()
    apple_daily_series.loc[day3, 'daily_roberta_pos'] = weeklydatachunked[2]['roberta_pos'].mean()
    apple_daily_series.loc[day4, 'daily_roberta_pos'] = weeklydatachunked[3]['roberta_pos'].mean()
    apple_daily_series.loc[day5, 'daily_roberta_pos'] = weeklydatachunked[4]['roberta_pos'].mean()
    #neutrality
    apple_daily_series.loc[day1, 'daily_roberta_neu'] = weeklydatachunked[0]['roberta_neu'].mean()
    apple_daily_series.loc[day2, 'daily_roberta_neu'] = weeklydatachunked[1]['roberta_neu'].mean()
    apple_daily_series.loc[day3, 'daily_roberta_neu'] = weeklydatachunked[2]['roberta_neu'].mean()
    apple_daily_series.loc[day4, 'daily_roberta_neu'] = weeklydatachunked[3]['roberta_neu'].mean()
    apple_daily_series.loc[day5, 'daily_roberta_neu'] = weeklydatachunked[4]['roberta_neu'].mean()
    #Dates
    apple_daily_series.loc[day1, 'Date'] = financialdata.loc[day1-1, 'Date']
    apple_daily_series.loc[day2, 'Date'] = financialdata.loc[day2-1, 'Date']
    apple_daily_series.loc[day3, 'Date'] = financialdata.loc[day3-1, 'Date']
    apple_daily_series.loc[day4, 'Date'] = financialdata.loc[day4-1, 'Date']
    apple_daily_series.loc[day5, 'Date'] = financialdata.loc[day5-1, 'Date']
    #abnormal returns
    apple_daily_series.loc[day1, 'apple_daily_abnormal_return'] = financialdata.loc[day1-1, 'AAPL abnormal returns']
    apple_daily_series.loc[day2, 'apple_daily_abnormal_return'] = financialdata.loc[day2-1, 'AAPL abnormal returns']
    apple_daily_series.loc[day3, 'apple_daily_abnormal_return'] = financialdata.loc[day3-1, 'AAPL abnormal returns']
    apple_daily_series.loc[day4, 'apple_daily_abnormal_return'] = financialdata.loc[day4-1, 'AAPL abnormal returns']
    apple_daily_series.loc[day5, 'apple_daily_abnormal_return'] = financialdata.loc[day5-1, 'AAPL abnormal returns']
    #returns
    apple_daily_series.loc[day1, 'apple_daily_return'] = financialdata.loc[day1-1, 'AAPL returns']
    apple_daily_series.loc[day2, 'apple_daily_return'] = financialdata.loc[day2-1, 'AAPL returns']
    apple_daily_series.loc[day3, 'apple_daily_return'] = financialdata.loc[day3-1, 'AAPL returns']
    apple_daily_series.loc[day4, 'apple_daily_return'] = financialdata.loc[day4-1, 'AAPL returns']
    apple_daily_series.loc[day5, 'apple_daily_return'] = financialdata.loc[day5-1, 'AAPL returns']
    if day1 > 1:
        apple_daily_series.loc[day1, 'daily_sensitivity'] = apple_daily_series.loc[day1, 'daily_roberta_sum'] - apple_daily_series.loc[day1-1, 'daily_roberta_sum']
        apple_daily_series.loc[day2, 'daily_sensitivity'] = apple_daily_series.loc[day2, 'daily_roberta_sum'] - apple_daily_series.loc[day2-1, 'daily_roberta_sum']
        apple_daily_series.loc[day3, 'daily_sensitivity'] = apple_daily_series.loc[day3, 'daily_roberta_sum'] - apple_daily_series.loc[day3-1, 'daily_roberta_sum']
        apple_daily_series.loc[day4, 'daily_sensitivity'] = apple_daily_series.loc[day4, 'daily_roberta_sum'] - apple_daily_series.loc[day4-1, 'daily_roberta_sum']
        apple_daily_series.loc[day5, 'daily_sensitivity'] = apple_daily_series.loc[day5, 'daily_roberta_sum'] - apple_daily_series.loc[day5-1, 'daily_roberta_sum']
    else:
        apple_daily_series.loc[day2, 'daily_sensitivity'] = apple_daily_series.loc[day2, 'daily_roberta_sum'] - apple_daily_series.loc[day2-1, 'daily_roberta_sum']
        apple_daily_series.loc[day3, 'daily_sensitivity'] = apple_daily_series.loc[day3, 'daily_roberta_sum'] - apple_daily_series.loc[day3-1, 'daily_roberta_sum']
        apple_daily_series.loc[day4, 'daily_sensitivity'] = apple_daily_series.loc[day4, 'daily_roberta_sum'] - apple_daily_series.loc[day4-1, 'daily_roberta_sum']
        apple_daily_series.loc[day5, 'daily_sensitivity'] = apple_daily_series.loc[day5, 'daily_roberta_sum'] - apple_daily_series.loc[day5-1, 'daily_roberta_sum']
    

#Load Blizzard_ent daily data

financialdata = pd.read_csv('Blizzard_Ent Daily abnormal return series.csv')
# financialdata = financialdata.dropna()
financialdata.rename(columns={'ATVI abnormal returns' : 'blizzard_ent_abnormal_returns'})


blizzard_ent_daily_series = pd.DataFrame(columns=['Date','blizzard_ent_daily_abnormal_return','daily_roberta_sum'])
blizzard_ent_daily_series.reset_index()


for i in range(Weeks):
    weeknumeral = str(i+1)
    day1 = (i * 5) + 1
    day2 = (i * 5) + 2
    day3 = (i * 5) + 3
    day4 = (i * 5) + 4
    day5 = (i * 5) + 5
    weeklydatafixed = pd.read_csv('Week ' + weeknumeral + ' - Processed tweets addressed toward ' + Config.namer2 + ' date cleansed.csv')
    weeklydatachunked = split_dataframe(weeklydatafixed, chunk_size=250)
    #sentiment
    blizzard_ent_daily_series.loc[day1, 'daily_roberta_sum'] = weeklydatachunked[0]['roberta_sum'].mean()
    blizzard_ent_daily_series.loc[day2, 'daily_roberta_sum'] = weeklydatachunked[1]['roberta_sum'].mean()
    blizzard_ent_daily_series.loc[day3, 'daily_roberta_sum'] = weeklydatachunked[2]['roberta_sum'].mean()
    blizzard_ent_daily_series.loc[day4, 'daily_roberta_sum'] = weeklydatachunked[3]['roberta_sum'].mean()
    blizzard_ent_daily_series.loc[day5, 'daily_roberta_sum'] = weeklydatachunked[4]['roberta_sum'].mean()
    #negativity
    blizzard_ent_daily_series.loc[day1, 'daily_roberta_neg'] = weeklydatachunked[0]['roberta_neg'].mean()
    blizzard_ent_daily_series.loc[day2, 'daily_roberta_neg'] = weeklydatachunked[1]['roberta_neg'].mean()
    blizzard_ent_daily_series.loc[day3, 'daily_roberta_neg'] = weeklydatachunked[2]['roberta_neg'].mean()
    blizzard_ent_daily_series.loc[day4, 'daily_roberta_neg'] = weeklydatachunked[3]['roberta_neg'].mean()
    blizzard_ent_daily_series.loc[day5, 'daily_roberta_neg'] = weeklydatachunked[4]['roberta_neg'].mean()
    #positivity
    blizzard_ent_daily_series.loc[day1, 'daily_roberta_pos'] = weeklydatachunked[0]['roberta_pos'].mean()
    blizzard_ent_daily_series.loc[day2, 'daily_roberta_pos'] = weeklydatachunked[1]['roberta_pos'].mean()
    blizzard_ent_daily_series.loc[day3, 'daily_roberta_pos'] = weeklydatachunked[2]['roberta_pos'].mean()
    blizzard_ent_daily_series.loc[day4, 'daily_roberta_pos'] = weeklydatachunked[3]['roberta_pos'].mean()
    blizzard_ent_daily_series.loc[day5, 'daily_roberta_pos'] = weeklydatachunked[4]['roberta_pos'].mean()
    #neutrality
    blizzard_ent_daily_series.loc[day1, 'daily_roberta_neu'] = weeklydatachunked[0]['roberta_neu'].mean()
    blizzard_ent_daily_series.loc[day2, 'daily_roberta_neu'] = weeklydatachunked[1]['roberta_neu'].mean()
    blizzard_ent_daily_series.loc[day3, 'daily_roberta_neu'] = weeklydatachunked[2]['roberta_neu'].mean()
    blizzard_ent_daily_series.loc[day4, 'daily_roberta_neu'] = weeklydatachunked[3]['roberta_neu'].mean()
    blizzard_ent_daily_series.loc[day5, 'daily_roberta_neu'] = weeklydatachunked[4]['roberta_neu'].mean()
    #Dates
    blizzard_ent_daily_series.loc[day1, 'Date'] = financialdata.loc[day1-1, 'Date']
    blizzard_ent_daily_series.loc[day2, 'Date'] = financialdata.loc[day2-1, 'Date']
    blizzard_ent_daily_series.loc[day3, 'Date'] = financialdata.loc[day3-1, 'Date']
    blizzard_ent_daily_series.loc[day4, 'Date'] = financialdata.loc[day4-1, 'Date']
    blizzard_ent_daily_series.loc[day5, 'Date'] = financialdata.loc[day5-1, 'Date']
    #abnormal returns
    blizzard_ent_daily_series.loc[day1, 'blizzard_ent_daily_abnormal_return'] = financialdata.loc[day1-1, 'ATVI abnormal returns']
    blizzard_ent_daily_series.loc[day2, 'blizzard_ent_daily_abnormal_return'] = financialdata.loc[day2-1, 'ATVI abnormal returns']
    blizzard_ent_daily_series.loc[day3, 'blizzard_ent_daily_abnormal_return'] = financialdata.loc[day3-1, 'ATVI abnormal returns']
    blizzard_ent_daily_series.loc[day4, 'blizzard_ent_daily_abnormal_return'] = financialdata.loc[day4-1, 'ATVI abnormal returns']
    blizzard_ent_daily_series.loc[day5, 'blizzard_ent_daily_abnormal_return'] = financialdata.loc[day5-1, 'ATVI abnormal returns']
    #returns
    blizzard_ent_daily_series.loc[day1, 'blizzard_ent_daily_return'] = financialdata.loc[day1-1, 'ATVI returns']
    blizzard_ent_daily_series.loc[day2, 'blizzard_ent_daily_return'] = financialdata.loc[day2-1, 'ATVI returns']
    blizzard_ent_daily_series.loc[day3, 'blizzard_ent_daily_return'] = financialdata.loc[day3-1, 'ATVI returns']
    blizzard_ent_daily_series.loc[day4, 'blizzard_ent_daily_return'] = financialdata.loc[day4-1, 'ATVI returns']
    blizzard_ent_daily_series.loc[day5, 'blizzard_ent_daily_return'] = financialdata.loc[day5-1, 'ATVI returns']
    if day1 > 1:
        blizzard_ent_daily_series.loc[day1, 'daily_sensitivity'] = blizzard_ent_daily_series.loc[day1, 'daily_roberta_sum'] - blizzard_ent_daily_series.loc[day1-1, 'daily_roberta_sum']
        blizzard_ent_daily_series.loc[day2, 'daily_sensitivity'] = blizzard_ent_daily_series.loc[day2, 'daily_roberta_sum'] - blizzard_ent_daily_series.loc[day2-1, 'daily_roberta_sum']
        blizzard_ent_daily_series.loc[day3, 'daily_sensitivity'] = blizzard_ent_daily_series.loc[day3, 'daily_roberta_sum'] - blizzard_ent_daily_series.loc[day3-1, 'daily_roberta_sum']
        blizzard_ent_daily_series.loc[day4, 'daily_sensitivity'] = blizzard_ent_daily_series.loc[day4, 'daily_roberta_sum'] - blizzard_ent_daily_series.loc[day4-1, 'daily_roberta_sum']
        blizzard_ent_daily_series.loc[day5, 'daily_sensitivity'] = blizzard_ent_daily_series.loc[day5, 'daily_roberta_sum'] - blizzard_ent_daily_series.loc[day5-1, 'daily_roberta_sum']
    else:
        blizzard_ent_daily_series.loc[day2, 'daily_sensitivity'] = blizzard_ent_daily_series.loc[day2, 'daily_roberta_sum'] - blizzard_ent_daily_series.loc[day2-1, 'daily_roberta_sum']
        blizzard_ent_daily_series.loc[day3, 'daily_sensitivity'] = blizzard_ent_daily_series.loc[day3, 'daily_roberta_sum'] - blizzard_ent_daily_series.loc[day3-1, 'daily_roberta_sum']
        blizzard_ent_daily_series.loc[day4, 'daily_sensitivity'] = blizzard_ent_daily_series.loc[day4, 'daily_roberta_sum'] - blizzard_ent_daily_series.loc[day4-1, 'daily_roberta_sum']
        blizzard_ent_daily_series.loc[day5, 'daily_sensitivity'] = blizzard_ent_daily_series.loc[day5, 'daily_roberta_sum'] - blizzard_ent_daily_series.loc[day5-1, 'daily_roberta_sum']
    
#Build CSV files

apple_daily_series.set_index('Date')

apple_daily_series = apple_daily_series.dropna()

apple_daily_series.to_csv('Apple Daily Regression Dataset.csv')

blizzard_ent_daily_series = blizzard_ent_daily_series.dropna()

blizzard_ent_daily_series.set_index('Date')

blizzard_ent_daily_series.to_csv('Blizzard_Ent Daily Regression Dataset.csv')