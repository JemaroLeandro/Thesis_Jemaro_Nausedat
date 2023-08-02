import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import numpy as np
import Config

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import numpy as np

#define ticker symbols for peer groups

peer_group_1 = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "INTC", "CSCO", "IBM", "NVDA", "ADBE", "005930.KS"]
peer_group_2 = ["ATVI", "EA", "TTWO", "UBI.PA", "NTDOY", "NEXOF", "NTES", "TCEHY", "SQNXF", "CCOEF", "NCBDF"]


# save list length for later iteration

pgl = len(peer_group_1)

# download peer groups financial data from yahoo finance

peer_group_1_full = yf.download(peer_group_1, period='1wk', start='2023-05-26',end='2023-07-31')

peer_group_2_full = yf.download(peer_group_2, period='1wk', start='2023-05-29',end='2023-07-31')


# reduce dataframes to adjusted close prices

peer_group_1_df = peer_group_1_full['Adj Close']

peer_group_2_df = peer_group_2_full['Adj Close']


#initial return calculation and dataframe setup for peer group 1

company = 'AAPL'

key_company1 = 'AAPL'

columngenerator = 'Daily Returns of ' + company
returnseries = peer_group_1_df[company]
returndataframe = returnseries.to_frame()
returndataframe.rename(columns={list(returndataframe)[0] : 'Adjusted Closing Price'}, inplace=True)
returndataframe[columngenerator] = (returndataframe['Adjusted Closing Price'] / returndataframe['Adjusted Closing Price'].shift(1)) - 1
key_results1 = (returndataframe['Adjusted Closing Price'] / returndataframe['Adjusted Closing Price'].shift(1)) - 1
key_results1 = key_results1.to_frame()
key_results1 = key_results1.rename(columns={'Adjusted Closing Price' : company + ' returns'})
returndataframe = returndataframe[columngenerator]
returndataframe = returndataframe.to_frame()
#returndataframe = returndataframe.reset_index().rename(columns={'index': 'Statistics'})


#define function for iteration of peer group 1

def returncalculation1(company):
    columngenerator = 'Daily Returns of ' + company
    returnseries1 = peer_group_1_df[company]
    returndataframe1 = returnseries1.to_frame()
    returndataframe1.rename(columns={list(returndataframe1)[0] : 'Adjusted Closing Price'}, inplace=True)
    returndataframe1[columngenerator] = (returndataframe1['Adjusted Closing Price'] / returndataframe1['Adjusted Closing Price'].shift(1)) - 1
    returndataframe1 = returndataframe1[columngenerator]
    returndataframe1 = returndataframe1.to_frame()
    #returndataframe1 = returndataframe1.reset_index().rename(columns={'index': 'Statistics'})
    return returndataframe1

#iterate through peer group 1

for i in range(pgl):

    company = peer_group_1[i-1]
    columngenerator = 'Daily Returns of ' + company
    returndataframe[columngenerator] = (returncalculation1(company))[columngenerator]

#store iteration results in dataframes for peer group 1 and transform them

peer_group_1_results = returndataframe

peer_group_1_summary = peer_group_1_results.T.describe()

peer_group_1_summary = peer_group_1_summary.T

peer_group_1_summary = peer_group_1_summary.rename(columns={ 'count' : 'Counted firm returns', 'mean' : 'Average peer group return', 'std' : 'Standard deviation of peer group return', 'min' : 'Minimum return', 'max' : 'Maximum return'})

peer_group_1_summary.index.rename('Day', inplace=True)

#calculate abnormal returns with group results

abnormal_returns1 = key_results1[key_company1 + ' returns'] - peer_group_1_summary['Average peer group return']

abnormal_returns1 = abnormal_returns1.to_frame()

abnormal_returns1 = abnormal_returns1.rename(columns={ list(abnormal_returns1)[0] : key_company1 + ' abnormal returns'})

#initial return calculation and dataframe setup for peer group 2s

company = 'ATVI'

key_company2 = 'ATVI'

columngenerator = 'Daily Returns of ' + company
returnseries = peer_group_2_df[company]
returndataframe = returnseries.to_frame()
returndataframe.rename(columns={list(returndataframe)[0] : 'Adjusted Closing Price'}, inplace=True)
returndataframe[columngenerator] = (returndataframe['Adjusted Closing Price'] / returndataframe['Adjusted Closing Price'].shift(1)) - 1
key_results2 = (returndataframe['Adjusted Closing Price'] / returndataframe['Adjusted Closing Price'].shift(1)) - 1
key_results2 = key_results2.to_frame()
key_results2 = key_results2.rename(columns={'Adjusted Closing Price' : company + ' returns'})
returndataframe = returndataframe[columngenerator]
returndataframe = returndataframe.to_frame()
#returndataframe = returndataframe.reset_index().rename(columns={'index': 'Statistics'})



#define function for iteration of peer group 2

def returncalculation2(company):
    columngenerator = 'Daily Returns of ' + company
    returnseries2 = peer_group_2_df[company]
    returndataframe2 = returnseries2.to_frame()
    returndataframe2.rename(columns={list(returndataframe2)[0] : 'Adjusted Closing Price'}, inplace=True)
    returndataframe2[columngenerator] = (returndataframe2['Adjusted Closing Price'] / returndataframe2['Adjusted Closing Price'].shift(1)) - 1
    returndataframe2 = returndataframe2[columngenerator]
    returndataframe2 = returndataframe2.to_frame()
    #returndataframe2 = returndataframe2.reset_index().rename(columns={'index': 'Statistics'})
    return returndataframe2

#iterate through peer group 2

for i in range(pgl):

    company = peer_group_2[i-1]
    columngenerator = 'Daily Returns of ' + company
    returndataframe[columngenerator] = (returncalculation2(company))[columngenerator]


#store iteration results in dataframes for peer group 2 and transform them

peer_group_2_results = returndataframe

peer_group_2_summary = peer_group_2_results.T.describe()

peer_group_2_summary = peer_group_2_summary.T

peer_group_2_summary = peer_group_2_summary.rename(columns={ 'count' : 'Counted firm returns', 'mean' : 'Average peer group return', 'std' : 'Standard deviation of peer group return', 'min' : 'Minimum return', 'max' : 'Maximum return'})

peer_group_2_summary.index.rename('Day', inplace=True)

#calculate abnormal returns with group results

abnormal_returns2 = key_results2[key_company2 + ' returns'] - peer_group_2_summary['Average peer group return']

abnormal_returns2 = abnormal_returns2.to_frame()

abnormal_returns2 = abnormal_returns2.rename(columns={ list(abnormal_returns2)[0] : key_company2 + ' abnormal returns'})


# define function to split daily dataframe data into weekly chunks

def split_dataframe(df, chunk_size = 10000): 
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

# split abnormal return table 1 into weeks

abnormal_returns_weekly1 = split_dataframe(abnormal_returns1, chunk_size=5)
returns_weekly1 = split_dataframe(key_results1, chunk_size=5)

#Note amount of weeks for downloaded sample

weeks = len(abnormal_returns_weekly1)

#define final dataframe 1 of stock analysis

abnormal_returns_finalized_1 = pd.DataFrame({'Statistics' : ['Weekly abnormal return', 'Average volatility', 'Weekly return']})
abnormal_returns_finalized_1 = abnormal_returns_finalized_1.set_index('Statistics')

# Iterate through every week and create final results table 1

for i in range(weeks):
    namenumeral = str(i+1)
    abnormal_returns_single_week = abnormal_returns_weekly1[i].describe()
    abnormal_returns_finalized_1['Week ' + namenumeral] = ""
    abnormal_returns_finalized_1.loc['Weekly abnormal return', ('Week ' + namenumeral)] = abnormal_returns_weekly1[i][key_company1 + ' abnormal returns'].sum()
    abnormal_returns_finalized_1.loc['Weekly return', ('Week ' + namenumeral)] = returns_weekly1[i][key_company1 + ' returns'].sum()
    abnormal_returns_finalized_1.loc['Average volatility', ('Week ' + namenumeral)] = abnormal_returns_single_week.loc['std', (key_company1 + ' abnormal returns')]


# split abnormal return table 2 into weeks

abnormal_returns_weekly2 = split_dataframe(abnormal_returns2, chunk_size=5)
returns_weekly2 = split_dataframe(key_results2, chunk_size=5)

#Note amount of weeks for downloaded sample

weeks = len(abnormal_returns_weekly2)

#define final dataframe 1 of stock analysis

abnormal_returns_finalized_2 = pd.DataFrame({'Statistics' : ['Weekly abnormal return', 'Average volatility']})
abnormal_returns_finalized_2 = abnormal_returns_finalized_2.set_index('Statistics')

# Iterate through every week and create final results table 1

for i in range(weeks):
    namenumeral = str(i+1)
    abnormal_returns_single_week = abnormal_returns_weekly2[i].describe()
    abnormal_returns_finalized_2['Week ' + namenumeral] = ""
    abnormal_returns_finalized_2.loc['Weekly abnormal return', ('Week ' + namenumeral)] = abnormal_returns_weekly2[i][key_company2 + ' abnormal returns'].sum()
    abnormal_returns_finalized_2.loc['Weekly return', ('Week ' + namenumeral)] = returns_weekly2[i][key_company2 + ' returns'].sum()
    abnormal_returns_finalized_2.loc['Average volatility', ('Week ' + namenumeral)] = abnormal_returns_single_week.loc['std', (key_company2 + ' abnormal returns')]

abnormal_returns_finalized_1.to_csv(Config.financial_file_1)

abnormal_returns_finalized_2.to_csv(Config.financial_file_2)

peer_group_1_summary.to_csv('Peer Group 1 Summary.csv')
peer_group_2_summary.to_csv('Peer Group 2 Summary.csv')

abnormal_returns1['AAPL returns'] = key_results1['AAPL returns']

abnormal_returns2['ATVI returns'] = key_results2['ATVI returns']

abnormal_returns1.to_csv('Apple Daily abnormal return series.csv')

abnormal_returns2.to_csv('Blizzard_Ent Daily abnormal return series.csv')