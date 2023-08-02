import pandas as pd
import numpy as np
import Config

from datetime import datetime

today = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

Weeks = 9



#Load sentiment table for Apple and add financial returns

file_baseline_1 = pd.read_csv(Config.transformed_file_name1)

file_baseline_1 = file_baseline_1.set_index('Statistics')

file_financials_1 = pd.read_csv(Config.financial_file_1)

file_financials_1 = file_financials_1.set_index('Statistics')

#Append baseline file for final dataset for regression

for i in range(Weeks):
    w = str(i+1)
    w_lag = str(i)
    file_baseline_1.loc['apple_weekly_abnormal_return',('Week ' + w)] = file_financials_1.loc['Weekly abnormal return',('Week ' + w)]
    file_baseline_1.loc['apple_average_volatility',('Week ' + w)] = file_financials_1.loc['Average volatility',('Week ' + w)]
    file_baseline_1.loc['apple_weekly_return',('Week ' + w)] = file_financials_1.loc['Weekly return',('Week ' + w)]
    if i > 0 :
        file_baseline_1.loc['roberta_sum_lag_1',('Week ' + w)] = file_baseline_1.loc['roberta_sum',('Week ' + w)] - file_baseline_1.loc['roberta_sum',('Week ' + (w_lag))]
    if i > 1 : 
        file_baseline_1.loc['sensitivity_lag_1',('Week ' + w)] = file_baseline_1.loc['sensitivity',('Week ' + w)] - file_baseline_1.loc['sensitivity',('Week ' + (w_lag))]



    

#Generate final csv file

file_baseline_1.T.to_csv(Config.regression_ready_1)



#Load sentiment table for Activision Blizzard and add financial returns

file_baseline_2 = pd.read_csv(Config.transformed_file_name2)

file_baseline_2 = file_baseline_2.set_index('Statistics')

file_financials_2 = pd.read_csv(Config.financial_file_2)

file_financials_2 = file_financials_2.set_index('Statistics')

#Append baseline file for final dataset for regression

for i in range(Weeks):
    w = str(i+1)
    w_lag = str(i)
    file_baseline_2.loc['weekly_abnormal_return',('Week ' + w)] = file_financials_2.loc['Weekly abnormal return',('Week ' + w)]
    file_baseline_2.loc['average_volatility',('Week ' + w)] = file_financials_2.loc['Average volatility',('Week ' + w)]
    file_baseline_2.loc['weekly_return',('Week ' + w)] = file_financials_2.loc['Weekly return',('Week ' + w)]
    if i > 0 :
        file_baseline_2.loc['roberta_sum_lag_1',('Week ' + w)] = file_baseline_2.loc['roberta_sum',('Week ' + w)] - file_baseline_2.loc['roberta_sum',('Week ' + (w_lag))]
    if i > 1: 
        file_baseline_2.loc['sensitivity_lag_1',('Week ' + w)] = file_baseline_2.loc['sensitivity',('Week ' + w)] - file_baseline_2.loc['sensitivity',('Week ' + (w_lag))]



#Generate final csv file

file_baseline_2.T.to_csv(Config.regression_ready_2)