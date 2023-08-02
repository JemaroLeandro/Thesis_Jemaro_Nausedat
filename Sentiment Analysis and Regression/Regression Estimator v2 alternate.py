import pandas as pd
import numpy as np
import Config
import seaborn as sns
import matplotlib as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tools.tools as smt
import statsmodels.stats.diagnostic as smd

# Set up dataset and model for regression for Apple

apple_data_weekly = pd.read_csv(
    'Adjusted Regression Datasets - Apple Regression Dataset.csv')

apple_data_daily = pd.read_csv(
    'Adjusted Regression Datasets - Apple Daily Regression Dataset.csv')

# Models based on daily data

apple_model_linear_sentiment_return_daily = smf.ols(
    formula="apple_daily_abnormal_return ~ daily_roberta_sum", data=apple_data_daily).fit()

apple_model_linear_positivity_return_daily = smf.ols(
    formula="apple_daily_abnormal_return ~ daily_roberta_pos", data=apple_data_daily).fit()

apple_model_linear_negativity_return_daily = smf.ols(
    formula="apple_daily_abnormal_return ~ daily_roberta_neg", data=apple_data_daily).fit()

apple_model_linear_sensitivity_return_daily = smf.ols(
    formula="apple_daily_abnormal_return ~ daily_sensitivity", data=apple_data_daily).fit()


apple_model_linear_sentiment_lagged_return_daily = smf.ols(
    formula="apple_daily_abnormal_return ~ daily_roberta_sum + daily_roberta_sum_lag_1", data=apple_data_daily[1:]).fit()

apple_model_linear_sentiment_lagged_return_daily_sp500 = smf.ols(
    formula="apple_daily_abnormal_return ~ daily_roberta_sum + daily_roberta_sum_lag_1", data=apple_data_daily[1:]).fit()

apple_model_linear_positivity_lagged_return_daily = smf.ols(
    formula="apple_daily_abnormal_return ~ daily_roberta_pos + daily_roberta_pos_lag_1", data=apple_data_daily[1:]).fit()

apple_model_linear_negativity_lagged_return_daily = smf.ols(
    formula="apple_daily_abnormal_return ~ daily_roberta_neg + daily_roberta_neg_lag_1", data=apple_data_daily[1:]).fit()

apple_model_linear_sensitivity_lagged_return_daily = smf.ols(
    formula="apple_daily_abnormal_return ~ daily_sensitivity + daily_sensitivity_lag_1 ", data=apple_data_daily[1:]).fit()


apple_model_linear_sentiment_norm_daily = smf.ols(
    formula="apple_daily_return ~ daily_roberta_sum + sp500_daily_returns", data=apple_data_daily).fit()

apple_model_linear_positivity_norm_daily = smf.ols(
    formula="apple_daily_return ~ daily_roberta_pos + sp500_daily_returns", data=apple_data_daily).fit()

apple_model_linear_negativity_norm_daily = smf.ols(
    formula="apple_daily_return ~ daily_roberta_neg + sp500_daily_returns", data=apple_data_daily).fit()

apple_model_linear_sensitivity_norm_daily = smf.ols(
    formula="apple_daily_return ~ daily_sensitivity + sp500_daily_returns", data=apple_data_daily).fit()


apple_model_linear_sentiment_lagged_norm_daily = smf.ols(
    formula="apple_daily_return ~ daily_roberta_sum + daily_roberta_sum_lag_1 + sp500_daily_returns", data=apple_data_daily[1:]).fit()

apple_model_linear_positivity_lagged_norm_daily = smf.ols(
    formula="apple_daily_return ~ daily_roberta_pos + daily_roberta_pos_lag_1 + sp500_daily_returns", data=apple_data_daily[1:]).fit()

apple_model_linear_negativity_lagged_norm_daily = smf.ols(
    formula="apple_daily_return ~ daily_roberta_neg + daily_roberta_neg_lag_1 + sp500_daily_returns", data=apple_data_daily[1:]).fit()

apple_model_linear_sensitivity_lagged_norm_daily = smf.ols(
    formula="apple_daily_return ~ daily_sensitivity + daily_sensitivity_lag_1 + sp500_daily_returns", data=apple_data_daily[1:]).fit()


# ---sinh

apple_model_sinh_sentiment_return_daily = smf.ols(
    formula="apple_daily_abnormal_return ~ np.sinh(daily_roberta_sum)", data=apple_data_daily).fit()

apple_model_sinh_positivity_return_daily = smf.ols(
    formula="apple_daily_abnormal_return ~ np.sinh(daily_roberta_pos)", data=apple_data_daily).fit()

apple_model_sinh_negativity_return_daily = smf.ols(
    formula="apple_daily_abnormal_return ~ np.sinh(daily_roberta_neg)", data=apple_data_daily).fit()

apple_model_sinh_sensitivity_return_daily = smf.ols(
    formula="apple_daily_abnormal_return ~ np.sinh(daily_sensitivity)", data=apple_data_daily).fit()

apple_model_sinh_sentiment_lagged_return_daily = smf.ols(
    formula="apple_daily_abnormal_return ~ np.sinh(daily_roberta_sum) + np.sinh(daily_roberta_sum_lag_1)", data=apple_data_daily[1:]).fit()

apple_model_sinh_positivity_lagged_return_daily = smf.ols(
    formula="apple_daily_abnormal_return ~ np.sinh(daily_roberta_pos)+ np.sinh(daily_roberta_pos_lag_1)", data=apple_data_daily[1:]).fit()

apple_model_sinh_negativity_lagged_return_daily = smf.ols(
    formula="apple_daily_abnormal_return ~ np.sinh(daily_roberta_neg)+ np.sinh(daily_roberta_neg_lag_1)", data=apple_data_daily[1:]).fit()

apple_model_sinh_sensitivity_lagged_return_daily = smf.ols(
    formula="apple_daily_abnormal_return ~ np.sinh(daily_sensitivity)+ np.sinh(daily_sensitivity_lag_1)", data=apple_data_daily[1:]).fit()


apple_model_sinh_sentiment_norm_daily = smf.ols(
    formula="apple_daily_return ~ np.sinh(daily_roberta_sum) + np.sinh(sp500_daily_returns)", data=apple_data_daily).fit()

apple_model_sinh_positivity_norm_daily = smf.ols(
    formula="apple_daily_return ~ np.sinh(daily_roberta_pos) + np.sinh(sp500_daily_returns)", data=apple_data_daily).fit()

apple_model_sinh_negativity_norm_daily = smf.ols(
    formula="apple_daily_return ~ np.sinh(daily_roberta_neg) + np.sinh(sp500_daily_returns)", data=apple_data_daily).fit()

apple_model_sinh_sensitivity_norm_daily = smf.ols(
    formula="apple_daily_return ~ np.sinh(daily_sensitivity) + np.sinh(sp500_daily_returns)", data=apple_data_daily).fit()

apple_model_sinh_sentiment_lagged_norm_daily = smf.ols(
    formula="apple_daily_return ~ np.sinh(daily_roberta_sum) + np.sinh(daily_roberta_sum_lag_1) + np.sinh(sp500_daily_returns)", data=apple_data_daily[1:]).fit()

apple_model_sinh_positivity_lagged_norm_daily = smf.ols(
    formula="apple_daily_return ~ np.sinh(daily_roberta_pos)+ np.sinh(daily_roberta_pos_lag_1) + np.sinh(sp500_daily_returns)", data=apple_data_daily[1:]).fit()

apple_model_sinh_negativity_lagged_norm_daily = smf.ols(
    formula="apple_daily_return ~ np.sinh(daily_roberta_neg)+ np.sinh(daily_roberta_neg_lag_1) + np.sinh(sp500_daily_returns)", data=apple_data_daily[1:]).fit()

apple_model_sinh_sensitivity_lagged_norm_daily = smf.ols(
    formula="apple_daily_return ~ np.sinh(daily_sensitivity)+ np.sinh(daily_sensitivity_lag_1) + np.sinh(sp500_daily_returns)", data=apple_data_daily[1:]).fit()

# ---log


apple_model_log_positivity_return_daily = smf.ols(
    formula="apple_daily_abnormal_return ~ np.log(daily_roberta_pos)", data=apple_data_daily).fit()

apple_model_log_negativity_return_daily = smf.ols(
    formula="apple_daily_abnormal_return ~ np.log(daily_roberta_neg)", data=apple_data_daily).fit()

apple_model_log_positivity_lagged_return_daily = smf.ols(
    formula="apple_daily_abnormal_return ~ np.log(daily_roberta_pos)+ np.log(daily_roberta_pos_lag_1)", data=apple_data_daily[1:]).fit()

apple_model_log_negativity_lagged_return_daily = smf.ols(
    formula="apple_daily_abnormal_return ~ np.log(daily_roberta_neg)+ np.log(daily_roberta_neg_lag_1)", data=apple_data_daily[1:]).fit()


apple_model_log_positivity_norm_daily = smf.ols(
    formula="apple_daily_return ~ np.log(daily_roberta_pos) + np.log(sp500_daily_returns)", data=apple_data_daily).fit()

apple_model_log_negativity_norm_daily = smf.ols(
    formula="apple_daily_return ~ np.log(daily_roberta_neg) + np.log(sp500_daily_returns)", data=apple_data_daily).fit()

apple_model_log_positivity_lagged_norm_daily = smf.ols(
    formula="apple_daily_return ~ np.log(daily_roberta_pos)+ np.log(daily_roberta_pos_lag_1) + np.log(sp500_daily_returns)", data=apple_data_daily[1:]).fit()

apple_model_log_negativity_lagged_norm_daily = smf.ols(
    formula="apple_daily_return ~ np.log(daily_roberta_neg)+ np.log(daily_roberta_neg_lag_1) + np.log(sp500_daily_returns)", data=apple_data_daily[1:]).fit()

# ---sqrt


apple_model_sqrt_positivity_return_daily = smf.ols(
    formula="apple_daily_abnormal_return ~ np.sqrt(daily_roberta_pos)", data=apple_data_daily).fit()

apple_model_sqrt_negativity_return_daily = smf.ols(
    formula="apple_daily_abnormal_return ~ np.sqrt(daily_roberta_neg)", data=apple_data_daily).fit()

apple_model_sqrt_positivity_lagged_return_daily = smf.ols(
    formula="apple_daily_abnormal_return ~ np.sqrt(daily_roberta_pos)+ np.sqrt(daily_roberta_pos_lag_1)", data=apple_data_daily[1:]).fit()

apple_model_sqrt_negativity_lagged_return_daily = smf.ols(
    formula="apple_daily_abnormal_return ~ np.sqrt(daily_roberta_neg)+ np.sqrt(daily_roberta_neg_lag_1)", data=apple_data_daily[1:]).fit()


apple_model_sqrt_positivity_norm_daily = smf.ols(
    formula="apple_daily_return ~ np.sqrt(daily_roberta_pos) + np.sqrt(sp500_daily_returns)", data=apple_data_daily).fit()

apple_model_sqrt_negativity_norm_daily = smf.ols(
    formula="apple_daily_return ~ np.sqrt(daily_roberta_neg) + np.sqrt(sp500_daily_returns)", data=apple_data_daily).fit()

apple_model_sqrt_positivity_lagged_norm_daily = smf.ols(
    formula="apple_daily_return ~ np.sqrt(daily_roberta_pos)+ np.sqrt(daily_roberta_pos_lag_1) + np.sqrt(sp500_daily_returns)", data=apple_data_daily[1:]).fit()

apple_model_sqrt_negativity_lagged_norm_daily = smf.ols(
    formula="apple_daily_return ~ np.sqrt(daily_roberta_neg)+ np.sqrt(daily_roberta_neg_lag_1) + np.sqrt(sp500_daily_returns)", data=apple_data_daily[1:]).fit()


# Models based on weekly data:

apple_model_linear_sentiment_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ roberta_sum", data=apple_data_weekly).fit()

apple_model_linear_positivity_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ roberta_pos", data=apple_data_weekly).fit()

apple_model_linear_negativity_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ roberta_neg", data=apple_data_weekly).fit()

apple_model_linear_sensitivity_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ sensitivity", data=apple_data_weekly[1:]).fit()

apple_model_linear_sentiment_lagged_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ roberta_sum + roberta_sum_lag_1", data=apple_data_weekly[1:]).fit()

apple_model_linear_positivity_lagged_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ roberta_pos + roberta_pos_lag_1", data=apple_data_weekly[1:]).fit()

apple_model_linear_negativity_lagged_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ roberta_neg + roberta_neg_lag_1", data=apple_data_weekly[1:]).fit()

apple_model_linear_sensitivity_lagged_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ sensitivity + sensitivity_lag_1", data=apple_data_weekly[2:]).fit()


apple_model_linear_sentiment_volatility_weekly = smf.ols(
    formula="average_volatility ~ roberta_sum", data=apple_data_weekly).fit()

apple_model_linear_positivity_volatility_weekly = smf.ols(
    formula="average_volatility ~ roberta_pos", data=apple_data_weekly).fit()

apple_model_linear_negativity_volatility_weekly = smf.ols(
    formula="average_volatility ~ roberta_neg", data=apple_data_weekly).fit()

apple_model_linear_sensitivity_volatility_weekly = smf.ols(
    formula="average_volatility ~ sensitivity", data=apple_data_weekly[1:]).fit()

apple_model_linear_sentiment_lagged_volatility_weekly = smf.ols(
    formula="average_volatility ~ roberta_sum + roberta_sum_lag_1", data=apple_data_weekly[1:]).fit()

apple_model_linear_positivity_lagged_volatility_weekly = smf.ols(
    formula="average_volatility ~ roberta_pos + roberta_pos_lag_1", data=apple_data_weekly[1:]).fit()

apple_model_linear_negativity_lagged_volatility_weekly = smf.ols(
    formula="average_volatility ~ roberta_neg + roberta_neg_lag_1", data=apple_data_weekly[1:]).fit()

apple_model_linear_sensitivity_lagged_volatility_weekly = smf.ols(
    formula="average_volatility ~ sensitivity + sensitivity_lag_1", data=apple_data_weekly[2:]).fit()


apple_model_linear_sentiment_norm_weekly = smf.ols(
    formula="weekly_return ~ roberta_sum + sp500_weekly_returns_alternate", data=apple_data_weekly).fit()

apple_model_linear_positivity_norm_weekly = smf.ols(
    formula="weekly_return ~ roberta_pos + sp500_weekly_returns_alternate", data=apple_data_weekly).fit()

apple_model_linear_negativity_norm_weekly = smf.ols(
    formula="weekly_return ~ roberta_neg + sp500_weekly_returns_alternate", data=apple_data_weekly).fit()

apple_model_linear_sensitivity_norm_weekly = smf.ols(
    formula="weekly_return ~ sensitivity + sp500_weekly_returns_alternate", data=apple_data_weekly[1:]).fit()

apple_model_linear_sentiment_lagged_norm_weekly = smf.ols(
    formula="weekly_return ~ roberta_sum + roberta_sum_lag_1 + sp500_weekly_returns_alternate", data=apple_data_weekly[1:]).fit()

apple_model_linear_positivity_lagged_norm_weekly = smf.ols(
    formula="weekly_return ~ roberta_pos + roberta_pos_lag_1 + sp500_weekly_returns_alternate", data=apple_data_weekly[1:]).fit()

apple_model_linear_negativity_lagged_norm_weekly = smf.ols(
    formula="weekly_return ~ roberta_neg + roberta_neg_lag_1 + sp500_weekly_returns_alternate", data=apple_data_weekly[1:]).fit()

apple_model_linear_sensitivity_lagged_norm_weekly = smf.ols(
    formula="weekly_return ~ sensitivity + sensitivity_lag_1 + sp500_weekly_returns_alternate", data=apple_data_weekly[2:]).fit()

# ---sinh

apple_model_sinh_sentiment_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ np.sinh(roberta_sum)", data=apple_data_weekly).fit()

apple_model_sinh_positivity_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ np.sinh(roberta_pos)", data=apple_data_weekly).fit()

apple_model_sinh_negativity_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ np.sinh(roberta_neg)", data=apple_data_weekly).fit()

apple_model_sinh_sensitivity_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~  np.sinh(sensitivity)", data=apple_data_weekly[1:]).fit()

apple_model_sinh_sentiment_lagged_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~  np.sinh(roberta_sum) +  np.sinh(roberta_sum_lag_1)", data=apple_data_weekly[1:]).fit()

apple_model_sinh_positivity_lagged_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ np.sinh(roberta_pos) +  np.sinh(roberta_pos_lag_1)", data=apple_data_weekly[1:]).fit()

apple_model_sinh_negativity_lagged_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ np.sinh(roberta_neg) +  np.sinh(roberta_neg_lag_1)", data=apple_data_weekly[1:]).fit()

apple_model_sinh_sensitivity_lagged_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~  np.sinh(sensitivity) +  np.sinh(sensitivity_lag_1)", data=apple_data_weekly[2:]).fit()


apple_model_sinh_sentiment_volatility_weekly = smf.ols(
    formula="average_volatility ~ np.sinh(roberta_sum)", data=apple_data_weekly).fit()

apple_model_sinh_positivity_volatility_weekly = smf.ols(
    formula="average_volatility ~ np.sinh(roberta_pos)", data=apple_data_weekly).fit()

apple_model_sinh_negativity_volatility_weekly = smf.ols(
    formula="average_volatility ~ np.sinh(roberta_neg)", data=apple_data_weekly).fit()

apple_model_sinh_sensitivity_volatility_weekly = smf.ols(
    formula="average_volatility ~  np.sinh(sensitivity)", data=apple_data_weekly[1:]).fit()

apple_model_sinh_sentiment_lagged_volatility_weekly = smf.ols(
    formula="average_volatility ~  np.sinh(roberta_sum) +  np.sinh(roberta_sum_lag_1)", data=apple_data_weekly[1:]).fit()

apple_model_sinh_positivity_lagged_volatility_weekly = smf.ols(
    formula="average_volatility ~ np.sinh(roberta_pos) +  np.sinh(roberta_pos_lag_1)", data=apple_data_weekly[1:]).fit()

apple_model_sinh_negativity_lagged_volatility_weekly = smf.ols(
    formula="average_volatility ~ np.sinh(roberta_neg) +  np.sinh(roberta_neg_lag_1)", data=apple_data_weekly[1:]).fit()

apple_model_sinh_sensitivity_lagged_volatility_weekly = smf.ols(
    formula="average_volatility ~  np.sinh(sensitivity) +  np.sinh(sensitivity_lag_1)", data=apple_data_weekly[2:]).fit()


apple_model_sinh_sentiment_norm_weekly = smf.ols(
    formula="weekly_return ~ np.sinh(roberta_sum) + np.sinh(sp500_weekly_returns_alternate)", data=apple_data_weekly).fit()

apple_model_sinh_positivity_norm_weekly = smf.ols(
    formula="weekly_return ~ np.sinh(roberta_pos) + np.sinh(sp500_weekly_returns_alternate)", data=apple_data_weekly).fit()

apple_model_sinh_negativity_norm_weekly = smf.ols(
    formula="weekly_return ~ np.sinh(roberta_neg) + np.sinh(sp500_weekly_returns_alternate)", data=apple_data_weekly).fit()

apple_model_sinh_sensitivity_norm_weekly = smf.ols(
    formula="weekly_return ~  np.sinh(sensitivity) + np.sinh(sp500_weekly_returns_alternate)", data=apple_data_weekly[1:]).fit()

apple_model_sinh_sentiment_lagged_norm_weekly = smf.ols(
    formula="weekly_return ~  np.sinh(roberta_sum) +  np.sinh(roberta_sum_lag_1) + np.sinh(sp500_weekly_returns_alternate)", data=apple_data_weekly[1:]).fit()

apple_model_sinh_positivity_lagged_norm_weekly = smf.ols(
    formula="weekly_return ~ np.sinh(roberta_pos) +  np.sinh(roberta_pos_lag_1) + np.sinh(sp500_weekly_returns_alternate)", data=apple_data_weekly[1:]).fit()

apple_model_sinh_negativity_lagged_norm_weekly = smf.ols(
    formula="weekly_return ~ np.sinh(roberta_neg) +  np.sinh(roberta_neg_lag_1) + np.sinh(sp500_weekly_returns_alternate)", data=apple_data_weekly[1:]).fit()

apple_model_sinh_sensitivity_lagged_norm_weekly = smf.ols(
    formula="weekly_return ~  np.sinh(sensitivity) +  np.sinh(sensitivity_lag_1) + np.sinh(sp500_weekly_returns_alternate)", data=apple_data_weekly[2:]).fit()


# ---log


apple_model_log_positivity_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ np.log(roberta_pos)", data=apple_data_weekly).fit()

apple_model_log_negativity_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ np.log(roberta_neg)", data=apple_data_weekly).fit()

apple_model_log_positivity_lagged_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ np.log(roberta_pos) +  np.log(roberta_pos_lag_1)", data=apple_data_weekly[1:]).fit()

apple_model_log_negativity_lagged_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ np.log(roberta_neg) +  np.log(roberta_neg_lag_1)", data=apple_data_weekly[1:]).fit()


apple_model_log_positivity_volatility_weekly = smf.ols(
    formula="average_volatility ~ np.log(roberta_pos)", data=apple_data_weekly).fit()

apple_model_log_negativity_volatility_weekly = smf.ols(
    formula="average_volatility ~ np.log(roberta_neg)", data=apple_data_weekly).fit()

apple_model_log_positivity_lagged_volatility_weekly = smf.ols(
    formula="average_volatility ~ np.log(roberta_pos) +  np.log(roberta_pos_lag_1)", data=apple_data_weekly[1:]).fit()

apple_model_log_negativity_lagged_volatility_weekly = smf.ols(
    formula="average_volatility ~ np.log(roberta_neg) +  np.log(roberta_neg_lag_1)", data=apple_data_weekly[1:]).fit()


apple_model_log_positivity_norm_weekly = smf.ols(
    formula="weekly_return ~ np.log(roberta_pos) + np.log(sp500_weekly_returns_alternate)", data=apple_data_weekly).fit()

apple_model_log_negativity_norm_weekly = smf.ols(
    formula="weekly_return ~ np.log(roberta_neg) + np.log(sp500_weekly_returns_alternate)", data=apple_data_weekly).fit()

apple_model_log_positivity_lagged_norm_weekly = smf.ols(
    formula="weekly_return ~ np.log(roberta_pos) +  np.log(roberta_pos_lag_1) + np.log(sp500_weekly_returns_alternate)", data=apple_data_weekly[1:]).fit()

apple_model_log_negativity_lagged_norm_weekly = smf.ols(
    formula="weekly_return ~ np.log(roberta_neg) +  np.log(roberta_neg_lag_1) + np.log(sp500_weekly_returns_alternate)", data=apple_data_weekly[1:]).fit()

# ---sqrt

apple_model_sqrt_positivity_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ np.sqrt(roberta_pos)", data=apple_data_weekly).fit()

apple_model_sqrt_negativity_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ np.sqrt(roberta_neg)", data=apple_data_weekly).fit()

apple_model_sqrt_positivity_lagged_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ np.sqrt(roberta_pos) +  np.sqrt(roberta_pos_lag_1)", data=apple_data_weekly[1:]).fit()

apple_model_sqrt_negativity_lagged_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ np.sqrt(roberta_neg) +  np.sqrt(roberta_neg_lag_1)", data=apple_data_weekly[1:]).fit()


apple_model_sqrt_positivity_volatility_weekly = smf.ols(
    formula="average_volatility ~ np.sqrt(roberta_pos)", data=apple_data_weekly).fit()

apple_model_sqrt_negativity_volatility_weekly = smf.ols(
    formula="average_volatility ~ np.sqrt(roberta_neg)", data=apple_data_weekly).fit()

apple_model_sqrt_positivity_lagged_volatility_weekly = smf.ols(
    formula="average_volatility ~ np.sqrt(roberta_pos) +  np.sqrt(roberta_pos_lag_1)", data=apple_data_weekly[1:]).fit()

apple_model_sqrt_negativity_lagged_volatility_weekly = smf.ols(
    formula="average_volatility ~ np.sqrt(roberta_neg) +  np.sqrt(roberta_neg_lag_1)", data=apple_data_weekly[1:]).fit()


apple_model_sqrt_positivity_norm_weekly = smf.ols(
    formula="weekly_return ~ np.sqrt(roberta_pos) + np.sqrt(sp500_weekly_returns_alternate)", data=apple_data_weekly).fit()

apple_model_sqrt_negativity_norm_weekly = smf.ols(
    formula="weekly_return ~ np.sqrt(roberta_neg) + np.sqrt(sp500_weekly_returns_alternate)", data=apple_data_weekly).fit()

apple_model_sqrt_positivity_lagged_norm_weekly = smf.ols(
    formula="weekly_return ~ np.sqrt(roberta_pos) +  np.sqrt(roberta_pos_lag_1) + np.sqrt(sp500_weekly_returns_alternate)", data=apple_data_weekly[1:]).fit()

apple_model_sqrt_negativity_lagged_norm_weekly = smf.ols(
    formula="weekly_return ~ np.sqrt(roberta_neg) +  np.sqrt(roberta_neg_lag_1) + np.sqrt(sp500_weekly_returns_alternate)", data=apple_data_weekly[1:]).fit()


# Find out which Apple models are significant below 5%

apple_model_list = [apple_model_linear_sentiment_return_daily, apple_model_linear_positivity_return_daily, apple_model_linear_negativity_return_daily, apple_model_linear_sensitivity_return_daily, apple_model_linear_sentiment_lagged_return_daily, apple_model_linear_positivity_lagged_return_daily, apple_model_linear_negativity_lagged_return_daily, apple_model_linear_sensitivity_lagged_return_daily, apple_model_linear_sentiment_norm_daily, apple_model_linear_positivity_norm_daily, apple_model_linear_negativity_norm_daily, apple_model_linear_sensitivity_norm_daily, apple_model_linear_sentiment_lagged_norm_daily, apple_model_linear_positivity_lagged_norm_daily, apple_model_linear_negativity_lagged_norm_daily, apple_model_linear_sensitivity_lagged_norm_daily, apple_model_sinh_sentiment_return_daily, apple_model_sinh_positivity_return_daily, apple_model_sinh_negativity_return_daily, apple_model_sinh_sensitivity_return_daily, apple_model_sinh_sentiment_lagged_return_daily, apple_model_sinh_positivity_lagged_return_daily, apple_model_sinh_negativity_lagged_return_daily, apple_model_sinh_sensitivity_lagged_return_daily, apple_model_sinh_sentiment_norm_daily, apple_model_sinh_positivity_norm_daily, apple_model_sinh_negativity_norm_daily, apple_model_sinh_sensitivity_norm_daily, apple_model_sinh_sentiment_lagged_norm_daily, apple_model_sinh_positivity_lagged_norm_daily, apple_model_sinh_negativity_lagged_norm_daily, apple_model_sinh_sensitivity_lagged_norm_daily, apple_model_log_positivity_return_daily, apple_model_log_negativity_return_daily, apple_model_log_positivity_lagged_return_daily, apple_model_log_negativity_lagged_return_daily, apple_model_log_positivity_norm_daily, apple_model_log_negativity_norm_daily, apple_model_log_positivity_lagged_norm_daily, apple_model_log_negativity_lagged_norm_daily, apple_model_sqrt_positivity_return_daily, apple_model_sqrt_negativity_return_daily, apple_model_sqrt_positivity_lagged_return_daily, apple_model_sqrt_negativity_lagged_return_daily, apple_model_sqrt_positivity_norm_daily, apple_model_sqrt_negativity_norm_daily, apple_model_sqrt_positivity_lagged_norm_daily, apple_model_sqrt_negativity_lagged_norm_daily, apple_model_linear_sentiment_return_weekly, apple_model_linear_positivity_return_weekly, apple_model_linear_negativity_return_weekly, apple_model_linear_sensitivity_return_weekly, apple_model_linear_sentiment_lagged_return_weekly, apple_model_linear_positivity_lagged_return_weekly, apple_model_linear_negativity_lagged_return_weekly, apple_model_linear_sensitivity_lagged_return_weekly, apple_model_linear_sentiment_volatility_weekly, apple_model_linear_positivity_volatility_weekly, apple_model_linear_negativity_volatility_weekly, apple_model_linear_sensitivity_volatility_weekly, apple_model_linear_sentiment_lagged_volatility_weekly,
                    apple_model_linear_positivity_lagged_volatility_weekly, apple_model_linear_negativity_lagged_volatility_weekly, apple_model_linear_sensitivity_lagged_volatility_weekly, apple_model_linear_sentiment_norm_weekly, apple_model_linear_positivity_norm_weekly, apple_model_linear_negativity_norm_weekly, apple_model_linear_sensitivity_norm_weekly, apple_model_linear_sentiment_lagged_norm_weekly, apple_model_linear_positivity_lagged_norm_weekly, apple_model_linear_negativity_lagged_norm_weekly, apple_model_linear_sensitivity_lagged_norm_weekly, apple_model_sinh_sentiment_return_weekly, apple_model_sinh_positivity_return_weekly, apple_model_sinh_negativity_return_weekly, apple_model_sinh_sensitivity_return_weekly, apple_model_sinh_sentiment_lagged_return_weekly, apple_model_sinh_positivity_lagged_return_weekly, apple_model_sinh_negativity_lagged_return_weekly, apple_model_sinh_sensitivity_lagged_return_weekly, apple_model_sinh_sentiment_volatility_weekly, apple_model_sinh_positivity_volatility_weekly, apple_model_sinh_negativity_volatility_weekly, apple_model_sinh_sensitivity_volatility_weekly, apple_model_sinh_sentiment_lagged_volatility_weekly, apple_model_sinh_positivity_lagged_volatility_weekly, apple_model_sinh_negativity_lagged_volatility_weekly, apple_model_sinh_sensitivity_lagged_volatility_weekly, apple_model_sinh_sentiment_norm_weekly, apple_model_sinh_positivity_norm_weekly, apple_model_sinh_negativity_norm_weekly, apple_model_sinh_sensitivity_norm_weekly, apple_model_sinh_sentiment_lagged_norm_weekly, apple_model_sinh_positivity_lagged_norm_weekly, apple_model_sinh_negativity_lagged_norm_weekly, apple_model_sinh_sensitivity_lagged_norm_weekly, apple_model_log_positivity_return_weekly, apple_model_log_negativity_return_weekly, apple_model_log_positivity_lagged_return_weekly, apple_model_log_negativity_lagged_return_weekly, apple_model_log_positivity_volatility_weekly, apple_model_log_negativity_volatility_weekly, apple_model_log_positivity_lagged_volatility_weekly, apple_model_log_negativity_lagged_volatility_weekly, apple_model_log_positivity_norm_weekly, apple_model_log_negativity_norm_weekly, apple_model_log_positivity_lagged_norm_weekly, apple_model_log_negativity_lagged_norm_weekly, apple_model_sqrt_positivity_return_weekly, apple_model_sqrt_negativity_return_weekly, apple_model_sqrt_positivity_lagged_return_weekly, apple_model_sqrt_negativity_lagged_return_weekly, apple_model_sqrt_positivity_volatility_weekly, apple_model_sqrt_negativity_volatility_weekly, apple_model_sqrt_positivity_lagged_volatility_weekly, apple_model_sqrt_negativity_lagged_volatility_weekly, apple_model_sqrt_positivity_norm_weekly, apple_model_sqrt_negativity_norm_weekly, apple_model_sqrt_positivity_lagged_norm_weekly, apple_model_sqrt_negativity_lagged_norm_weekly]

apple_model_names = ['apple_model_linear_sentiment_return_daily', 'apple_model_linear_positivity_return_daily', 'apple_model_linear_negativity_return_daily', 'apple_model_linear_sensitivity_return_daily', 'apple_model_linear_sentiment_lagged_return_daily', 'apple_model_linear_positivity_lagged_return_daily', 'apple_model_linear_negativity_lagged_return_daily', 'apple_model_linear_sensitivity_lagged_return_daily', 'apple_model_linear_sentiment_norm_daily', 'apple_model_linear_positivity_norm_daily', 'apple_model_linear_negativity_norm_daily', 'apple_model_linear_sensitivity_norm_daily', 'apple_model_linear_sentiment_lagged_norm_daily', 'apple_model_linear_positivity_lagged_norm_daily', 'apple_model_linear_negativity_lagged_norm_daily', 'apple_model_linear_sensitivity_lagged_norm_daily', 'apple_model_sinh_sentiment_return_daily', 'apple_model_sinh_positivity_return_daily', 'apple_model_sinh_negativity_return_daily', 'apple_model_sinh_sensitivity_return_daily', 'apple_model_sinh_sentiment_lagged_return_daily', 'apple_model_sinh_positivity_lagged_return_daily', 'apple_model_sinh_negativity_lagged_return_daily', 'apple_model_sinh_sensitivity_lagged_return_daily', 'apple_model_sinh_sentiment_norm_daily', 'apple_model_sinh_positivity_norm_daily', 'apple_model_sinh_negativity_norm_daily', 'apple_model_sinh_sensitivity_norm_daily', 'apple_model_sinh_sentiment_lagged_norm_daily', 'apple_model_sinh_positivity_lagged_norm_daily', 'apple_model_sinh_negativity_lagged_norm_daily', 'apple_model_sinh_sensitivity_lagged_norm_daily', 'apple_model_log_positivity_return_daily', 'apple_model_log_negativity_return_daily', 'apple_model_log_positivity_lagged_return_daily', 'apple_model_log_negativity_lagged_return_daily', 'apple_model_log_positivity_norm_daily', 'apple_model_log_negativity_norm_daily', 'apple_model_log_positivity_lagged_norm_daily', 'apple_model_log_negativity_lagged_norm_daily', 'apple_model_sqrt_positivity_return_daily', 'apple_model_sqrt_negativity_return_daily', 'apple_model_sqrt_positivity_lagged_return_daily', 'apple_model_sqrt_negativity_lagged_return_daily', 'apple_model_sqrt_positivity_norm_daily', 'apple_model_sqrt_negativity_norm_daily', 'apple_model_sqrt_positivity_lagged_norm_daily', 'apple_model_sqrt_negativity_lagged_norm_daily', 'apple_model_linear_sentiment_return_weekly', 'apple_model_linear_positivity_return_weekly', 'apple_model_linear_negativity_return_weekly', 'apple_model_linear_sensitivity_return_weekly', 'apple_model_linear_sentiment_lagged_return_weekly', 'apple_model_linear_positivity_lagged_return_weekly', 'apple_model_linear_negativity_lagged_return_weekly', 'apple_model_linear_sensitivity_lagged_return_weekly', 'apple_model_linear_sentiment_volatility_weekly', 'apple_model_linear_positivity_volatility_weekly', 'apple_model_linear_negativity_volatility_weekly', 'apple_model_linear_sensitivity_volatility_weekly', 'apple_model_linear_sentiment_lagged_volatility_weekly',
                     'apple_model_linear_positivity_lagged_volatility_weekly', 'apple_model_linear_negativity_lagged_volatility_weekly', 'apple_model_linear_sensitivity_lagged_volatility_weekly', 'apple_model_linear_sentiment_norm_weekly', 'apple_model_linear_positivity_norm_weekly', 'apple_model_linear_negativity_norm_weekly', 'apple_model_linear_sensitivity_norm_weekly', 'apple_model_linear_sentiment_lagged_norm_weekly', 'apple_model_linear_positivity_lagged_norm_weekly', 'apple_model_linear_negativity_lagged_norm_weekly', 'apple_model_linear_sensitivity_lagged_norm_weekly', 'apple_model_sinh_sentiment_return_weekly', 'apple_model_sinh_positivity_return_weekly', 'apple_model_sinh_negativity_return_weekly', 'apple_model_sinh_sensitivity_return_weekly', 'apple_model_sinh_sentiment_lagged_return_weekly', 'apple_model_sinh_positivity_lagged_return_weekly', 'apple_model_sinh_negativity_lagged_return_weekly', 'apple_model_sinh_sensitivity_lagged_return_weekly', 'apple_model_sinh_sentiment_volatility_weekly', 'apple_model_sinh_positivity_volatility_weekly', 'apple_model_sinh_negativity_volatility_weekly', 'apple_model_sinh_sensitivity_volatility_weekly', 'apple_model_sinh_sentiment_lagged_volatility_weekly', 'apple_model_sinh_positivity_lagged_volatility_weekly', 'apple_model_sinh_negativity_lagged_volatility_weekly', 'apple_model_sinh_sensitivity_lagged_volatility_weekly', 'apple_model_sinh_sentiment_norm_weekly', 'apple_model_sinh_positivity_norm_weekly', 'apple_model_sinh_negativity_norm_weekly', 'apple_model_sinh_sensitivity_norm_weekly', 'apple_model_sinh_sentiment_lagged_norm_weekly', 'apple_model_sinh_positivity_lagged_norm_weekly', 'apple_model_sinh_negativity_lagged_norm_weekly', 'apple_model_sinh_sensitivity_lagged_norm_weekly', 'apple_model_log_positivity_return_weekly', 'apple_model_log_negativity_return_weekly', 'apple_model_log_positivity_lagged_return_weekly', 'apple_model_log_negativity_lagged_return_weekly', 'apple_model_log_positivity_volatility_weekly', 'apple_model_log_negativity_volatility_weekly', 'apple_model_log_positivity_lagged_volatility_weekly', 'apple_model_log_negativity_lagged_volatility_weekly', 'apple_model_log_positivity_norm_weekly', 'apple_model_log_negativity_norm_weekly', 'apple_model_log_positivity_lagged_norm_weekly', 'apple_model_log_negativity_lagged_norm_weekly', 'apple_model_sqrt_positivity_return_weekly', 'apple_model_sqrt_negativity_return_weekly', 'apple_model_sqrt_positivity_lagged_return_weekly', 'apple_model_sqrt_negativity_lagged_return_weekly', 'apple_model_sqrt_positivity_volatility_weekly', 'apple_model_sqrt_negativity_volatility_weekly', 'apple_model_sqrt_positivity_lagged_volatility_weekly', 'apple_model_sqrt_negativity_lagged_volatility_weekly', 'apple_model_sqrt_positivity_norm_weekly', 'apple_model_sqrt_negativity_norm_weekly', 'apple_model_sqrt_positivity_lagged_norm_weekly', 'apple_model_sqrt_negativity_lagged_norm_weekly']


apple_verified_models = []

length = len(apple_model_list)

for i in range(length):

    caller = i

    test = apple_model_list[caller].summary()

    testcont = test.tables[0].as_html()

    testcont2 = test.tables[1].as_html()

    testpandas = pd.read_html(testcont, header=0, index_col=0)[0]

    testpandas = testpandas.reset_index()

    testpandas2 = pd.read_html(testcont2, header=0, index_col=0)[0]

    testpandas2 = testpandas2.reset_index()

    testpandas.to_csv('Regression Summary ' + apple_model_names[caller]+'.csv')

    testpandas2.to_csv('Regression Variables ' + apple_model_names[caller]+'.csv')

    if testpandas.iloc[2, 3] < 0.05:
        print(apple_model_list[caller].summary())
        apple_verified_models.append(apple_model_names[caller])

print(apple_verified_models)

# Set up dataset and model for regression for Blizzard_Ent

blizzard_ent_data_weekly = pd.read_csv(
    'Adjusted Regression Datasets - Blizzard_Ent Regression Dataset.csv')

blizzard_ent_data_daily = pd.read_csv(
    'Adjusted Regression Datasets - Blizzard_Ent Daily Regression Dataset.csv')

# Models based on daily data

blizzard_ent_model_linear_sentiment_return_daily = smf.ols(
    formula="blizzard_ent_daily_abnormal_return ~ daily_roberta_sum", data=blizzard_ent_data_daily).fit()

blizzard_ent_model_linear_positivity_return_daily = smf.ols(
    formula="blizzard_ent_daily_abnormal_return ~ daily_roberta_pos", data=blizzard_ent_data_daily).fit()

blizzard_ent_model_linear_negativity_return_daily = smf.ols(
    formula="blizzard_ent_daily_abnormal_return ~ daily_roberta_neg", data=blizzard_ent_data_daily).fit()

blizzard_ent_model_linear_negativity_return_daily = smf.ols(
    formula="blizzard_ent_daily_abnormal_return ~ daily_roberta_neg", data=blizzard_ent_data_daily).fit()

blizzard_ent_model_linear_sensitivity_return_daily = smf.ols(
    formula="blizzard_ent_daily_abnormal_return ~ daily_sensitivity", data=blizzard_ent_data_daily).fit()


blizzard_ent_model_linear_sentiment_lagged_return_daily = smf.ols(
    formula="blizzard_ent_daily_abnormal_return ~ daily_roberta_sum + daily_roberta_sum_lag_1", data=blizzard_ent_data_daily[1:]).fit()

blizzard_ent_model_linear_sentiment_lagged_return_daily_sp500 = smf.ols(
    formula="blizzard_ent_daily_abnormal_return ~ daily_roberta_sum + daily_roberta_sum_lag_1", data=blizzard_ent_data_daily[1:]).fit()

blizzard_ent_model_linear_positivity_lagged_return_daily = smf.ols(
    formula="blizzard_ent_daily_abnormal_return ~ daily_roberta_pos + daily_roberta_pos_lag_1", data=blizzard_ent_data_daily[1:]).fit()

blizzard_ent_model_linear_negativity_lagged_return_daily = smf.ols(
    formula="blizzard_ent_daily_abnormal_return ~ daily_roberta_neg + daily_roberta_neg_lag_1", data=blizzard_ent_data_daily[1:]).fit()

blizzard_ent_model_linear_sensitivity_lagged_return_daily = smf.ols(
    formula="blizzard_ent_daily_abnormal_return ~ daily_sensitivity + daily_sensitivity_lag_1 ", data=blizzard_ent_data_daily[1:]).fit()


blizzard_ent_model_linear_sentiment_norm_daily = smf.ols(
    formula="blizzard_ent_daily_return ~ daily_roberta_sum + sp500_daily_returns", data=blizzard_ent_data_daily).fit()

blizzard_ent_model_linear_positivity_norm_daily = smf.ols(
    formula="blizzard_ent_daily_return ~ daily_roberta_pos + sp500_daily_returns", data=blizzard_ent_data_daily).fit()

blizzard_ent_model_linear_negativity_norm_daily = smf.ols(
    formula="blizzard_ent_daily_return ~ daily_roberta_neg + sp500_daily_returns", data=blizzard_ent_data_daily).fit()

blizzard_ent_model_linear_sensitivity_norm_daily = smf.ols(
    formula="blizzard_ent_daily_return ~ daily_sensitivity + sp500_daily_returns", data=blizzard_ent_data_daily).fit()


blizzard_ent_model_linear_sentiment_lagged_norm_daily = smf.ols(
    formula="blizzard_ent_daily_return ~ daily_roberta_sum + daily_roberta_sum_lag_1 + sp500_daily_returns", data=blizzard_ent_data_daily[1:]).fit()

blizzard_ent_model_linear_positivity_lagged_norm_daily = smf.ols(
    formula="blizzard_ent_daily_return ~ daily_roberta_pos + daily_roberta_pos_lag_1 + sp500_daily_returns", data=blizzard_ent_data_daily[1:]).fit()

blizzard_ent_model_linear_negativity_lagged_norm_daily = smf.ols(
    formula="blizzard_ent_daily_return ~ daily_roberta_neg + daily_roberta_neg_lag_1 + sp500_daily_returns", data=blizzard_ent_data_daily[1:]).fit()

blizzard_ent_model_linear_sensitivity_lagged_norm_daily = smf.ols(
    formula="blizzard_ent_daily_return ~ daily_sensitivity + daily_sensitivity_lag_1 + sp500_daily_returns", data=blizzard_ent_data_daily[1:]).fit()


# ---sinh

blizzard_ent_model_sinh_sentiment_return_daily = smf.ols(
    formula="blizzard_ent_daily_abnormal_return ~ np.sinh(daily_roberta_sum)", data=blizzard_ent_data_daily).fit()

blizzard_ent_model_sinh_positivity_return_daily = smf.ols(
    formula="blizzard_ent_daily_abnormal_return ~ np.sinh(daily_roberta_pos)", data=blizzard_ent_data_daily).fit()

blizzard_ent_model_sinh_negativity_return_daily = smf.ols(
    formula="blizzard_ent_daily_abnormal_return ~ np.sinh(daily_roberta_neg)", data=blizzard_ent_data_daily).fit()

blizzard_ent_model_sinh_sensitivity_return_daily = smf.ols(
    formula="blizzard_ent_daily_abnormal_return ~ np.sinh(daily_sensitivity)", data=blizzard_ent_data_daily).fit()

blizzard_ent_model_sinh_sentiment_lagged_return_daily = smf.ols(
    formula="blizzard_ent_daily_abnormal_return ~ np.sinh(daily_roberta_sum) + np.sinh(daily_roberta_sum_lag_1)", data=blizzard_ent_data_daily[1:]).fit()

blizzard_ent_model_sinh_positivity_lagged_return_daily = smf.ols(
    formula="blizzard_ent_daily_abnormal_return ~ np.sinh(daily_roberta_pos)+ np.sinh(daily_roberta_pos_lag_1)", data=blizzard_ent_data_daily[1:]).fit()

blizzard_ent_model_sinh_negativity_lagged_return_daily = smf.ols(
    formula="blizzard_ent_daily_abnormal_return ~ np.sinh(daily_roberta_neg)+ np.sinh(daily_roberta_neg_lag_1)", data=blizzard_ent_data_daily[1:]).fit()

blizzard_ent_model_sinh_sensitivity_lagged_return_daily = smf.ols(
    formula="blizzard_ent_daily_abnormal_return ~ np.sinh(daily_sensitivity)+ np.sinh(daily_sensitivity_lag_1)", data=blizzard_ent_data_daily[1:]).fit()


blizzard_ent_model_sinh_sentiment_norm_daily = smf.ols(
    formula="blizzard_ent_daily_return ~ np.sinh(daily_roberta_sum) + np.sinh(sp500_daily_returns)", data=blizzard_ent_data_daily).fit()

blizzard_ent_model_sinh_positivity_norm_daily = smf.ols(
    formula="blizzard_ent_daily_return ~ np.sinh(daily_roberta_pos) + np.sinh(sp500_daily_returns)", data=blizzard_ent_data_daily).fit()

blizzard_ent_model_sinh_negativity_norm_daily = smf.ols(
    formula="blizzard_ent_daily_return ~ np.sinh(daily_roberta_neg) + np.sinh(sp500_daily_returns)", data=blizzard_ent_data_daily).fit()

blizzard_ent_model_sinh_sensitivity_norm_daily = smf.ols(
    formula="blizzard_ent_daily_return ~ np.sinh(daily_sensitivity) + np.sinh(sp500_daily_returns)", data=blizzard_ent_data_daily).fit()

blizzard_ent_model_sinh_sentiment_lagged_norm_daily = smf.ols(
    formula="blizzard_ent_daily_return ~ np.sinh(daily_roberta_sum) + np.sinh(daily_roberta_sum_lag_1) + np.sinh(sp500_daily_returns)", data=blizzard_ent_data_daily[1:]).fit()

blizzard_ent_model_sinh_positivity_lagged_norm_daily = smf.ols(
    formula="blizzard_ent_daily_return ~ np.sinh(daily_roberta_pos)+ np.sinh(daily_roberta_pos_lag_1) + np.sinh(sp500_daily_returns)", data=blizzard_ent_data_daily[1:]).fit()

blizzard_ent_model_sinh_negativity_lagged_norm_daily = smf.ols(
    formula="blizzard_ent_daily_return ~ np.sinh(daily_roberta_neg)+ np.sinh(daily_roberta_neg_lag_1) + np.sinh(sp500_daily_returns)", data=blizzard_ent_data_daily[1:]).fit()

blizzard_ent_model_sinh_sensitivity_lagged_norm_daily = smf.ols(
    formula="blizzard_ent_daily_return ~ np.sinh(daily_sensitivity)+ np.sinh(daily_sensitivity_lag_1) + np.sinh(sp500_daily_returns)", data=blizzard_ent_data_daily[1:]).fit()

# ---log


blizzard_ent_model_log_positivity_return_daily = smf.ols(
    formula="blizzard_ent_daily_abnormal_return ~ np.log(daily_roberta_pos)", data=blizzard_ent_data_daily).fit()

blizzard_ent_model_log_negativity_return_daily = smf.ols(
    formula="blizzard_ent_daily_abnormal_return ~ np.log(daily_roberta_neg)", data=blizzard_ent_data_daily).fit()

blizzard_ent_model_log_positivity_lagged_return_daily = smf.ols(
    formula="blizzard_ent_daily_abnormal_return ~ np.log(daily_roberta_pos)+ np.log(daily_roberta_pos_lag_1)", data=blizzard_ent_data_daily[1:]).fit()

blizzard_ent_model_log_negativity_lagged_return_daily = smf.ols(
    formula="blizzard_ent_daily_abnormal_return ~ np.log(daily_roberta_neg)+ np.log(daily_roberta_neg_lag_1)", data=blizzard_ent_data_daily[1:]).fit()


blizzard_ent_model_log_positivity_norm_daily = smf.ols(
    formula="blizzard_ent_daily_return ~ np.log(daily_roberta_pos) + np.log(sp500_daily_returns)", data=blizzard_ent_data_daily).fit()

blizzard_ent_model_log_negativity_norm_daily = smf.ols(
    formula="blizzard_ent_daily_return ~ np.log(daily_roberta_neg) + np.log(sp500_daily_returns)", data=blizzard_ent_data_daily).fit()

blizzard_ent_model_log_positivity_lagged_norm_daily = smf.ols(
    formula="blizzard_ent_daily_return ~ np.log(daily_roberta_pos)+ np.log(daily_roberta_pos_lag_1) + np.log(sp500_daily_returns)", data=blizzard_ent_data_daily[1:]).fit()

blizzard_ent_model_log_negativity_lagged_norm_daily = smf.ols(
    formula="blizzard_ent_daily_return ~ np.log(daily_roberta_neg)+ np.log(daily_roberta_neg_lag_1) + np.log(sp500_daily_returns)", data=blizzard_ent_data_daily[1:]).fit()

# ---sqrt


blizzard_ent_model_sqrt_positivity_return_daily = smf.ols(
    formula="blizzard_ent_daily_abnormal_return ~ np.sqrt(daily_roberta_pos)", data=blizzard_ent_data_daily).fit()

blizzard_ent_model_sqrt_negativity_return_daily = smf.ols(
    formula="blizzard_ent_daily_abnormal_return ~ np.sqrt(daily_roberta_neg)", data=blizzard_ent_data_daily).fit()

blizzard_ent_model_sqrt_positivity_lagged_return_daily = smf.ols(
    formula="blizzard_ent_daily_abnormal_return ~ np.sqrt(daily_roberta_pos)+ np.sqrt(daily_roberta_pos_lag_1)", data=blizzard_ent_data_daily[1:]).fit()

blizzard_ent_model_sqrt_negativity_lagged_return_daily = smf.ols(
    formula="blizzard_ent_daily_abnormal_return ~ np.sqrt(daily_roberta_neg)+ np.sqrt(daily_roberta_neg_lag_1)", data=blizzard_ent_data_daily[1:]).fit()


blizzard_ent_model_sqrt_positivity_norm_daily = smf.ols(
    formula="blizzard_ent_daily_return ~ np.sqrt(daily_roberta_pos) + np.sqrt(sp500_daily_returns)", data=blizzard_ent_data_daily).fit()

blizzard_ent_model_sqrt_negativity_norm_daily = smf.ols(
    formula="blizzard_ent_daily_return ~ np.sqrt(daily_roberta_neg) + np.sqrt(sp500_daily_returns)", data=blizzard_ent_data_daily).fit()

blizzard_ent_model_sqrt_positivity_lagged_norm_daily = smf.ols(
    formula="blizzard_ent_daily_return ~ np.sqrt(daily_roberta_pos)+ np.sqrt(daily_roberta_pos_lag_1) + np.sqrt(sp500_daily_returns)", data=blizzard_ent_data_daily[1:]).fit()

blizzard_ent_model_sqrt_negativity_lagged_norm_daily = smf.ols(
    formula="blizzard_ent_daily_return ~ np.sqrt(daily_roberta_neg)+ np.sqrt(daily_roberta_neg_lag_1) + np.sqrt(sp500_daily_returns)", data=blizzard_ent_data_daily[1:]).fit()


# Models based on weekly data:

blizzard_ent_model_linear_sentiment_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ roberta_sum", data=blizzard_ent_data_weekly).fit()

blizzard_ent_model_linear_positivity_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ roberta_pos", data=blizzard_ent_data_weekly).fit()

blizzard_ent_model_linear_negativity_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ roberta_neg", data=blizzard_ent_data_weekly).fit()

blizzard_ent_model_linear_sensitivity_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ sensitivity", data=blizzard_ent_data_weekly[1:]).fit()

blizzard_ent_model_linear_sentiment_lagged_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ roberta_sum + roberta_sum_lag_1", data=blizzard_ent_data_weekly[1:]).fit()

blizzard_ent_model_linear_positivity_lagged_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ roberta_pos + roberta_pos_lag_1", data=blizzard_ent_data_weekly[1:]).fit()

blizzard_ent_model_linear_negativity_lagged_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ roberta_neg + roberta_neg_lag_1", data=blizzard_ent_data_weekly[1:]).fit()

blizzard_ent_model_linear_sensitivity_lagged_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ sensitivity + sensitivity_lag_1", data=blizzard_ent_data_weekly[2:]).fit()


blizzard_ent_model_linear_sentiment_volatility_weekly = smf.ols(
    formula="average_volatility ~ roberta_sum", data=blizzard_ent_data_weekly).fit()

blizzard_ent_model_linear_positivity_volatility_weekly = smf.ols(
    formula="average_volatility ~ roberta_pos", data=blizzard_ent_data_weekly).fit()

blizzard_ent_model_linear_negativity_volatility_weekly = smf.ols(
    formula="average_volatility ~ roberta_neg", data=blizzard_ent_data_weekly).fit()

blizzard_ent_model_linear_sensitivity_volatility_weekly = smf.ols(
    formula="average_volatility ~ sensitivity", data=blizzard_ent_data_weekly[1:]).fit()

blizzard_ent_model_linear_sentiment_lagged_volatility_weekly = smf.ols(
    formula="average_volatility ~ roberta_sum + roberta_sum_lag_1", data=blizzard_ent_data_weekly[1:]).fit()

blizzard_ent_model_linear_positivity_lagged_volatility_weekly = smf.ols(
    formula="average_volatility ~ roberta_pos + roberta_pos_lag_1", data=blizzard_ent_data_weekly[1:]).fit()

blizzard_ent_model_linear_negativity_lagged_volatility_weekly = smf.ols(
    formula="average_volatility ~ roberta_neg + roberta_neg_lag_1", data=blizzard_ent_data_weekly[1:]).fit()

blizzard_ent_model_linear_sensitivity_lagged_volatility_weekly = smf.ols(
    formula="average_volatility ~ sensitivity + sensitivity_lag_1", data=blizzard_ent_data_weekly[2:]).fit()


blizzard_ent_model_linear_sentiment_norm_weekly = smf.ols(
    formula="weekly_return ~ roberta_sum + sp500_weekly_returns_alternate", data=blizzard_ent_data_weekly).fit()

blizzard_ent_model_linear_positivity_norm_weekly = smf.ols(
    formula="weekly_return ~ roberta_pos + sp500_weekly_returns_alternate", data=blizzard_ent_data_weekly).fit()

blizzard_ent_model_linear_negativity_norm_weekly = smf.ols(
    formula="weekly_return ~ roberta_neg + sp500_weekly_returns_alternate", data=blizzard_ent_data_weekly).fit()

blizzard_ent_model_linear_sensitivity_norm_weekly = smf.ols(
    formula="weekly_return ~ sensitivity + sp500_weekly_returns_alternate", data=blizzard_ent_data_weekly[1:]).fit()

blizzard_ent_model_linear_sentiment_lagged_norm_weekly = smf.ols(
    formula="weekly_return ~ roberta_sum + roberta_sum_lag_1 + sp500_weekly_returns_alternate", data=blizzard_ent_data_weekly[1:]).fit()

blizzard_ent_model_linear_positivity_lagged_norm_weekly = smf.ols(
    formula="weekly_return ~ roberta_pos + roberta_pos_lag_1 + sp500_weekly_returns_alternate", data=blizzard_ent_data_weekly[1:]).fit()

blizzard_ent_model_linear_negativity_lagged_norm_weekly = smf.ols(
    formula="weekly_return ~ roberta_neg + roberta_neg_lag_1 + sp500_weekly_returns_alternate", data=blizzard_ent_data_weekly[1:]).fit()

blizzard_ent_model_linear_sensitivity_lagged_norm_weekly = smf.ols(
    formula="weekly_return ~ sensitivity + sensitivity_lag_1 + sp500_weekly_returns_alternate", data=blizzard_ent_data_weekly[2:]).fit()

# ---sinh

blizzard_ent_model_sinh_sentiment_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ np.sinh(roberta_sum)", data=blizzard_ent_data_weekly).fit()

blizzard_ent_model_sinh_positivity_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ np.sinh(roberta_pos)", data=blizzard_ent_data_weekly).fit()

blizzard_ent_model_sinh_negativity_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ np.sinh(roberta_neg)", data=blizzard_ent_data_weekly).fit()

blizzard_ent_model_sinh_sensitivity_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~  np.sinh(sensitivity)", data=blizzard_ent_data_weekly[1:]).fit()

blizzard_ent_model_sinh_sentiment_lagged_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~  np.sinh(roberta_sum) +  np.sinh(roberta_sum_lag_1)", data=blizzard_ent_data_weekly[1:]).fit()

blizzard_ent_model_sinh_positivity_lagged_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ np.sinh(roberta_pos) +  np.sinh(roberta_pos_lag_1)", data=blizzard_ent_data_weekly[1:]).fit()

blizzard_ent_model_sinh_negativity_lagged_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ np.sinh(roberta_neg) +  np.sinh(roberta_neg_lag_1)", data=blizzard_ent_data_weekly[1:]).fit()

blizzard_ent_model_sinh_sensitivity_lagged_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~  np.sinh(sensitivity) +  np.sinh(sensitivity_lag_1)", data=blizzard_ent_data_weekly[2:]).fit()


blizzard_ent_model_sinh_sentiment_volatility_weekly = smf.ols(
    formula="average_volatility ~ np.sinh(roberta_sum)", data=blizzard_ent_data_weekly).fit()

blizzard_ent_model_sinh_positivity_volatility_weekly = smf.ols(
    formula="average_volatility ~ np.sinh(roberta_pos)", data=blizzard_ent_data_weekly).fit()

blizzard_ent_model_sinh_negativity_volatility_weekly = smf.ols(
    formula="average_volatility ~ np.sinh(roberta_neg)", data=blizzard_ent_data_weekly).fit()

blizzard_ent_model_sinh_sensitivity_volatility_weekly = smf.ols(
    formula="average_volatility ~  np.sinh(sensitivity)", data=blizzard_ent_data_weekly[1:]).fit()

blizzard_ent_model_sinh_sentiment_lagged_volatility_weekly = smf.ols(
    formula="average_volatility ~  np.sinh(roberta_sum) +  np.sinh(roberta_sum_lag_1)", data=blizzard_ent_data_weekly[1:]).fit()

blizzard_ent_model_sinh_positivity_lagged_volatility_weekly = smf.ols(
    formula="average_volatility ~ np.sinh(roberta_pos) +  np.sinh(roberta_pos_lag_1)", data=blizzard_ent_data_weekly[1:]).fit()

blizzard_ent_model_sinh_negativity_lagged_volatility_weekly = smf.ols(
    formula="average_volatility ~ np.sinh(roberta_neg) +  np.sinh(roberta_neg_lag_1)", data=blizzard_ent_data_weekly[1:]).fit()

blizzard_ent_model_sinh_sensitivity_lagged_volatility_weekly = smf.ols(
    formula="average_volatility ~  np.sinh(sensitivity) +  np.sinh(sensitivity_lag_1)", data=blizzard_ent_data_weekly[2:]).fit()


blizzard_ent_model_sinh_sentiment_norm_weekly = smf.ols(
    formula="weekly_return ~ np.sinh(roberta_sum) + np.sinh(sp500_weekly_returns_alternate)", data=blizzard_ent_data_weekly).fit()

blizzard_ent_model_sinh_positivity_norm_weekly = smf.ols(
    formula="weekly_return ~ np.sinh(roberta_pos) + np.sinh(sp500_weekly_returns_alternate)", data=blizzard_ent_data_weekly).fit()

blizzard_ent_model_sinh_negativity_norm_weekly = smf.ols(
    formula="weekly_return ~ np.sinh(roberta_neg) + np.sinh(sp500_weekly_returns_alternate)", data=blizzard_ent_data_weekly).fit()

blizzard_ent_model_sinh_sensitivity_norm_weekly = smf.ols(
    formula="weekly_return ~  np.sinh(sensitivity) + np.sinh(sp500_weekly_returns_alternate)", data=blizzard_ent_data_weekly[1:]).fit()

blizzard_ent_model_sinh_sentiment_lagged_norm_weekly = smf.ols(
    formula="weekly_return ~  np.sinh(roberta_sum) +  np.sinh(roberta_sum_lag_1) + np.sinh(sp500_weekly_returns_alternate)", data=blizzard_ent_data_weekly[1:]).fit()

blizzard_ent_model_sinh_positivity_lagged_norm_weekly = smf.ols(
    formula="weekly_return ~ np.sinh(roberta_pos) +  np.sinh(roberta_pos_lag_1) + np.sinh(sp500_weekly_returns_alternate)", data=blizzard_ent_data_weekly[1:]).fit()

blizzard_ent_model_sinh_negativity_lagged_norm_weekly = smf.ols(
    formula="weekly_return ~ np.sinh(roberta_neg) +  np.sinh(roberta_neg_lag_1) + np.sinh(sp500_weekly_returns_alternate)", data=blizzard_ent_data_weekly[1:]).fit()

blizzard_ent_model_sinh_sensitivity_lagged_norm_weekly = smf.ols(
    formula="weekly_return ~  np.sinh(sensitivity) +  np.sinh(sensitivity_lag_1) + np.sinh(sp500_weekly_returns_alternate)", data=blizzard_ent_data_weekly[2:]).fit()


# ---log


blizzard_ent_model_log_positivity_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ np.log(roberta_pos)", data=blizzard_ent_data_weekly).fit()

blizzard_ent_model_log_negativity_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ np.log(roberta_neg)", data=blizzard_ent_data_weekly).fit()

blizzard_ent_model_log_positivity_lagged_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ np.log(roberta_pos) +  np.log(roberta_pos_lag_1)", data=blizzard_ent_data_weekly[1:]).fit()

blizzard_ent_model_log_negativity_lagged_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ np.log(roberta_neg) +  np.log(roberta_neg_lag_1)", data=blizzard_ent_data_weekly[1:]).fit()


blizzard_ent_model_log_positivity_volatility_weekly = smf.ols(
    formula="average_volatility ~ np.log(roberta_pos)", data=blizzard_ent_data_weekly).fit()

blizzard_ent_model_log_negativity_volatility_weekly = smf.ols(
    formula="average_volatility ~ np.log(roberta_neg)", data=blizzard_ent_data_weekly).fit()

blizzard_ent_model_log_positivity_lagged_volatility_weekly = smf.ols(
    formula="average_volatility ~ np.log(roberta_pos) +  np.log(roberta_pos_lag_1)", data=blizzard_ent_data_weekly[1:]).fit()

blizzard_ent_model_log_negativity_lagged_volatility_weekly = smf.ols(
    formula="average_volatility ~ np.log(roberta_neg) +  np.log(roberta_neg_lag_1)", data=blizzard_ent_data_weekly[1:]).fit()


blizzard_ent_model_log_positivity_norm_weekly = smf.ols(
    formula="weekly_return ~ np.log(roberta_pos) + np.log(sp500_weekly_returns_alternate)", data=blizzard_ent_data_weekly).fit()

blizzard_ent_model_log_negativity_norm_weekly = smf.ols(
    formula="weekly_return ~ np.log(roberta_neg) + np.log(sp500_weekly_returns_alternate)", data=blizzard_ent_data_weekly).fit()

blizzard_ent_model_log_positivity_lagged_norm_weekly = smf.ols(
    formula="weekly_return ~ np.log(roberta_pos) +  np.log(roberta_pos_lag_1) + np.log(sp500_weekly_returns_alternate)", data=blizzard_ent_data_weekly[1:]).fit()

blizzard_ent_model_log_negativity_lagged_norm_weekly = smf.ols(
    formula="weekly_return ~ np.log(roberta_neg) +  np.log(roberta_neg_lag_1) + np.log(sp500_weekly_returns_alternate)", data=blizzard_ent_data_weekly[1:]).fit()

# ---sqrt

blizzard_ent_model_sqrt_positivity_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ np.sqrt(roberta_pos)", data=blizzard_ent_data_weekly).fit()

blizzard_ent_model_sqrt_negativity_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ np.sqrt(roberta_neg)", data=blizzard_ent_data_weekly).fit()

blizzard_ent_model_sqrt_positivity_lagged_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ np.sqrt(roberta_pos) +  np.sqrt(roberta_pos_lag_1)", data=blizzard_ent_data_weekly[1:]).fit()

blizzard_ent_model_sqrt_negativity_lagged_return_weekly = smf.ols(
    formula="weekly_abnormal_return ~ np.sqrt(roberta_neg) +  np.sqrt(roberta_neg_lag_1)", data=blizzard_ent_data_weekly[1:]).fit()


blizzard_ent_model_sqrt_positivity_volatility_weekly = smf.ols(
    formula="average_volatility ~ np.sqrt(roberta_pos)", data=blizzard_ent_data_weekly).fit()

blizzard_ent_model_sqrt_negativity_volatility_weekly = smf.ols(
    formula="average_volatility ~ np.sqrt(roberta_neg)", data=blizzard_ent_data_weekly).fit()

blizzard_ent_model_sqrt_positivity_lagged_volatility_weekly = smf.ols(
    formula="average_volatility ~ np.sqrt(roberta_pos) +  np.sqrt(roberta_pos_lag_1)", data=blizzard_ent_data_weekly[1:]).fit()

blizzard_ent_model_sqrt_negativity_lagged_volatility_weekly = smf.ols(
    formula="average_volatility ~ np.sqrt(roberta_neg) +  np.sqrt(roberta_neg_lag_1)", data=blizzard_ent_data_weekly[1:]).fit()


blizzard_ent_model_sqrt_positivity_norm_weekly = smf.ols(
    formula="weekly_return ~ np.sqrt(roberta_pos) + np.sqrt(sp500_weekly_returns_alternate)", data=blizzard_ent_data_weekly).fit()

blizzard_ent_model_sqrt_negativity_norm_weekly = smf.ols(
    formula="weekly_return ~ np.sqrt(roberta_neg) + np.sqrt(sp500_weekly_returns_alternate)", data=blizzard_ent_data_weekly).fit()

blizzard_ent_model_sqrt_positivity_lagged_norm_weekly = smf.ols(
    formula="weekly_return ~ np.sqrt(roberta_pos) +  np.sqrt(roberta_pos_lag_1) + np.sqrt(sp500_weekly_returns_alternate)", data=blizzard_ent_data_weekly[1:]).fit()

blizzard_ent_model_sqrt_negativity_lagged_norm_weekly = smf.ols(
    formula="weekly_return ~ np.sqrt(roberta_neg) +  np.sqrt(roberta_neg_lag_1) + np.sqrt(sp500_weekly_returns_alternate)", data=blizzard_ent_data_weekly[1:]).fit()



# Find out which Blizzard Ent models are significant below 5%

blizzard_ent_model_list = [blizzard_ent_model_linear_sentiment_return_daily, blizzard_ent_model_linear_positivity_return_daily, blizzard_ent_model_linear_negativity_return_daily, blizzard_ent_model_linear_sensitivity_return_daily, blizzard_ent_model_linear_sentiment_lagged_return_daily, blizzard_ent_model_linear_positivity_lagged_return_daily, blizzard_ent_model_linear_negativity_lagged_return_daily, blizzard_ent_model_linear_sensitivity_lagged_return_daily, blizzard_ent_model_linear_sentiment_norm_daily, blizzard_ent_model_linear_positivity_norm_daily, blizzard_ent_model_linear_negativity_norm_daily, blizzard_ent_model_linear_sensitivity_norm_daily, blizzard_ent_model_linear_sentiment_lagged_norm_daily, blizzard_ent_model_linear_positivity_lagged_norm_daily, blizzard_ent_model_linear_negativity_lagged_norm_daily, blizzard_ent_model_linear_sensitivity_lagged_norm_daily, blizzard_ent_model_sinh_sentiment_return_daily, blizzard_ent_model_sinh_positivity_return_daily, blizzard_ent_model_sinh_negativity_return_daily, blizzard_ent_model_sinh_sensitivity_return_daily, blizzard_ent_model_sinh_sentiment_lagged_return_daily, blizzard_ent_model_sinh_positivity_lagged_return_daily, blizzard_ent_model_sinh_negativity_lagged_return_daily, blizzard_ent_model_sinh_sensitivity_lagged_return_daily, blizzard_ent_model_sinh_sentiment_norm_daily, blizzard_ent_model_sinh_positivity_norm_daily, blizzard_ent_model_sinh_negativity_norm_daily, blizzard_ent_model_sinh_sensitivity_norm_daily, blizzard_ent_model_sinh_sentiment_lagged_norm_daily, blizzard_ent_model_sinh_positivity_lagged_norm_daily, blizzard_ent_model_sinh_negativity_lagged_norm_daily, blizzard_ent_model_sinh_sensitivity_lagged_norm_daily, blizzard_ent_model_log_positivity_return_daily, blizzard_ent_model_log_negativity_return_daily, blizzard_ent_model_log_positivity_lagged_return_daily, blizzard_ent_model_log_negativity_lagged_return_daily, blizzard_ent_model_log_positivity_norm_daily, blizzard_ent_model_log_negativity_norm_daily, blizzard_ent_model_log_positivity_lagged_norm_daily, blizzard_ent_model_log_negativity_lagged_norm_daily, blizzard_ent_model_sqrt_positivity_return_daily, blizzard_ent_model_sqrt_negativity_return_daily, blizzard_ent_model_sqrt_positivity_lagged_return_daily, blizzard_ent_model_sqrt_negativity_lagged_return_daily, blizzard_ent_model_sqrt_positivity_norm_daily, blizzard_ent_model_sqrt_negativity_norm_daily, blizzard_ent_model_sqrt_positivity_lagged_norm_daily, blizzard_ent_model_sqrt_negativity_lagged_norm_daily, blizzard_ent_model_linear_sentiment_return_weekly, blizzard_ent_model_linear_positivity_return_weekly, blizzard_ent_model_linear_negativity_return_weekly, blizzard_ent_model_linear_sensitivity_return_weekly, blizzard_ent_model_linear_sentiment_lagged_return_weekly, blizzard_ent_model_linear_positivity_lagged_return_weekly, blizzard_ent_model_linear_negativity_lagged_return_weekly, blizzard_ent_model_linear_sensitivity_lagged_return_weekly, blizzard_ent_model_linear_sentiment_volatility_weekly, blizzard_ent_model_linear_positivity_volatility_weekly, blizzard_ent_model_linear_negativity_volatility_weekly, blizzard_ent_model_linear_sensitivity_volatility_weekly, blizzard_ent_model_linear_sentiment_lagged_volatility_weekly,
                           blizzard_ent_model_linear_positivity_lagged_volatility_weekly, blizzard_ent_model_linear_negativity_lagged_volatility_weekly, blizzard_ent_model_linear_sensitivity_lagged_volatility_weekly, blizzard_ent_model_linear_sentiment_norm_weekly, blizzard_ent_model_linear_positivity_norm_weekly, blizzard_ent_model_linear_negativity_norm_weekly, blizzard_ent_model_linear_sensitivity_norm_weekly, blizzard_ent_model_linear_sentiment_lagged_norm_weekly, blizzard_ent_model_linear_positivity_lagged_norm_weekly, blizzard_ent_model_linear_negativity_lagged_norm_weekly, blizzard_ent_model_linear_sensitivity_lagged_norm_weekly, blizzard_ent_model_sinh_sentiment_return_weekly, blizzard_ent_model_sinh_positivity_return_weekly, blizzard_ent_model_sinh_negativity_return_weekly, blizzard_ent_model_sinh_sensitivity_return_weekly, blizzard_ent_model_sinh_sentiment_lagged_return_weekly, blizzard_ent_model_sinh_positivity_lagged_return_weekly, blizzard_ent_model_sinh_negativity_lagged_return_weekly, blizzard_ent_model_sinh_sensitivity_lagged_return_weekly, blizzard_ent_model_sinh_sentiment_volatility_weekly, blizzard_ent_model_sinh_positivity_volatility_weekly, blizzard_ent_model_sinh_negativity_volatility_weekly, blizzard_ent_model_sinh_sensitivity_volatility_weekly, blizzard_ent_model_sinh_sentiment_lagged_volatility_weekly, blizzard_ent_model_sinh_positivity_lagged_volatility_weekly, blizzard_ent_model_sinh_negativity_lagged_volatility_weekly, blizzard_ent_model_sinh_sensitivity_lagged_volatility_weekly, blizzard_ent_model_sinh_sentiment_norm_weekly, blizzard_ent_model_sinh_positivity_norm_weekly, blizzard_ent_model_sinh_negativity_norm_weekly, blizzard_ent_model_sinh_sensitivity_norm_weekly, blizzard_ent_model_sinh_sentiment_lagged_norm_weekly, blizzard_ent_model_sinh_positivity_lagged_norm_weekly, blizzard_ent_model_sinh_negativity_lagged_norm_weekly, blizzard_ent_model_sinh_sensitivity_lagged_norm_weekly, blizzard_ent_model_log_positivity_return_weekly, blizzard_ent_model_log_negativity_return_weekly, blizzard_ent_model_log_positivity_lagged_return_weekly, blizzard_ent_model_log_negativity_lagged_return_weekly, blizzard_ent_model_log_positivity_volatility_weekly, blizzard_ent_model_log_negativity_volatility_weekly, blizzard_ent_model_log_positivity_lagged_volatility_weekly, blizzard_ent_model_log_negativity_lagged_volatility_weekly, blizzard_ent_model_log_positivity_norm_weekly, blizzard_ent_model_log_negativity_norm_weekly, blizzard_ent_model_log_positivity_lagged_norm_weekly, blizzard_ent_model_log_negativity_lagged_norm_weekly, blizzard_ent_model_sqrt_positivity_return_weekly, blizzard_ent_model_sqrt_negativity_return_weekly, blizzard_ent_model_sqrt_positivity_lagged_return_weekly, blizzard_ent_model_sqrt_negativity_lagged_return_weekly, blizzard_ent_model_sqrt_positivity_volatility_weekly, blizzard_ent_model_sqrt_negativity_volatility_weekly, blizzard_ent_model_sqrt_positivity_lagged_volatility_weekly, blizzard_ent_model_sqrt_negativity_lagged_volatility_weekly, blizzard_ent_model_sqrt_positivity_norm_weekly, blizzard_ent_model_sqrt_negativity_norm_weekly, blizzard_ent_model_sqrt_positivity_lagged_norm_weekly, blizzard_ent_model_sqrt_negativity_lagged_norm_weekly]

blizzard_ent_model_names = ['blizzard_ent_model_linear_sentiment_return_daily', 'blizzard_ent_model_linear_positivity_return_daily', 'blizzard_ent_model_linear_negativity_return_daily', 'blizzard_ent_model_linear_sensitivity_return_daily', 'blizzard_ent_model_linear_sentiment_lagged_return_daily', 'blizzard_ent_model_linear_positivity_lagged_return_daily', 'blizzard_ent_model_linear_negativity_lagged_return_daily', 'blizzard_ent_model_linear_sensitivity_lagged_return_daily', 'blizzard_ent_model_linear_sentiment_norm_daily', 'blizzard_ent_model_linear_positivity_norm_daily', 'blizzard_ent_model_linear_negativity_norm_daily', 'blizzard_ent_model_linear_sensitivity_norm_daily', 'blizzard_ent_model_linear_sentiment_lagged_norm_daily', 'blizzard_ent_model_linear_positivity_lagged_norm_daily', 'blizzard_ent_model_linear_negativity_lagged_norm_daily', 'blizzard_ent_model_linear_sensitivity_lagged_norm_daily', 'blizzard_ent_model_sinh_sentiment_return_daily', 'blizzard_ent_model_sinh_positivity_return_daily', 'blizzard_ent_model_sinh_negativity_return_daily', 'blizzard_ent_model_sinh_sensitivity_return_daily', 'blizzard_ent_model_sinh_sentiment_lagged_return_daily', 'blizzard_ent_model_sinh_positivity_lagged_return_daily', 'blizzard_ent_model_sinh_negativity_lagged_return_daily', 'blizzard_ent_model_sinh_sensitivity_lagged_return_daily', 'blizzard_ent_model_sinh_sentiment_norm_daily', 'blizzard_ent_model_sinh_positivity_norm_daily', 'blizzard_ent_model_sinh_negativity_norm_daily', 'blizzard_ent_model_sinh_sensitivity_norm_daily', 'blizzard_ent_model_sinh_sentiment_lagged_norm_daily', 'blizzard_ent_model_sinh_positivity_lagged_norm_daily', 'blizzard_ent_model_sinh_negativity_lagged_norm_daily', 'blizzard_ent_model_sinh_sensitivity_lagged_norm_daily', 'blizzard_ent_model_log_positivity_return_daily', 'blizzard_ent_model_log_negativity_return_daily', 'blizzard_ent_model_log_positivity_lagged_return_daily', 'blizzard_ent_model_log_negativity_lagged_return_daily', 'blizzard_ent_model_log_positivity_norm_daily', 'blizzard_ent_model_log_negativity_norm_daily', 'blizzard_ent_model_log_positivity_lagged_norm_daily', 'blizzard_ent_model_log_negativity_lagged_norm_daily', 'blizzard_ent_model_sqrt_positivity_return_daily', 'blizzard_ent_model_sqrt_negativity_return_daily', 'blizzard_ent_model_sqrt_positivity_lagged_return_daily', 'blizzard_ent_model_sqrt_negativity_lagged_return_daily', 'blizzard_ent_model_sqrt_positivity_norm_daily', 'blizzard_ent_model_sqrt_negativity_norm_daily', 'blizzard_ent_model_sqrt_positivity_lagged_norm_daily', 'blizzard_ent_model_sqrt_negativity_lagged_norm_daily', 'blizzard_ent_model_linear_sentiment_return_weekly', 'blizzard_ent_model_linear_positivity_return_weekly', 'blizzard_ent_model_linear_negativity_return_weekly', 'blizzard_ent_model_linear_sensitivity_return_weekly', 'blizzard_ent_model_linear_sentiment_lagged_return_weekly', 'blizzard_ent_model_linear_positivity_lagged_return_weekly', 'blizzard_ent_model_linear_negativity_lagged_return_weekly', 'blizzard_ent_model_linear_sensitivity_lagged_return_weekly', 'blizzard_ent_model_linear_sentiment_volatility_weekly', 'blizzard_ent_model_linear_positivity_volatility_weekly', 'blizzard_ent_model_linear_negativity_volatility_weekly', 'blizzard_ent_model_linear_sensitivity_volatility_weekly', 'blizzard_ent_model_linear_sentiment_lagged_volatility_weekly',
                            'blizzard_ent_model_linear_positivity_lagged_volatility_weekly', 'blizzard_ent_model_linear_negativity_lagged_volatility_weekly', 'blizzard_ent_model_linear_sensitivity_lagged_volatility_weekly', 'blizzard_ent_model_linear_sentiment_norm_weekly', 'blizzard_ent_model_linear_positivity_norm_weekly', 'blizzard_ent_model_linear_negativity_norm_weekly', 'blizzard_ent_model_linear_sensitivity_norm_weekly', 'blizzard_ent_model_linear_sentiment_lagged_norm_weekly', 'blizzard_ent_model_linear_positivity_lagged_norm_weekly', 'blizzard_ent_model_linear_negativity_lagged_norm_weekly', 'blizzard_ent_model_linear_sensitivity_lagged_norm_weekly', 'blizzard_ent_model_sinh_sentiment_return_weekly', 'blizzard_ent_model_sinh_positivity_return_weekly', 'blizzard_ent_model_sinh_negativity_return_weekly', 'blizzard_ent_model_sinh_sensitivity_return_weekly', 'blizzard_ent_model_sinh_sentiment_lagged_return_weekly', 'blizzard_ent_model_sinh_positivity_lagged_return_weekly', 'blizzard_ent_model_sinh_negativity_lagged_return_weekly', 'blizzard_ent_model_sinh_sensitivity_lagged_return_weekly', 'blizzard_ent_model_sinh_sentiment_volatility_weekly', 'blizzard_ent_model_sinh_positivity_volatility_weekly', 'blizzard_ent_model_sinh_negativity_volatility_weekly', 'blizzard_ent_model_sinh_sensitivity_volatility_weekly', 'blizzard_ent_model_sinh_sentiment_lagged_volatility_weekly', 'blizzard_ent_model_sinh_positivity_lagged_volatility_weekly', 'blizzard_ent_model_sinh_negativity_lagged_volatility_weekly', 'blizzard_ent_model_sinh_sensitivity_lagged_volatility_weekly', 'blizzard_ent_model_sinh_sentiment_norm_weekly', 'blizzard_ent_model_sinh_positivity_norm_weekly', 'blizzard_ent_model_sinh_negativity_norm_weekly', 'blizzard_ent_model_sinh_sensitivity_norm_weekly', 'blizzard_ent_model_sinh_sentiment_lagged_norm_weekly', 'blizzard_ent_model_sinh_positivity_lagged_norm_weekly', 'blizzard_ent_model_sinh_negativity_lagged_norm_weekly', 'blizzard_ent_model_sinh_sensitivity_lagged_norm_weekly', 'blizzard_ent_model_log_positivity_return_weekly', 'blizzard_ent_model_log_negativity_return_weekly', 'blizzard_ent_model_log_positivity_lagged_return_weekly', 'blizzard_ent_model_log_negativity_lagged_return_weekly', 'blizzard_ent_model_log_positivity_volatility_weekly', 'blizzard_ent_model_log_negativity_volatility_weekly', 'blizzard_ent_model_log_positivity_lagged_volatility_weekly', 'blizzard_ent_model_log_negativity_lagged_volatility_weekly', 'blizzard_ent_model_log_positivity_norm_weekly', 'blizzard_ent_model_log_negativity_norm_weekly', 'blizzard_ent_model_log_positivity_lagged_norm_weekly', 'blizzard_ent_model_log_negativity_lagged_norm_weekly', 'blizzard_ent_model_sqrt_positivity_return_weekly', 'blizzard_ent_model_sqrt_negativity_return_weekly', 'blizzard_ent_model_sqrt_positivity_lagged_return_weekly', 'blizzard_ent_model_sqrt_negativity_lagged_return_weekly', 'blizzard_ent_model_sqrt_positivity_volatility_weekly', 'blizzard_ent_model_sqrt_negativity_volatility_weekly', 'blizzard_ent_model_sqrt_positivity_lagged_volatility_weekly', 'blizzard_ent_model_sqrt_negativity_lagged_volatility_weekly', 'blizzard_ent_model_sqrt_positivity_norm_weekly', 'blizzard_ent_model_sqrt_negativity_norm_weekly', 'blizzard_ent_model_sqrt_positivity_lagged_norm_weekly', 'blizzard_ent_model_sqrt_negativity_lagged_norm_weekly']


blizzard_verified_models = []


length = len(blizzard_ent_model_list)

for i in range(length):

    caller = i

    test = blizzard_ent_model_list[caller].summary()

    testcont = test.tables[0].as_html()

    testcont2 = test.tables[1].as_html()

    testpandas = pd.read_html(testcont, header=0, index_col=0)[0]

    testpandas = testpandas.reset_index()

    testpandas2 = pd.read_html(testcont2, header=0, index_col=0)[0]

    testpandas2 = testpandas2.reset_index()

    testpandas.to_csv('Regression Summary ' + blizzard_ent_model_names[caller] + '.csv')
    testpandas2.to_csv('Regression Variables ' + blizzard_ent_model_names[caller] + '.csv')

    if testpandas.iloc[2, 3] < 0.05:
        print(blizzard_ent_model_list[caller].summary())
        blizzard_verified_models.append(blizzard_ent_model_names[caller])

print(apple_verified_models)

print(blizzard_verified_models)

apple_significant_models = len(apple_verified_models)

blizzard_significant_models = len(blizzard_verified_models)

apple_all_models = len(apple_model_list)

blizzard_all_models = len(blizzard_ent_model_list)