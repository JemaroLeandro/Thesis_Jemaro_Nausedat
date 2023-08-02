import pandas as pd
import numpy as np
import Config
import seaborn as sns
import matplotlib as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tools.tools as smt
import statsmodels.stats.diagnostic as smd


weeklydescriptives1 = pd.read_csv('Adjusted Regression Datasets - Apple Regression Dataset.csv')

dailydescriptives1 = pd.read_csv('Adjusted Regression Datasets - Apple Daily Regression Dataset.csv')

weeklydescriptives1 = weeklydescriptives1.describe().T

dailydescriptives1 = dailydescriptives1.describe().T

weeklydescriptives2 = pd.read_csv('Adjusted Regression Datasets - Blizzard_Ent Regression Dataset.csv')

dailydescriptives2 = pd.read_csv('Adjusted Regression Datasets - Blizzard_Ent Daily Regression Dataset.csv')

weeklydescriptives2 = weeklydescriptives2.describe().T

dailydescriptives2 = dailydescriptives2.describe().T

weeklydescriptives1.to_csv('Descriptive Statistics of Weekly Variables - Apple.csv')

weeklydescriptives2.to_csv('Descriptive Statistics of Weekly Variables - Activision Blizzard.csv')

dailydescriptives1.to_csv('Descriptive Statistics of Daily Variables - Apple.csv')

dailydescriptives2.to_csv('Descriptive Statistics of Daily Variables - Activision Blizzard.csv')