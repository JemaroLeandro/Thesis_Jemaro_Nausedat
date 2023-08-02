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

sandp500 = yf.download("^GSPC", period='1wk', start='2023-05-26',end='2023-07-31')

cols = ['Open', 'Adj Close']

sp500_df = sandp500[cols]

sp500_df['daily_return'] = (sp500_df['Adj Close'] / sp500_df['Open']) - 1

sp500_df.to_csv('sandp500 daily returns.csv')