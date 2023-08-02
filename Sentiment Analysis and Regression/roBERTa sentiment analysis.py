
import Config
import transformers
import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

import os


from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
from tqdm.notebook import tqdm

from datetime import datetime

today = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

# Download roBERTa model

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# read in data and choose company

df = pd.read_csv(Config.file_name1)
df2 = pd.read_csv(Config.file_name2)

example = df['Text'][0]

# define function for transforming text into tokens, tokens into numbers and numbers into probabilities


def twitter_sample_analysis(example):
    encoded_text = tokenizer(example, return_tensors ='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2],
        'roberta_sum' : scores[2] - scores[0],
        }
    return scores_dict

# Iterate through dataset and create dictionary for results

res1 = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['ID']
    created_at = row['Created_At']
    result = twitter_sample_analysis(text)
    res1[myid] = result

res2 = {}
for i, row in tqdm(df2.iterrows(), total=len(df2)):
    text = row['Text']
    myid = row['ID']
    created_at = row['Created_At']
    result = twitter_sample_analysis(text)
    res2[myid] = result



# Transform dictionary to pandas dataframe and merge on original dataset

results_df = pd.DataFrame(res1).T
results_df = results_df.reset_index().rename(columns={'index': 'ID'})
results_df = results_df.merge(df, how='left')

results_df2 = pd.DataFrame(res2).T
results_df2 = results_df2.reset_index().rename(columns={'index': 'ID'})
results_df2 = results_df2.merge(df2, how='left')


sns.histplot(data=results_df, x = 'roberta_sum')

plt.show()


results_df.to_csv(Config.processed_file_name1)

results_df2.to_csv(Config.processed_file_name2)

