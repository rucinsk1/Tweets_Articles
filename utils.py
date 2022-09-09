import re
from signal import raise_signal
from typing import List, Optional
import pandas as pd
import numpy as np
import re
import string
from os.path import exists
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os

DATA_FILENAME_FORMAT = "data/{}.csv"
STOP_WORDS = set(stopwords.words('english'))

def load_data(csv_filename : str, columns : Optional[List[str]] = None) -> pd.DataFrame:
    filepath = DATA_FILENAME_FORMAT.format(csv_filename)
    if exists(filepath):
        if columns is not None:
            df = pd.read_csv(filepath, columns = columns)
        else:
            df = pd.read_csv(filepath)
        return df
    else:
        raise FileNotFoundError(f"There is no file {filepath}")

def preprocess_tweet_text(tweet : pd.Series) -> pd.Series:
    tweet = tweet.lower()
    # Remove urls
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet
    tweet = re.sub(r'\@\w+|\#','', tweet)
    # Remove punctuations
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [w for w in tweet_tokens if not w in STOP_WORDS]
    
    return " ".join(filtered_words)

def delete_columns(df : pd.DataFrame, columns : List[str]) -> pd.DataFrame:
    if set(columns).issubset(df.columns):
        return df.drop(columns = columns)      
    else:
        raise Exception("Cannot delete columns that are not in dataframe!")