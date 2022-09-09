import pytest
import pandas as pd
from utils import load_data, delete_columns, preprocess_tweet_text

TEST_DATA = "iris"

def test_load_data() -> None:
    #do not crash
    df_1 = load_data(TEST_DATA)
    
    with pytest.raises(FileNotFoundError):
       df_2 = load_data("FAKE")
    
    

def test_prepreocess_data() -> None:
    d = {'col1': ["dog, cat", "I am the HERO."]}
    df = pd.DataFrame(data=d)
    res = df["col1"].apply(preprocess_tweet_text).tolist()
    
    assert res[0] == "dog cat"
    assert res[1] == "hero"

def test_delete_columns() -> None:
    df = pd.read_csv("data/iris.csv")
    df2 = delete_columns(df, ["petal.length"])
    
    assert df.shape[1] == 5
    assert df2.shape[1] == 4

    with pytest.raises(Exception):
       df_3 = delete_columns(df, ["FAKE"])

# python -m pytest tests/test_utils.py