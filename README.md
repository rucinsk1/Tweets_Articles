# Tweets_Articles
Sentiment analysis of tweets, Fake news detection

## Before running the code
Python 3.9.6 is required. After installation run:
```
pip install requirements.txt
```
After succesfully intalling packages, launch the cell in preparations.ipynb

## Solution
Two models (Logistics Regression and Naive Bayes) were built and compared for each problem.
- Fake news detection data preparation + model creation are in ex_1.ipynb
- Tweet sentiment extraction data preparation + model creation are in ex_2.ipynb

## Tests
Run ``` python -m pytest tests/test_utils.py ``` from Tweets_Articles folder.
