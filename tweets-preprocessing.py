import preprocessor as p
from nltk.corpus import stopwords
import numpy as np


def filter_tweets():

    with open('./train_semeval2018task2/train_semeval2018task2/crawler/data/tweet_by_ID_12_12_2017__11_01_28.txt.text') as f:
        content = f.readlines()

    for i in range(0, len(content)):
        content[i] = p.clean(content[i])
        filtered_tweets = [word for word in content[i] if word not in stopwords.words('english')]

    return filtered_tweets

filtered_tweets = filter_tweets()
print filtered_tweets[0]