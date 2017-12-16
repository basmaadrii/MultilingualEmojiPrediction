import preprocessor as p
from nltk.corpus import stopwords
import string
import numpy as np


def filter_tweets():

    with open('./train_semeval2018task2/train_semeval2018task2/crawler/data/tweet_by_ID_12_12_2017__11_01_28.txt.text') as f:
        tweets = f.readlines()

    f = open('filtered_tweets.txt', 'w')

    for i in range(0, len(tweets)):
        print '\n\ntweet #' + str(i)
        print 'before:'
        print tweets[i]
        tweets[i] = p.clean(tweets[i])
        tweets[i] = filter(lambda x: x in set(string.printable), tweets[i])
        tweets[i] = tweets[i].encode().split()
        tweets[i] = [word for word in tweets[i] if word not in stopwords.words('english')]
        print 'after:'
        print tweets[i]
        for w in tweets[i]:
            f.write(w + ' ')
        f.write('\n')

    return tweets

filtered_tweets = filter_tweets()