import preprocessor as p
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import string


def filter_tweets():

    lmtzr = WordNetLemmatizer()
    ps = PorterStemmer()
    tokenizer = RegexpTokenizer(r'\w+')

    with open('./train_semeval2018task2/train_semeval2018task2/crawler/data/tweet_by_ID_12_12_2017__11_01_28.txt.text') as f:
        tweets = f.readlines()

    f = open('filtered_tweets.txt', 'w')

    for i in range(0, len(tweets)):
        print '\n\ntweet #' + str(i)
        print 'before:'
        print tweets[i]
        tweets[i] = p.clean(tweets[i])
        tweets[i] = filter(lambda x: x in set(string.printable), tweets[i])
        tweets[i] = tokenizer.tokenize(tweets[i].encode())
        tweets[i] = [word.lower() for word in tweets[i] if word.lower() not in stopwords.words('english')]
        tweets[i] = [lmtzr.lemmatize(word) for word in tweets[i]]
        tweets[i] = [lmtzr.lemmatize(word, 'v') for word in tweets[i]]

        print 'after:'
        print tweets[i]
        for w in tweets[i]:
            f.write(w + ' ')
        f.write('\n')

    return tweets

filtered_tweets = filter_tweets()