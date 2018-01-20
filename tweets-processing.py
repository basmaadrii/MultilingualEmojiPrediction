import preprocessor as p
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import string
import fasttext
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

def filter_tweets(fname='./train_semeval2018task2/train_semeval2018task2/crawler/data/tweet_by_ID_12_12_2017__11_01_28.txt.text'):
    lmtzr = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r'\w+')

    with open(fname) as f:
        tweets = f.readlines()

    f = open('filtered_tweets.txt', 'w')

    maximum = 0
    m = len(tweets)

    p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION)
    temp = list(tweets)
    for i in range(0, m):
        print '\n\ntweet #' + str(i)
        print 'before:'
        print tweets[i]
        tweets[i] = p.clean(tweets[i])
        tweets[i] = filter(lambda x: x in set(string.printable), tweets[i])
        tweets[i] = tokenizer.tokenize(tweets[i].encode())

        temp[i] = list([])
        for j in range(0, len(tweets[i])):
            words = split_on_uppercase(tweets[i][j])
            for k in range(0, len(words)):
                temp[i].append(words[k])
        tweets[i] = list(temp[i])

        tweets[i] = [word.lower() for word in tweets[i] if word.lower() not in stopwords.words('english')]
        tweets[i] = [lmtzr.lemmatize(word) for word in tweets[i]]
        tweets[i] = [lmtzr.lemmatize(word, 'v') for word in tweets[i]]

        print 'after:'
        print tweets[i]
        for w in tweets[i]:
            f.write(w + ' ')
        f.write('\n')
        if len(tweets[i]) > maximum:
            maximum = len(tweets[i])
    print maximum


def word_embedding(fname='filtered_tweets.txt'):
    vec_dim = 10
    maximum = 24
    tokenizer = RegexpTokenizer(r'\w+')

    with open(fname) as f:
        tweets = f.readlines()
    m = len(tweets)

    for i in range(0, m):
        tweets[i] = tokenizer.tokenize(tweets[i])

    model = fasttext.skipgram(fname, 'model', dim=vec_dim)

    for i in range(0, m):
        print '\n\ntweet #' + str(i)
        tweets[i] = np.array([model[w] for w in tweets[i]])
        if len(tweets[i]) == 0:
            tweets[i] = np.zeros([maximum, vec_dim])
        elif len(tweets[i]) < maximum:
            padlen = maximum - len(tweets[i])
            padding = np.zeros([padlen, vec_dim])
            tweets[i] = np.concatenate((np.array(tweets[i]), padding), axis=0)
        tweets[i] = tweets[i].flatten()

    return tweets


def data_partition(data):

    with open('./train_semeval2018task2/train_semeval2018task2/crawler/data/tweet_by_ID_12_12_2017__11_01_28.txt.labels') as f:
        labels = [int(x) for x in f]
    j = 0
    k = 0

    train = np.zeros([341080, 240])
    test = np.zeros([146175, 240])

    train_labels = np.zeros([341080,])
    test_labels = np.zeros([146175,])

    for i in range(0, len(data)):
        if i % 10 < 7:
            train[j][:] = data[i]
            train_labels[j] = labels[i]
            j += 1
        else:
            test[k][:] = data[i]
            test_labels[k] = labels[i]
            k += 1
    return train, test, train_labels, test_labels


def split_on_uppercase(s, keep_contiguous=True):
    """

    Args:
        s (str): string
        keep_contiguous (bool): flag to indicate we want to
                                keep contiguous uppercase chars together

    Returns:

    """

    string_length = len(s)
    is_lower_around = (lambda: s[i-1].islower() or
                       string_length > (i + 1) and s[i + 1].islower())

    start = 0
    parts = []
    for i in range(1, string_length):
        if s[i].isupper() and (not keep_contiguous or is_lower_around()):
            parts.append(s[start: i])
            start = i
    parts.append(s[start:])

    return parts


filter_tweets()
filtered_tweets = word_embedding()
np.save('data', filtered_tweets)
print '\ndata saved...\n'
#filtered_tweets = np.load('data.npy')
[train, test, train_labels, test_labels] = data_partition(filtered_tweets)

f = open('gold.txt', 'w')
for x in test_labels:
    f.write(str(x))
    f.write('\n')


clf = tree.DecisionTreeClassifier()
clf.fit(train, train_labels)
y_pred = clf.predict(test)
print '\nDecision Tree Accuracy: '
print accuracy_score(test_labels, y_pred)
f = open('output1.txt', 'w')
for x in y_pred:
    f.write(str(x))
    f.write('\n')

clf = BernoulliNB()
clf.fit(train, train_labels)
y_pred = clf.predict(test)
print '\nNaive Bayes accuracy: '
print accuracy_score(test_labels, y_pred)
f = open('output2.txt', 'w')
for x in y_pred:
    f.write(str(x))
    f.write('\n')

clf = RandomForestClassifier(max_depth=20)
clf.fit(train, train_labels)
y_pred = clf.predict(test)
print '\nRandom Forest accuracy: '
print accuracy_score(test_labels, y_pred)
f = open('output3.txt', 'w')
for x in y_pred:
    f.write(str(x))
    f.write('\n')

clf = AdaBoostClassifier(n_estimators=100)
clf.fit(train, train_labels)
y_pred = clf.predict(test)
print '\nAdaboost accuracy: '
print accuracy_score(test_labels, y_pred)
f = open('output4.txt', 'w')
for x in y_pred:
    f.write(str(x))
    f.write('\n')

