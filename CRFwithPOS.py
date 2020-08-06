# adapted from https://towardsdatascience.com/named-entity-recognition-and-classification-with-scikit-learn-f05372f07ba2
# by Susan Li

import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from collections import Counter

df = pd.read_csv('data/trainDataWithPOS.csv', encoding = "ISO-8859-1")
dfTest = pd.read_csv('data/testDataWithPOS.csv', encoding = "ISO-8859-1")

df = df.fillna(method='ffill')
dfTest = dfTest.fillna(method='ffill')
# df['Sentence #'].nunique(), df.Word.nunique(), df.Tag.nunique()

df.groupby('Word').size().reset_index(name='counts')
dfTest.groupby('Word').size().reset_index(name='counts')

X = df.drop('Word', axis=1)
Xtest = dfTest.drop('Word', axis=1)
v = DictVectorizer(sparse=False)
X = v.fit_transform(X.to_dict('records'))
Xtest = v.fit_transform(Xtest.to_dict('records'))
y = df.Tag.values
yTest = dfTest.Tag.values

# classes = np.unique(y)
# classes = classes.tolist()
#
# new_classes = classes.copy()
# new_classes.pop()
#
# new_classes_test = classesTest.copy()
# new_classes_test.pop()


class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s['Word'].values.tolist(),
                                                           s['POS'].values.tolist(),
                                                           s['Tag'].values.tolist())]
        self.grouped = self.data.groupby('Sentence #').apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped['Sentence: {}'.format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None
getter = SentenceGetter(df)
sentences = getter.sentences



def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]
def sent2labels(sent):
    return [label for token, postag, label in sent]
def sent2tokens(sent):
    return [token for token, postag, label in sent]


getter = SentenceGetter(df)
sentences = getter.sentences

getterTest = SentenceGetter(dfTest)
sentencesTest = getterTest.sentences

X_train = [sent2features(s) for s in sentences]
y_train = [sent2labels(s) for s in sentences]
X_test = [sent2features(s) for s in sentencesTest]
y_test = [sent2labels(s) for s in sentencesTest]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.2,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)

crf.fit(X_train, y_train)

y_pred = crf.predict(X_test)

# for input, prediction, label in zip(sentences, y_pred, y_test):
#
#     if prediction != label:
#         print(input)
#         print(prediction, ' should be ')
#         print(label)

print(metrics.flat_classification_report(y_test, y_pred))
#labels = new_classes
