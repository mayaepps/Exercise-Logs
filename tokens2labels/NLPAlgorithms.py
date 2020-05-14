import random

import pandas as pd
import numpy as np

from fuzzywuzzy.process import extractOne

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.feature_extraction.text import CountVectorizer

from scipy.sparse import csr_matrix

from py_stringmatching.similarity_measure.monge_elkan import MongeElkan
from py_stringmatching.similarity_measure.jaro_winkler import JaroWinkler
from py_stringmatching.similarity_measure.tfidf import TfIdf
from py_stringmatching.similarity_measure.soft_tfidf import SoftTfIdf
from py_stringmatching.tokenizer.alphanumeric_tokenizer import AlphanumericTokenizer

def stringMatching(data, label_options, print_output = False):
    
    processed_rows, correct_rows = 0, 0
    total_rows = len(data.index)
    
    predictions, scores, correct_matches = [], [], []
    
    for row_index, row in data.iterrows():
        
        prediction, score = extractOne(row['text'], label_options)
        predictions.append(prediction)
        scores.append(score)
        
        if prediction == row['label']:
            correct_rows += 1
            correct_matches.append(True)
        else:
            correct_matches.append(False)
        
        if print_output:
            print('\rProgress: Completed: {:.2%} --- Accuracy: {:.2%}'.format((row_index+1)/total_rows, correct_rows/(row_index+1)), end='')
    
    results = data
    results['predicted-label'] = predictions
    results['label-score'] = scores
    results['correct_match'] = correct_matches

    accuracy = correct_rows/total_rows

    if print_output:
        print('\nFinal Accuracy: {:.2%}'.format(accuracy))

    return accuracy, results

def logisticMulticlass(train_data, test_data, print_output=False, dummy_labels = False):
    train_text = train_data['text']
    train_label = train_data['label']
    train_text_dummie = pd.get_dummies(train_text)

    test_text = test_data['text']
    test_label = test_data['label']
    test_text_dummie = pd.get_dummies(test_text)

    if dummy_labels:
        train_label_dummie = pd.get_dummies(train_label)
        test_label_dummie = pd.get_dummies(test_label)
    else:
        train_label_dummie = train_label
        test_label_dummie = test_label

    classifier = OneVsRestClassifier(LogisticRegression(solver = 'lbfgs', random_state = 0))
    classifier.fit(train_text_dummie, train_label_dummie)

    test_label_predictions = classifier.predict(test_text_dummie)
    classification_report_result = classification_report(test_label_dummie, test_label_predictions)

    if print_output:
        print('Classification Report for Multiclass Logistic Regression:')
        print(classification_report_result)
    return classifier, classification_report_result 

def data2BOW_match(data, label_options, bow_vec):
    vec_data = []
    match_data = []

    for row_index, row in data.iterrows():

        vec_row = bow_vec.transform([row['text']+' '+str(row['label'])])
        vec_data.append(vec_row.toarray()[0])
        match_data.append(csr_matrix([1], dtype='int64').toarray()[0])

        while True:
            new_label = random.choice(label_options)
            if new_label != row['label']:
                break

        row['label'] = new_label
        vec_row = bow_vec.transform([row['text']+' '+row['label']])
        vec_data.append(vec_row.toarray()[0])
        match_data.append(csr_matrix([0], dtype='int64').toarray()[0])

    return np.array(vec_data), np.array(match_data).ravel()

def logisticBinaryMatch(train_data, test_data, label_options, print_output=False):
    
    corpus = list(train_data['text']) + list(test_data['text']) + [str(x) for x in train_data['label']] + [str(x) for x in test_data['label']]

    bow_vec = CountVectorizer()
    bow_vec.fit(corpus)

    vec_train_data, match_train_data = data2BOW_match(train_data, label_options, bow_vec)
    vec_test_data, match_test_data = data2BOW_match(test_data, label_options, bow_vec)

    classifier = LogisticRegression(solver='lbfgs', random_state=0, max_iter=10000)
    classifier.fit(vec_train_data, match_train_data)

    match_test_predictions = classifier.predict(vec_test_data)

    confusion_matrix_result = confusion_matrix(match_test_data, match_test_predictions)
    classification_report_result = classification_report(match_test_data, match_test_predictions)

    if print_output:
        print('Confusion Matrix:')
        print(confusion_matrix_result)
        print('\nClassification Report:')
        print(classification_report_result)

    return classifier, confusion_matrix_result, classification_report_result

def data2BOW_multiclass(data, bow_vec, column):
    vec_data = []

    for row_index, row in data.iterrows():

        vec_row = bow_vec.transform([row[column]])
        vec_data.append(vec_row.toarray()[0])

    return np.array(vec_data)

def logisticMulticlassBOW(train_data, test_data, print_output=False, dummy_labels = False):

    text_corpus = list(train_data['text']) + list(test_data['text'])
    bow_vec_text = CountVectorizer()
    bow_vec_text.fit(text_corpus)

    train_label = train_data['label']
    test_label = test_data['label']

    vec_train_data = data2BOW_multiclass(train_data, bow_vec_text, 'text')
    vec_test_data = data2BOW_multiclass(test_data, bow_vec_text, 'text')

    if dummy_labels:
        label_corpus = list(train_data['label']) + list(test_data['label'])
        bow_vec_label = CountVectorizer()
        bow_vec_label.fit(label_corpus)
        train_label_dummie = data2BOW_multiclass(train_data, bow_vec_label, 'label')
        test_label_dummie = data2BOW_multiclass(test_data, bow_vec_label, 'label')
    else:
        train_label_dummie = train_label
        test_label_dummie = test_label

    classifier = OneVsRestClassifier(LogisticRegression(solver = 'lbfgs', random_state = 0))
    classifier.fit(vec_train_data, train_label_dummie)

    test_label_predictions = classifier.predict(vec_test_data)
    classification_report_result = classification_report(test_label_dummie, test_label_predictions)

    if print_output:
        print('Classification Report for Multiclass Logistic Regression:')
        print(classification_report_result)
    return classifier, classification_report_result 

def calculate_features(str1, str2):

    me = MongeElkan()
    jw = JaroWinkler()
    tfidf = TfIdf(dampen = False)
    stdidf = SoftTfIdf()
    tokenizer = AlphanumericTokenizer()

    str1 = str1.casefold()
    str2 = str2.casefold()
    bag1 = tokenizer.tokenize(str1)
    bag2 =tokenizer.tokenize(str2)
    
    monge_elkan = me.get_raw_score(bag1, bag2)
    jaro_winkler = jw.get_sim_score(str1, str2)
    tf_idf = tfidf.get_raw_score(bag1, bag2)
    soft_tfidf = stdidf.get_raw_score(bag1, bag2)
    
    return monge_elkan, jaro_winkler, tf_idf, soft_tfidf

def data2features(data, label_options):
    features_data = []
    match_data = []

    for row_index, row in data.iterrows():

        features_row = calculate_features(row['text'], str(row['label']))
        features_data.append(features_row)
        match_data.append(1)

        while True:
            new_label = random.choice(label_options)
            if new_label != row['label']:
                break

        row['label'] = new_label
        features_row = calculate_features(row['text'], str(row['label']))
        features_data.append(features_row)
        match_data.append(0)

    return features_data, match_data

def logisticStrComparisson(train_data, test_data, label_options, print_output=False):

    features_train_data, match_train_data = data2features(train_data, label_options)
    features_test_data, match_test_data = data2features(test_data, label_options)

    classifier = LogisticRegression(solver='lbfgs', random_state=0)
    classifier.fit(features_train_data, match_train_data)

    match_test_predictions = classifier.predict(features_test_data)

    confusion_matrix_result = confusion_matrix(match_test_data, match_test_predictions)
    classification_report_result = classification_report(match_test_data, match_test_predictions)

    if print_output:
        print('Confusion Matrix:')
        print(confusion_matrix_result)
        print('\nClassification Report:')
        print(classification_report_result)

    return classifier, confusion_matrix_result, classification_report_result