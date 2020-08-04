'''
Description: Uses sklearn classifiers and neural network models to predict the matching exercise.
Author: Mandy Korpusik
Date: 4/21/2020
Reference: https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/
'''

import pandas
import numpy
import string
import json
import textblob

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn import decomposition, ensemble


def textblob_tokenizer(str_input):
    '''Tokenizer that also does lowercasing and stemming.'''
    blob = textblob.TextBlob(str_input.lower())
    tokens = blob.words
    words = [token.stem() for token in tokens]
    return words


def get_exercise_data():
    '''Extracts the input exercise logs, and outputs the label "exercise".'''
    labels, sents = [], []
    train_data = json.load(open('trainDataValues.json'))
    test_data = json.load(open('testDataValues.json'))
    for data in [train_data, test_data]:
        for sent_id in data:
            tokens = data[sent_id]['sentence']
            exercise_tokens = data[sent_id]['exercise segment']
            feeling_tokens = data[sent_id]['feeling segment']
            exercise = data[sent_id]['exercise value']
            feeling = data[sent_id]['exercise value']
            labels.append("exercise")
            sents.append(tokens)
    return sents, labels


def get_food_data(sents, labels):
    '''Extracts the input food logs, and outputs the label "food".'''
    num_exercise_sents = len(sents) # keep the data balanced
    for line in open('food_logs').readlines()[:num_exercise_sents]:
        sents.append(line.strip())
        labels.append("food")
    return sents, labels


def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    '''Trains and tests the given classifier on given input features and output labels.'''
    classifier.fit(feature_vector_train, label)

    # Predict the labels on validation dataset.
    predictions = classifier.predict(feature_vector_valid)

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)

    return metrics.accuracy_score(predictions, valid_y)


def ablation_study(model, model_name):
    '''Trains and tests a given model on various feature sets.'''
    
    # Count vectors
    accuracy = train_model(model, xtrain_count, train_y, xvalid_count)
    print("\n" + model_name, "Count vectors: ", accuracy)

    # Word-level TF-IDF vectors
    accuracy = train_model(model, xtrain_tfidf, train_y, xvalid_tfidf)
    print(model_name, "Unigram TF-IDF: ", accuracy)

    # Bigram (2-word) TF-IDF vectors
    accuracy = train_model(model, xtrain_tfidf_bigram, train_y, xvalid_tfidf_bigram)
    print(model_name, "Bigram TF-IDF: ", accuracy)

    # Trigram (3-word) TF-IDF vectors
    accuracy = train_model(model, xtrain_tfidf_trigram, train_y, xvalid_tfidf_trigram)
    print(model_name, "Trigram TF-IDF: ", accuracy)

    # All n-grams (1-, 2-, and 3-word) TF-IDF vectors
    accuracy = train_model(model, xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
    print(model_name, "All n-gram TF-IDF: ", accuracy)

    # Character-level TF-IDF vectors
    accuracy = train_model(model, xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
    print(model_name, "Char-level n-gram TF-IDF: ", accuracy)

    # N-grams and Character-level TF-IDF vectors
    accuracy = train_model(model, xtrain_tfidf_ngram_word_char, train_y, xvalid_tfidf_ngram_word_char)
    print(model_name, "Word and char n-gram TF-IDF: ", accuracy)

    # N-grams and counts
    accuracy = train_model(model, xtrain_tfidf_ngrams_counts, train_y, xvalid_tfidf_ngrams_counts)
    print(model_name, "Word and char n-gram TF-IDF and counts: ", accuracy)


if __name__ == '__main__':

    ########### LOAD THE DATA AND FEATURES #############

    sents, labels = get_exercise_data()
    sents, labels = get_food_data(sents, labels)

    # create a dataframe using texts and lables
    trainDF = pandas.DataFrame()
    trainDF['text'] = sents
    trainDF['label'] = labels

    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'], test_size=0.1)

    # label encode the target variable (i.e., turns exercise string into int label)
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    valid_y = encoder.fit_transform(valid_y)

    # create a count vectorizer object
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', tokenizer=textblob_tokenizer)
    count_vect.fit(trainDF['text'])
    # transform the training and validation data using count vectorizer object
    xtrain_count =  count_vect.transform(train_x)
    xvalid_count =  count_vect.transform(valid_x)

    # word-level tf-idf
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000, tokenizer=textblob_tokenizer)
    tfidf_vect.fit(trainDF['text'])
    xtrain_tfidf =  tfidf_vect.transform(train_x)
    xvalid_tfidf =  tfidf_vect.transform(valid_x)

    # bigram-level tf-idf
    tfidf_vect_bigram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,2), max_features=5000, tokenizer=textblob_tokenizer)
    tfidf_vect_bigram.fit(trainDF['text'])
    xtrain_tfidf_bigram =  tfidf_vect_bigram.transform(train_x)
    xvalid_tfidf_bigram =  tfidf_vect_bigram.transform(valid_x)

    # triigram-level tf-idf
    tfidf_vect_trigram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(3,3), max_features=5000, tokenizer=textblob_tokenizer)
    tfidf_vect_trigram.fit(trainDF['text'])
    xtrain_tfidf_trigram =  tfidf_vect_trigram.transform(train_x)
    xvalid_tfidf_trigram =  tfidf_vect_trigram.transform(valid_x)

    # all n-grams tf-idf
    tfidf_vect_ngram = FeatureUnion([('unigram vectorizer', tfidf_vect),
                                     ('bigram vectorizer', tfidf_vect_bigram),
                                     ('trigram vectorizer', tfidf_vect_trigram)])
    tfidf_vect_ngram.fit(trainDF['text'])
    xtrain_tfidf_ngram = tfidf_vect_ngram.transform(train_x)
    xvalid_tfidf_ngram = tfidf_vect_ngram.transform(valid_x)

    # character-level (bigram and tri-gram) tf-idf
    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000, tokenizer=textblob_tokenizer)
    tfidf_vect_ngram_chars.fit(trainDF['text'])
    xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x)
    xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x)

    # word and char ngrams tf-idf
    tfidf_vect_ngram_word_char = FeatureUnion([('unigram vectorizer', tfidf_vect),
                                     ('bigram vectorizer', tfidf_vect_bigram),
                                     ('trigram vectorizer', tfidf_vect_trigram),
                                     ('char vectorizer', tfidf_vect_ngram_chars)])
    tfidf_vect_ngram_word_char.fit(trainDF['text'])
    xtrain_tfidf_ngram_word_char = tfidf_vect_ngram_word_char.transform(train_x)
    xvalid_tfidf_ngram_word_char = tfidf_vect_ngram_word_char.transform(valid_x)

    # word and char ngrams tf-idf and word counts
    tfidf_vect_ngrams_counts = FeatureUnion([('Counter vectorizer', count_vect),
                                     ('unigram vectorizer', tfidf_vect),
                                     ('bigram vectorizer', tfidf_vect_bigram),
                                     ('trigram vectorizer', tfidf_vect_trigram),
                                     ('char vectorizer', tfidf_vect_ngram_chars)])
    tfidf_vect_ngrams_counts.fit(trainDF['text'])
    xtrain_tfidf_ngrams_counts = tfidf_vect_ngrams_counts.transform(train_x)
    xvalid_tfidf_ngrams_counts = tfidf_vect_ngrams_counts.transform(valid_x)




    ########### TRAIN AND TEST CLASSIFIERS #############

    # logistic regression and random forest get over 99% accuracy on test set
    ablation_study(linear_model.LogisticRegression(), "Logistic Regression")
    ablation_study(ensemble.RandomForestClassifier(), "Random Forest")
