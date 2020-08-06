'''
Description: Maps exercise logs to exercise/feeling values using embeddings.
Author: Mandy korpusik
Date: 8/5/2020
'''

import json
import nnets
import numpy as np

from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity


def get_x_y(data):
    '''Loads exercise/feeling tokens and labels, given input data.'''
    exercise_tokens, feeling_tokens = [], []
    exercise_labels, feeling_labels = [], []
    for sent_id in data:
        tokens = data[sent_id]['sentence']
        # Option to use the full sentence, instead of just the segments.
        if FULL_SENTENCE:
            exercise_tokens.append(tokens.lower())
            feeling_tokens.append(tokens.lower())
        else:
            exercise_tokens.append(data[sent_id]['exercise segment'].lower())
            feeling_tokens.append(data[sent_id]['feeling segment'].lower())
        exercise_labels.append(data[sent_id]['exercise value'].lower())
        feeling_value = int(data[sent_id]['feeling value'])

        # If the do_binarize flag is set, feeling labels are set to 0 or 1.
        if DO_BINARIZE and feeling_value > 5:
            feeling_labels.append(1)
        elif DO_BINARIZE:
            feeling_labels.append(0)
        else:
            feeling_labels.append(feeling_value)
    return exercise_tokens, feeling_tokens, exercise_labels, feeling_labels


def nearest_neighbor(embed, label_vecs):
    '''Rank label vectors by cosine dist with the given segment emedding.'''
    best_match = None
    best_similarity = 0
    # Skip unknown segments.
    if type(embed) == np.float64:
        return
    for label in label_vecs:
        # Skip unknown labels.
        if type(label_vecs[label]) == np.float64:
            continue
        new_similarity = cosine_similarity([embed], [label_vecs[label]])[0][0]
        if new_similarity > best_similarity:
            best_match = label
            best_similarity = new_similarity
    return best_match


def load_label_vecs(word_vecs, labels):
    '''Maps from all exercise/feeling labels to embedding ectors.'''
    label_vecs = {}
    for label in labels:
        label_vecs[label] = get_vec(word_vecs, label)
    return label_vecs


def get_vec(word_vecs, text, do_stem=False):
    '''Sums the embeddings for each token in the given text.'''
    vecs_list = []
    for token in text.split():
        # Stem tokens before looking up embeddings.
        if do_stem:
            token = stemmer.stem(token)
        if token in word_vecs:
            vecs_list.append(word_vecs[token])
    return np.sum(vecs_list, axis=0)


if __name__ == '__main__':

    DO_BINARIZE = True
    FULL_SENTENCE = False
    EMBEDDING = "glove"

    ################## LOAD THE DATA ##################

    stemmer = PorterStemmer()

    train_data = json.load(open('trainDataValues.json'))
    test_data = json.load(open('testDataValues.json'))

    tr_ex_text, tr_feel_text, tr_ex_vals, tr_feel_vals = get_x_y(train_data)
    test_ex_text, test_feel_text, test_ex_vals, test_feel_vals = get_x_y(test_data)

    # Gets each unique exercise label and feeling tokens (seen in training).
    all_ex_labels = set(tr_ex_vals + test_ex_vals)
    all_feel_labels = {} # maps from feeling segment to numeric value
    for feel_seg, feel_val in zip(tr_feel_text, tr_feel_vals):
        all_feel_labels[feel_seg] = feel_val

    ############### EMBEDDING SIMILARITY ###############

    if EMBEDDING == "glove":
        word_vecs = nnets.load_glove()
    elif EMBEDDING == "word2vec":
        word_vecs = nnets.load_word2vec()
    elif EMBEDDING == "fastText":
        word_vecs = nnets.load_fastText()

    ex_label_vecs = load_label_vecs(word_vecs, all_ex_labels)
    feel_vecs = load_label_vecs(word_vecs, all_feel_labels.keys())

    top_1_ex = 0
    top_1_feel = 0
    total_ex = len(test_ex_vals)
    total_feel = len(test_feel_vals)

    # Iterates through each test sample for exercise and feeling segments.
    for ex_seg, feel_seg, ex_true, feel_true in zip(test_ex_text, test_feel_text, 
                                                    test_ex_vals, test_feel_vals):
        ex_embed = get_vec(word_vecs, ex_seg)
        feel_embed = get_vec(word_vecs, feel_seg)

        # Finds the nearest neighbor label embedding to segment embedding.
        ex_pred = nearest_neighbor(ex_embed, ex_label_vecs)
        feel_pred = nearest_neighbor(feel_embed, feel_vecs)

        # Convert feeling description to a numeric value.
        if feel_pred is not None:
            feel_pred = all_feel_labels[feel_pred]
        #print('\n' + ex_seg)
        #print(ex_pred)

        #print('\n' + feel_seg)
        #print(feel_pred)

        # Increments number of correct matches.
        if ex_pred == ex_true:
            top_1_ex += 1
        if feel_pred == feel_true:
            top_1_feel += 1

    print("Embedding type:", EMBEDDING)
    print("Do binarize:", DO_BINARIZE)
    print("Full sentence:", FULL_SENTENCE)

    print('\nExercise embedding top-1 recall:', top_1_ex / total_ex)
    print('Feeling embedding top-1 recall:', top_1_feel / total_feel)

