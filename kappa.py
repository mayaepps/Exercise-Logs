'''
Description: Computes the kappa score for per-token inter-annotator agreement.
Author: Mandy Korpusik
Date: 2020-08-04
'''

import csv
import json
from collections import defaultdict


def read_tags_data(filename):
    '''Builds a dictionary mapping tokens to tag frequencies.'''
    token_tags = defaultdict(lambda: defaultdict(int))

    with open(filename) as csvfile:
        csvreader = csv.reader(csvfile)

        # List of all the fields.                                                                                     
        fields = csvreader.__next__()
        word_index = fields.index('Word')
        tag_index = fields.index('Tag')
        
        for row in csvreader:
            if len(row) == 0:
                continue
            token = row[word_index]
            tag = row[tag_index]
            token_tags[token][tag] += 1
    return token_tags


def read_exercise_data():
    '''Builds a dictionary mapping tokens to exercise value frequencies.'''
    token_vals = defaultdict(lambda: defaultdict(int))
    train_data = json.load(open('trainDataValues.json'))
    test_data = json.load(open('testDataValues.json'))
    for data in [train_data, test_data]:
        for sent_id in data:
            exercise_tokens = data[sent_id]['exercise segment'].lower()
            exercise = data[sent_id]['exercise value']
            for token in exercise_tokens.split():
                token_vals[token][exercise] += 1
    return token_vals


def read_feeling_data(binary=False):
    '''
    Builds a dictionary mapping tokens to feeling value frequencies.
    The binary flag indicates whether to map values to positive or negative.
    '''
    token_vals = defaultdict(lambda: defaultdict(int))
    train_data = json.load(open('trainDataValues.json'))
    test_data = json.load(open('testDataValues.json'))
    for data in [train_data, test_data]:
        for sent_id in data:
            feeling_tokens = data[sent_id]['feeling segment'].lower()
            feeling = data[sent_id]['feeling value']
            if binary:
                if int(feeling) > 5:
                    feeling = 1
                else:
                    feeling = 0
            for token in feeling_tokens.split():
                token_vals[token][feeling] += 1
    return token_vals


def compute_kappa(token_tags):
    '''Computes the kappa agreement score given each tag per token.'''

    # Extract all possible tags/values.
    tags = set()
    for token in token_tags:
        for tag in token_tags[token]:
            tags.add(tag)

    P_mean = compute_P_mean(token_tags, tags)
    P_e = compute_P_e(token_tags, tags)
    kappa = (P_mean - P_e) / (1 - P_e)
    return kappa


def compute_P_mean(token_tags, tags):
    '''Computes the mean of each P_i (i.e., agreement for token i).'''
    P_i_list = []
    tokens = list(token_tags.keys())
    for token in tokens:
        # Remove tokens that were only labeled once (to avoid dividing by 0).
        if len(token_tags[token]) <= 1:
            del token_tags[token]
            continue
        P_i_list.append(compute_P_i(token_tags[token], tags))
    P_mean = sum(P_i_list) / len(P_i_list)
    return P_mean


def compute_P_i(tag_dict, tags):
    '''Computes the agreement among tags for a single token i.'''
    total = 0
    n = sum(tag_dict.values())
    for tag in tags:
        n_ij = tag_dict[tag]
        total += n_ij * (n_ij - 1)
    P_i = total * (1 / (n * (n-1)))
    return P_i


def compute_P_e(token_tags, tags):
    '''Sums each squared P_j (i.e., proportion of tokens labeled tag j).'''
    P_j_list = []
    for tag in tags:
        P_j_list.append(compute_P_j(tag, token_tags))
    P_e = sum([P_j**2 for P_j in P_j_list])
    return P_e


def compute_P_j(tag, token_tags):
    '''Computes the proportion of tokens assigned tag j.'''
    N = len(token_tags)
    total = 0
    for token in token_tags:
        n = sum(token_tags[token].values())
        n_ij = token_tags[token][tag]
        total += n_ij / n
    P_j = (1 / N)
    return P_j


if __name__ == '__main__':
    token_tags = read_tags_data('dataWithPOS.csv')
    token_exercise_vals = read_exercise_data()
    token_feeling_vals = read_feeling_data(binary=True)

    # Scores from 0.01-0.2 are slight, 0.21-0.4 fair, 0.41-0.6 moderate, 
    # 0.61-0.8 substantial, and 0.81-1 near perfect.
    kappa_tags = compute_kappa(token_tags)
    print('The kappa score for tags is:', kappa_tags)

    kappa_exercise = compute_kappa(token_exercise_vals)
    print('The kappa score for exercise is:', kappa_exercise)

    kappa_feeling = compute_kappa(token_feeling_vals)
    print('The kappa score for feelings is:', kappa_feeling)
