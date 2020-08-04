'''
Description: Computes the kappa score for per-token inter-annotator agreement.
Author: Mandy Korpusik
Date: 2020-08-04
'''

import csv
from collections import defaultdict


def read_data(filename):
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


def compute_kappa(token_tags, tags):
    '''Computes the kappa agreement score given each tag per token.'''
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
    tags = ['BE', 'IE', 'BF', 'IF', 'O']

    token_tags = read_data('dataWithPOS.csv')

    # Scores from 0.01-0.2 are slight, 0.21-0.4 fair, 0.41-0.6 moderate, 
    # 0.61-0.8 substantial, and 0.81-1 near perfect.
    kappa = compute_kappa(token_tags, tags)
    print('The kappa score for tags is:', kappa)
    