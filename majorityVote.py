'''
processCSV.py
Maya Epps
October 4, 2019
Most common sematic tag ("majority vote") for a baseline, using csv from processCSVforMajorityVote.py
'''

#for reference, in the csv file from processCSVforMajorityVote.py:

    # column[0] --> the log

    # column[1] --> the exercise

    # column[2] --> the feeling

import csv
import spacy
import operator
from itertools import islice


nlp = spacy.load("en_core_web_sm")

data_file_path = "../majorityVoteData.csv"

all_words = {}
final_words = {}


def get_length_of_csv():
    with open(data_file_path) as f:
        return sum(1 for line in f)

# 80% data is for training, 20% for testing
train_set_length = int(get_length_of_csv() * 0.8)
test_set_length = get_length_of_csv() - train_set_length


def tokenize(row):
    log_token_list = ([token.text for token in nlp(row[0])])

    ex_token_list = ([token.text for token in nlp(row[1])])

    feel_token_list = ([token.text for token in nlp(row[2])])

    return (log_token_list, ex_token_list, feel_token_list)

# dictionary of every word in the dataset, each word maps to a dictionary of a count for
# how many times that word has had each possible tag
def create_all_words_dict():
    with open(data_file_path) as csvfile:
        csvreader = csv.reader(csvfile)

        for row in islice(csvreader, 0, train_set_length):

            log, exercises, feelings = tokenize(row)

            for token in log:
                if token not in all_words:
                    all_words[token] = {"BE": 0,
                                        "IE": 0,
                                        "BF": 0,
                                        "IF": 0,
                                        "O": 0
                                        }
                else:
                    tag = get_tag(token, log, exercises, feelings)
                    all_words[token][tag] += 1

# given a word, the sentence it is in, the exercise words and the feeling words in
# that sentence, get_tag returns what tag the given word should have
def get_tag(token, sentence, exercises, feelings):
    index = sentence.index(token)
    if token in exercises:
        if index > 0 and sentence[index - 1] in exercises:
            return "IE"
        else:
            return "BE"

    if token in feelings:
        if index > 0 and sentence[index - 1] in feelings:
            return "IF"
        else:
            return "BF"

    if token not in feelings and token not in exercises:
            return "O"


# creates a dictionary (final_words) with the most common sem tag for every word
def create_final_dict():
    for word in all_words:
        final_words[word] = max(all_words[word], key=all_words[word].get)

# retrives the tag for a given word in a dictionary,
# if it is not in the dictionary it returns "O"
def guess(word):
    token = nlp(word).text
    if token not in final_words:
        return "O"
    else:
        return final_words[token]

# returns percent correct using the testing data to check how often the model's tag prediction
# matches the actual tag
def test():

    tags = 0
    correct = 0

    with open(data_file_path) as csvfile:
        csvreader = csv.reader(csvfile)

        for row in islice(csvreader, train_set_length, None):

            log, exercises, feelings = tokenize(row)

            for word in log:
                tags += 1
                if guess(word) == get_tag(word, log, exercises, feelings):
                    correct += 1

    return correct / tags * 100


create_all_words_dict()
create_final_dict()


print("Percent Correct:", test())


# interactive mode! Type your own logs and see what it predicts! (type "exit" to end interactive mode)

# while True:
#     user_log = input("Hi! How did you exercise today? ")
#     if user_log == "exit":
#         break
#     for word in user_log.split():
#         print(word+":", guess(word))
