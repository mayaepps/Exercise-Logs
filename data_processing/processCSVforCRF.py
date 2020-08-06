'''
processCSVforCRF.py
Maya Epps
April 16, 2020
Processes the CSV files from Qualtrics containing the results of
HITs. Creates 2 CSV files (train and test) with POS tags for both CRFs
'''

import csv, json
from itertools import islice
import spacy

nlp = spacy.load("en_core_web_sm")

input_csv_file_path = "../data/unprocessedFinalData.csv"
train_output_csv_file_path = "../data/trainDataWithPOS.csv"
test_output_csv_file_path = "../data/testDataWithPOS.csv"
train_output_values_file_path = "../data/trainDataValues.json"
test_output_values_file_path = "../data/testDataValues.json"

logs = []
exerciseTags = []
feelingTags = []
exerciseTagValues = []
feelingTagValues = []


with open(input_csv_file_path,encoding='cp1252') as csvfile:
    csvreader = csv.reader(csvfile)

    #list of all the fields
    fields = csvreader.__next__()

    # gets the indexes of each column based on how Qualtrics named the questions
    indexOldLog1 = fields.index("Q20")
    indexOldLog2 = fields.index("Q21")
    indexExerciseLog1 = fields.index("Q3")
    indexExerciseLog2 = fields.index("Q5")
    indexFeelingLog1 = fields.index("Q4")
    indexFeelingLog2 = fields.index("Q6")
    indexExerciseTag1 = fields.index("Q8")
    indexExerciseTag2 = fields.index("Q9")
    indexFeelingTag1 = fields.index("Q11")
    indexFeelingTag2 = fields.index("Q12")
    indexExerciseTagValue1 = fields.index("Q14")
    indexOther1 = fields.index("Q15")
    indexExerciseTagValue2 = fields.index("Q16")
    indexOther2 = fields.index("Q17")
    indexFeelingTagValue1 = fields.index("Q13_1")
    indexFeelingTagValue2 = fields.index("Q13_2")

    # builds the lists (logs, exerciseTags, feelingTags, exerciseTagValues, feelingTagValues)
    # note: we initially had the exercise log and the feeling logs combined in ONE log
    # however, this didn't work as well, so we seperated the two types of logs and concatinate them here
    for row in islice(csvreader, 1, None, 2):
        if row[indexOldLog1] == ("") and row[indexOldLog2] == (""):
            logs.append(row[indexExerciseLog1] + " " + row[indexFeelingLog1])
            logs.append(row[indexExerciseLog2] + " " + row[indexFeelingLog2])
            exerciseTags.append(row[indexExerciseTag1].replace(",", ""))
            exerciseTags.append(row[indexExerciseTag2].replace(",", ""))
            feelingTags.append(row[indexFeelingTag1].replace(",", ""))
            feelingTags.append(row[indexFeelingTag2].replace(",", ""))
            exerciseTagValues.append(row[indexExerciseTagValue1])
            exerciseTagValues.append(row[indexExerciseTagValue2])
            feelingTagValues.append(row[indexFeelingTagValue1])
            feelingTagValues.append(row[indexFeelingTagValue2])

        else:
            # old questions (exercise and feeling in one log)
            logs.append(row[indexOldLog1])
            logs.append(row[indexOldLog2])
            exerciseTags.append(row[indexExerciseTag1].replace(",", ""))
            exerciseTags.append(row[indexExerciseTag2].replace(",", ""))
            feelingTags.append(row[indexFeelingTag1].replace(",", ""))
            feelingTags.append(row[indexFeelingTag2].replace(",", ""))
            exerciseTagValues.append(row[indexExerciseTagValue1])
            exerciseTagValues.append(row[indexExerciseTagValue2])
            feelingTagValues.append(row[indexFeelingTagValue1])
            feelingTagValues.append(row[indexFeelingTagValue2])


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




# writes the new csv file, which has four columns: the sentence number,
# the word, the POS, and the word's tag
def createCSV(path, rangeOfData):
    with open(path, 'w') as csvfile:

        sentence_number = 1

        log_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        log_writer.writerow(["Sentence #", "Word", "POS", "Tag"])

        for i in rangeOfData:

            ex_tokens = [token.text for token in nlp(exerciseTags[i])]
            feel_tokens = [token.text for token in nlp(feelingTags[i])]
            tokenized_sentence = nlp(logs[i])
            list_tokenized_sentence = [token.text for token in nlp(logs[i])]

            for token in tokenized_sentence:
                tag = get_tag(token.text, list_tokenized_sentence, ex_tokens, feel_tokens)


                log_writer.writerow(["Sentence: " + str(sentence_number), token.text, token.pos_, tag])

            sentence_number += 1


def createJSON(path, rangeOfData):
    data = {} # maps from tokens to tagged segments and exercise/feeling values
    for i in rangeOfData:
        list_tokenized_sentence = " ".join([token.text for token in nlp(logs[i])])
        ex_tokens = " ".join([token.text for token in nlp(exerciseTags[i])])
        feel_tokens = " ".join([token.text for token in nlp(feelingTags[i])])
        exercise = exerciseTagValues[i]
        feeling = feelingTagValues[i]
        data[i] = {'sentence':list_tokenized_sentence}
        data[i]['exercise segment'] = ex_tokens
        data[i]['feeling segment'] = feel_tokens
        data[i]['exercise value'] = exercise
        data[i]['feeling value'] = int(feeling)

    print('Number of samples:', len(data))

    # write to output json file
    with open(path, 'w') as outfile:
        json.dump(data, outfile)



TRAIN_LENGTH = (int)(0.8 * len(logs))

# createCSV(train_output_csv_file_path, range(0, TRAIN_LENGTH))
# createCSV(test_output_csv_file_path, range(TRAIN_LENGTH, len(logs)))

createJSON(train_output_values_file_path, range(0, TRAIN_LENGTH))
createJSON(test_output_values_file_path, range(TRAIN_LENGTH, len(logs)))
