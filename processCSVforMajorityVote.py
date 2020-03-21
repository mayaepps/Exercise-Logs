'''
processCSV.py
Maya Epps
October 4, 2019
Processes the CSV files from Qualtrics containing the results of
HITs. Creates a CSV file called majorityVoteData.csv that is used for mostCommonSemTag.py
'''

import csv
from itertools import islice

logs = []
exerciseTags = []
feelingTags = []
exerciseTagValues = []
feelingTagValues = []


with open("./../data/1502 exercise data.csv") as csvfile:
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
    # however, this didn't work as well, so we seperated the two types of logs
    for row in islice(csvreader, 2, None):

        # new questions (spereate exercise and feeling logs)
        if row[indexOldLog1] == ("") and row[indexOldLog2] == (""):
            logs.append(row[indexExerciseLog1] + " " + row[indexFeelingLog1])
            logs.append(row[indexExerciseLog2] + " " + row[indexFeelingLog2])
            exerciseTags.append(row[indexExerciseTag1])
            exerciseTags.append(row[indexExerciseTag2])
            feelingTags.append(row[indexFeelingTag1])
            feelingTags.append(row[indexFeelingTag2])
            exerciseTagValues.append(row[indexExerciseTagValue1])
            exerciseTagValues.append(row[indexExerciseTagValue2])
            feelingTagValues.append(row[indexFeelingTagValue1])
            feelingTagValues.append(row[indexFeelingTagValue2])

        else:
            # old questions (exercise and feeling in one log)
            logs.append(row[indexOldLog1])
            logs.append(row[indexOldLog2])
            exerciseTags.append(row[indexExerciseTag1])
            exerciseTags.append(row[indexExerciseTag2])
            feelingTags.append(row[indexFeelingTag1])
            feelingTags.append(row[indexFeelingTag2])
            exerciseTagValues.append(row[indexExerciseTagValue1])
            exerciseTagValues.append(row[indexExerciseTagValue2])
            feelingTagValues.append(row[indexFeelingTagValue1])
            feelingTagValues.append(row[indexFeelingTagValue2])


# lowercase and strip all logs of extra spaces
for i in range(len(logs)):
    feelingTags[i] = feelingTags[i].strip().lower()
    exerciseTags[i] = exerciseTags[i].strip().lower()
    logs[i] = logs[i].strip().lower()

# writes the new csv file, which has three columns: the sentence (log),
# the word selected/tagged as exercise, and the word(s) selected/tagged as how they felt
# during/after the exercise
with open('majorityVoteData.csv', 'w') as csvfile:

    log_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    log_writer.writerow(["Log", "Exercise Tag", "Feeling Tag"])
    for i in range(len(logs)):
        log_writer.writerow([logs[i], exerciseTags[i], feelingTags[i]])
