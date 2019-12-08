'''
processCSV.py
Maya Epps
October 4, 2019
Processes the CSV files from Amazon Mechanical Turk containing the results of
HITs. Prints the worker IDs and their responses for now.
'''

import csv
from itertools import islice

#list of all the MTurk responses
logs = []
exerciseTags = []
feelingTags = []
exerciseTagValues = []
feelingTagValues = []

with open("../data/test_data.csv") as csvfile:
    csvreader = csv.reader(csvfile)

    #list of all the fields
    fields = csvreader.__next__()

    indexLog1 = fields.index("Q3")
    indexLog2 = fields.index("Q4")
    indexExerciseTag1 = fields.index("Q7")
    indexExerciseTag2 = fields.index("Q8")
    indexFeelingTag1 = fields.index("Q10")
    indexFeelingTag2 = fields.index("Q11")
    indexExerciseTagValue1 = fields.index("Q14")
    indexOther1 = fields.index("Q15")
    indexExerciseTagValue2 = fields.index("Q16")
    indexOther2 = fields.index("Q17")
    indexFeelingTagValue1 = fields.index("Q13_1")
    indexFeelingTagValue2 = fields.index("Q13_2")

    for row in islice(csvreader, 2, None):

        logs.append(row[indexLog1])
        logs.append(row[indexLog2])
        exerciseTags.append(row[indexExerciseTag1])
        exerciseTags.append(row[indexExerciseTag2])
        feelingTags.append(row[indexFeelingTag1])
        feelingTags.append(row[indexFeelingTag2])
        if row[indexExerciseTagValue1] == "" or row[indexExerciseTagValue1] == "Other":
            exerciseTagValues.append("Other: " + row[indexOther1])
        else:
            exerciseTagValues.append(row[indexExerciseTagValue1])
        if row[indexExerciseTagValue2] == "" or row[indexExerciseTagValue2] == "Other":
            exerciseTagValues.append("Other: " + row[indexOther2])
        else:
            exerciseTagValues.append(row[indexExerciseTagValue2])
        feelingTagValues.append(row[indexFeelingTagValue1])
        feelingTagValues.append(row[indexFeelingTagValue2])

print("Logs:", logs)
print("Exercises:", exerciseTags)
print("Exercise values:",exerciseTagValues)
print("Feelings:",feelingTags)
print("Feeling values:", feelingTagValues)
