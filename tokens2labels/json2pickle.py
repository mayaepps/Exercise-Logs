import pandas, json

'''
Extracts the input sentence and output exercise value label, given data dict.
'''
def get_x_y(data):
    labels, sents = [], []
    for sent_id in data:
        tokens = data[sent_id]['sentence']
        exercise_tokens = data[sent_id]['exercise segment'] # Real exercise value
        feeling_tokens = data[sent_id]['feeling segment'] # Real feeling value
        exercise = data[sent_id]['exercise value'] # True exercise value
        feeling = data[sent_id]['feeling value'] # True feeling value
        labels.append(feeling)
        sents.append(feeling_tokens) # TODO: compare to score with only exercise_tokens
    return sents, labels


if __name__ == '__main__':

    ################## LOAD THE DATA ##################

    train_data = json.load(open('trainDataValues.json'))
    test_data = json.load(open('testDataValues.json'))

    train_sents, train_labels = get_x_y(train_data)
    test_sents, test_labels = get_x_y(test_data)

    # create a dataframe using texts and lables
    trainDF = pandas.DataFrame()
    testDF = pandas.DataFrame()
    trainDF['text'] = train_sents
    trainDF['label'] = train_labels
    testDF['text'] = test_sents
    testDF['label'] = test_labels
    trainDF.to_pickle('data/sentence2feeling-train-df.pkl')
    testDF.to_pickle('data/sentence2feeling-test-df.pkl')
    print('--------Train Dataframe--------')
    print(trainDF)
    print('\n--------Test Dataframe--------')
    print(testDF)