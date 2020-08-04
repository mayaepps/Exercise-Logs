from NLPAlgorithms import calculate_features, logisticStrComparisson, logisticBinaryMatch
import pandas as pd
import numpy as np

def logitsticBowPredictor(classifier, bow_vec, data, label_options, tolerance = 1):
    correct_rows = 0
    total_rows = len(data)
    for row_index, row in data.iterrows():
        vec_data = []
        for label_option in label_options:
            vec_row = bow_vec.transform([row['text']+' '+label_option])
            vec_data.append(vec_row.toarray()[0])
        
        predictions = classifier.predict(vec_data)
        predictions_proba = classifier.predict_proba(vec_data)
        predictions_proba = np.delete(predictions_proba, 0, axis=1)
        predictions_proba = predictions_proba.flatten()

        max_indexes = (-predictions_proba).argsort()[:tolerance]
        predicted_labels = [label_options[idx] for idx in max_indexes]
        if str(row['label']) in predicted_labels:
            correct_rows += 1
        accuracy = correct_rows/(row_index+1)
        print('\rProgress: Completed: {:.2%} --- Accuracy: {:.2%}'.format((row_index+1)/total_rows, accuracy), end='')

    return accuracy

def logitsticCompPredictor(classifier, data, label_options, tolerance = 1):
    correct_rows = 0
    total_rows = len(data)
    for row_index, row in data.iterrows():
        vec_data = []
        for label_option in label_options:
            vec_row = calculate_features(row['text'], label_option)
            vec_data.append(vec_row)
        
        predictions = classifier.predict(vec_data)
        predictions_proba = classifier.predict_proba(vec_data)
        predictions_proba = np.delete(predictions_proba, 0, axis=1)
        predictions_proba = predictions_proba.flatten()

        max_indexes = (-predictions_proba).argsort()[:tolerance]
        predicted_labels = [label_options[idx] for idx in max_indexes]
        if str(row['label']) in predicted_labels:
            correct_rows += 1
        accuracy = correct_rows/(row_index+1)
        print('\rProgress: Completed: {:.2%} --- Accuracy: {:.2%}'.format((row_index+1)/total_rows, accuracy), end='')

    return accuracy


if __name__ == "__main__":
    token = 'sentence'
    label = 'exercise'

    train_df = pd.read_pickle('data/{}2{}-train-df.pkl'.format(token, label))
    test_df = pd.read_pickle('data/{}2{}-test-df.pkl'.format(token, label))

    exercise_list_df = pd.read_pickle('./data/exercise-list-df.pkl')
    str_label_options = list(exercise_list_df['exercise-label'])#[str(x) for x in range(1,11)]#


    max_acc = 0
    for x in range(10):
        classifier, bowvec, _, _ = logisticBinaryMatch(train_df, test_df, str_label_options, print_output = False)
        acc = logitsticBowPredictor(classifier, bowvec, test_df, str_label_options, tolerance=1)
        print('Final acc: {}'.format(acc))
        if acc > max_acc:
            max_acc = acc