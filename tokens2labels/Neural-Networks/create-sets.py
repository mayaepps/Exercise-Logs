import pandas as pd
import random

def multiclass2binary(data, label_options):
    new_data = []

    for row_index, row in data.iterrows():
        new_row = [row['text'], row['label'], 1]
        new_data.append(new_row)
        while True:
            new_label = random.choice(label_options)
            if new_label != row['label']:
                break
        new_row = [row['text'], new_label, 0]
        new_data.append(new_row)

    new_data_df = pd.DataFrame(new_data, columns=['text', 'label', 'match'])

    return new_data_df


if __name__ == "__main__":

    ### sentence2feeling and feeling2feeling are the same as before

    ### sentence2exercise and exercise2exercise
    
    tokens = ['sentence', 'exercise']

    exercise_list_df = pd.read_pickle('../data/exercise-list-df.pkl')
    label_options = list(exercise_list_df['exercise-label'])

    for token in tokens:
        train_df = pd.read_pickle('../data/{}2{}-train-df.pkl'.format(token, 'exercise'))
        test_df = pd.read_pickle('../data/{}2{}-test-df.pkl'.format(token, 'exercise'))

        train_match_df = multiclass2binary(train_df, label_options)
        test_match_df = multiclass2binary(test_df, label_options)

        train_match_df.to_pickle('./data/{}2{}-match-train-df.pkl'.format(token, 'exercise'))
        test_match_df.to_pickle('./data/{}2{}-match-test-df.pkl'.format(token, 'exercise'))


    ### sentence2both is same as before