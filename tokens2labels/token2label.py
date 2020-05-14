import pandas as pd
from NLPAlgorithms import stringMatching, logisticMulticlass, logisticBinaryMatch, logisticMulticlassBOW, logisticStrComparisson

if __name__ == '__main__':
    ### Importing data and defining token2label program
    print('\n\n\t\t-----Welcome to NLP Algorithms tests-----\n\n') 

    while True:
        token_i = input("\nWhich token do you want to use?\n\t(s) Sentence\n\t(e) Exercise\n\t(f) Feeling\n")
        tswitcher = {'s':'sentence','e':'exercise', 'f':'feeling'}
        token = tswitcher.get(token_i, 'wrong_i')
        if token != 'wrong_i':
            print('\n\tUsing {} as token'.format(token))
            break
        print('{} is not a token'.format(token_i))

    while True:
        label_i = input("\nWhich label do you want to use?\n\t(e) Exercise\n\t(f) Feeling\n")
        lswitcher = {'e':'exercise', 'f':'feeling'}
        label = lswitcher.get(label_i, 'wrong_i')
        if label != 'wrong_i':
            print('\n\tUsing {} as label'.format(label))
            break
        print('{} is not a label'.format(label_i))

    train_df = pd.read_pickle('./data/{}2{}-train-df.pkl'.format(token, label))
    test_df = pd.read_pickle('./data/{}2{}-test-df.pkl'.format(token, label))

    if label == 'exercise':
        exercise_list_df = pd.read_pickle('./data/exercise-list-df.pkl')
        label_options = list(exercise_list_df['exercise-label'])
        str_label_options = label_options
        dummy_labels = True
    else:
        label_options = [x for x in range(1,11)]
        str_label_options = [str(x) for x in range(1,11)]
        dummy_labels = False
    
    ### Trial 1: Exact String Matching

    print('\n\tTRIAL 1\n')
    while True:
        run = input('Do you want to test Exact string matching? (y/n)\n\t')
        if run == 'y':
            print('Training set:')
            train_accuracy, train_results = stringMatching(train_df, label_options, print_output = True)
            print('testing set:')
            test_accuracy, test_results = stringMatching(test_df, label_options, print_output = True)
            break
        elif run =='n':
            break

    ### Trial 2: Multiclass logistic regression
    #       Doesn't work properly because train and test are divided before converted
    #       into dummy variables.
    #       Not developed further because of bias in amount of labels

    # print('\n\tTRIAL 2\n')
    # if token == 'sentence':
    #     print('Trial 2 not supported for sentence token because of the sparcity of the dummy variable')
    # else:
    #     while True:
    #         run = input('Do you want to test Multiclass logistic regression? (y/n)\n\t')
    #         if run == 'y':
    #             print('The algorithm does not work properly because\ntrain and test are divided before converted into\ndummy variables')
    #             multiclass_classifier, classification_report = logisticMulticlass(train_df, train_df, print_output=True, dummy_labels=dummy_labels)
    #             break
    #         elif run =='n':
    #             break

    ### Trial 4: Logistic Multiclass with bow
    print('\n\tTRIAL 2\n')
    while True:
        run = input('Do you want to test logistic multiclass with BOW? (y/n)\n\t')
        if run == 'y':
            logisticMulticlassBOW(train_df, test_df, print_output=True, dummy_labels=dummy_labels)
            break
        elif run =='n':
            break
        
    ### Trial 3: Logistic binary match with bow
    print('\n\tTRIAL 3\n')
    while True:
        run = input('Do you want to test logistic binary match with BOW? (y/n)\n\t')
        if run == 'y':
            logisticBinaryMatch(train_df, test_df, str_label_options, print_output = True)
            break
        elif run =='n':
            break

    ### Trial 5: Logistic binary string match
    print('\n\tTRIAL 4\n')
    while True:
        run = input('Do you want to test logistic binary string match with features? (y/n)\n\t')
        if run == 'y':
            logisticStrComparisson(train_df, test_df, str_label_options, print_output=True)
            break
        elif run =='n':
            break