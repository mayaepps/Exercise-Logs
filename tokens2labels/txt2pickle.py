import pandas

if __name__ == '__main__':
    exercise_list = open('exercise-labels.txt').read().split('\n')
    exercise_list = [exercise for exercise in exercise_list if exercise != '']
    exercise_df = pandas.DataFrame(exercise_list, columns= ['exercise-label'])
    exercise_df.to_pickle('data\exercise-list-df.pkl')
    print(exercise_df)