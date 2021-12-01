import pandas as pd
import numpy as np

if __name__ == '__main__':
    train_ids_file = '../data/splits/train_semeval_parids-labels.csv'
    val_ids_file = '../data/splits/dev_semeval_parids-labels.csv'

    # columns = [par_id, label]
    train_ids = pd.read_csv(train_ids_file)
    train_ids = train_ids.rename(columns={'label': 'label_in_split_file'})
    val_ids = pd.read_csv(val_ids_file)
    val_ids = val_ids.rename(columns={'label': 'label_in_split_file'})
    print(f'Number of train samples: {len(train_ids)}')
    print(f'Number of val samples: {len(val_ids)}')
    #print(train_ids.columns)
    #print(val_ids.columns)

    task1_data_file = '../data/full_baseline/task1.csv'
    task2_data_file = '../data/full_baseline/task2.csv'

    task1_df = pd.read_csv(task1_data_file)
    task2_df = pd.read_csv(task2_data_file)
    print(f'Number of task 1 samples: {len(task1_df)}')
    print(f'Number of task 2 samples: {len(task2_df)}')
    #print(task1_df.columns)
    #print(task2_df.columns)

    task1_train = pd.merge(task1_df, train_ids, how='right', on='par_id')
    task1_val = pd.merge(task1_df, val_ids, how='right', on='par_id')
    print(f'Number of task 1 train: {len(task1_train)}')
    print(f'Number of task 1 val: {len(task1_val)}')
    assert len(task1_train) + len(task1_val) == len(task1_df)
    #print(task1_train.head())
    assert np.sum(task1_train['label'].isnull()) == 0
    assert np.sum(task1_val['label'].isnull()) == 0

    task1_train.to_csv('../data/split_baseline/task1_train.csv', index=False)
    task1_val.to_csv('../data/split_baseline/task1_val.csv', index=False)

    task2_train = pd.merge(task2_df, train_ids, how='inner', on='par_id')
    task2_val = pd.merge(task2_df, val_ids, how='inner', on='par_id')
    print(f'Number of task 2 train: {len(task2_train)}')
    print(f'Number of task 2 val: {len(task2_val)}')

    assert len(task2_train) + len(task2_val) == len(task2_df)

    #assert np.sum(task2_train['label'].isnull()) + np.sum(task2_val['label'].isnull()) == np.sum(task1_df['label'] == 0)

    task2_train.to_csv('../data/split_baseline/task2_train.csv', index=False)
    task2_val.to_csv('../data/split_baseline/task2_val.csv', index=False)

    # create toy examples for model development
    # Task 1: select 50 samples for each class
    task1_toy_df = task1_train.groupby('label').sample(n=50, random_state=1)
    # Task 2: select 100 samples in total, regardless of labels
    task2_toy_df = task2_train.sample(n=100, random_state=1)

    task1_toy_df.to_csv('../data/new_toy_baseline/task1.csv')
    task2_toy_df.to_csv('../data/new_toy_baseline/task2.csv')


