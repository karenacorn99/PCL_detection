import pandas as pd
import numpy as np
import ast
from spacy.lang.en import English

def find_subarray_indices(array, subarray):
    subarray_len = len(subarray)
    for i in range(0, len(array) - subarray_len + 1):
        if array[i:i+subarray_len] == subarray:
            return i, i + subarray_len - 1
    return -1, -1

def find_unfound_labels(df):
    text = df['text'].iloc[0]
    text_tokens = text.split()

    with open('../raw_data/unfound_tokens.txt', 'a+') as f:
        for _, row in df.iterrows():
            start = row['start']
            finish = row['finish']
            label_tokens = text[start:finish].split()
            # find span in text
            b_start, b_end = find_subarray_indices(text_tokens, label_tokens)
            if b_start == -1:
                print(f"UNFOUND {row['par_id']}")
                output = f"{row['par_id']}\n{text_tokens}\n{label_tokens}\n"
                f.write(output)
                f.flush()
    return

def tag_tokens(df):
    text = df['text'].iloc[0]
    text_tokens = text.split()
    labels = ['O'] * len(text_tokens)
    label_len = 0

    for _, row in df.iterrows():
        start = row['start']
        finish = row['finish']
        label_tokens = text[start:finish].split()
        label_len += len(label_tokens)
        category = row['label']
        # find span in text
        b_start, b_end = find_subarray_indices(text_tokens, label_tokens)
        if b_start == -1:
            pass
        else:
            labels[b_start] = f'B-{category}'
            labels[b_end] = f'B-{category}'
            for i in range(b_start + 1, b_end):
                labels[i] = f'I-{category}'

    #print(labels)
    return [row['par_id'], row['art_id'], row['text'], labels, row['keyword'], row['country']]

if __name__ == '__main__':

    # print('generate files with required labels')
    # raw_data_dir = '../raw_data/dontpatronizeme_categories.tsv'
    # raw_data_df = pd.read_csv(raw_data_dir, sep='\t', skiprows=4)
    # raw_data_df.columns = ['par_id', 'art_id', 'text', 'keyword', 'country', 'start', 'finish',
    #                        'text_span', 'label', 'num_annotators']
    # print(raw_data_df.columns)
    #
    # # group by paragraph id
    # paragraphs = raw_data_df.groupby('par_id')
    #
    # # find unfound label spans
    # # paragraphs.apply(find_unfound_labels)
    #
    # # print(paragraphs.groups)
    # tags = paragraphs.apply(tag_tokens)
    # data = tags.tolist()
    # for d in data:
    #     print(d)
    # ner_df = pd.DataFrame(data, columns=['par_id', 'art_id', 'text', 'ner_labels', 'keyword', 'country'])
    # #ner_df.to_csv('../raw_data/task2_ner_tags.csv', index=False)

    # create toy data for testing model functionality
    # file = '../raw_data/task2_ner_tags_updated.csv'
    # df = pd.read_csv(file)
    #
    # par_179 = df[df['par_id'] == 179].iloc[0]
    # print(len(par_179['text'].split()))
    # print(len(ast.literal_eval(par_179['ner_labels'])))
    #
    # for _, row in df.iterrows():
    #     assert len(row['text'].split()) == len(ast.literal_eval(row['ner_labels']))
    #
    # # create toy set
    # ner_df = df.sample(n=100, random_state=1)
    # ner_df.to_csv('../data/ner_toy/task2.csv', index=False)

    # convert text to list of tokens
    files = ['../data/split_ner/task2_train.csv', '../data/split_ner/task2_val.csv', '../data/ner_toy/task2.csv']
    for file in files:
        df = pd.read_csv(file)
        df['tokens'] = df['text'].apply(lambda x : x.split())
        for _, row in df.iterrows():
            assert len(ast.literal_eval(row['ner_labels'])) == len(row['tokens'])
        df.to_csv(file, index=False)