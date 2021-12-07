from transformers import AutoTokenizer
import torch
import numpy as np

# features is a list of dictionaries
# [{'text': text, 'label': label}]
def data_collator_baseline(features, args):
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    labels = [feature.pop('label') for feature in features]
    texts = [feature.pop('text') for feature in features]
    batch = tokenizer(texts, truncation=True, padding=True, max_length=args.max_length, return_tensors='pt')
    batch['labels'] = torch.tensor(np.array(labels), dtype=torch.int64)
    return batch

def token_label2id(label_list):
    label2id = {'O': 0,
                'B-Unbalanced_power_relations': 1, 'I-Unbalanced_power_relations': 2,
                'B-Shallow_solution': 3, 'I-Shallow_solution': 4,
                'B-Presupposition': 5, 'I-Presupposition': 6,
                'B-Authority_voice': 7, 'I-Authority_voice': 8,
                'B-Metaphors': 9, 'I-Metaphors': 10,
                'B-Compassion': 11, 'I-Compassion': 12,
                'B-The_poorer_the_merrier': 13, 'I-The_poorer_the_merrier': 14}
    labels = [label2id[x] for x in label_list]
    return labels

def data_collator_multi_class_token(features, args):
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, add_prefix_space=True)

    # tokenize and align labels
    tokens = [feature.pop('text') for feature in features]
    tags = [feature.pop('label') for feature in features]
    tags = list(map(token_label2id, tags))

    tokenized_inputs = tokenizer(tokens, truncation=True, padding=True, is_split_into_words=True)

    label_all_tokens = True

    aligned_labels = []
    for i, label in enumerate(tags):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        #print(word_ids)
        previous_word_idx = None
        labels_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                labels_ids .append(-100)
            elif word_idx != previous_word_idx:
                labels_ids.append(label[word_idx])
            else:
                labels_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        aligned_labels.append(labels_ids)

    for k, v in tokenized_inputs.items():
        tokenized_inputs[k] = torch.tensor(v)
    tokenized_inputs['labels'] = torch.tensor(aligned_labels)
    #print(tokenized_inputs['input_ids'])
    #print(tokenized_inputs['labels'])

    for i in range(len(features)):
        assert len(tokenized_inputs['input_ids'][i]) == len(tokenized_inputs['labels'][i])
        #print(len(tokenized_inputs['input_ids'][i]), len(tokenized_inputs['labels'][i]))

    return tokenized_inputs