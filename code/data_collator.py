from transformers import AutoTokenizer
import torch
import numpy as np

# features is a list of dictionaries
# [{'text': text, 'label':label}]
def data_collator_baseline(features, args):
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    labels = [feature.pop('label') for feature in features]
    texts = [feature.pop('text') for feature in features]
    batch = tokenizer(texts, truncation=True, padding=True, max_length=args.max_length, return_tensors='pt')
    batch['labels'] = torch.tensor(np.array(labels), dtype=torch.int64)
    return batch
