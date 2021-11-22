import torch
from transformers import AutoTokenizer

class PCLDataset:
    def __init__(self, texts, labels, args):
        self.texts = texts
        self.labels = labels
        self.args = args

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return {'text': self.texts[index], 'label': self.labels[index]}


