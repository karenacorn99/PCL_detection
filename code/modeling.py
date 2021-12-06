from transformers import AutoModel
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import torch


class Task1Baseline(nn.Module):
    def __init__(self, args):
        super(Task1Baseline, self).__init__()

        self.num_classes = 2
        self.bert = AutoModel.from_pretrained(args.pretrained_model, return_dict=False, output_hidden_states=True)
        self.dropout = nn.Dropout(0.1) # TODO: confirm dropout prob

        self.classifier = nn.Linear(768, self.num_classes)


    def forward(self, input_ids, attention_mask, labels, class_weight):

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1] # shape=(batch_size, 768)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss_fct = CrossEntropyLoss(class_weight)
        loss = loss_fct(logits, labels)

        print(logits.shape)
        print(labels.shape)

        return logits, loss

class Task2Baseline(nn.Module):
    def __init__(self, args):
        super(Task2Baseline, self).__init__()

        self.num_classes = 7
        self.bert = AutoModel.from_pretrained(args.pretrained_model, return_dict=False, output_hidden_states=True)
        self.dropout = nn.Dropout(0.1)

        self.classifier = nn.Linear(768, self.num_classes)

    def forward(self, input_ids, attention_mask, labels, class_weight):

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss_fct = BCEWithLogitsLoss(class_weight)
        loss = loss_fct(logits, labels.float())

        print(logits.shape)
        print(labels.shape)

        return logits, loss

class Task2MultiClassTokenClassifier(nn.Module):
    def __init__(self, args):

        super(Task2MultiClassTokenClassifier, self).__init__()

        self.num_classes = 15 # 7 classes * 2 (I & B) + 1 (O)
        self.bert = AutoModel.from_pretrained(args.pretrained_model, return_dict=False, output_hidden_states=True)
        self.dropout = nn.Dropout(0.1)

        self.classifier = nn.Linear(768, self.num_classes)

    def forward(self, input_ids, attention_mask, labels, class_weight):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        print(logits.shape)
        print(logits.view(-1, self.num_classes).shape)
        print(labels.shape)
        print(labels.float().view(-1).shape)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        return logits, loss
