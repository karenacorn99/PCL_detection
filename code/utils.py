import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import os

def process_data(data_dir, args):
    data = pd.read_csv(data_dir)
    texts = data['text'].tolist()
    labels = data['label'].tolist()
    # compute class weights
    class_weight = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return texts, labels, torch.tensor(class_weight, dtype=torch.float)

def train(dataloader, model, device, optimizer, scheduler, class_weight):
    model.train()
    print('Training mode')

    output_loss = 0

    for data in tqdm(dataloader, total=len(dataloader)):
        for k, v in data.items():
            data[k] = v.to(device)
        class_weight = class_weight.to(device)
        optimizer.zero_grad()
        logits, loss = model(input_ids=data['input_ids'],
                             attention_mask=data['attention_mask'],
                             labels=data['labels'],
                             class_weight=class_weight)
        loss.backward()
        optimizer.step()
        scheduler.step()
        output_loss += loss.item()

    output_loss = output_loss / len(dataloader)

    return output_loss

def evaluate(dataloader, model, device, class_weight):
    model.eval()
    print('Evaluation mode')
    output_loss = 0

    y_true = []
    y_pred = []

    for data in tqdm(dataloader, total=len(dataloader)):
        for k, v in data.items():
            data[k] = v.to(device)
        y_true.extend(data['labels'].cpu().numpy())
        class_weight = class_weight.to(device)
        logits, loss = model(input_ids=data['input_ids'],
                             attention_mask=data['attention_mask'],
                             labels=data['labels'],
                             class_weight=class_weight)
        logits = logits.detach().cpu().numpy()
        y_pred.extend(np.argmax(logits, axis=1))
        output_loss += loss.item()

    output_loss = output_loss / len(dataloader)

    # get metrics
    performance_metric = {'accuracy': accuracy_score(y_true, y_pred),
                          'precision': precision_score(y_true, y_pred, pos_label=1, average='binary'),
                          'recall': recall_score(y_true, y_pred, pos_label=1, average='binary'),
                          'f1': f1_score(y_true, y_pred, pos_label=1, average='binary')}

    return output_loss, performance_metric

def test(dataloader, model, device, class_weight, test_texts, test_labels, output_dir):
    model.eval()
    print('Test mode')
    output_loss = 0

    y_true = []
    y_pred = []

    logits_dim_0 = []
    logits_dim_1 = []

    for data in tqdm(dataloader, total=len(dataloader)):
        for k, v in data.items():
            data[k] = v.to(device)
        y_true.extend(data['labels'].cpu().numpy())
        class_weight = class_weight.to(device)
        logits, loss = model(input_ids=data['input_ids'],
                             attention_mask=data['attention_mask'],
                             labels=data['labels'],
                             class_weight=class_weight)
        logits = logits.detach().cpu().numpy()
        y_pred.extend(np.argmax(logits, axis=1))
        # TODO: normalize logits
        logits_dim_0.extend(logits[:, 0])
        logits_dim_1.extend(logits[:, 1])
        output_loss += loss.item()

    output_loss = output_loss / len(dataloader)

    # get metrics
    performance_metric = {'accuracy': accuracy_score(y_true, y_pred),
                          'precision': precision_score(y_true, y_pred, pos_label=1, average='binary'),
                          'recall': recall_score(y_true, y_pred, pos_label=1, average='binary'),
                          'f1': f1_score(y_true, y_pred, pos_label=1, average='binary')}

    prediction_df = pd.DataFrame({'text': test_texts, 'y_true': test_labels,
                                  'y_pred': y_pred, 'logits_dim_0': logits_dim_0, 'logits_dim_1': logits_dim_1})

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    prediction_df.to_csv(output_dir + 'prediction.csv', index=False)

    return output_loss, performance_metric