import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import os
import ast

def process_data_task1(data_dir):
    data = pd.read_csv(data_dir)
    texts = data['text'].tolist()
    labels = data['label'].tolist()
    class_weight = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return texts, labels, torch.tensor(class_weight, dtype=torch.float)

def process_data_task2(data_dir):
    data = pd.read_csv(data_dir)
    texts = data['text'].tolist()
    #label_matrix = np.array(data['label'].apply(lambda x : x[1:-1].strip().split()).tolist(), dtype=np.int64)
    label_matrix = np.array(data['label_in_split_file'].apply(lambda x : ast.literal_eval(x)).tolist(), dtype=np.int64)
    class_freq = np.apply_along_axis(lambda x : sum(x==1), 0, label_matrix)
    # TODO: confirm balanced class weight
    class_weight = np.sum(class_freq) / 7 * class_freq.astype(np.float64)
    return texts, label_matrix, torch.tensor(class_weight, dtype=torch.float)

def process_data_multi_class_token(data_dir):
    data = pd.read_csv(data_dir)
    text_tokens = data['tokens'].apply(lambda x : ast.literal_eval(x)).tolist()
    ner_tags = data['ner_labels'].apply(lambda x : ast.literal_eval(x)).tolist()
    return text_tokens, ner_tags, None

def softmax(logits):
    return np.exp(logits) / np.sum(np.exp(logits))

def sigmoid(logits):
    return 1 / (1 + np.exp(-logits))

def train(dataloader, model, device, optimizer, scheduler, class_weight):
    model.train()
    print('Training mode')

    output_loss = 0

    for data in tqdm(dataloader, total=len(dataloader)):
        for k, v in data.items():
            data[k] = v.to(device)
        if class_weight is not None:
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

def evaluate(dataloader, model, device, class_weight, task, args):
    model.eval()
    print('Evaluation mode')
    output_loss = 0

    if task == 1:
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

        performance_metric = {'accuracy': accuracy_score(y_true, y_pred),
                              'precision': precision_score(y_true, y_pred, pos_label=1, average='binary'),
                              'recall': recall_score(y_true, y_pred, pos_label=1, average='binary'),
                              'f1': f1_score(y_true, y_pred, pos_label=1, average='binary')}

    elif task == 2:

        if args.config == 'baseline':
            y_true = []
            y_pred = []

            for data in tqdm(dataloader, total=len(dataloader)):
                for k, v in data.items():
                    data[k] = v.to(device)
                y_true.extend(data['labels'].cpu().numpy())
                if class_weight is not None:
                    class_weight = class_weight.to(device)
                logits, loss = model(input_ids=data['input_ids'],
                                     attention_mask=data['attention_mask'],
                                     labels=data['labels'],
                                     class_weight=class_weight)
                logits = logits.detach().cpu().numpy()
                logits = sigmoid(logits)

                for logit in logits:
                    y_pred.append(list(map(lambda x : 1 if x > 0.5 else 0, logit))) # TODO: confirm criterion

                output_loss += loss.item()

            output_loss = output_loss / len(dataloader)

            y_true = np.array(y_true)
            y_pred = np.array(y_pred)

            performance_metric = {}
            for i in range(7):
                class_metric = {}
                f1_scores = []
                class_metric['precision'] = precision_score(y_true[:, i], y_pred[:, i], pos_label=1, average='binary')
                class_metric['recall'] = recall_score(y_true[:, i], y_pred[:, i], pos_label=1,  average='binary')
                f1 = f1_score(y_true[:, i], y_pred[:, i], pos_label=1, average='binary')
                class_metric['f1'] = f1
                f1_scores.append(f1)
                performance_metric[f'class {i}'] = class_metric
                performance_metric['macro f1'] = np.mean(f1_scores)
        elif args.config == 'multi-class-token':
            y_true = []
            y_pred = []

            for data in tqdm(dataloader, total=len(dataloader)):
                for k, v in data.items():
                    data[k] = v.to(device)
                #y_true.extend(data['labels'].cpu().numpy())
                if class_weight is not None:
                    class_weight = class_weight.to(device)
                logits, loss = model(input_ids=data['input_ids'],
                                     attention_mask=data['attention_mask'],
                                     labels=data['labels'],
                                     class_weight=class_weight)
                logits = logits.detach().cpu().numpy()
                predictions = np.argmax(logits, axis=2)
                #print(predictions)
                #print(predictions.shape)

                # remove ignored index (special tokens)
                labels = data['labels'].cpu().numpy()
                true_labels = [[l for l in label if l != -100] for label in labels]
                #print(true_labels)
                y_true.extend(true_labels)

                predictions = [
                    [p for p, l in zip(prediction, label) if l != -100]
                    for prediction, label in zip(predictions, true_labels)
                ]
                #print(predictions)
                y_pred.extend(predictions)

            #print(len(y_true), len(y_pred))
            for i in range(len(y_true)):
                assert len(y_true[i]) == len(y_pred[i])
                #print(len(y_true[i]), len(y_pred[i]))

            # transform to vector labels
            true_vector_labels = list(map(get_vector_label, y_true))
            predicted_vector_labels = list(map(get_vector_label, y_pred))
            y_true = np.array(true_vector_labels)
            y_pred = np.array(predicted_vector_labels)
            #print(y_true)
            #print(y_pred)

            performance_metric = {}
            for i in range(7):
                class_metric = {}
                f1_scores = []
                class_metric['precision'] = precision_score(y_true[:, i], y_pred[:, i], pos_label=1, average='binary')
                class_metric['recall'] = recall_score(y_true[:, i], y_pred[:, i], pos_label=1, average='binary')
                f1 = f1_score(y_true[:, i], y_pred[:, i], pos_label=1, average='binary')
                class_metric['f1'] = f1
                f1_scores.append(f1)
                performance_metric[f'class {i}'] = class_metric
                performance_metric['macro f1'] = np.mean(f1_scores)

    return output_loss, performance_metric

def get_vector_label(token_preds):
    id2label = {0: 'O',
                1: 'B-Unbalanced_power_relations', 2: 'I-Unbalanced_power_relations',
                3: 'B-Shallow_solution', 4: 'I-Shallow_solution',
                5: 'B-Presupposition', 6: 'I-Presupposition',
                7: 'B-Authority_voice', 8: 'I-Authority_voice',
                9: 'B-Metaphors', 10: 'I-Metaphors',
                11: 'B-Compassion', 12: 'I-Compassion',
                13: 'B-The_poorer_the_merrier', 14: 'I-The_poorer_the_merrier'}
    label_to_position = {'Unbalanced_power_relations':0,
				         'Shallow_solution':1,
				         'Presupposition':2,
				         'Authority_voice':3,
				         'Metaphors':4,
				         'Compassion':5,
				         'The_poorer_the_merrier':6}

    predicted = set([id2label[pred][2:] for pred in token_preds if pred != 0])
    #print(predicted)
    vector_label = [0] * 7

    for p in predicted:
        vector_label[label_to_position[p]] = 1

    return vector_label

def test(dataloader, model, device, class_weight, test_texts, test_labels, output_dir, task, submission_dir, args):
    model.eval()
    print('Test mode')
    output_loss = 0

    if task == 1:
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
            logits = np.apply_along_axis(softmax, 1, logits)
            y_pred.extend(np.argmax(logits, axis=1))
            logits_dim_0.extend(logits[:, 0])
            logits_dim_1.extend(logits[:, 1])
            output_loss += loss.item()

        output_loss = output_loss / len(dataloader)

        # get metrics
        performance_metric = {'accuracy': accuracy_score(y_true, y_pred),
                              'precision': precision_score(y_true, y_pred, pos_label=1, average='binary'),
                              'recall': recall_score(y_true, y_pred, pos_label=1, average='binary'),
                              'f1': f1_score(y_true, y_pred, pos_label=1, average='binary')}

        # write output file
        prediction_df = pd.DataFrame({'text': test_texts, 'y_true': test_labels,
                                      'y_pred': y_pred, 'logits_dim_0': logits_dim_0, 'logits_dim_1': logits_dim_1})

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        prediction_df.to_csv(output_dir + 'prediction.csv', index=False)

        # write to submission file
        if not os.path.exists(submission_dir):
            os.makedirs(submission_dir)

        with open(submission_dir + 'task1.txt', 'w') as f:
            f.write('\n'.join(np.array(y_pred).astype(str)))

    elif task == 2:
        if args.config == 'baseline':
            y_true = []
            y_pred = []

            sigmoid_logits = []

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
                logits = sigmoid(logits)
                sigmoid_logits.extend(logits)

                for logit in logits:
                    y_pred.append(list(map(lambda x: 1 if x > 0.5 else 0, logit)))  # TODO: confirm criterion

                output_loss += loss.item()

            output_loss = output_loss / len(dataloader)

            y_true = np.array(y_true)
            y_pred = np.array(y_pred)

            performance_metric = {}
            for i in range(7):
                class_metric = {}
                f1_scores = []
                class_metric['precision'] = precision_score(y_true[:, i], y_pred[:, i], pos_label=1, average='binary')
                class_metric['recall'] = recall_score(y_true[:, i], y_pred[:, i], pos_label=1, average='binary')
                f1 = f1_score(y_true[:, i], y_pred[:, i], pos_label=1, average='binary')
                class_metric['f1'] = f1
                f1_scores.append(f1)
                performance_metric[f'class {i}'] = class_metric
                performance_metric['macro f1'] = np.mean(f1_scores)

            y_true = np.apply_along_axis(lambda x : str(x), 1, y_true)
            y_pred = np.apply_along_axis(lambda x : str(x), 1, y_pred)
            sigmoid_logits = np.apply_along_axis(lambda x : str(x), 1, sigmoid_logits)

            # write output file
            prediction_df = pd.DataFrame({'text': test_texts, 'y_true': y_true,
                                          'y_pred': y_pred, 'sigmoid_logits': sigmoid_logits})

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            prediction_df.to_csv(output_dir + 'prediction.csv', index=False)

            # write to submission file
            if not os.path.exists(submission_dir):
                os.makedirs(submission_dir)

            with open(submission_dir + 'task2.txt', 'w') as f:
                f.write('\n'.join(y_pred))
        elif args.config == 'multi-class-token':
            y_true = []
            y_pred = []

            for data in tqdm(dataloader, total=len(dataloader)):
                for k, v in data.items():
                    data[k] = v.to(device)
                #y_true.extend(data['labels'].cpu().numpy())
                if class_weight is not None:
                    class_weight = class_weight.to(device)
                logits, loss = model(input_ids=data['input_ids'],
                                     attention_mask=data['attention_mask'],
                                     labels=data['labels'],
                                     class_weight=class_weight)
                logits = logits.detach().cpu().numpy()
                predictions = np.argmax(logits, axis=2)
                #print(predictions)
                #print(predictions.shape)

                # remove ignored index (special tokens)
                labels = data['labels'].cpu().numpy()
                true_labels = [[l for l in label if l != -100] for label in labels]
                #print(true_labels)
                y_true.extend(true_labels)

                predictions = [
                    [p for p, l in zip(prediction, label) if l != -100]
                    for prediction, label in zip(predictions, true_labels)
                ]
                #print(predictions)
                y_pred.extend(predictions)

            #print(len(y_true), len(y_pred))
            for i in range(len(y_true)):
                assert len(y_true[i]) == len(y_pred[i])
                #print(len(y_true[i]), len(y_pred[i]))

            # transform to vector labels
            true_vector_labels = list(map(get_vector_label, y_true))
            predicted_vector_labels = list(map(get_vector_label, y_pred))
            y_true = np.array(true_vector_labels)
            y_pred = np.array(predicted_vector_labels)
            #print(y_true)
            #print(y_pred)

            performance_metric = {}
            for i in range(7):
                class_metric = {}
                f1_scores = []
                class_metric['precision'] = precision_score(y_true[:, i], y_pred[:, i], pos_label=1, average='binary')
                class_metric['recall'] = recall_score(y_true[:, i], y_pred[:, i], pos_label=1, average='binary')
                f1 = f1_score(y_true[:, i], y_pred[:, i], pos_label=1, average='binary')
                class_metric['f1'] = f1
                f1_scores.append(f1)
                performance_metric[f'class {i}'] = class_metric
                performance_metric['macro f1'] = np.mean(f1_scores)

            # write output file
            y_true = np.apply_along_axis(lambda x: str(x), 1, y_true)
            y_pred = np.apply_along_axis(lambda x: str(x), 1, y_pred)
            prediction_df = pd.DataFrame({'text': test_texts, 'y_true': y_true, 'y_pred': y_pred})

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            prediction_df.to_csv(output_dir + 'prediction.csv', index=False)

            # write to submission file
            if not os.path.exists(submission_dir):
                os.makedirs(submission_dir)

            with open(submission_dir + 'task2.txt', 'w') as f:
                f.write('\n'.join(y_pred))

    return output_loss, performance_metric