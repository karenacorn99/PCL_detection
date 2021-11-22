import argparse
from modeling import Task1BinaryClassification
from dataset import PCLDataset
from utils import process_data, train, evaluate, test
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
import os
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', type=str, default='roberta-base')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--train_data_dir', type=str, default='../data/toy/task1.csv')
    parser.add_argument('--validation_data_dir', type=str, default='../data/toy/task1.csv')
    parser.add_argument('--test_data_dir', type=str, default='../data/toy/task1.csv')
    parser.add_argument('--do_train', action='store_true', default=True)
    parser.add_argument('--model_dir', type=str, default='./output/model/')
    parser.add_argument('--output_dir', type=str, default='./output/pred/')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    print(f'Read data from {args.train_data_dir}')
    train_texts, train_labels, class_weight = process_data(args.train_data_dir, args)
    validation_texts, validation_labels, _ = process_data(args.validation_data_dir, args)
    test_texts, test_labels, _ = process_data(args.test_data_dir, args)
    print(f'Class weight = {class_weight}')

    print(f'Encode dataset using {args.pretrained_model}')
    train_dataset = PCLDataset(train_texts, train_labels, args)
    validation_dataset = PCLDataset(validation_texts, validation_labels, args)
    test_dataset = PCLDataset(test_texts, test_labels, args)

    # features is a list of dictionaries
    # [{'text': text, 'label':label}]
    def data_collator(features):
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        labels = [feature.pop('label') for feature in features]
        texts = [feature.pop('text') for feature in features]
        batch = tokenizer(texts, truncation=True, padding=True, max_length=args.max_length, return_tensors='pt')
        batch['labels'] = torch.tensor(labels, dtype=torch.int64)
        return batch

    print(f'Create dataloader with batch size {args.batch_size}')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collator)
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, collate_fn=data_collator)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=data_collator)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    # training loop
    if args.do_train:
        model = Task1BinaryClassification(args)

        model.to(device)

        num_training_steps = int(len(train_dataset) / args.batch_size * args.epoch)
        print(f'num_training_steps={num_training_steps}')
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

        best_loss = np.inf
        best_metric = 0
        for epoch in range(args.epoch):
            train_loss = train(train_dataloader, model, device, optimizer, scheduler, class_weight)
            post_train_loss, _ = evaluate(train_dataloader, model, device, class_weight)
            print(post_train_loss)
            validation_loss, validation_performance_metric = evaluate(validation_dataloader, model, device, class_weight)
            print(f"Epoch = {epoch+1} Train Loss = {train_loss} Validation Loss = {validation_loss}")
            print(validation_performance_metric)

            if best_metric < validation_performance_metric['f1']:
                print('Better performance on validation set, save model')
                if not os.path.exists(args.model_path):
                    print('output directory does not exist')
                    os.makedirs(args.model_path)
                torch.save(model.state_dict(), args.model_path + 'model.bin')

                best_metric = validation_performance_metric['f1']

    # test loop
    print('Running test loop')
    best_model = Task1BinaryClassification(args)

    print('Loading best model')
    best_model.load_state_dict(torch.load(args.model_dir + 'model.bin'))
    best_model.to(device)

    test_loss, test_performance_metric = test(test_dataloader, best_model, device, class_weight, test_texts, test_labels, args.output_dir)



















