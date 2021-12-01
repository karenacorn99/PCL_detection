import argparse
from modeling import Task1Baseline, Task2Baseline
from dataset import PCLDatasetBaseline
from utils import process_data_task1, process_data_task2, train, evaluate, test
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
import os
import numpy as np
from data_collator import data_collator_baseline


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', type=str, default='roberta-base')
    parser.add_argument('--task', type=int, default=2)
    parser.add_argument('--config', type=str, default='baseline')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--train_data_dir', type=str, default='../data/new_toy_baseline/task2.csv')
    parser.add_argument('--validation_data_dir', type=str, default='../data/new_toy_baseline/task2.csv')
    parser.add_argument('--test_data_dir', type=str, default='../data/new_toy_baseline/task2.csv')
    parser.add_argument('--do_train', action='store_true', default=False)
    parser.add_argument('--model_dir', type=str, default='./output/model/')
    parser.add_argument('--output_dir', type=str, default='./output/pred/')
    parser.add_argument('--submission_dir', type=str, default='./output/submission/')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    if args.task == 1:
        train_texts, train_labels, class_weight = process_data_task1(args.train_data_dir)
        validation_texts, validation_labels, _ = process_data_task1(args.validation_data_dir)
        test_texts, test_labels, _ = process_data_task1(args.test_data_dir)

        train_dataset = PCLDatasetBaseline(train_texts, train_labels, args)
        validation_dataset = PCLDatasetBaseline(validation_texts, validation_labels, args)
        test_dataset = PCLDatasetBaseline(test_texts, test_labels, args)

    elif args.task == 2:
        train_texts, train_labels, class_weight = process_data_task2(args.train_data_dir)
        validation_texts, validation_labels, _ = process_data_task2(args.validation_data_dir)
        test_texts, test_labels, _ = process_data_task2(args.test_data_dir)

        train_dataset = PCLDatasetBaseline(train_texts, train_labels, args)
        validation_dataset = PCLDatasetBaseline(validation_texts, validation_labels, args)
        test_dataset = PCLDatasetBaseline(test_texts, test_labels, args)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=lambda x: data_collator_baseline(x, args), shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, collate_fn=lambda x: data_collator_baseline(x, args))
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=lambda x: data_collator_baseline(x, args))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    # training loop
    if args.do_train:
        if args.task == 1:
            if args.config == 'baseline':
                model = Task1Baseline(args)
        elif args.task == 2:
            if args.config == 'baseline':
                model = Task2Baseline(args)

        model.to(device)

        num_training_steps = int(len(train_dataset) / args.batch_size * args.epoch)
        print(f'num_training_steps={num_training_steps}')
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

        best_loss = np.inf
        best_metric = 0
        for epoch in range(args.epoch):
            train_loss = train(train_dataloader, model, device, optimizer, scheduler, class_weight)
            #post_train_loss, _ = evaluate(train_dataloader, model, device, class_weight, args.task)
            validation_loss, validation_performance_metric = evaluate(validation_dataloader, model, device, class_weight, args.task)
            print(f"Epoch = {epoch+1} Train Loss = {train_loss} Validation Loss = {validation_loss}")
            print(validation_performance_metric)

            best_criterion = 'f1' if args.task == 1 else 'macro f1'
            print(best_criterion)
            if best_metric <= validation_performance_metric[best_criterion]:
                print('Better performance on validation set, save model')
                if not os.path.exists(args.model_dir):
                    print('output directory does not exist')
                    os.makedirs(args.model_dir)
                torch.save(model.state_dict(), args.model_dir + 'model.bin')

                best_metric = validation_performance_metric[best_criterion]

    # test loop
    print('Running test loop')
    if args.task == 1:
        if args.config == 'baseline':
            best_model = Task1Baseline(args)
    elif args.task == 2:
        if args.config == 'baseline':
            best_model = Task2Baseline(args)

    print('Loading best model')
    best_model.load_state_dict(torch.load(args.model_dir + 'model.bin'))
    best_model.to(device)

    test_loss, test_performance_metric = test(test_dataloader, best_model, device, class_weight, test_texts, test_labels, args.output_dir, args.task, args.submission_dir)
