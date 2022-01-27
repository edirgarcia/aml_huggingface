import argparse
import pandas as pd
import numpy as np
import time
import os

from sklearn.model_selection import train_test_split
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, AutoModelForSequenceClassification
# can it be AutoTokenizer as well?
from transformers import RobertaTokenizer, RobertaTokenizerFast
from datasets import load_dataset, load_metric

def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, max_length=512)

def compute_metrics(eval_preds):
    metric_names = ["accuracy", "f1", "precision", "recall"]
    return_dict = {}
    for metric_name in metric_names:
        metric = load_metric(metric_name)
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        curr_dict = metric.compute(predictions=predictions, references=labels)
        return_dict.update(curr_dict)

    return return_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default = 'data/imdb_train.csv')
    parser.add_argument('--test_data', type=str, default = 'data/imdb_test.csv')
    parser.add_argument('--pretrained_model_store', type=str, default = 'models')
    parser.add_argument('--pretrained_model_path', type=str, default = 'imdb_roberta_ft')
    parser.add_argument('--output_path', type=str, default = 'models/imdb_sentiment')
    parser.add_argument('--val_size', type=float, default = 0.05)
    parser.add_argument('--epochs', type=int, default = 2)

    args = parser.parse_args()

    #print all pass in arguments
    print("Pass in arguments", args)

    #schema is text, label
    train_df = pd.read_csv(args.train_data) 
    test_df = pd.read_csv(args.test_data) 

    print("Training data:")
    print(train_df.head())
    print(train_df.info())
    print(train_df.label.value_counts())

    print("Test data:")
    print(test_df.head())
    print(test_df.info())
    print(test_df.label.value_counts())

    os.makedirs("data", exist_ok=True)

    train_df.to_csv("data/train_data.csv", index=False)
    test_df.to_csv("data/test_data.csv", index=False)

    pretrained_model_path = args.pretrained_model_store + '/' + args.pretrained_model_path
    print("Pretrained model path:", pretrained_model_path)

    tokenizer = RobertaTokenizerFast.from_pretrained(pretrained_model_path)

    #this has to be the data path above
    data_files = {"train": "data/train_data.csv", "test": "data/test_data.csv"}
    dataset = load_dataset("csv", data_files=data_files)

    print("Created dataset")
    print(dataset)

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    print("Tokenized dataset")
    print(tokenized_dataset)

    # this could be RobertaForSequenceClassification too?                                                   
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_path, type_vocab_size=1)

    # #trying with the specific RobertaForSequenceClassification
    # config = RobertaConfig.from_pretrained(pretrained_model_path)
    # #for some reason when loading the config it still has the 4 not the 1 in type vocab size, so setting manually
    # config.type_vocab_size = 1
    # model = RobertaForSequenceClassification.from_pretrained(pretrained_model_path, config=config)

    #print("Model config: ")
    #print(model.config)

    logging_dir = os.path.join(args.output_path, 'logs')

    #all of this can be extracted to an argument to this script
    training_args = TrainingArguments(
        output_dir=args.output_path,     # output directory
        num_train_epochs=args.epochs,              # total number of training epochs
        per_device_train_batch_size=8,   # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir=logging_dir,         # directory for storing logs
        fp16=True,                       # whether to use 16-bit (mixed) precision
        metric_for_best_model='f1',      # which metric to track for saving best model
        logging_steps= 1000,             # interval for logging
        save_steps=10000,                # interval for saving model
        label_names=['labels']           # list of label names
    )

    trainer = Trainer(
        model=model,                                # the instantiated 🤗 Transformers model to be trained
        args=training_args,                         # training arguments, defined above
        train_dataset=tokenized_dataset["train"],   # training dataset
        eval_dataset=tokenized_dataset["test"],     # evaluation dataset
        tokenizer=tokenizer,                        # tokenizer for the model
        compute_metrics=compute_metrics             # metrics to track during training
    )

    trainer.train()

    eval_data = trainer.evaluate()
    print("Evaluation results:")
    print(eval_data)

    #save pretrained model
    trainer.save_model(args.output_path)
    print("Model saved to", args.output_path)

    print("Done fine tuning")

