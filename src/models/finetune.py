import os
import time
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from sklearn.metrics import average_precision_score, accuracy_score

from utils import convert_canonical_smiles_to_selfies

def load_data(task_filepath):
    
    # Load a task
    task = pd.read_parquet(task_filepath)

    # Convert canonical SMILES to SELFIES
    selfies = task["canonical_smiles"].apply(convert_canonical_smiles_to_selfies)

    # Split task into support and query sets
    support_selfies = selfies[task["support_query"] == 0].tolist()
    query_selfies = selfies[task["support_query"] == 1].tolist()

    # Get support and query labels
    support_labels = torch.tensor(task["ground_truth"][task["support_query"] == 0].reset_index(drop=True))
    query_labels = torch.tensor(task["ground_truth"][task["support_query"] == 1].reset_index(drop=True))
    
    return support_selfies, support_labels, query_selfies, query_labels

def tokenize_data(tokenizer, data):

    # Tokenize the SELFIES
    return tokenizer(
        data, padding=True, truncation=True, return_tensors="pt", max_length=256
    )

def finetune_model(support_dataset, query_dataset, model, output_path):

    # Define Trainer
    args = TrainingArguments(
        output_dir=output_path,
        evaluation_strategy="steps",
        eval_steps=500,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        seed=0,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=support_dataset,
        eval_dataset=query_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    
    return trainer


# Create a PyTorch dataset for each task
class ChemDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])
    

def compute_metrics(p):
    pred, labels = p
    auc_pr = average_precision_score(y_true=labels, y_pred=pred)
    
    return {"auc_pr": auc_pr}


def finetune_on_tasks(run_dir, model_checkpoint, output_dir):
    # Get a list of task filepaths
    task_filepaths = [f"{run_dir}/{task}" for task in os.listdir(run_dir)]

    # Tokenize the SELFIES
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Adding a padding token
    tokenizer.pad_token = "[PAD]"

    for i, task_filepath in enumerate(task_filepaths):
        # Get the task identifier from basename
        task_basename = os.path.basename(task_filepath)

        # Load data
        support_selfies, support_labels, query_selfies, query_labels = load_data(task_filepath)
        
        # Tokenize data
        support_encodings = tokenize_data(tokenizer, support_selfies)
        query_encodings = tokenize_data(tokenizer, query_selfies)

        # Create support and query data 
        support_dataset = ChemDataset(support_encodings, support_labels)
        query_dataset = ChemDataset(query_encodings, query_labels)
        
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
        model.config.pad_token_id = tokenizer.pad_token_id

        # Create output path by concatenating output_dir and task_id
        output_path = os.path.join(output_dir, task_basename)
        
        # Finetune model
        trainer = finetune_model(support_dataset, query_dataset, model, output_path)
        
        # Save model 
        trainer.save_model(output_path)

        print("%d finetuned!", i+1)


# Set the tokenizer
model_checkpoint = "ncfrey/ChemGPT-4.7M"

# Set the path to the task folder
run_dir = "/rds/general/user/ssh22/home/FS-Tox/outputs/2023-08-10/16-27-54/data/processed/task"

# Set the path for the output directory
output_dir = "/rds/general/user/ssh22/home/FS-Tox/models/"

time_start = time.time()
finetune_on_tasks(run_dir, model_checkpoint, output_dir)

time_end = time.time()

time_difference = time_end - time_start

print(time_difference)