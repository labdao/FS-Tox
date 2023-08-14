import os

import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import evaluate
import torch

from utils import convert_canonical_smiles_to_selfies

# Set the device
device = torch.device("mps")

# Set the metric to evaluate on
metric = evaluate.load("accuracy")

# Set the tokenizer
checkpoint = "ncfrey/ChemGPT-4.7M"

path = "/Users/sethhowes/Desktop/FS-Tox/outputs/2023-08-14/12-53-58/data/processed/task"

# Get a list of task filepaths
task_filepaths = [f"{path}/{task}" for task in os.listdir(path)]

# Load a task
task = pd.read_parquet(task_filepaths[0])

# Tokekize the SELFIES
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Adding a padding token
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

# Convert canonical SMILES to SELFIES
selfies = task["canonical_smiles"].apply(convert_canonical_smiles_to_selfies)

# Split task into support and query sets
support_selfies = selfies[task["support_query"] == 0].tolist()
query_selfies = selfies[task["support_query"] == 1].tolist()

# Tokenize the SELFIES
support_encodings = tokenizer(
    support_selfies, padding=True, truncation=True, return_tensors="pt"
).to(device)
query_encodings = tokenizer(
    query_selfies, padding=True, truncation=True, return_tensors="pt"
).to(device)

# Get support and query labels
support_labels = torch.tensor(task["ground_truth"][task["support_query"] == 0].reset_index(drop=True), device=device)
query_labels = torch.tensor(task["ground_truth"][task["support_query"] == 1].reset_index(drop=True), device=device)

# Create a PyTorch dataset for each task
class ChemDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        print(item)
        return item

    def __len__(self):
        return len(self.labels)


# Create a PyTorch dataset for each task
support_dataset = ChemDataset(support_encodings, support_labels)
query_dataset = ChemDataset(query_encodings, query_labels)

# Finetune model with Trainer
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
model.to(device)

training_args = TrainingArguments(
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    do_train=True,
    do_eval=True,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=100,
    output_dir="./results",
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=support_dataset,
    eval_dataset=query_dataset,
)

trainer.train()