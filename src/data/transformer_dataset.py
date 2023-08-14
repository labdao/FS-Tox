import os

import pandas as pd
from transformers import AutoTokenizer
import torch

from .utils import convert_canonical_smiles_to_selfies

import selfies as sf


# Set the tokenizer
tokenizer_name = "ncfrey/ChemGPT-4.7M"

path = "/Users/sethhowes/Desktop/FS-Tox/outputs/2023-08-14/12-53-58/data/processed/task"

# Get a list of task filepaths
task_filepaths = [f"{path}/{task}" for task in os.listdir(path)]

# Load a task
task = pd.read_parquet(task_filepaths[0])

# Tokekize the SELFIES
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Adding a padding token
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

# Convert canonical SMILES to SELFIES
selfies = task["canonical_smiles"].apply(convert_canonical_smiles_to_selfies)

# Split task into support and query sets
support_selfies = selfies[task['support_query'] == 0].tolist()
query_selfies = selfies[task['support_query'] == 1].tolist()

# Tokenize the SELFIES
support_encodings = tokenizer(support_selfies, padding=True, truncation=True, return_tensors="pt")
query_encodings = tokenizer(query_selfies, padding=True, truncation=True, return_tensors="pt")

# Get support and query labels
support_labels = task["ground_truth"][task['support_query'] == 0]
query_labels = task["ground_truth"][task['support_query'] == 1]

# Create a PyTorch dataset for each task
class ChemDataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

task_dataset = ChemDataset(support_encodings, support_labels)
print(task_dataset[0])

# Finetune model with Trainer
