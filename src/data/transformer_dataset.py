import os

from transformers import AutoTokenizer
import torch

# Set the tokenizer
tokenizer_name = "ncfrey/ChemGPT-4.7M")

# Get a list of task filepaths
task_filepaths = os.listdir()

# Create a PyTorch dataset for each task
class ChemDataset(torch.utils.data.Dataset):

    def __init__(self, task_filepath, tokenizer_name):
        self.task = pd.read_parquet(task_filepath)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self):
        return len(self.task)
    
    def __getitem__(self, idx):
        return self.task.loc[idx, "ground_truth"]

# Finetune model with Trainer