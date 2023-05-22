from transformers import AutoTokenizer, AutoModel
import torch

import logging

def chemberta_encode(smiles):

    # Load pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

    # Tokenize the SMILES string and get the input IDs
    input_ids = tokenizer.encode(smiles, return_tensors='pt')

    # Feed input IDs through the model. This will return the hidden states for each token in the input
    with torch.no_grad():
        outputs = model(input_ids)
        
    # Take the hidden state of the [CLS] token (first token) as the embedding of the whole SMILES string
    embedding = outputs[0][0, 0]

    return embedding

def main():
    logger = logging.getLogger(__name__)

    # Example SMILES string
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"

    # Get the embedding of a SMILES string
    embedding = chemberta_encode(smiles)

if __name__ == "__main__":
    main()
