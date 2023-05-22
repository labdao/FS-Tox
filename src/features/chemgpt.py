import logging
import torch
from transformers import AutoTokenizer, AutoModel


def chemgpt_encode(selfies):
    tokenizer = AutoTokenizer.from_pretrained("ncfrey/ChemGPT-4.7M")
    model = AutoModel.from_pretrained("ncfrey/ChemGPT-4.7M")

    # Encode the SMILES string
    inputs = tokenizer(selfies, return_tensors="pt")
    print(inputs)

    # Feed the inputs to the model
    with torch.no_grad():  # disable gradient calculations to save memory
        outputs = model(**inputs)

    return outputs
    # Output is a tuple, where the first item are the hidden states
    hidden_states = outputs[0]

    return hidden_states


def main():
    logger = logging.getLogger(__name__)

    selfies = "C1=CC=C(C=C1)C=O"
    embeddings = chemgpt_encode(selfies)
    print(embeddings)

if __name__ == "__main__":
    main()
