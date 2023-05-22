import torch
from transformers import AutoTokenizer, AutoModel

import logging
import pandas as pd
import click
import duckdb


def chemgpt_encode(smiles):
    tokenizer = AutoTokenizer.from_pretrained("ncfrey/ChemGPT-4.7M")
    model = AutoModel.from_pretrained("ncfrey/ChemGPT-4.7M")

    # Adding a padding token
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Encode the SMILES string
    inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True)

    # @follow-up Warning message: Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
    # Really not sure of the rationale behind these parameters, need to check they make sense

    # Feed the inputs to the model
    with torch.no_grad():  # disable gradient calculations to save memory
        outputs = model(**inputs)

    # Output is a tuple, where the first item are the hidden states
    hidden_states = outputs[0]

    # Take the average of the hidden states for each token of the SMILES string
    average_embeddings = torch.mean(hidden_states, dim=1)

    # @follow-up I'm not sure if this is the best way to get the embedding of the whole SMILES string

    return average_embeddings


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    logger = logging.getLogger(__name__)
    logger.info("creating embeddings from selfies")

    # Connect to a database in memory
    connection = duckdb.connect(database=":memory:")

    # Load the data into the database as list of tuples
    smiles = connection.execute(
    f"""
    SELECT DISTINCT canonical_smiles
    FROM '{input_filepath}/assay_*.parquet'
    """
    ).fetchall()

    # Convert list of tuples to list of strings
    smiles_list = [x[0] for x in smiles]

    # Get the embeddings for each SMILES string
    embeddings = chemgpt_encode(smiles_list)

    # Convert tensor to list of lists (each sub-list is an embedding)
    embeddings_list = embeddings.numpy().tolist()

    # Create a pandas dataframe to store the SMILES strings and embeddings
    df = pd.DataFrame(list(zip(smiles_list, embeddings_list)), columns=['smiles', 'embeddings'])

    # Save the dataframe as a parquet file
    df.to_parquet(f"{output_filepath}/chemgpt_embeddings.parquet")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
