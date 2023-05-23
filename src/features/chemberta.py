from transformers import AutoTokenizer, AutoModel
import torch

import duckdb
import pandas as pd
import click
import logging


def chemberta_encode(smiles_list):
    # Load pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

    # Create a list to store the embeddings
    embeddings = []

    # @follow-up this code is much slower than the ChemGPT version as iterating instead of batching inputs??
    # More efficient way to process this?

    for smiles in smiles_list:
        # Tokenize the SMILES string and get the input IDs
        input_id = tokenizer.encode(
            smiles, return_tensors="pt", padding=True, truncation=True
        )

        # Feed input IDs through the model. This will return the hidden states for each token in the input
        with torch.no_grad():
            outputs = model(input_id)

        # Take the hidden state of the [CLS] token (first token) as the embedding of the whole SMILES string
        tensor_embedding = outputs[0][0, 0]

        embedding = tensor_embedding.tolist()

        # Append the embedding to the list
        embeddings.append(embedding)

    return embeddings


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    logger = logging.getLogger(__name__)

    # Connect to a database in memory
    connection = duckdb.connect(database=":memory:")

    # Load the data into the database as list of tuples
    smiles = connection.execute(
        f"""
    SELECT DISTINCT canonical_smiles
    FROM '{input_filepath}/array_*.parquet'
    """
    ).fetchall()

    # Convert list of tuples to list of strings
    smiles_list = [x[0] for x in smiles]

    logger.info("creating embeddings from selfies...")

    # Get the embedding of a SMILES string
    embeddings_list = chemberta_encode(smiles_list)

    # Create column names
    embedding_names = [
        "embedding_" + str(i + 1) for i in range(len(embeddings_list[0]))
    ]

    # Convert list of embeddings to DataFrame where each element of the list becomes a separate column
    embeddings_df = pd.DataFrame(embeddings_list, columns=embedding_names)

    # # Add the SMILES strings to the DataFrame
    embeddings_df["smiles"] = smiles_list

    # Rearrange the columns so that 'smiles' column comes first
    embeddings_df = embeddings_df[
        ["smiles"] + [col for col in embeddings_df.columns if col != "smiles"]
    ]

    # Save the dataframe as a parquet file
    embeddings_df.to_parquet(f"{output_filepath}/chemberta_embeddings.parquet")



if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
