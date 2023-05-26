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

    for smiles in smiles_list:
        # Tokenize the SMILES string and get the input IDs
        input_id = tokenizer.encode(
            smiles, return_tensors="pt", padding=True, truncation=True
        )

        # Feed input IDs through the model. This will return the hidden states for each token in the input
        with torch.no_grad():
            outputs = model(input_id)

       # Get the hidden states of the last layer for all tokens
        tensor_embeddings = outputs[0][0]

        # Compute the mean along the dimension corresponding to the tokens (dimension 0)
        average_tensor_embedding = tensor_embeddings.mean(dim=0)

        embedding = average_tensor_embedding.tolist()

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
    FROM '{input_filepath}/assay_*.parquet'
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
    embeddings_df["canonical_smiles"] = smiles_list

    # Rearrange the columns so that 'smiles' column comes first
    embeddings_df = embeddings_df[
        ["canonical_smiles"] + [col for col in embeddings_df.columns if col != "canonical_smiles"]
    ]

    # Save the dataframe as a parquet file
    embeddings_df.to_parquet(f"{output_filepath}/feature_chemberta_embeddings.parquet")

    logger.info("emeddings written to %s", output_filepath)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
