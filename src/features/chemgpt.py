import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

import torch
from transformers import AutoTokenizer, AutoModel

import click
import duckdb


def chemgpt_encode(selfies):
    tokenizer = AutoTokenizer.from_pretrained("ncfrey/ChemGPT-4.7M")
    model = AutoModel.from_pretrained("ncfrey/ChemGPT-4.7M")

    # Encode the SMILES string
    inputs = tokenizer(
        selfies,
        return_tensors="pt"
    )

    # Feed the inputs to the model
    with torch.no_grad():  # disable gradient calculations to save memory
        outputs = model(**inputs)

    # Output is a tuple, where the first item are the hidden states
    hidden_states = outputs[0]

    return hidden_states


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
def main(input_filepath):
    logger = logging.getLogger(__name__)
    logger.info("creating embeddings from selfies")

    # Connect to a database in memory
    connection = duckdb.connect(database=":memory:")

    # Load the data into the database as list of tuples
    selfies = connection.execute(
        """
    SELECT DISTINCT selfies
    FROM '/Users/sethhowes/Desktop/FS-Tox/data/processed/*_toxcast_2023.parquet'
    """
    ).fetchall()

    # Convert list of tuples to list of strings
    selfies_list = [x[0] for x in selfies]

    #test_selfies = "[C][N][C][Branch1_2][C][=O][C][=C][Branch2_1][Ring2][Branch1_3][C][=C][C][=C][Branch1_1][Ring2][S][Ring1][Branch1_1][C][S][C][Branch1_1][N][C][=N][C][=C][Branch1_1][Ring1][C][#N][S][Ring1][Branch1_3][=C][C][Expl=Ring1][N][C][Ring1][S][O][C][C][O][Ring1][Branch1_1][N][Branch1_1][C][C][C][Branch1_2][C][=O][C][Ring2][Ring1][=N][=C][Ring2][Ring1][P][C][=C][C][=C][Branch1_1][Ring2][S][Ring1][Branch1_1][C][S][C][Branch1_1][N][C][=N][C][=C][Branch1_1][Ring1][C][#N][S][Ring1][Branch1_3][=C][C][Expl=Ring1][N][C][Ring1][S][O][C][C][O][Ring1][Branch1_1]"
    embeddings = chemgpt_encode(selfies_list)


if __name__ == "__main__":
    main()
