import os
import logging
import pyarrow.parquet as pq
import click
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
import selfies as sf
import numpy as np

def assign_test_train(df_len):
    """Creates pd.Series with random assignment of each row to test or train."""

    # Create a list with 80% 1s and 20% 0s
    test_train = [1]*int(df_len*0.8) + [0]*int(df_len*0.2)

    # Randomly shuffle the list
    np.random.shuffle(test_train)

    # Create the pandas Series
    return pd.Series(test_train)

def safe_encoder(smiles):
    try:
        return sf.encoder(smiles)
    except Exception as e:
        return None

def convert_to_assay(input_filepath, output_filepath):
    """Converts an unprocessed parquet file to processed parquet files
    for each assay.

    Args:
        input_filepath (str): Path to the unprocessed parquet file.

    Returns:
        int: The number of assays successfully converted.
        int: The number of SMILES that could not be converted to canonical SMILES.

    Writes:
        Parquet files for each assay with the following attributes:

            smiles (str)
            canonical_smiles (str)
            selfies (str)
            source_id
            assay_id
            publication
            ground_truth
            train_test
    """
    # Suppress RDKit warnings
    RDLogger.DisableLog("rdApp.*")

    # Read in the unprocessed parquet file
    df = pq.read_table(input_filepath).to_pandas()

    # Get assay names
    all_columns = df.columns.tolist()
    assay_names = [col for col in all_columns if col != 'smiles']

    # Convert smiles to canonical smiles
    df['canonical_smiles'] = df['smiles'].apply(Chem.MolFromSmiles)
    df['canonical_smiles'] = df['canonical_smiles'].apply(lambda x: Chem.MolToSmiles(x) if x is not None else None)

    # Create selfie column
    df["selfies"] = df["canonical_smiles"].apply(lambda x: safe_encoder(x) if x is not None else None)
    
    # Get number of smiles that could not be converted to canonical smiles
    smiles_errors = df["canonical_smiles"].isnull().sum()

    # Get number of canonical smiles that could not be converted to selfies
    selfies_errors = df["selfies"].isnull().sum()

    # # Drop rows with null canonical smiles or selfies
    df.dropna(subset=["canonical_smiles"], inplace=True)

    # Get the filename without the suffix
    filename = os.path.splitext(os.path.basename(input_filepath))[0]

    # Add source_id column
    df["source_id"] = filename

    # Loop through each assay
    for assay_name in assay_names:
        # Filter out rows with null values for the assay
        assay_df = df.dropna(subset=[assay_name])
        
        # Randomly assign each row to train or test
        assay_df['test_train'] = assign_test_train(len(assay_df))

        # Change assay column label as ground_truth
        assay_df.rename(columns={'assay': 'ground_truth'}, inplace=True)

        # Write each assay to a parquet file
        assay_df.to_parquet(f"{output_filepath}/{assay_name}_{filename}.parquet")

    # Return error numbers
    return smiles_errors, selfies_errors, len(df)


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    logger = logging.getLogger(__name__)
    logger.info("converting raw data to individual assay parquet files")
    smiles_errors, selfies_errors, assay_num = convert_to_assay(input_filepath, output_filepath)
    logger.info(
        "successfully created %d assay parquet files. %d SMILES could not be converted to canonical SMILES. %d canonical SMILES could not be converted to selfies.",
        assay_num,
        smiles_errors,
        selfies_errors
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
