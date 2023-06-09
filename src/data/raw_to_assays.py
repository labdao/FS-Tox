import os
import logging
import pyarrow.parquet as pq
import click
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
import selfies as sf
import numpy as np


def process_tox21(input_filepath):
    """Process the tox21 dataset."""
    df = pd.read_csv(input_filepath)

    # Drop the mol_id column
    df.drop(columns="mol_id", inplace=True)

    return df


def process_clintox(input_filepath):
    """Process the clintox dataset."""
    df = pd.read_csv(input_filepath)

    return df


def process_toxcast(input_filepath):
    """Process the toxcast dataset."""
    df = pd.read_csv(input_filepath)

    return df

def process_bbbp(input_filepath):
    """Process the bbbp dataset."""
    df = pd.read_csv(input_filepath)

    # Remove molecule name and num column
    df.drop(columns=["num", "name"], inplace=True)
    print(df.head())


    return df

def assign_test_train(df_len):
    """
    Creates a pandas Series with random assignment of each row to test or train.

    Parameters:
    df_len (int): The length of the DataFrame to create the test-train split for.

    Returns:
    pd.Series: A pandas Series with random assignment of each row to test (0) or train (1).
    """

    # Set the proportions for 0s and 1s
    proportions = [0.8, 0.2]

    # Create a random series with the desired proportions
    test_train = pd.Series(np.random.choice([0, 1], size=df_len, p=proportions))

    # Create the pandas Series
    return test_train


def safe_encoder(smiles):
    """
    Encodes a SMILES string using the selfies library, handling any exceptions that may occur.

    Parameters:
    smiles (str): The SMILES string to encode.

    Returns:
    str or None: The encoded SELFIES string, or None if an exception occurred during encoding.
    """
    try:
        return sf.encoder(smiles)
    except Exception as e:
        return None


def convert_to_assay(df, source_id, output_filepath):
    """
    Converts an unprocessed DataFrame to individual parquet files for each assay.

    This function processes the input DataFrame, extracting each assay name, converting 'smiles' to 'canonical_smiles'
    and 'selfies', adding a source_id, and writing the resulting DataFrame for each assay to individual parquet files.
    It also returns the number of 'smiles' that could not be converted to 'canonical_smiles' and the number of
    'canonical_smiles' that could not be converted to 'selfies'.

    Args:
        df (pd.DataFrame): The unprocessed input DataFrame. Must contain 'smiles' and assay columns.
        source_id (str): The source identifier to be added to the DataFrame.
        output_filepath (str): The directory where the resulting parquet files will be saved.

    Returns:
        tuple: A tuple containing the number of 'smiles' that could not be converted to 'canonical_smiles' (int),
               the number of 'canonical_smiles' that could not be converted to 'selfies' (int),
               and the number of assays successfully converted (int).
    """

    # Suppress RDKit warnings
    RDLogger.DisableLog("rdApp.*")

    # Get assay names
    all_columns = df.columns.tolist()
    assay_names = [col for col in all_columns if col != "smiles"]

    # Convert smiles to canonical smiles
    mol_objects = df["smiles"].apply(Chem.MolFromSmiles)
    df["canonical_smiles"] = mol_objects.apply(
        lambda x: Chem.MolToSmiles(x) if x is not None else None
    )

    # Create selfie column
    df["selfies"] = df["canonical_smiles"].apply(
        lambda x: safe_encoder(x) if x is not None else None
    )

    # Get number of smiles that could not be converted to canonical smiles
    smiles_errors = df["canonical_smiles"].isnull().sum()

    # Get number of canonical smiles that could not be converted to selfies
    selfies_errors = df["selfies"].isnull().sum()

    # Drop rows with null canonical smiles or selfies
    df.dropna(subset=["canonical_smiles"], inplace=True)

    # Add source_id column
    df["source_id"] = source_id

    # Loop through each assay
    for assay_name in assay_names:
        # Get dataframe with assay and smiles columns
        assay_df = df[["canonical_smiles", assay_name, "selfies", "source_id"]].copy()

        # Filter out rows with null values for the assay
        assay_df.dropna(subset=[assay_name], inplace=True)
        assay_df.reset_index(drop=True, inplace=True)

        # Randomly assign each row to train or test
        assay_df["test_train"] = assign_test_train(len(assay_df))

        # Change assay column label as ground_truth
        assay_df.rename(columns={assay_name: "ground_truth"}, inplace=True)

        # Convert ground_truth column to int from float
        assay_df["ground_truth"] = assay_df["ground_truth"].astype(
            int
        )

        # Create a column for the assay name
        assay_df["assay_id"] = assay_name

        # Write each assay to a parquet file
        assay_df.to_parquet(f"{output_filepath}/{assay_name}_{source_id}.parquet")

    # Return error numbers
    return smiles_errors, selfies_errors, len(assay_names)


@click.command(help="This command converts raw data to individual assay parquet files.")
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.option(
    "-d",
    "--dataset",
    type=click.Choice(["tox21_2023", "clintox_2023", "toxcast_2023", "bbbp_2023"]),
    help="The name of the dataset to wrangle. This must be one of 'tox21_2023', 'clintox_2023', 'toxcast_2023', or 'bbbp_2023'.",
)
def main(input_filepath, output_filepath, dataset):
    logger = logging.getLogger(__name__)
    logger.info("converting raw data to individual assay parquet files")

    # Create interim parquet file for each dataset
    if dataset == "tox21_2023":
        df = process_tox21(input_filepath)
    elif dataset == "clintox_2023":
        df = process_clintox(input_filepath)
    elif dataset == "toxcast_2023":
        df = process_toxcast(input_filepath)
    elif dataset == "bbbp_2023":
        df = process_bbbp(input_filepath)

    # Get the source_id from the input filepath
    source_id = os.path.splitext(os.path.basename(input_filepath))[0]

    smiles_errors, selfies_errors, assay_num = convert_to_assay(
        df, source_id, output_filepath
    )
    logger.info(
        """successfully created %d assay parquet files.
    %d SMILES could not be converted to canonical SMILES.
    %d canonical SMILES could not be converted to selfies.""",
        assay_num,
        smiles_errors,
        selfies_errors,
    )

if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
