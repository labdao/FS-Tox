import os
import logging
import click
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
import selfies as sf
import numpy as np

from utils import convert_to_smiles


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


def process_toxval(input_filepath, output_filepath, identifier):
    """Process the toxval dataset."""

    # Create list of columns to include
    vars_to_extract = [
        "dtxsid",
        "casrn",
        "long_ref",
        "toxval_type",
        "common_name",
        "exposure_route",
        "toxval_units",
        "study_type",
        "source",
        "toxval_numeric",
        "toxval_numeric_qualifier",
    ]

    df = pd.read_csv(input_filepath, usecols=vars_to_extract)

    # Replace '-' with np.nan
    df.replace("-", np.nan, inplace=True)

    assay_components = [
        "long_ref",
        "toxval_type",
        "common_name",
        "exposure_route",
        "toxval_units",
        "study_type",
        "source",
    ]

    # Drop rows with null values for key variables
    df.dropna(subset=assay_components + ["toxval_numeric"], inplace=True)

    # Drop long refs with different NA formats
    na_list = ["- - - NA", "- - - -", "- Unnamed - NA", "Unknown", "- Unnamed - -"]
    df = df[~df["long_ref"].isin(na_list)]

    # Remove those records with toxval_numeric_qualifier not equal to "="
    df = df[df["toxval_numeric_qualifier"] != "="]

    # Drop the toxval_numeric_qualifier column as it is no longer needed
    df.drop(columns="toxval_numeric_qualifier", inplace=True)

    # Read in the identifiers
    identifiers = pd.read_csv(identifier)

    # Merge tox data with the molecule identifiers
    df_with_inchis = df.merge(identifiers, how="left", on="dtxsid")

    # Apply the function to each value in the Series
    smiles_series = df_with_inchis["inchi"].astype(str).apply(convert_to_smiles)

    # Reset index to ensure smiles series appends correctly
    df = df.reset_index(drop=True)

    # Add the smiles column to the DataFrame
    df["smiles"] = smiles_series

    # Drop rows where smiles column is equal to 'InvalidInChI'
    df = df[df["smiles"] != "InvalidInChI"]

    # Get records that belong to a group of greater than 10 members
    df = df.groupby(assay_components).filter(lambda x: len(x) >= 24)

    # Replace all '_' with '-' in long_ref column
    df["long_ref"] = df["long_ref"].str.replace("_", "-")

    # Create a new column that is a combination of the assay_components
    df["combined"] = df[assay_components].apply(
        lambda row: "_".join(row.values.astype(str)), axis=1
    )

    # Create a pivot table where each unique combination forms a separate column
    df_pivoted = df.pivot_table(
        index="smiles",
        columns="combined",
        values="toxval_numeric",
        aggfunc=np.mean,
    )

    def binarize_series(series):
        median = series.median()
        return series.apply(
            lambda x: 1 if x > median else (0 if pd.notnull(x) else np.nan)
        )

    # Apply the function to each column
    df_pivoted = df_pivoted.apply(binarize_series)

    # Remove all columns where the value is identical for all non-null rows
    df_pivoted = df_pivoted.loc[:, df_pivoted.nunique() != 1]

    # Remove all columns where there are fewer than 24 non-null values
    df_pivoted = df_pivoted.dropna(axis=1, thresh=24)

    # Convert smiles index to column
    df_pivoted = df_pivoted.reset_index()

    # Create a lookup table for the assay names
    assay_names = pd.DataFrame(df_pivoted.columns[1:], columns=["combined"])

    # Split the 'combined' column at '_', expanding into new columns
    split_df = assay_names["combined"].str.split("_", expand=True)

    # Name the new columns
    split_df.columns = assay_components

    # Save the assay names as a lookup table
    split_df.to_csv(os.path.join("./data/external/assay_lookup.csv"), index=False)

    # Simplify the column names
    df_pivoted.columns = [
        f"assay_{i}" if col != "smiles" else col
        for i, col in enumerate(df_pivoted.columns)
    ]

    return df_pivoted


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
        assay_df["ground_truth"] = assay_df["ground_truth"].astype(int)

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
    type=click.Choice(["tox21", "clintox", "toxcast", "bbbp", "toxval"]),
    help="The name of the dataset to wrangle. This must be one of 'tox21', 'clintox', 'toxcast', 'bbbp' or 'toxval'.",
)
@click.option(
    "-i",
    "--identifier",
    type=click.Path(),
    help="Filepath for chemical identifiers for toxvaldb.",
)
def main(input_filepath, output_filepath, dataset, identifier):
    logger = logging.getLogger(__name__)
    logger.info("converting raw data to individual assay parquet files")

    # Create interim parquet file for each dataset
    if dataset == "tox21":
        df = process_tox21(input_filepath)
    elif dataset == "clintox":
        df = process_clintox(input_filepath)
    elif dataset == "toxcast":
        df = process_toxcast(input_filepath)
    elif dataset == "bbbp":
        df = process_bbbp(input_filepath)
    elif dataset == "toxval":
        df = process_toxval(input_filepath, output_filepath, identifier)

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
