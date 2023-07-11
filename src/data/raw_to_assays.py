import os
import logging
import click
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
import numpy as np


from utils import (
    filter_by_active_ratio,
    inchi_to_smiles,
    smiles_to_canonical_smiles,
    assign_test_train,
    pivot_assays,
    binarize_assays,
    filter_by_range,
    drug_name_to_smiles
)


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


def process_toxval(input_filepath, identifier):
    """Process the toxval dataset."""

    # List of columns to extract from raw data
    identifiers = [
        "dtxsid",
        "casrn"
    ]

    additional_cols = "toxval_numeric_qualifier"

    outcome = "toxval_numeric"

    assay_components = [
        "long_ref",
        "toxval_type",
        "common_name",
        "exposure_route",
        "toxval_units",
        "study_type",
        "source",
    ]

    df = pd.read_csv(input_filepath, usecols=identifiers + [outcome] + assay_components + [additional_cols])

    # Replace '-' in outcome col with np.nan
    df["toxval_numeric"].replace("-", np.nan, inplace=True)

    # Drop rows with null values for key variables
    df.dropna(subset=assay_components + [outcome] + [additional_cols], inplace=True)

    # Drop long refs with different NA formats
    na_list = ["- - - NA", "- - - -", "- Unnamed - NA", "Unknown", "- Unnamed - -"]
    df = df[~df["long_ref"].isin(na_list)]

    # Remove those records with toxval_numeric_qualifier not equal to "="
    df = df[df["toxval_numeric_qualifier"] != "="]

    # Drop the toxval_numeric_qualifier column
    df.drop(columns="toxval_numeric_qualifier", inplace=True)

    # Read in the identifiers
    identifiers = pd.read_csv(identifier)

    # Merge tox data with the molecule identifiers
    df = df.merge(identifiers, how="left", on="dtxsid")

    # Apply the function to each value in the Series
    df = inchi_to_smiles(df)

    # Reset index so smiles series and df index align correctly
    df = df.reset_index(drop=True)

    # Replace all '_' with '-' in long_ref column
    df["long_ref"] = df["long_ref"].str.replace("_", "-")

    # Apply -log to toxval numeric column
    df["toxval_numeric"] = -np.log(df["toxval_numeric"])

    # Pivot the DataFrame so that each column is a unique assay
    df = pivot_assays(df, assay_components, "toxval_numeric")

    # Keep assays where the order of magnitude outcome range is greater than 2
    df = filter_by_range(df)

    # Binarize the assays
    df = binarize_assays(df)

    # Convert smiles index to a column
    df.reset_index(inplace=True)

    return df


def process_nci60(input_filepath, identifier_filepath, group_size=32):
    df = pd.read_csv(input_filepath)

    # Define assay components columns
    assay_components = ["PANEL_NAME", "CELL_NAME", "CONCENTRATION_UNIT", "EXPID"]

    # Remove AVERAGE values greater than -3.5
    df = df[df["AVERAGE"] < 3.5]

    # Read in the identifiers
    identifier_col_names = ["nsc", "casrn", "smiles"]
    identifiers = pd.read_csv(
        identifier_filepath, delim_whitespace=True, names=identifier_col_names
    )

    # Merge the filtered DataFrame with the identifiers
    df = pd.merge(df, identifiers, left_on="NSC", right_on="nsc", how="inner")

    # Remove records that belong to a group of fewer than specified size
    df = df.groupby(assay_components).filter(lambda x: len(x) >= group_size)

    # Pivot the DataFrame so that each column is a unique assay
    df = pivot_assays(df, assay_components, "AVERAGE")

    # Keep assays where the order of magnitude outcome range is greater than 2
    df = filter_by_range(df)

    # Binarize the assays
    df = binarize_assays(df)

    # Convert smiles index to a column
    df.reset_index(inplace=True)

    return df


def process_cancerrx(input_filepath):
    df = pd.read_excel(input_filepath)

    # Change "SMILES" column name to "smiles"
    df.rename(columns={"SMILES": "smiles"}, inplace=True)
    
    # Convert drug names to smiles
    df = drug_name_to_smiles(df)

    # Set the assay components
    assay_components = ["CELL_LINE_NAME"]

    # Convert the LN_IC50 column to negative
    df["LN_IC50"] = -(df["LN_IC50"])

    # Pivot the DataFrame so that each column is a unique assay
    df = pivot_assays(df, assay_components, "LN_IC50")

    # Keep assays where the order of magnitude outcome range is greater than 2
    df = filter_by_range(df)

    # Binarize the assays
    df = binarize_assays(df)

    # Convert smiles index to a column
    df.reset_index(inplace=True)

    return df


def process_prism(input_filepath):
    df = pd.read_csv(input_filepath, dtype={14: str, 15: str})

    # Set the assay components
    assay_components = ["ccle_name"]

    # Convert ec50 to negative log
    df["ec50"] = -np.log(df["ec50"])

    # Pivot the DataFrame so that each column is a unique assay
    df = pivot_assays(df, assay_components, "ec50")

    # Remove columns where the absolute range is less than 2
    df = filter_by_range(df)

    # Binarize the assays
    df = binarize_assays(df)

    # Convert smiles index to a column
    df.reset_index(inplace=True)

    return df


def convert_to_parquets(df, source_id, output_filepath):
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
    """

    # Suppress RDKit warnings
    RDLogger.DisableLog("rdApp.*")

    assay_names = [col for col in df.columns if col != "canonical_smiles"]

    # Loop through each assay
    for assay_name in assay_names:
        # Get dataframe with assay and smiles columns
        assay_df = df[["canonical_smiles", assay_name]].copy()

        # Filter out rows with null values for the assay
        assay_df.dropna(subset=[assay_name], inplace=True)
        assay_df.reset_index(drop=True, inplace=True)

        # Randomly assign each row to train or test
        assay_df["support_query"] = assign_test_train(len(assay_df))

        # Rename assay column label to 'ground_truth'
        assay_df.rename(columns={assay_name: "ground_truth"}, inplace=True)

        # Convert ground_truth column to int from float
        assay_df["ground_truth"] = assay_df["ground_truth"].astype(int)

        # Create a column for the assay name
        assay_df["assay_id"] = assay_name

        # Add source_id column
        assay_df["source_id"] = source_id

        # Write each assay to a parquet file
        assay_df.to_parquet(f"{output_filepath}/{assay_name}_{source_id}.parquet")


@click.command(help="This command converts raw data to individual assay parquet files.")
@click.argument("input_filepath", type=click.Path())
@click.argument("output_filepath", type=click.Path(), default="data/processed/assays")
@click.option(
    "-d",
    "--dataset",
    type=click.Choice(
        ["tox21", "clintox", "toxcast", "bbbp", "toxval", "nci60", "cancerrx", "prism"]
    ),
    help="The name of the dataset to wrangle. This must be one of 'tox21', 'clintox', 'toxcast', 'bbbp', 'toxval', 'nci60', 'cancerrx', or 'prism'.",
)
@click.option(
    "-i",
    "--identifier",
    type=click.Path(),
    help="Filepath for chemical identifiers for toxvaldb and nci60.",
)
@click.option(
    "-s",
    "--size",
    type=int,
    help="The minimum number of records for an assay to be included.",
    default=32,
)
def main(input_filepath, output_filepath, dataset, identifier, size):
    logger = logging.getLogger(__name__)
    logger.info("converting raw data to individual assay parquet files...")

    # Return DataFrame with binary outcomes for each assay and a lookup table
    if dataset == "tox21":
        df = process_tox21(input_filepath)
    elif dataset == "clintox":
        df = process_clintox(input_filepath)
    elif dataset == "toxcast":
        df = process_toxcast(input_filepath)
    elif dataset == "bbbp":
        df = process_bbbp(input_filepath)
    elif dataset == "toxval":
        df = process_toxval(input_filepath, identifier)
    elif dataset == "nci60":
        df = process_nci60(input_filepath, identifier)
    elif dataset == "cancerrx":
        df = process_cancerrx(input_filepath)
    elif dataset == "prism":
        df = process_prism(input_filepath)
    else:
        raise ValueError("dataset must be one of 'tox21', 'clintox', 'toxcast', 'bbbp', 'toxval', 'nci60', 'cancerrx', or 'prism'.")

    # Remove columns not in active ratio range
    df = filter_by_active_ratio(df)

    # Get assay names
    assay_names = [col for col in df.columns if col != "smiles"]

    # Convert smiles to canonical_smiles
    df = smiles_to_canonical_smiles(df)

    # Remove columns with fewer non-null values than specified size
    df = df.loc[:, (df.count() >= size).values]

    # Convert the list of column names to a lookup DataFrame
    lookup_df = pd.DataFrame(assay_names, columns=["assay"])

    # Save lookup_df
    lookup_df.to_csv(
        os.path.join(f"./data/processed/assay_lookup/{dataset}_lookup.csv"), index=False
    )

    # Convert column names to standard identifiers
    df.columns = [
        f"assay_{i+1}" if col != "canonical_smiles" else col
        for i, col in enumerate(df.columns)
    ]

    # Convert the assay DataFrame to individual parquet files
    convert_to_parquets(df, dataset, output_filepath)

    logger.info("created %d individual assay parquet files.", df.shape[1] - 1)


if __name__ == "__main__":
    LOG_FMT = "%(asctime)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    main()
