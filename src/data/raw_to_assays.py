import logging
import os

import click
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from tdc.single_pred import Tox

from joblib import Memory

# Setup joblib caching configuration
cache_dir = os.path.join(os.getcwd(), ".assay_cache")
memory = Memory(cache_dir, verbose=0)


from .utils import (
    assign_test_train,
    binarize_assays,
    drug_name_to_smiles,
    filter_by_active_ratio,
    filter_by_range,
    get_sha256_snippet,
    inchi_to_smiles,
    pivot_assays,
    smiles_to_canonical_smiles,
)

@memory.cache
def process_tox21(input_filepath):
    """Process the tox21 dataset."""
    df = pd.read_csv(f"{input_filepath}/tox21.csv")

    # Drop the mol_id column
    df.drop(columns="mol_id", inplace=True)

    return df

@memory.cache
def process_clintox(input_filepath):
    """Process the clintox dataset."""
    df = pd.read_csv(f"{input_filepath}/clintox.csv")

    return df

@memory.cache
def process_toxcast(input_filepath):
    """Process the toxcast dataset."""
    df = pd.read_csv(f"{input_filepath}/toxcast.csv")

    return df

@memory.cache
def process_bbbp(input_filepath):
    """Process the bbbp dataset."""
    df = pd.read_csv(f"{input_filepath}/bbbp.csv")

    # Remove molecule name and num column
    df.drop(columns=["num", "name"], inplace=True)
    print(df.head())

    return df


@memory.cache
def process_toxval(input_filepath, identifier):
    """Process the toxval dataset."""

    # List of columns to extract from raw data
    identifiers = ["dtxsid", "casrn"]

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

    df = pd.read_csv(
        f"{input_filepath}/toxval.csv",
        usecols=identifiers + [outcome] + assay_components + [additional_cols],
    )

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

    # Filter out any rows with 0 for toxval_numeric
    df = df[df["toxval_numeric"] != 0]

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


@memory.cache
def process_nci60(input_filepath, identifier_filepath, assay_size):
    df = pd.read_csv(f"{input_filepath}/nci60.csv")

    # Define assay components columns
    assay_components = ["CELL_NAME", "CONCENTRATION_UNIT"]

    # Remove AVERAGE values greater than -3.5
    df = df[df["AVERAGE"] < -3.5]

    # Read in the identifiers
    identifier_col_names = ["nsc", "casrn", "smiles"]
    identifiers = pd.read_csv(
        identifier_filepath, delim_whitespace=True, names=identifier_col_names
    )

    # Merge the filtered DataFrame with the identifiers
    df = pd.merge(df, identifiers, left_on="NSC", right_on="nsc", how="inner")

    # Remove records that belong to a group of fewer than specified size
    df = df.groupby(assay_components).filter(lambda x: len(x) >= assay_size)

    # Pivot the DataFrame so that each column is a unique assay
    df = pivot_assays(df, assay_components, "AVERAGE")

    # Keep assays where the order of magnitude outcome range is greater than 2
    df = filter_by_range(df)

    # Binarize the assays
    df = binarize_assays(df)

    # Convert smiles index to a column
    df.reset_index(inplace=True)

    return df


@memory.cache
def process_cancerrx(input_filepath):
    df = pd.read_excel(f"{input_filepath}/cancerrx.xlsx")

    # Change "SMILES" column name to "smiles"
    df.rename(columns={"SMILES": "smiles"}, inplace=True)

    # Convert drug names to smiles
    df = drug_name_to_smiles(df, "DRUG_NAME")

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


@memory.cache
def process_prism(input_filepath):
    df = pd.read_csv(f"{input_filepath}/prism.csv", dtype={14: str, 15: str})

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


@memory.cache
def process_acute_oral_toxicity():
    # Load data
    data = Tox(name = 'LD50_Zhu')
    df = data.get_data()
    df.drop(columns="Drug_ID", inplace=True)
    df.rename(columns={"Drug": "smiles"}, inplace=True)
    
    # Change outcome column to negative log from positive log
    df["Y"] = -df["Y"]
    
    # Set the SMILES column as the index
    df.set_index("smiles", inplace=True)

    # Keep assays where the order of magnitude outcome range is greater than 2
    df = filter_by_range(df)

    # Binarize the assays
    df = binarize_assays(df)

    # Convert smiles index to a column
    df.reset_index(inplace=True)

    return df


@memory.cache
def process_meic(input_filepath):
    df = pd.read_csv(f"{input_filepath}/meic.csv")

    # Remove molecular weight column
    df.drop("Molecular weight", axis=1, inplace=True)

    # Set the outcome column names
    outcome_cols = ["Rat LD50 (log mol/kg)", "MouseLD50 (log mol/kg)" ,"Human lethal dose (log mol/kg)"]

    # Convert log-ld50 to negative log-ld50
    df[outcome_cols] = -df[outcome_cols]

    # Get smiles from Chemical column
    df = drug_name_to_smiles(df, "Chemical")

    df.drop("Chemical", axis=1, inplace=True)

    # Set smiles column as index
    df.set_index("smiles", inplace=True)

    # Remove columns where the absolute range is less than 2
    df = filter_by_range(df)

    # Binarize the assays
    df = binarize_assays(df)

    # Convert smiles index to a column
    df.reset_index(inplace=True)

    return df

def load_data(input_filepath, dataset, identifier, assay_size):
    # Return DataFrame with binary outcomes for each assay and a lookup table
    if dataset == "tox21":
        return process_tox21(input_filepath)
    elif dataset == "clintox":
        return process_clintox(input_filepath)
    elif dataset == "toxcast":
        return process_toxcast(input_filepath)
    elif dataset == "bbbp":
        return process_bbbp(input_filepath)
    elif dataset == "toxval":
        return process_toxval(input_filepath, f"{identifier}/toxval_identifiers.csv")
    elif dataset == "nci60":
        return process_nci60(
            input_filepath, f"{identifier}/nci60_identifiers.txt", assay_size
        )
    elif dataset == "cancerrx":
        return process_cancerrx(input_filepath)
    elif dataset == "prism":
        return process_prism(input_filepath)
    elif dataset == "acute_oral_toxicity":
        return process_acute_oral_toxicity()
    elif dataset == "meic":
        return process_meic(input_filepath)
    else:
        raise ValueError(
            "dataset must be one of 'tox21', 'clintox', 'toxcast', 'bbbp', 'toxval', 'nci60', 'cancerrx', or 'prism'."
        )

def preprocess_data(df, assay_size):
     # Remove columns not in active ratio range
    df = filter_by_active_ratio(df)

    # Convert smiles to canonical_smiles
    df = smiles_to_canonical_smiles(df)

    # Remove columns with fewer non-null values than specified size
    df = df.loc[:, (df.count() >= assay_size).values]

    return df

def convert_to_parquets(
    df, source_id, output_filepath, support_set_size, test_prob
):
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

    assay_ids = [col for col in df.columns if col != "canonical_smiles"]

    # Loop through each assay
    for i, assay_id in enumerate(assay_ids):
        # Get dataframe with assay and smiles columns
        assay_df = df[["canonical_smiles", assay_id]].copy()

        # Filter out rows with null values for the assay
        assay_df.dropna(subset=[assay_id], inplace=True)
        assay_df.reset_index(drop=True, inplace=True)

        # Randomly assign each row to train or test
        assay_df["support_query"] = assign_test_train(len(assay_df), support_set_size)

        # Rename assay column label to 'ground_truth'
        assay_df.rename(columns={assay_id: "ground_truth"}, inplace=True)

        # Convert ground_truth column to int from float
        assay_df["ground_truth"] = assay_df["ground_truth"].astype(int)

        # Create a column for the assay name
        assay_df["assay_id"] = assay_id

        # Add source_id column
        assay_df["source_id"] = source_id

        # Write each assay to a parquet file
        assay_df.to_parquet(f"{output_filepath}/{assay_id}.parquet")


def make_assays(
    input_filepath,
    output_filepath,
    assay_id_path,
    dataset,
    identifier,
    assay_size,
    support_set_size,
    test_prob,
):
    logger = logging.getLogger(__name__)
    logger.info("converting %s raw data to individual assay parquet files...", dataset)
    
    # Load data 
    df = load_data(input_filepath, dataset, identifier, assay_size)

    # Filter by active ratio, convert smiles to canonical_smiles, and remove columns with fewer non-null values than specified size
    df = preprocess_data(df, assay_size)

    # Get assay names
    assay_components = [col for col in df.columns if col != "canonical_smiles"]

    # Create assay_identifiers using hash of column name
    assay_ids = [
        get_sha256_snippet(col) for col in df.columns if col != "canonical_smiles"
    ]

    # Convert the list of column names to a lookup DataFrame
    lookup_df = pd.DataFrame({"assay_name": assay_components, "assay_id": assay_ids})

    # Save lookup_df
    lookup_df.to_parquet(
        os.path.join(f"{assay_id_path}/{dataset}.parquet"),
        index=False,
    )

    # Convert column names to standard identifiers
    df.columns = [get_sha256_snippet(col) if col != "canonical_smiles" else col for col in df.columns]

    # Convert the assay DataFrame to individual parquet files
    convert_to_parquets(
        df, dataset, output_filepath, support_set_size, test_prob
    )

    logger.info("created %d individual assay parquet files.", df.shape[1] - 1)
