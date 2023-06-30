import os
import logging
import click
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
import numpy as np


from utils import inchi_to_smiles, smiles_to_canonical_smiles, selfies_encoder, assign_test_train, pivot_assays

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
    canonical_smiles = df_with_inchis["inchi"].astype(str).apply(inchi_to_smiles)

    # Reset index to ensure smiles series appends correctly
    df = df.reset_index(drop=True)

    # Add the smiles column to the DataFrame
    df["canonical_smiles"] = canonical_smiles

    # Drop rows where smiles column is equal to 'InvalidInChI'
    df = df[df["canonical_smiles"] != "InvalidInChI"]

    # Get records that belong to a group of greater than 10 members
    df = df.groupby(assay_components).filter(lambda x: len(x) >= 32)

    # Replace all '_' with '-' in long_ref column
    df["long_ref"] = df["long_ref"].str.replace("_", "-")

    # Apply -log to toxval numeric column
    df["toxval_numeric"] = -np.log(df["toxval_numeric"])

    # Pivot the DataFrame so that each column is a unique assay
    assay_df, lookup_df = pivot_assays(df, assay_components, "toxval_numeric")

    # Save the lookup table
    lookup_df.to_csv(os.path.join(f"./data/processed/assay_lookup/toxval_lookup.csv"), index=False)

    return assay_df


def process_nci60(input_filepath, identifier_filepath):
    df = pd.read_csv(input_filepath)
    
    # Columns to group by
    assay_components = ["PANEL_NAME", "CELL_NAME", "CONCENTRATION_UNIT", "EXPID"]

    # Remove values with higher than -3.5 AVERAGE conc
    df = df[df["AVERAGE"] < 3.5]

    # Remove records that belong to a group of fewer than 24 members
    df = df.groupby(assay_components).filter(lambda x: len(x) >= 32)

    # Read in the identifiers
    identifier_col_names = ["nsc", "casrn", "smiles"]
    identifiers = pd.read_csv(identifier_filepath, delim_whitespace=True, names=identifier_col_names)

    # Merge the filtered DataFrame with the identifiers
    df = pd.merge(df, identifiers, left_on="NSC", right_on="nsc", how="inner")

    if os.path.isfile('temp_data.pkl'):
        df = pd.read_pickle('temp_data.pkl')
    else:
        # Get canonical smiles
        df = smiles_to_canonical_smiles(df)
        df.to_pickle('temp_data.pkl')
        
    # Drop rows where canonical_smiles is null
    df.dropna(subset=["canonical_smiles"], inplace=True)

    # Remove records that belong to a group of fewer than 24 members after removing null canonical_smiles
    df = df.groupby(assay_components).filter(lambda x: len(x) >= 32)

    # Pivot the DataFrame so that each column is a unique assay
    df, lookup_df = pivot_assays(df, assay_components, "AVERAGE")

    # Save the lookup table
    lookup_df.to_csv(os.path.join("./data/processed/assay_lookup/nci60_lookup.csv"), index=False)

    return df

def process_cancerrx(input_filepath):
    df = pd.read_csv(input_filepath)

    # Change "SMILES" column name to "smiles"
    df.rename(columns={"SMILES": "smiles"}, inplace=True)

    # Set the assay components
    assay_components = ["CELL_LINE_NAME"]
    
    # Convert the LN_IC50 column to negative
    df["LN_IC50"] = -(df["LN_IC50"])

    # Convert the SMILES column to canonical smiles
    df = smiles_to_canonical_smiles(df)

    # Drop rows where canonical_smiles is null
    df.dropna(subset=["canonical_smiles"], inplace=True)

    # Pivot the DataFrame so that each column is a unique assay
    df, lookup_df = pivot_assays(df, assay_components, "LN_IC50")

    # Save the lookup table
    lookup_df.to_csv(os.path.join("./data/processed/assay_lookup/cancerrx_lookup.csv"), index=False)
    
    return df


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

    # Convert smiles to canonical smiles if not already done
    if "canonical_smiles" not in df.columns:
        df = smiles_to_canonical_smiles(df, "smiles")

    # Get assay names
    all_columns = df.columns.tolist()
    assay_names = [col for col in all_columns if col != "canonical_smiles"]

    # Create selfie column
    df["selfies"] = df["canonical_smiles"].apply(
        lambda x: selfies_encoder(x) if x is not None else None
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
        assay_df["support_query"] = assign_test_train(len(assay_df))

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
@click.argument("input_filepath", type=click.Path())
@click.argument("output_filepath", type=click.Path(), default="data/processed/assays")
@click.option(
    "-d",
    "--dataset",
    type=click.Choice(["tox21", "clintox", "toxcast", "bbbp", "toxval", "nci60", "cancerrx"]),
    help="The name of the dataset to wrangle. This must be one of 'tox21', 'clintox', 'toxcast', 'bbbp', 'toxval', 'nci60', 'cancerrx'.",
)
@click.option(
    "-i",
    "--identifier",
    type=click.Path(),
    help="Filepath for chemical identifiers for toxvaldb and nci60.",
)
def main(input_filepath, output_filepath, dataset, identifier):
    logger = logging.getLogger(__name__)
    logger.info("converting raw data to individual assay parquet files...")

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
        df = process_toxval(input_filepath, identifier)
    elif dataset == "nci60":
        df = process_nci60(input_filepath, identifier)
    elif dataset == "cancerrx":
        df = process_cancerrx(input_filepath)

    # Set source_id as the dataset name
    source_id = dataset

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
    log_fmt = "%(asctime)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()