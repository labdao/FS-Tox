from rdkit import RDLogger
from rdkit import Chem

import pandas as pd
import numpy as np
import selfies as sf
import os
from rdkit import RDLogger

# Suppress RDKit warnings
RDLogger.DisableLog("rdApp.*")

def smiles_to_canonical_smiles(df):
    # Convert smiles to canonical smiles
    mol_objects = df["smiles"].apply(Chem.MolFromSmiles)
    df["canonical_smiles"] = mol_objects.apply(
        lambda x: Chem.MolToSmiles(x) if x is not None else None
    )

    return df


def binarize_df(df):
    # Calculate the median of each column, while ignoring NaNs
    medians = df.median()

    # Subtract the medians from the DataFrame (broadcasted along columns)
    diff = df.subtract(medians)

    # Values less than or equal to 0 become 0, and greater than 0 become 1. NaNs remain NaNs.
    binarized_df = np.where(diff >= 0, 1, np.where(diff < 0, 0, np.nan))

    # Convert back to DataFrame
    binarized_df = pd.DataFrame(binarized_df, columns=df.columns, index=df.index)

    # Count 1s and 0s in each column
    count_1s = (binarized_df == 1).sum()
    count_0s = (binarized_df == 0).sum()

    # Compute the proportion
    proportion = count_1s / count_0s

    # Identify columns to drop
    columns_to_drop = proportion[(proportion > 2.333) | (proportion < 0.43)].index

    # Drop the columns
    binarized_df.drop(columns_to_drop, axis=1, inplace=True)

    return binarized_df


def selfies_encoder(smiles):
    """
    Encodes a SMILES string using the selfies library, handling any exceptions that may occur.
    Parameters:
    smiles (str): The SMILES string to encode.
 
    Returns:
    str or None: The encoded SELFIES string, or None if an exception occurred during encoding."""
    try:
        return sf.encoder(smiles)
    except Exception as e:
        return None

# Define a function to convert InChI to SMILES
def inchi_to_smiles(inchi):

    # Suppress RDKit warnings
    RDLogger.DisableLog("rdApp.*")
    
    mol = Chem.MolFromInchi(inchi)
    if mol is None:
        return 'InvalidInChI'  # Placeholder for invalid InChI
    smiles = Chem.MolToSmiles(mol)
    return smiles


def pivot_assays(df, assay_components, outcome_col_name):
    # Create a new column that is a combination of the assay_components
    df["combined"] = df[assay_components].apply(
        lambda row: "_".join(row.values.astype(str)), axis=1

    )
    # Create a pivot table where each unique combination forms a separate column
    df = df.pivot_table(
        index="canonical_smiles",
        columns="combined",
        values=outcome_col_name,
        aggfunc=np.mean,
    )

     # Apply the function to each column
    df = binarize_df(df)

    # Convert smiles index to column
    df = df.reset_index()

    # Remove all columns where the value is identical for all non-null rows
    df = df.loc[:, df.nunique() != 1]

    # Create a lookup table for the assay names
    assay_names = pd.DataFrame(df.columns[1:], columns=["combined"])

    # Split the 'combined' column at '_', expanding into new columns
    split_df = assay_names["combined"].str.split("_", expand=True)

    # Name the new columns
    split_df.columns = assay_components

    # Save the assay names as a lookup table
    split_df.to_csv(os.path.join("./data/processed/assay_lookup/assay_lookup.csv"), index=False)

    # Simplify the column names
    df.columns = [
        f"assay_{i}" if col != "canonical_smiles" else col
        for i, col in enumerate(df.columns)
    ]

    return df


def assign_test_train(df_len):
    """
    Creates a pandas Series with random assignment of each row to test or train.

    Parameters:
    df_len (int): The length of the DataFrame to create the test-train split for.

    Returns:
    pd.Series: A pandas Series with random assignment of each row to test (0) or train (1).
    """

    # Set seed for reproducibility
    np.random.seed(42)

    # Set the proportions for 0s and 1s
    proportions = [0.8, 0.2]

    # Create a random series with the desired proportions
    test_train = pd.Series(np.random.choice([0, 1], size=df_len, p=proportions))

    return test_train