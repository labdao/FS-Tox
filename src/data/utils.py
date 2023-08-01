import hashlib
import os
import time

import numpy as np
import pandas as pd
import requests
from rdkit import Chem, RDLogger
from joblib import Memory

# Suppress RDKit warnings
RDLogger.DisableLog("rdApp.*")

 # Setup joblib caching configuration
cache_dir = os.path.join(os.getcwd(), ".assay_cache")
memory = Memory(cache_dir, verbose=0)

@memory.cache
def drug_name_to_smiles(df, drug_name_colname):

    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    
    smiles_dict = {}

    # Get the drug names from drug name col
    drug_names = df[drug_name_colname].unique()  

    for drug_name in drug_names:
        url = f"{base_url}/compound/name/{drug_name}/property/IsomericSMILES/JSON"

        # Make a request to the PubChem API
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        # If an HTTP error occurs (e.g., 404, 500):
        except requests.HTTPError as http_err:
            continue
        # If a connection error occurs (e.g., DNS failure, refused connection, etc):
        except Exception as err:
            continue
        
        smiles = response.json()['PropertyTable']['Properties'][0]['IsomericSMILES']
        if smiles is not None:
            smiles_dict[drug_name] = smiles
        
        # Wait for 0.2 before sending another request
        time.sleep(0.2)

    # Add the smiles values as a new column
    df['smiles'] = df[drug_name_colname].map(smiles_dict)

    # Remove rows where 'SMILES' is NaN
    df = df.dropna(subset=['smiles'])

    return df
    

def filter_by_range(df: pd.DataFrame):
    # Get absolute value of the difference between the max and min of each column
    abs_diff = (df.max() - df.min()).abs()

    # Keep assays where the order of magnitude range is greater than 2
    df = df.loc[:, (abs_diff > 2).values]

    return df


def filter_by_active_ratio(df: pd.DataFrame):
    # Temporarily remove the smiles column
    smiles = df.pop("smiles")

    # Remove columns with no inactive compounds
    df = df.loc[:, (df != 1).any(axis=0)]

    # Sum the number of active compounds in a column
    active_compounds = df.sum()

    # Sum the number of inactive compounds in a column
    inactive_compounds = (df == 0).sum()

    # Calculate the ratio of active to inactive compounds
    active_ratio = active_compounds / inactive_compounds

    # Remove columns with active ratio greater than 2.333 or less than 0.43
    active_ratio_mask = (active_ratio < 2.333) & (active_ratio > 0.43)
    df = df.loc[:, active_ratio_mask]

    # Add the smiles column back
    df["smiles"] = smiles

    # If there are no columns in the active ratio range, raise an exception
    if len(df.columns) == 1 and "smiles" in df.columns:
        raise ValueError("No assays have active proportions between 0.3 and 0.7.")

    return df


def smiles_to_canonical_smiles(df):
    # Convert smiles to canonical smiles
    mol_objects = df["smiles"].apply(Chem.MolFromSmiles)
    df["canonical_smiles"] = mol_objects.apply(
        lambda x: Chem.MolToSmiles(x) if x is not None else None
    )

    # Drop the smiles column
    df.drop("smiles", axis=1, inplace=True)

    # Remove rows with null canonical smiles
    df.dropna(subset=["canonical_smiles"], inplace=True)

    return df

# Define a function to convert InChI to SMILES
def inchi_to_smiles(df: pd.DataFrame) -> pd.DataFrame:
    # Suppress RDKit warnings
    RDLogger.DisableLog("rdApp.*")

    # Convert InChI to SMILES
    mol_objects = df["inchi"].astype(str).apply(Chem.MolFromInchi)
    df["smiles"] = mol_objects.apply(
        lambda x: Chem.MolToSmiles(x) if x is not None else None
    )

    # Drop the inchi column
    df.drop("inchi", axis=1, inplace=True)

    # Remove rows with null canonical smiles
    df.dropna(subset=["smiles"], inplace=True)

    return df




def pivot_assays(df, assay_components, outcome_col_name):
    # Create a new column that is a combination of the assay_components
    df["combined"] = df[assay_components].apply(
        lambda row: "_".join(row.values.astype(str)), axis=1
    )
    # Create a pivot table where each unique combination forms a separate column
    df = df.pivot_table(
        index="smiles",
        columns="combined",
        values=outcome_col_name,
        aggfunc=np.mean,
    )

    return df


def binarize_assays(df):
    # Get column names of the assays
    assay_names = df.columns

    # Get the smiles from the index
    smiles = df.index

    # Calculate the median of each column
    medians = df.median()

    # Subtract the medians from the DataFrame (broadcasted along columns)
    diff = df.subtract(medians)

    # Values less than or equal to 0 become 0, and greater than 0 become 1. NaNs remain NaNs.
    df = np.where(diff >= 0, 1, np.where(diff < 0, 0, np.nan))

    # Convert back to DataFrame from numpy array
    df = pd.DataFrame(df, columns=assay_names, index=smiles)

    return df


def get_sha256_snippet(input_string):
    return hashlib.sha256(input_string.encode()).hexdigest()[:15]


def assign_test_train(df_len, size_train):
    """
    Creates a pandas Series with random assignment of each row to test or train.

    Parameters:
    df_len (int): The length of the DataFrame to create the test-train split for.
    size_train (int): The size of the train set.

    Returns:
    pd.Series: A pandas Series with random assignment of each row to test (0) or train (1).
    """

    # Set seed for reproducibility
    np.random.seed(42)

    # Create a list with size_train 0s and (df_len - size_train) 1s
    assignment_list = [0]*size_train + [1]*(df_len - size_train)
    
    # Shuffle the list
    np.random.shuffle(assignment_list)
    
    # Convert the list to a pandas Series
    test_train = pd.Series(assignment_list)

    return test_train