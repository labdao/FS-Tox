import logging
import os

import click
import duckdb
import pandas as pd
from rdkit.Chem import AllChem, MolFromSmiles


def smiles_to_ecfp4(smiles_string, nBits):
    """
    Converts a SMILES (Simplified Molecular Input Line Entry System) string into ECFP4 (Extended-Connectivity Fingerprints 4) representation.

    Args:
        smiles_string (str): A string representing a chemical compound in SMILES format.

    Returns:
        str or None: A string representing the chemical compound in ECFP4 format, or None if the SMILES string cannot be parsed.
    """
    mol = MolFromSmiles(smiles_string)
    if mol is None:  # If RDKit couldn't parse the SMILES string
        return None
    else:
        ecfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nBits)
        ecfp4 = list(
            map(int, ecfp4.ToBitString())
        )  # Convert the BitVector to a Python list of ints
        return "".join(map(str, ecfp4))  # Convert the list of ints to a string

def generate(input_filepath, output_filepath, bits):

    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    logger = logging.getLogger(__name__)
    logger.info("creating ECFP4 fingerprints...")

    # Connect to a database in memory
    connection = duckdb.connect(database=":memory:")
    df = connection.execute(
    f"""
    SELECT DISTINCT canonical_smiles
    FROM read_parquet('{input_filepath}/*')
    """
    ).df()
    

    # Apply the function to df
    df["ECFP4"] = df["canonical_smiles"].apply(lambda x: smiles_to_ecfp4(x, bits))

    # Count number of SMILES that could not be converted to ECFP4
    num_parse_err = df["ECFP4"].isna().sum()

    # Remove variables with NA or empty string for molecule
    df.dropna(subset=["ECFP4"], inplace=True)
    df.drop(df[df["ECFP4"] == ""].index, inplace=True)

    # Convert each ECFP4 so that each bit is a column
    df_ecfp4_bits = pd.DataFrame(df['ECFP4'].apply(list).tolist()).astype(int)
    df_ecfp4_bits.columns = [f'bit_{i}' for i in range(df_ecfp4_bits.shape[1])]

    # Concatenate original df with df_ecfp4_bits
    df = pd.concat([df.reset_index(drop=True), df_ecfp4_bits], axis=1)

    # Drop the ECFP4 column
    df.drop(columns=['ECFP4'], inplace=True)

    # Add a column with the representation name
    df['representation'] = f'ecfp4_{bits}'

    # Get number of successful conversions
    num_success = len(df)

    # Save the ECFP4 fingerprints to a parquet file
    df.to_parquet(f"{output_filepath}/ecfp4_{bits}.parquet")

    logger.info("%d SMILES successfully converted to ECFP4.", num_success)
    logger.info("%d SMILES could not be converted to ECFP4",num_parse_err) if num_parse_err > 0 else None