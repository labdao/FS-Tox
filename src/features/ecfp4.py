import pandas as pd
import click
import os
import logging
from rdkit.Chem import MolFromSmiles, AllChem
import duckdb


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


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.option(
    "-b",
    "--bits",
    default=1024,
    type=int,
    help="Length of the ECFP4 fingerprint bitstring",
)
def main(input_filepath, output_filepath, bits):
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # Connect to a database in memory
    connection = duckdb.connect(database=":memory:")
    df = connection.execute(
    f"""
    SELECT DISTINCT canonical_smiles
    FROM '{input_filepath}*.parquet'
    """
    ).df()
    

    # Apply the function to df
    df["ECFP4"] = df["canonical_smiles"].apply(lambda x: smiles_to_ecfp4(x, bits))

    # Count number of SMILES that could not be converted to ECFP4
    num_parse_err = df["ECFP4"].isna().sum()

    # Remove variables with NA or empty string for molecule
    df.dropna(subset=["ECFP4"], inplace=True)
    df.drop(df[df["ECFP4"] == ""].index, inplace=True)

    # Get number of successful conversions
    num_success = len(df)

    # Save the ECFP4 fingerprints to a parquet file
    df.to_parquet(f"{output_filepath}/ecfp4_{bits}.parquet")

    logger.info("%d SMILES successfully converted.", num_success)
    logger.info("%d SMILES could not be converted to ECFP4",num_parse_err) if num_parse_err > 0 else None


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
    