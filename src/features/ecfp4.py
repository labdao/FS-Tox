import pandas as pd
import click
import os
import logging
from rdkit.Chem import MolFromSmiles, AllChem

def smiles_to_ecfp4(smiles_string, nBits=nBits):
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
        ecfp4 = list(map(int, ecfp4.ToBitString()))  # Convert the BitVector to a Python list of ints
        return ''.join(map(str, ecfp4))  # Convert the list of ints to a string

@click.command()
@click.argument("file_path", type=str, help="Path to the directory of the parquet file")
@click.option("-b", "--bits", default=1024, type=int, help="Length of the ECFP4 fingerprint bitstring")
def main(file_path, bits):
    """
    Read a csv file and add a column with ECFP4
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Read in data
    df = pd.read_csv(file_path)

    # Apply the function to df
    df['ECFP4'] = df['smiles'].apply(lambda x: smiles_to_ecfp4(x, bits))

    # Count number of SMILES that could not be converted to ECFP4
    num_parse_err = df['ECFP4'].isna().sum()
    
    # Remove variables with NA or empty string for molecule
    df.dropna(subset=['ECFP4'], inplace=True)
    df.drop(df[df['ECFP4'] == ''].index, inplace=True)

    # Get number of successful conversions
    num_success = len(df)

    logginer.info("""%d SMILES successfully converted.
    %d SMILES could not be converted to ECFP4""",
    num_success,
    num_parse_err)

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()