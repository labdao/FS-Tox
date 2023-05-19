from rdkit.Chem import MolFromSmiles, AllChem
import pandas as pd
import argparse
import os

# Convert SMILES column to ECFP4 bitstring
def smiles_to_ecfp4(smiles_string):
    mol = MolFromSmiles(smiles_string)
    if mol is None:  # If RDKit couldn't parse the SMILES string
        return None
    else:
        ecfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        ecfp4 = list(map(int, ecfp4.ToBitString()))  # Convert the BitVector to a Python list of ints
        return ''.join(map(str, ecfp4))  # Convert the list of ints to a string
    
# Convert from pd.Series of ECFP4 bitstrings to pd.DataFrame of ints
def ecfp4_to_ints(ecfp4_series):

     # Convert each bit of bitstring to feature
    return ecfp4_series.apply(lambda x: pd.Series(list(x))).astype(int)


def main():
    # Initialize parser
    parser = argparse.ArgumentParser(description="Read a csv file and add a column with ECFP4")

    # Add arguments
    parser.add_argument("file_path", type=str, help="Path to the csv file")

    # Parse arguments
    args = parser.parse_args()
    
    # Read in data
    df = pd.read_csv(args.file_path)

    # Apply the function to df
    df['ECFP4'] = df['smiles'].apply(smiles_to_ecfp4)

    # Remove variables with NA or empty string for molecule
    df.dropna(subset=['ECFP4'], inplace=True)
    df.drop(df[df['ECFP4'] == ''].index, inplace=True)

    # Get the base name of the csv file path
    base_name = os.path.basename(args.file_path)  # Get the base name
    csv_name = os.path.splitext(base_name)[0]  # Split the extension and get the name only

    # Save the df to interim
    df.to_csv(f'data/interim/{csv_name}_ecfp4.csv', index=False)

    # Display the number of smiles successfully converted
    print(f"Converted {len(df)} smiles to ECFP4")

if __name__ == '__main__':
    main()