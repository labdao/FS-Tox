import click
import logging
from pathlib import Path
import pandas as pd

import sys
import os

# Get the absolute path of the src directory
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the src directory to the system path
sys.path.append(src_dir)

from features.ecfp4_featurisation import ecfp4_to_ints, smiles_to_ecfp4

def make_parquets(df):
    """ Takes in a pandas DataFrame, and outputs three parquet files:
        - A parquet file for molecule metadata
        - A parquet file for molecule representations
        - A parquet file for assay data
    """

    # Add a column with ECFP4 fingerprints
    df['ecfp4'] = df['smiles'].apply(smiles_to_ecfp4)
    # Drop rows with NA or empty string for molecule
    df.dropna(subset=['ecfp4'], inplace=True)

    # Convert Series to DataFrame for .to_parquet method
    molecule_metadata = df["smiles"].to_frame()

    # Convert ECFP4 bitstrings to ints
    molecule_representation = ecfp4_to_ints(df['ecfp4'])
    
    # Rename columns to strings for .to_parquet method
    molecule_representation.columns = molecule_representation.columns.astype(str)

    # Drop the smiles and ecfp4 columns to get assay data only
    assays = df.drop(["smiles", "ecfp4"], axis=1)

    # Save the parquet files
    molecule_metadata.to_parquet("data/processed/molecule_metadata.parquet", engine="pyarrow")
    molecule_representation.to_parquet("data/processed/molecule_representation.parquet", engine="pyarrow")
    assays.to_parquet("data/processed/assays.parquet", engine="pyarrow")

@click.command()
@click.argument('input_directory', type=click.Path(exists=True))
@click.argument('output_directory', type=click.Path())
@click.argument('output_format', type=click.Choice(['csv', 'parquet']))
def main(input_directory, output_directory, output_format):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('reading CSV files from directory: %s', input_directory)

    # Get a list of all CSV files in the input directory
    csv_files = [os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.endswith('.csv')]

   # Read each CSV file into a Pandas DataFrame then add it to a list
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)

    # Create counter
    i = 0

    # Create a flag column for each dataset and set it to 1
    for i, df in enumerate(dfs):

        # Get the name of the CSV file
        csv_name = csv_files[i].split('/')[-1].split('.')[0]

        # Create a new column with the modified name and set all values to 1
        col_name = csv_name + '_moleculenet_2023'
        df[col_name] = 1
    
    logger.info('merging dataframes')

    # Loop through DataFrames and merge them
    merged = dfs[0]
    for i in range(1, len(dfs)):

        # Merge the two DataFrames with an outer join
        merged = merged.merge(dfs[i], on="smiles", how='outer')

    # Remove non-assay columns
    merged = merged.drop("mol_id", axis=1)

    # Save pandas DataFrame if output format is CSV
    if output_format == 'csv':

        # Join the output directory and file name)
        output_directory = os.path.join(output_directory, "combined-datasets.csv")
        
        # Save DataFrame to CSV
        merged.to_csv(output_directory, index=False)

    logger.info('saved result to file: %s', output_directory)

    # Save as parquet files if output format is parquet
    if output_format == 'parquet':
        make_parquets(merged)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()