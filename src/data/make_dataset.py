# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import os
import pandas as pd


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

        # Create a new column with the CSV name and set all values to 1
        df[csv_name] = 1
    
    logger.info('merging dataframes')

    # Loop through DataFrames and merge them
    merged = dfs[0]
    for i in range(1, len(dfs)):

        # Merge the two DataFrames with an outer join
        merged = merged.merge(dfs[i], on="smiles", how='outer')

    # Remove non-assay columns
    merged = merged.drop("smiles", axis=1)

    # Save pandas DataFrame if output format is CSV
    if output_format == 'csv':

        # Join the output directory and file name)
        output_directory = os.path.join(output_directory, "combined-datasets.csv")
        
        # Save DataFrame to CSV
        merged.to_csv(output_directory, index=False)


    logger.info('saved result to file: %s', output_directory)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()