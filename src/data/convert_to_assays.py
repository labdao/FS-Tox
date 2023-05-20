import logging
import pyarrow.parquet as pq
import click
import pandas as pd


def convert_to_assay(input_filepath):
    """Converts an unprocessed parquet file to a processed parquet
    file for each assay.

    Args:
        input_filepath (str): Path to the unprocessed parquet file.

    Returns:
        A parquet file with the following attribute names:

            smiles (str)
            canonical_smiles
            selfie
            source_id
            assay_id
            publication
            ground_truth
            train_test
    """

    # Read in the unprocessed parquet file
    df = pq.read_table(input_filepath).to_pandas()

    # Convert to parquet format
    df.to_parquet(output_filepath)


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
def main(input_filepath):
    convert_to_assay(input_filepath)


if __name__ == "__main__":
    main()
