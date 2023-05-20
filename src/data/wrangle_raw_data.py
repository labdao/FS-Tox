import click
import logging
import pandas as pd


def process_tox21(input_filepath, output_filepath):
    """Process the tox21 dataset."""
    df = pd.read_csv(input_filepath)

    # Drop the mol_id column
    df.drop(columns="mol_id", inplace=True)

    # Convert to parquet format
    df.to_parquet(output_filepath)


def process_clintox(input_filepath, output_filepath):
    """Process the clintox dataset."""
    df = pd.read_csv(input_filepath)

    # Convert to parquet format
    df.to_parquet(output_filepath)


def process_toxcast(input_filepath, output_filepath):
    """Process the toxcast dataset."""
    df = pd.read_csv(input_filepath)

    # Convert to parquet format
    df.to_parquet(output_filepath)


@click.command()
@click.argument("input-filepath", type=click.Path(exists=True))
@click.argument("output-filepath", type=click.Path())
@click.option(
    "-i", "--input-dataset", type=click.Choice(["tox21", "clintox", "toxcast"])
)
def main(input_filepath, output_filepath, input_dataset):
    """Runs data processing script to convert raw csv file (../raw) into
    parquet file (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("convert raw data to parquet format")

    if input_dataset == "tox21":
        process_tox21(input_filepath, output_filepath)
    elif input_dataset == "clintox":
        process_clintox(input_filepath, output_filepath)
    elif input_dataset == "toxcast":
        process_toxcast(input_filepath, output_filepath)

    logger.info("finished converting raw data to parquet format")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
