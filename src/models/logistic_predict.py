import logging
import os
import click
import duckdb
import pandas as pd

@click.command()
@click.argument("representation_filepath", type=click.Path(exists=True))
@click.argument("assay_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.option("-r", "--representation", multiple=True)
@click.option("-d", "--dataset", multiple=True)
def main(representation_filepath, assay_filepath, output_filepath, representation, dataset):
    logger = logging.getLogger(__name__)
    logger.info("loading data...")
    


    


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()