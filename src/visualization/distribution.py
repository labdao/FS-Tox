import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import duckdb
import click
import logging


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True), default="data/processed/scores")
@click.option("-f", "--feature", default="ecfp4_1024")
@click.option("-d", "--dataset")
def main(input_filepath, feature, dataset):
    con = duckdb.connect()

    logging.info(f"Loading evaluation metrics from {input_filepath}...")

    # Create filepath pattern for each feature
    score_filepaths = [f"{input_filepath}/*{dataset}*{feature}.parquet"]

    # Convert list to string so it can be incorporated into the SQL query
    score_filepaths_as_str = str(score_filepaths)

    # Load scoring data from parquet files
    pred_df = con.execute(
        f"""
            SELECT feature, auc_roc
            FROM read_parquet({score_filepaths_as_str})
            """
    ).fetchdf()
    
    logging.info("Creating distribution...")

    # Create distribution plot
    sns.distplot(pred_df["auc_roc"], kde=False, rug=True)
    plt.title(f"Distribution of AUC-ROC scores for assays in {dataset}")
    plt.xlabel("AUC-ROC score")
    plt.ylabel("Count")
    plt.show()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()