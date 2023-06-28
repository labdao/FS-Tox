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

    if len(pred_df) > 600:
        # Randomly sample 600 rows from the DataFrame
        pred_df = pred_df.sample(n=600, random_state=42)

    logging.info("Creating swarmplot...")
    
    # Create swarmplot
    sns.swarmplot(x=pred_df["feature"], y=pred_df["auc_roc"], hue=pred_df["feature"], palette='Set2')
    plt.title(f"Swarmplot of AUC-ROC scores for assays in {dataset}")

    plt.show()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()