import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import duckdb
import click
import logging
import neptune

run = neptune.init_run()


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True), default="data/processed/scores")
@click.argument("output_filepath", type=click.Path(), default="data/processed/visualizations")
@click.option("-f", "--feature", default="ecfp4_1024")
@click.option("-d", "--dataset", default="tox21")
@click.option("-m", "--metric", default="auc_roc")
def main(input_filepath, output_filepath, feature, dataset, metric):
    con = duckdb.connect()

    logging.info(f"Loading evaluation metrics from {input_filepath}...")

    # Create filepath pattern for each feature
    score_filepaths = [f"{input_filepath}/*{dataset}*{feature}*.parquet"]

    # Convert list to string so it can be incorporated into the SQL query
    score_filepaths_as_str = str(score_filepaths)

    # Load scoring data from parquet files
    pred_df = con.execute(
        f"""
            SELECT feature, {metric}
            FROM read_parquet({score_filepaths_as_str})
            """
    ).fetchdf()

    pred_df.to_csv(f"{output_filepath}/{dataset}_{feature}_{metric}_distribution.csv")
    
    logging.info("Creating distribution...")

    # Create distribution plot
    sns.displot(pred_df[metric], kde=False, rug=True)
    plt.title(f"Distribution of {metric} scores for assays in {dataset}")
    plt.xlabel(f"{metric} score")
    plt.ylabel("Count")
    plt.savefig(f"{output_filepath}/{dataset}_{feature}_{metric}_distribution.png")
    run["chart"].upload(f"{output_filepath}/{dataset}_{feature}_{metric}_distribution.png")

    run.stop()

if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()