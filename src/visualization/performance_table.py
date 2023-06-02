import duckdb
import logging
import click
import os
import pandas as pd


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
def main(input_filepath):
    # Get list of score files
    files = os.listdir(input_filepath)
    score_files = [f for f in files if f.startswith("score_")]

    associated_datasets = []

    # Get dataset name associated with each score file
    for score_file in score_files:
        parts = score_file.split("_")
        dataset = "_".join(parts[-3:-2])
        associated_datasets.append(dataset)

    evaluation = pd.DataFrame()
    # Loop through each file and load the results into a dataframe
    for score_file in score_files:
        df = pd.read_parquet(f"{input_filepath}/{score_file}")
        evaluation = pd.concat([evaluation, df])

    # Add datasets as a new column
    evaluation["dataset"] = associated_datasets

    # Group by dataset and feature and calculate the mean AUC-ROC
    grouped_df = evaluation.groupby(["dataset", "feature"])["auc_roc"].mean()

    # Reshape the DataFrame
    reshaped_df = grouped_df.reset_index().pivot_table(
        index="feature", columns="dataset", values="auc_roc"
    )

    # Save reshaped DataFrame to a parquet file
    reshaped_df.to_parquet(f"{input_filepath}/evaluation_by_dataset.parquet")

if __name__ == "__main__":
    main()
