import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import click
import duckdb

@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.option("-f", "--features", multiple=True)
def main(input_filepath, features):
    con = duckdb.connect()

    # Create a glob pattern to match the features
    features_pattern = "|".join(features)

    # Load scoring data from parquet files
    pred_df = con.execute(
        f"""
            SELECT feature, auc_roc
            FROM read_parquet('{input_filepath}/score_*{{{features_pattern}}}.parquet'), filename=true
            """
    ).fetchdf()

    sns.violinplot(x=pred_df["feature"], y=pred_df["auc_roc"])


if __name__ == "__main__":
    main()