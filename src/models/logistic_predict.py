import logging
import os
import click
import duckdb
import pandas as pd
from sklearn.linear_model import LogisticRegression

from utils import (
    construct_query,
    mod_test_train_split,
    load_representations,
    load_assays,
)


def model_fit_predict(X_train, X_test, y_train, y_test):
    # Create a Logistic Regression object
    log_reg = LogisticRegression()

    # Fit the model to the training data
    log_reg.fit(X_train, y_train)

    # Generate predictions on the test set
    preds = log_reg.predict(X_test)

    # Generate prediction probabilities
    preds_proba = log_reg.predict_proba(X_test)[:, 1]

    return preds_proba, preds


@click.command()
@click.argument("representation_filepath", type=click.Path(exists=True))
@click.argument("assay_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.option("-r", "--representation", multiple=True)
@click.option("-d", "--dataset", multiple=True)
@click.option("-a", "--assay", multiple=True)
def main(
    representation_filepath,
    assay_filepath,
    output_filepath,
    representation,
    dataset,
    assay,
):
    logger = logging.getLogger(__name__)
    logger.info("loading data...")

    # Create a SQL query as a string to select relevant representations
    representation_query = construct_query(representation_filepath, representation)

    # Load representations from parquet files
    representation_df = load_representations(representation_query)

    # Load the assays
    assay_dfs = load_assays(assay_filepath, dataset, assay)

    # Evaluate each assay
    for i, (assay_df, assay_filename) in enumerate(assay_dfs):
        # Merge the representations and assays
        merged_df = pd.merge(
            representation_df, assay_df, on="canonical_smiles", how="inner"
        )

        # Conduct test train split
        X_train, X_test, y_train, y_test = mod_test_train_split(merged_df)

        # Fit the model and generate predictions
        preds_proba, preds = model_fit_predict(X_train, X_test, y_train, y_test)

        logger.info(f"creating predictions for assay {i+1}...")

        # Get canonical smiles for index of output prediction parquet file
        test_canonical_smiles = merged_df.loc[
            merged_df["test_train"] == 1, "canonical_smiles"
        ]

        # Add predictions to dataframe
        preds_df = pd.DataFrame(
            {
                "canonical_smiles": test_canonical_smiles,
                "preds": preds,
                "preds_proba": preds_proba,
                "ground_truth": y_test,
                "model": "logistic_regression",
            }
        )

        # Convert the representations tuple into a string with elements separated by '_'
        representation_str = "_".join(representation)

        # Save the predictions to a parquet file
        preds_df.to_parquet(
            f"{output_filepath}/{assay_filename}_logistic_regression_{representation_str}.parquet"
        )

    logger.info(f"predictions saved to {output_filepath}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
