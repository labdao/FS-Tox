import logging
import pickle

import duckdb
import pandas as pd

from .utils import (construct_query, load_assays, load_representations,
                    mod_test_train_split)


def generate_predictions(
    model_filepath,
    assay_filepath,
    feature_filepath,
    prediction_filepath,
    representation,
    model,
    dataset,
    support_set_size,
):
    logger = logging.getLogger(__name__)

    # Create a SQL query as a string to select relevant representations
    representation_query = construct_query(feature_filepath, representation)

    # Load representations from parquet files
    representation_df = load_representations(representation_query)

    # Load the assays
    assay_dfs = load_assays(assay_filepath, dataset)

    logger.info("creating predictions for %s...", dataset)
    # Evaluate each assay
    for i, (assay_df, assay_filename) in enumerate(assay_dfs):

        # Merge the representations and assays
        merged_df = pd.merge(
            representation_df, assay_df, on="canonical_smiles", how="inner"
        )

        # Conduct test train split
        _, X_test, _, y_test = mod_test_train_split(merged_df)

        # Get model filepath
        trained_model_filepath = f"{model_filepath}/{assay_filename}_{model}_{representation}_support_{support_set_size}.pkl"

        # Load the model
        with open(trained_model_filepath, "rb") as f:
            trained_model = pickle.load(f)

        # Generate predictions on the test set
        preds = trained_model.predict(X_test)

        # Generate prediction probabilities
        preds_proba = trained_model.predict_proba(X_test)[:, 1]

        # Get canonical smiles for index of output prediction parquet file
        test_canonical_smiles = merged_df.loc[
            merged_df["support_query"] == 1, "canonical_smiles"
        ]

        # Add predictions to dataframe
        preds_df = pd.DataFrame(
            {
                "canonical_smiles": test_canonical_smiles,
                "preds": preds,
                "preds_proba": preds_proba,
                "ground_truth": y_test,
                "model": model,
            }
        )

        # Save the predictions to a parquet file
        preds_df.to_parquet(
            f"{prediction_filepath}/{assay_filename}_{model}_{representation}_support_{support_set_size}.parquet"
        )
    logger.info(f"predictions created for {i+1} models.")