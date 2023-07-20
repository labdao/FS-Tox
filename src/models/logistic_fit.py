import logging
import os
import pickle

import click
import duckdb
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .utils import (construct_query, load_assays, load_representations,
                    mod_test_train_split)


def train(
    representation_filepath, assay_filepath, output_filepath, representation, dataset
):
    log_fmt = "%(asctime)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    logger = logging.getLogger(__name__)
    logger.info("loading features...")

    # Create a SQL query as a string to select relevant representations
    representation_query = construct_query(representation_filepath, representation)

    # Load representations from parquet files
    representation_df = load_representations(representation_query)

    # Load the assays
    assay_dfs = load_assays(assay_filepath, dataset)

    logger.info("fitting models to assay data...")

    # Evaluate each assay
    for i, (assay_df, assay_id) in enumerate(assay_dfs):

        # Merge the representations and assays
        merged_df = pd.merge(
            representation_df, assay_df, on="canonical_smiles", how="inner"
        )

        # Conduct test train split
        X_train, _, y_train, _ = mod_test_train_split(merged_df)

        # Check if y_train has more than one unique class
        if len(pd.unique(y_train)) < 2:
            logger.info("Skipping model training for %s due to lack of classes.", assay_id)
            continue


        # Create a Logistic Regression object
        log_reg = LogisticRegression(max_iter=1000)

        
        # Fit the model to the training data
        log_reg.fit(X_train, y_train)

        # Create a filename for the model
        model_path = f"{output_filepath}/{assay_id}.pkl"

        # Save model to a pickle file
        with open(model_path, "wb") as f:
            pickle.dump(log_reg, f)

    logger.info(f"successfully trained {i+1} logistic models.")
