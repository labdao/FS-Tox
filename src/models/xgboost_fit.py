import logging
import os
import pickle
import statistics

import omegaconf
import click
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

from .utils import (construct_query, load_assays, load_representations,
                    mod_test_train_split)


def param_search(X_train, y_train, xgboost_params):
    # Define the parameters for the XGBoost model
    param_grid = {
        "max_depth": omegaconf.OmegaConf.to_container(xgboost_params.max_depth),
        "gamma": omegaconf.OmegaConf.to_container(xgboost_params.gamma),
        "eta": omegaconf.OmegaConf.to_container(xgboost_params.eta),
    }

    # Check outcome imbalance - toxic(1) vs non-toxic(0)
    control_case_num = y_train.value_counts()
    control_case_ratio = control_case_num[0] / control_case_num[1]

    # Create a XGBoost classifier with 10x weighting to positive cases
    xgb_model = xgb.XGBClassifier(
        eval_metric="logloss", scale_pos_weight=control_case_ratio
    )

    # Setup the random search with 4-fold cross validation
    random_search = RandomizedSearchCV(
        xgb_model, param_grid, cv=4, n_iter=20, random_state=42
    )

    random_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = random_search.best_params_

    return best_params


def train(
    representation_filepath,
    assay_filepath,
    output_filepath,
    representation,
    dataset,
    xgboost_params
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

    # Create empty list for results of hyperparameter search
    best_params_list = []

    # Evaluate each assay
    for i, (assay_df, assay_id) in enumerate(assay_dfs):
        # Merge the representations and assays
        merged_df = pd.merge(
            representation_df, assay_df, on="canonical_smiles", how="inner"
        )

        # Conduct test train split
        X_train, _, y_train, _ = mod_test_train_split(merged_df)

        if i < 5:
            logger.info("conducting hyperparameter search for assay %d...", i+1)

            # Conduct hyperparameter search
            best_params = param_search(X_train, y_train, xgboost_params)

            # Add best_params to best_params_list
            best_params_list.append(best_params)

        if i == 5:
            # Use modal best_params for remaining assays
            tmp_params = {}

            for key in best_params_list[0].keys():
                try:
                    tmp_params[key] = statistics.mode(
                        [d[key] for d in best_params_list]
                    )
                except statistics.StatisticsError:
                    logger.warning(
                        "Couldn't find a unique mode for key '%d'. You might want to handle this case differently.", key
                    )

            best_params = tmp_params

        logger.info("fitting model for assay %d...", i+1)

        # Train the XGBoost model with the best parameters
        model = xgb.XGBClassifier(**best_params, eval_metric="logloss")
        model.fit(X_train, y_train)

        # Create a filename for the model
        model_path = f"{output_filepath}/{assay_id}.pkl"

        # Save model to a pickle file
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        logger.info("successfully trained %d xgboost models.", i+1)
