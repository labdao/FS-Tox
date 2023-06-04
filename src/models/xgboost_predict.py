import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import statistics

import os
import logging
import click

from utils import load_assays
from utils import load_representations, construct_query, mod_test_train_split


def param_search(X_train, y_train):
    # Define the parameters for the XGBoost model
    param_grid = {
        "max_depth": [3, 4, 5, 6, 7, 8, 9],
        "gamma": [0.01, 0.1, 0.2, 0.5, 1, 2, 5],
        "eta": [0.1, 0.2, 0.3, 0.4, 0.5],
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


def model_fit_predict(X_train, X_test, y_train, best_params):
    # Train the XGBoost model with the best parameters
    num_round = 20
    model = xgb.XGBClassifier(**best_params, eval_metric="logloss")
    model.fit(X_train, y_train)

    # Train the XGBoost model with the best parameters
    num_round = 20
    model = xgb.XGBClassifier(**best_params, eval_metric="logloss")
    model.fit(X_train, y_train)

    # Make predictions on the test set
    preds_proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    return preds_proba, preds


@click.command()
@click.argument("representation_filepath", type=click.Path(exists=True))
@click.argument("assay_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.option("-r", "--representation", multiple=True)
@click.option("-d", "--dataset", multiple=True)
def main(representation_filepath, assay_filepath, output_filepath, representation, dataset):
    logger = logging.getLogger(__name__)
    logger.info("loading data...")
    
    # Create a SQL query as a string to select relevant representations 
    representation_query = construct_query(representation_filepath, representation)

    # Load representations from parquet files
    representation_df = load_representations(representation_query)

    # Load the assays
    assay_dfs = load_assays(assay_filepath, dataset)

    # Create empty list for results of hyperparameter search
    best_params_list = []

    
    # Evaluate each assay
    for i, (assay_df, assay_filename) in enumerate(assay_dfs):
        # Merge the representations and assays
        merged_df = pd.merge(representation_df, assay_df, on="canonical_smiles", how="inner")

        # Conduct test train split
        X_train, X_test, y_train, y_test = mod_test_train_split(
            merged_df
        )

        if i < 5:
            logger.info(f"conducting hyperparameter search for assay {i+1}...")

            # Conduct hyperparameter search
            best_params = param_search(X_train, y_train)

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
                    logger.warn(
                        f"Couldn't find a unique mode for key '{key}'. You might want to handle this case differently."
                    )

            best_params = tmp_params

        logger.info(f"fitting model for assay {i+1}...")

        # Fit model and predict using best hyperparameters
        preds_proba, preds = model_fit_predict(X_train, X_test, y_train, best_params)

        logger.info(f"creating predictions for assay {i+1}...")
        
        # Get canonical smiles for index of output prediction parquet file
        test_canonical_smiles = merged_df.loc[merged_df["test_train"] == 1, "canonical_smiles"]
        
        # Add predictions to dataframe
        preds_df = pd.DataFrame(
            {"canonical_smiles": test_canonical_smiles, "preds": preds, "preds_proba": preds_proba, "ground_truth": y_test, 'model': 'xgboost'}
        )
        
        # Convert the representations tuple into a string with elements separated by '_'
        representation_str = '_'.join(representation)

        # Save the predictions to a parquet file
        preds_df.to_parquet(
            f"{output_filepath}/{assay_filename}_xgboost_{representation_str}.parquet"
        )

    logger.info(f"predictions saved to {output_filepath}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
