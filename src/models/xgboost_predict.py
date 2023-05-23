import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

import logging
import click
import duckdb

# Lookup table for feature parquet file names
feature_dict = {
    "chemberta": "chemberta_embeddings",
    "chemgpt": "chemgpt_embeddings",
    "ecfp4": "ecfp4_1024",
}

# ASSUMING 1 FEATURE
def load_data(input_filepath, features, dataset):
    # Connect to a database in memory
    connection = duckdb.connect(database=":memory:")

    # Set table 1 filepath
    table_1 = f"{input_filepath}/*{dataset}.parquet"

    # Set table 2 filepath
    table_2 = f"{input_filepath}/{feature_dict[features]}.parquet"

    df = connection.execute(
        f"""
    SELECT DISTINCT canonical_smiles, embeddings AS {feature_dict[features]}, ground_truth
    FROM '{table_1}' AS table_1 INNER JOIN '{table_2}' AS table_2 ON (table_1.canonical_smiles = table_2.smiles)
    """
    ).df()

    return df


# ASSUMING ONE FEATURE
def param_search(X_train, y_train, features):
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
        xgb_model, param_grid, cv=4, n_iter=50, random_state=42
    )

    random_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = random_search.best_params_

    return best_params


def model_fit_predict(X_train, X_test, y_train, best_params):

    # Train the XGBoost model with the best parameters
    num_round = 20
    model = xgb.XGBClassifier(
        **best_params, eval_metric="logloss"
    )
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
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.option("--features", default="ECFP4")
@click.option("--dataset")
def main(input_filepath, output_filepath, features, dataset):
    logger = logging.getLogger(__name__)
    logger.info("loading data...")

    # Load the data from parquet files
    df = load_data(input_filepath, features, dataset)

    # Get the features
    X = df[feature_dict[features]]

    # Get the label
    y = df["ground_truth"]

    # Conduct test-train split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Conduct hyperparameter search
    best_params = param_search(X_train, y_train, features)

    # Predict using best parameters
    preds_proba, preds = model_fit_predict(X_train, X_test, y_train, best_params)

    print(preds)

if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
