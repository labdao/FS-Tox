import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

import os
import logging
import click
import duckdb

# Lookup table for feature parquet file names
feature_dict = {
    "chemberta": "chemberta_embeddings",
    "chemgpt": "chemgpt_embeddings",
    "ecfp4": "ecfp4_1024",
}


def load_features(input_filepath, features):

   # Check if features is a list
    if isinstance(features, tuple):
        # Get a list of filepaths for features
        feature_filepaths = [os.path.join(input_filepath, f"feature_{feature_dict[feature]}.parquet") for feature in features]
    else:
        # Get a single filepath for the feature
        feature_filepaths = os.path.join(input_filepath, f"feature_{feature_dict[features]}.parquet")

        # Convert to list
        feature_filepaths = [feature_filepaths]

    # Create a database connection
    con = duckdb.connect()

    # Create empty list to store each feature df
    dfs = []

    # Create a SQL query as a string that joins feature tables using inner joins
    for feature_filepath in feature_filepaths:
        df = con.execute(f"SELECT * FROM '{feature_filepath}'").df()

        dfs.append(df)

    # Start with the first dataframe in the list
    joined_dfs = dfs[0].copy()

    # Ensure 'smiles' column is of type string
    joined_dfs['smiles'] = joined_dfs['smiles'].astype(str)

    # Inner join with the rest of the dataframes
    for i, df in enumerate(dfs[1:], start=1):
        # Ensure 'smiles' column is of type string
        df['smiles'] = df['smiles'].astype(str)
        
        # Create a copy of df to avoid modifying the original dataframe
        df_copy = df.copy()

        # Modify the column names in the copy to include a unique identifier
        df_copy.columns = [f"{col}_{i}" if col != 'smiles' else col for col in df_copy.columns]

        # Perform the join
        joined_dfs = joined_dfs.join(df_copy.set_index('smiles'), on='smiles', how='inner')

    # Return the resultant dataframe
    return joined_dfs


def load_assays(input_filepath, dataset):

    # Create a DuckDB connection
    con = duckdb.connect()

    # Query all parquet files in the directory, and include a "filename" column
    query = f"SELECT * FROM read_parquet('{input_filepath}/assay_*.parquet', filename=true)"

    # Execute the query
    result = con.execute(query).fetch_df()

    # Filter the result where source_id is "test"
    filtered_result = result[result['source_id'].isin(dataset)]

    # Retrieve the filenames
    filenames = filtered_result['filename'].unique()

    # Create list of dataframes for each assay
    dfs = []

    # Read each assay into a dataframe
    for filename in filenames:
        df = pd.read_parquet(os.path.join(input_filepath, filename))

        # Drop the source_id and selfies columns
        df.drop(['source_id', 'selfies'], axis=1, inplace=True)

        dfs.append(df)
    
    return dfs


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
        xgb_model, param_grid, cv=4, n_iter=50, random_state=42
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
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.option("-f", "--features", multiple=True)
@click.option("-d", "--dataset", multiple=True)
def main(input_filepath, output_filepath, features, dataset):
    logger = logging.getLogger(__name__)
    logger.info("loading data...")

    # Load features from parquet files
    feature_df = load_features(input_filepath, features)

    # Load the assays
    assay_dfs = load_assays(input_filepath, dataset)

    # Evaluate each assay
    for i, assay_df in enumerate(assay_dfs):
        # Merge the features and assays
        merged_df = pd.merge(feature_df, assay_df, left_on="smiles", right_on="canonical_smiles", how="inner")

        # Split data into train and test - y-test not needed at scoring takes place in separate script
        y_train = merged_df.loc[merged_df['test_train'] == 0, 'ground_truth']
        y_test = merged_df.loc[merged_df['test_train'] == 1, 'ground_truth']
        X_train = merged_df.loc[merged_df['test_train'] == 0].drop(['smiles', 'canonical_smiles', 'ground_truth', 'test_train'], axis=1)
        X_test = merged_df.loc[merged_df['test_train'] == 1].drop(['smiles', 'canonical_smiles', 'ground_truth', 'test_train'], axis=1)

        logger.info(f"conducting hyperparameter search for assay {i+1}...")

        # Conduct hyperparameter search
        best_params = param_search(X_train, y_train)

        logger.info(f"fitting model {i+1}...")
    
        # Fit model and predict using best hyperparameters
        preds_proba, preds = model_fit_predict(X_train, X_test, y_train, best_params)

        # Add predictions to dataframe
        preds_df = pd.DataFrame({'preds': preds, 'preds_proba': preds_proba, 'ground_truth': y_test})

        # Save the predictions to a csv file
        preds_df.to_csv(f"{output_filepath}/preds_xgboost_{i+1}.csv", index=False) 

    logger.info(f"predictions saved to {output_filepath}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
