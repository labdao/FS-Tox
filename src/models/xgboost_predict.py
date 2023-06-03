import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import statistics

import os
import logging
import click
import duckdb


def construct_query(input_filepath, representations):
    if len(representations) == 1:
        return "SELECT * FROM '" + os.path.join(input_filepath, f"{representations[0]}.parquet'")
    
    base_query = "SELECT * FROM "
    joins = []

    for i in range(len(representations) - 1):
            join = f"'{input_filepath}/{representations[i]}.parquet' INNER JOIN '{input_filepath}/{representations[i+1]}.parquet' ON {representations[i]}.canonical_smiles == {representations[i+1]}.canonical_smiles"
            joins.append(join)

    return base_query + " ".join(joins)


def load_representations(representation_query):
    # Create a database connection
    con = duckdb.connect()

    # Execute the query
    representations_df = con.execute(representation_query).df()

    # Drop all columns starting with 'canonical_smiles'
    canonical_cols = [col for col in representations_df.columns if col.startswith('canonical_smiles_')]
    representations_df = representations_df.drop(canonical_cols, axis=1)

    # Drop the columns starting with 'representation_'
    representation_cols = [col for col in representations_df.columns if col.startswith('representation')]
    representations_df = representations_df.drop(representation_cols, axis=1)

    # Return the resultant dataframe
    return representations_df


def load_assays(input_filepath, dataset):
    # Create a DuckDB connection
    con = duckdb.connect()

    # Convert the dataset tuple to a string in the format ('item1', 'item2', ...)
    if len(dataset) == 1:
        dataset_str = f"('{dataset[0]}')"
    else:
        dataset_str = str(dataset)

    # Query all parquet files in the directory, and include a "filename" column
    query = (
        f"SELECT * FROM read_parquet('{input_filepath}/*', filename=true) WHERE source_id IN {dataset_str}"
    )

    # Execute the query
    result = con.execute(query).df()

    # Retrieve the filenames for the relevant assays
    filenames = result["filename"].unique()

    # Create list of dataframes for each assay
    dfs = []

    # Read each assay into a dataframe
    for filename in filenames:
        df = pd.read_parquet(filename)

        # Drop the source_id and selfies columns
        df.drop(["source_id", "selfies"], axis=1, inplace=True)

        # Get file basename
        assay_basename = os.path.basename(filename)
        
        # Remove the file extension
        assay_basename = os.path.splitext(assay_basename)[0]

        dfs.append((df, assay_basename))

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

        # Split data into train and test - y-test not needed at scoring takes place in separate script
        y_train = merged_df.loc[merged_df["test_train"] == 0, "ground_truth"]
        y_test = merged_df.loc[merged_df["test_train"] == 1, "ground_truth"]
        X_train = merged_df.loc[merged_df["test_train"] == 0].drop(
            ["canonical_smiles", "ground_truth", "test_train", "assay_id"], axis=1
        )
        X_test = merged_df.loc[merged_df["test_train"] == 1].drop(
            ["canonical_smiles", "ground_truth", "test_train", "assay_id"], axis=1
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
            {"canonical_smiles": test_canonical_smiles, "preds": preds, "preds_proba": preds_proba, "ground_truth": y_test}
        )
        
        # Convert the representations tuple into a string with elements separated by '_'
        representation_str = '_'.join(representation)

        # Save the predictions to a parquet file
        preds_df.to_parquet(
            f"{output_filepath}/{assay_filename}_{representation_str}.csv"
        )

    logger.info(f"predictions saved to {output_filepath}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
