import logging
import click
import pickle
import duckdb
import pandas as pd

from utils import (
    load_assays,
    load_representations,
    construct_query,
    mod_test_train_split,
)


@click.command()
@click.argument("model_filepath", type=click.Path(exists=True))
@click.argument("assay_filepath", type=click.Path(exists=True))
@click.argument("representation_filepath", type=click.Path(exists=True))
@click.argument("prediction_filepath", type=click.Path(exists=True))
@click.option("-t", "--test", is_flag=True)
@click.option("-r", "--representation", default="ecfp4_1024")
@click.option("-m", "--model", default="logistic")
@click.option("-d", "--dataset", default="tox21")
@click.option("-a", "--assay", multiple=True)
def main(
    model_filepath,
    assay_filepath,
    representation_filepath,
    prediction_filepath,
    test,
    representation,
    model,
    dataset,
    assay,
):
    logger = logging.getLogger(__name__)
    logger.info("loading data...")

    # Create a SQL query as a string to select relevant representations
    representation_query = construct_query(representation_filepath, representation)

    # Load representations from parquet files
    representation_df = load_representations(representation_query)

    if test:
        # Load the assays
        assay_dfs = load_assays(assay_filepath, dataset)

        # Evaluate each assay
        for i, (assay_df, assay_filename) in enumerate(assay_dfs):
            logger.info(f"creating predictions for assay {i+1}...")

            # Merge the representations and assays
            merged_df = pd.merge(
                representation_df, assay_df, on="canonical_smiles", how="inner"
            )

            # Conduct test train split
            _, X_test, _, y_test = mod_test_train_split(merged_df)

            # Get model filepath
            trained_model_filepath = f"{model_filepath}/{assay_filename}_{model}_{representation}.pkl"

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
                    "model": model[0],
                }
            )

            # Save the predictions to a parquet file
            preds_df.to_parquet(
                f"{prediction_filepath}/{assay_filename}_{model}_{representation}.parquet"
            )

    else:
        # Get model filepath
        trained_model_filepath = f"{model_filepath}/{assay[0]}_{model[0]}_{representation[0]}.pkl"
        
        # Load the model
        with open(trained_model_filepath, "rb") as f:
            trained_model = pickle.load(f)
        
        # Save canonical smiles to a list
        canonical_smiles = representation_df["canonical_smiles"].tolist()

        # Drop the canonical_smiles column from the representation dataframe
        representation_df.drop(columns=["canonical_smiles"], inplace=True)

        # Generate predictions on the dataset
        preds = trained_model.predict(representation_df)

        # Generate prediction probabilities
        preds_proba = trained_model.predict_proba(representation_df)[:, 1]

        # Add predictions to dataframe
        preds_df = pd.DataFrame(
            {
                "canonical_smiles": canonical_smiles,
                "preds": preds,
                "preds_proba": preds_proba,
                "model": model[0],
            }
        )

        # Convert the representations tuple into a string with elements separated by '_'
        representation_str = "_".join(representation)

        # Save the predictions to a parquet file
        preds_df.to_parquet(
            f"{prediction_filepath}/{assay[0]}_{model[0]}_{representation_str}.parquet"
        )

if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
