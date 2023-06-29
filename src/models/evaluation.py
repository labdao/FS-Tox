import logging
import os
import re
import click
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.option("-d", "--dataset")
def main(input_filepath, output_filepath, dataset):
    logger = logging.getLogger(__name__)
    logger.info("Reading data from %s", input_filepath)

    # Read prediction files
    pred_filenames = []

    if dataset:
        pred_filenames = [
            f for f in os.listdir(input_filepath) if dataset in f
        ]

    # Create a list to store the metric dictionaries
    feature_performance = []

    for pred_filename in pred_filenames:
        # Create empty dictionary to store metrics
        metrics_dict = {}

        # Load the predictions
        df = pd.read_parquet(os.path.join(input_filepath, pred_filename))

        # Get embeding name from filename
        filename_without_extension, _ = os.path.splitext(pred_filename)
        feature_name = re.search(r"[^_]*$", filename_without_extension).group()

        # Get the true labels and the predicted labels
        y_true = df["ground_truth"]
        y_pred = df["preds"]

        # Add feature name to metrics_dict
        metrics_dict["feature"] = feature_name
        # Calculate common evaluation metrics
        metrics_dict["accuracy"] = accuracy_score(y_true, y_pred)
        metrics_dict["precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics_dict["recall"] = recall_score(y_true, y_pred)
        metrics_dict["f1"] = f1_score(y_true, y_pred)
        try:
            y_score = df["preds_proba"]
            metrics_dict["auc_roc"] = roc_auc_score(y_true, y_score)
            metrics_dict["auc_pr"] = average_precision_score(
                y_true, y_score
            )  # calculate AUC-PR
        except ValueError as e:
            logger.warn(
                "Cannot compute ROC AUC score because ground truth data contains only one class. Outputting NaN for ROC AUC"
            )
            metrics_dict["auc_roc"] = float("nan")

        # Convert the dictionary to a dataframe
        metrics_df = pd.DataFrame(metrics_dict, index=[0])

        # Get the assay name from the filename
        assay_feature_name = pred_filename.replace("preds_", "").split(".")[0]

        # Save inidividual assay metrics to a parquet file
        metrics_df.to_parquet(f"{output_filepath}/{assay_feature_name}.parquet")

    logger.info("Saved metrics to %s", output_filepath)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
