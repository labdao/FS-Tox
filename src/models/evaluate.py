import logging
import os
import re

import click
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)


def evaluate_predictions(input_filepath, output_filepath, assay_filepath):
    logger = logging.getLogger(__name__)
    logger.info("Reading data from %s", input_filepath)

    pred_filenames = os.listdir(input_filepath)
    assay_ids = [pred_filename.split(".")[0] for pred_filename in pred_filenames]

    for (pred_filename, assay_id) in zip(pred_filenames, assay_ids):
        # Create empty dictionary to store metrics
        metrics_dict = {}

        # Load the predictions
        df = pd.read_parquet(os.path.join(input_filepath, pred_filename))

        # Get the true labels and the predicted labels
        y_true = df["ground_truth"]
        y_pred = df["preds"]

        # Calculate common evaluation metrics
        metrics_dict["accuracy"] = accuracy_score(y_true, y_pred)
        metrics_dict["precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics_dict["recall"] = recall_score(y_true, y_pred)
        metrics_dict["f1"] = f1_score(y_true, y_pred)

        try:
            y_score = df["preds_proba"]
            metrics_dict["auc_roc"] = roc_auc_score(y_true, y_score)
            auc_pr = average_precision_score(y_true, y_score)

            # Load the ground truth data
            ratio_df = pd.read_parquet(
                f"{assay_filepath}/{assay_id}.parquet",
                columns=["ground_truth", "support_query"],
            )
            # Filter the ground truth data to only include the support query
            ratio_df = ratio_df[ratio_df["support_query"] == 1]
            # Calculate the delta AUC PR
            delta_auc = auc_pr - (ratio_df["ground_truth"].sum() / len(ratio_df))
            metrics_dict["delta_auc_pr"] = delta_auc

        except ValueError:
            logger.warning(
                "Cannot compute ROC AUC score because ground truth data contains only one class. Outputting NaN for ROC AUC"
            )
            metrics_dict["auc_roc"] = float("nan")

        # Convert the dictionary to a dataframe
        metrics_df = pd.DataFrame(metrics_dict, index=[0])

        # Save inidividual assay metrics to a parquet file
        metrics_df.to_parquet(f"{output_filepath}/{assay_id}.parquet")

    logger.info("Saved metrics to %s", output_filepath)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
