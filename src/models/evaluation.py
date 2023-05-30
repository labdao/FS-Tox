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
)


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.option("-a", "--assay")
def main(input_filepath, output_filepath, assay):
    logger = logging.getLogger(__name__)
    logger.info("Reading data from %s...", input_filepath)

    # Read prediction files
    pred_filenames = [f for f in os.listdir(input_filepath) if f.startswith('preds_') and assay in f]

    # Create a list to store the metric dictionaries
    feature_performance = []

    for pred_filename in pred_filenames:
        
        # Create empty dictionary to store metrics
        metrics_dict = {}

        # Load the predictions
        df = pd.read_csv(f"{input_filepath}/{pred_filename}")
        
        # Get embeding name from filename
        filename_without_extension, _ = os.path.splitext(pred_filename)
        feature_name = re.search(r'[^_]*$', filename_without_extension).group()

        # Get the true labels and the predicted labels
        y_true = df["ground_truth"]
        y_pred = df["preds"]

        # Add feature name to metrics_dict
        metrics_dict["feature"] = feature_name
        # Calculate common evaluation metrics
        metrics_dict["accuracy"] = accuracy_score(y_true, y_pred)
        metrics_dict["precision"] = precision_score(y_true, y_pred)
        metrics_dict["recall"] = recall_score(y_true, y_pred)
        metrics_dict["f1"] = f1_score(y_true, y_pred)
        try:
            y_score = df["preds_proba"]
            metrics_dict["auc_roc"] = roc_auc_score(y_true, y_score)
        except ValueError as e:
            logger.warn("Cannot compute ROC AUC score because ground truth data contains only one class. Outputting NaN for ROC AUC")
            metrics_dict["auc_roc"] = float('nan')

        # Convert metrics_dict to DataFrame and transpose it to have keys as columns
        feature_performance.append(metrics_dict)

    metrics_df = pd.DataFrame(feature_performance)
    
    # Save the metrics to a new CSV file
    metrics_df.to_csv(f"{output_filepath}/score_{assay}.csv", index=False)

    logger.info("Saved metrics to %s", output_filepath)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
