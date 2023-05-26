import logging
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
@click.option("-m", "--metrics", default="auc")
def main(input_filepath, output_filepath, metrics):
    logger = logging.getLogger(__name__)
    logger.info(f"Reading data from {input_filepath}...")

    # Read the data
    df = pd.read_csv(input_filepath)

    # Get the true labels and the predicted labels
    y_true = df["ground_truth"]
    y_pred = df["preds"]

    # Calculate the metrics
    metrics_dict = {}
    if "accuracy" in metrics:
        metrics_dict["accuracy"] = accuracy_score(y_true, y_pred)
    if "precision" in metrics:
        metrics_dict["precision"] = precision_score(y_true, y_pred)
    if "recall" in metrics:
        metrics_dict["recall"] = recall_score(y_true, y_pred)
    if "f1" in metrics:
        metrics_dict["f1"] = f1_score(y_true, y_pred)
    if "auc" in metrics:
        try:
            y_score = df["preds_proba"]
            metrics_dict["auc_roc"] = roc_auc_score(y_true, y_score)
        except ValueError as e:
            logger.warn("Cannot compute ROC AUC score because ground truth data contains only one class. Outputting NaN for ROC AUC.")
            metrics_dict["auc_roc"] = float('nan')  # or use np.nan if numpy is imported


    # Save the metrics to a new CSV file
    metrics_df = pd.DataFrame(metrics_dict, index=[0])
    metrics_df.to_csv(output_filepath, index=False)

    logger.info(f"Saved metrics to {output_filepath}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
