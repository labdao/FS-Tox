import click
import logging
import os
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.option("-a", "--assay")
def main(input_filepath, output_filepath, assay):
    # Get evaluation filenames
    pred_filenames = [f for f in os.listdir(input_filepath) if f.startswith('preds_') and assay in f]

    plt.figure()

    # Read each csv file, calculate ROC curve and plot it
    for pred_filename in pred_filenames:
        data = pd.read_csv(f"{input_filepath}/{pred_filename}")
        fpr, tpr, _ = roc_curve(data['ground_truth'], data['preds_proba'])
        roc_auc = auc(fpr, tpr)

        feature_name = pred_filename.split('_')[-1].split('.')[0]
        plt.plot(fpr, tpr, label=f'ROC curve for {feature_name} (AUC = {roc_auc:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic for {assay}')
    plt.legend(loc="lower right")

    plt.show()


if __name__ == "__main__":
    main()