import logging
import click
import xgboost_fit
import logistic_fit


@click.command()
@click.argument("feature_filepath", type=click.Path(exists=True), default="./data/processed/features")
@click.argument("assay_filepath", type=click.Path(exists=True), default="./data/processed/assays")
@click.argument("output_filepath", type=click.Path(), default="./data/processed/models")
@click.option("-d", "--dataset")
@click.option("-f", "--feature", default="ecfp4_1024")
@click.option("-m", "--model", default="logistic")
def main(feature_filepath, assay_filepath, output_filepath, feature, dataset, model):
    logger = logging.getLogger(__name__)
    logger.info("training models...")

    if model == "logistic":
        logistic_fit.train(feature_filepath, assay_filepath, output_filepath, feature, dataset)
    elif model == "xgboost":
        xgboost_fit.train(feature_filepath, assay_filepath, output_filepath, feature, dataset)

if __name__ == '__main__':
    log_fmt = "%(asctime)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()