import logging
import click
import xgboost_fit
import logistic_fit


@click.command()
@click.argument("representation_filepath", type=click.Path(exists=True))
@click.argument("assay_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.option("-r", "--representation")
@click.option("-d", "--dataset")
@click.option("-m", "--model")
def main(representation_filepath, assay_filepath, output_filepath, representation, dataset, model):
    logger = logging.getLogger(__name__)
    logger.info("training models...")

    if model == "logistic":
        logistic_fit.train(representation_filepath, assay_filepath, output_filepath, representation, dataset)
    elif model == "xgboost":
        xgboost_fit.train(representation_filepath, assay_filepath, output_filepath, representation, dataset)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()