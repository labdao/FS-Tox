import logging
import os

import hydra

from config import AssayConfig
from data.raw_to_assays import make_assays
from data.assays_to_tasks import make_tasks
from features import chemberta, chemgpt, ecfp4
from models import logistic_fit, xgboost_fit, random_forest_fit
from models.evaluate import evaluate_predictions
from models.predict import generate_predictions


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: AssayConfig) -> None:
    """
    Generates assay data from the raw dataset.

    Args:
        input_filepath (str): The filepath of the raw dataset.
        output_filepath (str): The filepath to save the generated assay parquet files.
        dataset (str): The name of the dataset.
        identifier (str): The identifier for the generated assay data.
        assay_size (int): The size of each assay.
        support_set_size (int): The size of the support set for each assay.
    """
        
    # Create pipeline directories in the outputs directory
    for _, value in cfg.paths.items():
        # Handles case where value is raw dataset path
        try:
            os.makedirs(value, exist_ok=True)
        except FileExistsError:
            continue

    make_assays(
        cfg.paths.raw,
        cfg.paths.assay,
        cfg.paths.assay_id,
        cfg.params.dataset,
        cfg.params.meta_id,
        cfg.files.identifier,
        cfg.params.assay_size
    )

    # Create tasks from assays
    make_tasks(
        cfg.paths.assay,
        cfg.paths.task
    )

    # Generate features
    if cfg.params.feature == "ecfp4_128":
        ecfp4.generate(cfg.paths.assay, cfg.paths.feature, 128)
    if cfg.params.feature == "ecfp4_256":
        ecfp4.generate(cfg.paths.assay, cfg.paths.feature, 256)
    if cfg.params.feature == "ecfp4_512":
        ecfp4.generate(cfg.paths.assay, cfg.paths.feature, 512)
    if cfg.params.feature == "ecfp4_1024":
        ecfp4.generate(cfg.paths.assay, cfg.paths.feature, 1024)
    if cfg.params.feature == "ecfp4_2048":
        ecfp4.generate(cfg.paths.assay, cfg.paths.feature, 2048)
    if cfg.params.feature == "chemberta":
        chemberta.generate(cfg.paths.assay, cfg.paths.feature)
    if cfg.params.feature == "chemgpt":
        chemgpt.generate(cfg.paths.assay, cfg.paths.feature, cfg.params.model_size)

    # Train models
    if cfg.params.model == "logistic":
        logistic_fit.train(
            cfg.paths.feature,
            cfg.paths.task,
            cfg.paths.model,
            cfg.params.feature,
            cfg.params.dataset,
        )
        # Train models
    elif cfg.params.model == "random-forest":
        random_forest_fit.train(
            cfg.paths.feature,
            cfg.paths.task,
            cfg.paths.model,
            cfg.params.feature,
            cfg.params.dataset,
        )
    elif cfg.params.model == "xgboost":
        xgboost_fit.train(
            cfg.paths.feature,
            cfg.paths.task,
            cfg.paths.model,
            cfg.params.feature,
            cfg.params.dataset,
            cfg.xgboost
        )

    # Make predictions
    generate_predictions(
        cfg.paths.model,
        cfg.paths.task,
        cfg.paths.feature,
        cfg.paths.prediction,
        cfg.params.feature,
        cfg.params.dataset,
    )

    # Evaluate predictions
    evaluate_predictions(
        cfg.paths.prediction,
        cfg.paths.score,
        cfg.paths.task,
    )


if __name__ == "__main__":
    LOG_FMT = "%(asctime)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    main()
