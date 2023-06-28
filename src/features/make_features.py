import chemberta
import chemgpt
import ecfp4

import click
import logging

@click.command(help="This command creates features from the small molecules contained in a given assay.")
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.option("--feature", "-f", default="ecfp4_1024", type=click.Choice(["ecfp4_1024", "ecfp4_2048", "chemberta", "chemgpt"]))
def main(input_filepath, output_filepath, feature):
    logger = logging.getLogger(__name__)

    if feature == "ecfp4_1024":
        ecfp4.generate(input_filepath, output_filepath, 1024)
    if feature == "ecfp4_2048":
        ecfp4.generate(input_filepath, output_filepath, 2048)
    if feature == "chemberta":
        chemberta.generate(input_filepath, output_filepath)
    if feature == "chemgpt":
        chemgpt.generate(input_filepath, output_filepath)

if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()