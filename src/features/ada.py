import openai
import pandas as pd
import logging
import click
import duckdb

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
main(input_filepath, output_filepath):
    # Connect to a database in memory
    connection = duckdb.connect(database=":memory:")

    # Load the data into the database as list of tuples
    df = connection.execute(
        f"""
    SELECT DISTINCT canonical_smiles
    FROM '{input_filepath}/assay_*.parquet'
    """
    ).df()

    df['ada_embedding'] = df.combined.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
    df.to_csv(f'{output_filepath}/feature_ada_embeddings.csv', index=False)


if __name__ == __main__:
    log_fmt = "%(asctime)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()