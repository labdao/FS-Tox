import os
import pandas as pd

from .utils import assign_test_train



def make_tasks(input_filepath, output_filepath):
    assay_filepaths= os.listdir(input_filepath)
    for assay_filepath in assay_filepaths:
        assay_id = os.path.basename(assay_filepath).split(".")[0]
        assay_df = pd.read_parquet(f"{input_filepath}/{assay_filepath}")
        dfs = assay_to_tasks(assay_df)
        for i, (support_set_size, df) in enumerate(dfs):
            df.to_parquet(f'{output_filepath}/{assay_id}_support-{support_set_size}_permutation-{i}.parquet', index=False)


def assay_to_tasks(df, target_support_set_sizes=[16,32,64,128,256], min_query_size=16, num_permutations=3):
    dfs = []
    available_support_set_sizes = get_available_support_sizes(len(df), min_query_size, target_support_set_sizes)
    for support_size in available_support_set_sizes:
        for i in range(num_permutations):
            df['support_query'] = assign_test_train(len(df), support_size, seed=i)
            df['permutation'] = i+1
            df['support_set_size'] = support_size
            dfs.append((support_size, df.copy()))
    return dfs


def get_available_support_sizes(df_size, min_query_size, support_set_sizes):
     adjusted_size = df_size - min_query_size
     return [size for size in support_set_sizes if adjusted_size >= size]