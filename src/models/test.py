import pandas as pd

# Read the Parquet file into a pandas DataFrame
df = pd.read_parquet('/Users/sethhowes/Desktop/FS-Tox/data/processed/assays/NR-AR_tox21.parquet')

# Display the DataFrame
print(df.head())