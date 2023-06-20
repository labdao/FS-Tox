#!/bin/bash
python ./src/data/raw_to_assays.py ./data/raw/toxvaldb_2023.csv ./data/processed/assays -d toxval -i ./data/external/DSSTox_Identifiers_and_CASRN_2021r1.csv

python ./src/features/ecfp4.py ./data/processed/assays ./data/processed/representations

python ./src/models/logistic_fit.py ./data/processed/representations ./data/processed/assays ./data/processed/models -d toxvaldb_2023 -r ecfp4_1024

python ./src/models/predict.py ./data/processed/models ./data/processed/assays ./data/processed/representations ./data/processed/predictions -t -d toxvaldb_2023

python ./src/models/evaluation.py ./data/processed/predictions ./data/processed/scores -d toxvaldb_2023