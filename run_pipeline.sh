#!/bin/bash

python ./src/data/raw_to_assays.py ./data/raw/nci60/LC50/LC50.csv ./data/processed/assays --dataset nci60 --identifier ./data/external/NCIOPENB_SMI.txt

python ./src/features/make_features.py ./data/processed/assays ./data/processed/features --feature ecfp4_1024

python ./src/models/train_model.py ./data/processed/features ./data/processed/assays ./data/processed/models --dataset nci60 --feature ecfp4_1024 --model logistic

python ./src/models/predict.py ./data/processed/models ./data/processed/assays ./data/processed/features ./data/processed/predictions -t --dataset nci60

python ./src/models/evaluation.py ./data/processed/predictions ./data/processed/scores --dataset nci60
