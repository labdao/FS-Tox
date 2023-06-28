#!/bin/bash
python3 ./src/data/raw_to_assays.py ./data/raw/nci60/LC50/LC50.csv ./data/processed/assays --dataset nci60

python3 ./src/features/make_features.py ./data/processed/assays ./data/processed/features --feature ecfp_1024

python3 ./src/models/train_model.py ./data/processed/features ./data/processed/assays ./data/processed/models -d nci60 -r ecfp4_1024 -m logistic

python3 ./src/models/predict.py ./data/processed/models ./data/processed/assays ./data/processed/features ./data/processed/predictions -t -d nci60 --representation ecfp4_1024 --model logistic

python3 ./src/models/evaluation.py ./data/processed/predictions ./data/processed/scores --dataset nci60