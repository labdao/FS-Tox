#!/bin/bash

python ./src/data/raw_to_assays.py ./data/raw/tox21.csv ./data/processed/assays -d tox21

python ./src/features/make_features.py ./data/processed/assays ./data/processed/features --feature ecfp4_1024

python ./src/models/train_model.py ./data/processed/features ./data/processed/assays ./data/processed/models --dataset tox21 --representation ecfp4_1024 --model logistic

python ./src/models/predict.py ./data/processed/models ./data/processed/assays ./data/processed/features ./data/processed/predictions -t --dataset tox21

python ./src/models/evaluation.py ./data/processed/predictions ./data/processed/scores -d tox21
