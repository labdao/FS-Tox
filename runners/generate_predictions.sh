#!/bin/bash

export NEPTUNE_PROJECT=sethhowes/fs-tox

export input_path=toxcast.csv
export dataset=toxcast
export assay_size=128
support_set_sizes=(16 32 64)

for support_set_size in ${support_set_sizes[@]}
do
    make clean

    python ./src/data/raw_to_assays.py ./data/raw/$input_path ./data/processed/assays --dataset $dataset

    python ./src/features/make_features.py ./data/processed/assays ./data/processed/features --feature ecfp4_1024

    python ./src/models/train_model.py ./data/processed/features ./data/processed/assays ./data/processed/models --dataset $dataset --feature ecfp4_1024 --model logistic

    python ./src/models/predict.py ./data/processed/models ./data/processed/assays ./data/processed/features ./data/processed/predictions -t --dataset $dataset

    python ./src/models/evaluation.py ./data/processed/predictions ./data/processed/scores --dataset $dataset

    python ./src/visualization/distribution.py --dataset $dataset --metric delta_auc_pr
done