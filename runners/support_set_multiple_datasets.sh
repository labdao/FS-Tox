#!/opt/homebrew/bin/bash

export NEPTUNE_PROJECT=sethhowes/fs-tox

export assay_size=128
support_set_sizes=(8 16 32 64)

declare -A datasets
datasets["toxcast"]="toxcast.csv"
datasets["toxval"]="toxval.csv"
datasets["cancerrx"]="cancerrx.xlsx"
datasets["prism"]="prism.csv"
datasets["nci60"]="nci60/LC50.csv"

for dataset in "${!datasets[@]}"
do 
    export input_path=${datasets[$dataset]}
    for support_set_size in ${support_set_sizes[@]}
    do
        python ./src/data/raw_to_assays.py ./data/raw/$input_path ./data/processed/assays --dataset $dataset --support_set_size $support_set_size --assay_size $assay_size

        python ./src/features/make_features.py ./data/processed/assays ./data/processed/features --feature ecfp4_1024

        python ./src/models/train_model.py ./data/processed/features ./data/processed/assays ./data/processed/models --dataset $dataset --feature ecfp4_1024 --model logistic

        python ./src/models/predict.py ./data/processed/models ./data/processed/assays ./data/processed/features ./data/processed/predictions -t --dataset $dataset

        python ./src/models/evaluation.py ./data/processed/predictions ./data/processed/scores --dataset $dataset --support-query $support_set_size

        python ./src/visualization/distribution.py --dataset $dataset --metric delta_auc_pr
    done
done