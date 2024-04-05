#!/usr/bin/env bash

DIR=$(dirname $0)
PROJECT_DIR="$(cd $DIR/..; pwd)"

LOSSES=("rankNet" "pointwise_rmse" "neuralNDCG" "approxNDCGLoss")
FOLDS=("Fold1" "Fold2" "Fold3" "Fold4" "Fold5")

# Loop over folds
for fold in "${FOLDS[@]}"; do
  # Loop over losses
  for loss in "${LOSSES[@]}"; do
    # Prepare the loss part of the configuration, handling pointwise_rmse specially
    if [[ "$loss" == "pointwise_rmse" ]]; then
      jq ".loss.name = \"$loss\" | .loss.args.no_of_levels = 5" local_config.json > temp.json
    else
      jq ".loss.name = \"$loss\" | .loss.args = {}" local_config.json > temp.json
    fi
    mv temp.json local_config.json

    # Update the data path
    jq ".data.path = \"/allrank/MSLR-WEB10K/$fold\"" local_config.json > temp.json && mv temp.json local_config.json

    # Run the Docker command
    docker run --gpus all -e PYTHONPATH=/allrank -v $PROJECT_DIR:/allrank allrank:latest /bin/sh -c "python allrank/main.py --config-file-name /allrank/scripts/local_config.json --run-id run_${loss}_${fold} --job-dir /allrank/MSLR-WEB10K/run_${loss}_${fold}"
  done
done