#!/usr/bin/env bash
DATASET_PATH="/tmp/nvflare/dataset/des.csv"
NUM_CLIENTS=2
valid_frac=1
echo "Generating job configs with data splits, reading from ${DATASET_PATH}"

task_name="sklearn_kmeans"

python3 utils/prepare_job_config.py \
    --task_name "${task_name}" \
    --data_path "${DATASET_PATH}" \
    --site_num ${NUM_CLIENTS} \
    --valid_frac ${valid_frac}
