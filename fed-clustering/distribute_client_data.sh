#!/bin/bash

# Parameters
NUM_CLIENTS=2
CLIENTS=()
for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    CLIENTS+=("site-$((i+1))")
done

DATASET_PREFIX="client_"
DATASET_FOLDER="/tmp/nvflare/dataset/dataset.csv"
LOCAL_DATA_PATH="/home/lylian/federated_clustering_FADE/fed-clustering/processed"

# Function to create folders and copy data
create_folders_and_copy() {
    for i in $(seq 0 $((NUM_CLIENTS - 1))); do
        CLIENT="${CLIENTS[$i]}"
        echo "Creating dataset folder in ${CLIENT}..."
        
        # Ensure the parent directory exists
        sudo docker exec "${CLIENT}" mkdir -p "$(dirname "${DATASET_FOLDER}")" || exit 1
        
        # Copy the processed CSV file to the container
        src_file="${LOCAL_DATA_PATH}/${DATASET_PREFIX}${i}_processed.csv"
        dest_path="${CLIENT}:${DATASET_FOLDER}"
        
        sudo docker cp "${src_file}" "${dest_path}" || exit 1
    done
}

# Main execution
create_folders_and_copy
echo "All client data distributed."
