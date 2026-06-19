import subprocess
import sys

# Parameters
NUM_CLIENTS = 2
CLIENTS = [f"site-{i+1}" for i in range(NUM_CLIENTS)] 
DATASET_PREFIX = "client_"  # e.g., client_0.csv, client_1.csv, ...
DATASET_FOLDER = "/tmp/nvflare/dataset/dataset.csv"
LOCAL_DATA_PATH = "/home/lylian/federated_clustering_FADE/fed-clustering/processed"  # Where the split CSVs are stored on host


def create_folders_and_copy():
    for i, client in enumerate(CLIENTS):
        print(f"Creating dataset folder in {client}...")
        # Ensure the parent directory exists
        subprocess.run(
            ["sudo", "docker", "exec", client, "mkdir", "-p", DATASET_FOLDER.rsplit("/", 1)[0]],
            check=True,
        )
        src_file = f"{LOCAL_DATA_PATH}/{DATASET_PREFIX}{i}_processed.csv"
        dest_path = f"{client}:{DATASET_FOLDER}"
        print(f"Copying {src_file} to {dest_path}")
        subprocess.run(["sudo", "docker", "cp", src_file, dest_path], check=True)


if __name__ == "__main__":
    create_folders_and_copy()
    print("All client data distributed.")
