from huggingface_hub import snapshot_download

if __name__ == "__main__":
    local_path = "/Users/kevin/Projects/fifa-challenge-2026/fifa-skeletal-tracking-2026/data"

    local_path = snapshot_download(
        repo_id="tijiang13/FIFA-Skeletal-Tracking-Light-2026",
        repo_type="dataset",
        local_dir=local_path,
    )

    print(local_path)
