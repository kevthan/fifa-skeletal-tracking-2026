import argparse
import os
import zipfile
from pathlib import Path

import numpy as np


def load_sequences(sequences_file: Path | str) -> list[str]:
    with open(sequences_file) as f:
        sequences = f.read().splitlines()
    sequences = filter(lambda x: not x.startswith("#"), sequences)
    sequences = [s.strip() for s in sequences]
    return sequences


def prepare_submission(output_dir: Path, pred_path: Path, split: str):
    data = np.load(pred_path)
    sequences = load_sequences(f"data/sequences_{split}.txt")

    submission = {}
    for k in sequences:
        submission[k] = data[k].astype(np.float32)

    submission_name = pred_path.stem
    with zipfile.ZipFile(output_dir / f"{submission_name}.zip", "w") as zipf:
        np.savez_compressed(output_dir / f"{submission_name}.npz", **submission)
        zipf.write(output_dir / f"{submission_name}.npz", arcname="submission.npz")
        os.remove(output_dir / f"{submission_name}.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare submission zip file.")
    parser.add_argument(
        "--predictions-path",
        type=Path,
        help="Path to the .npz file containing the predictions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/"),
        help="Directory where the predictions are stored and where the submission zip will be saved.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="full",
        choices=["val", "test", "full"],
        help="Which split to prepare the submission for.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    pred_path = args.predictions_path
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions not found at {pred_path}")

    predictions = np.load(pred_path)
    if args.split == "val" or args.split == "full":
        prepare_submission(output_dir, pred_path, "val")
    if args.split == "test" or args.split == "full":
        prepare_submission(output_dir, pred_path, "test")
