from pathlib import Path
import zipfile
import numpy as np
import os


def load_sequences(sequences_file: Path | str) -> list[str]:
    with open(sequences_file) as f:
        sequences = f.read().splitlines()
    sequences = filter(lambda x: not x.startswith("#"), sequences)
    sequences = [s.strip() for s in sequences]
    return sequences


def prepare_submission(output_dir: Path, split: str):
    data = np.load(output_dir / f"submission_full.npz")
    sequences = load_sequences(output_dir / f"sequences_{split}.txt")

    submission = {}
    for k in sequences:
        submission[k] = data[k].astype(np.float32)

    with zipfile.ZipFile(output_dir / f"submission_{split}.zip", "w") as zipf:
        np.savez_compressed(output_dir / f"submission_{split}.npz", **submission)
        zipf.write(output_dir / f"submission_{split}.npz", arcname="submission.npz")
        os.remove(output_dir / f"submission_{split}.npz")


if __name__ == "__main__":
    output_dir = Path("outputs/")
    predictions_dir = output_dir / "submission_full.npz"
    if not predictions_dir.exists():
        raise FileNotFoundError(f"Predictions not found at {predictions_dir}")

    predictions = np.load(predictions_dir)
    prepare_submission(predictions, output_dir, "val")
    prepare_submission(predictions, output_dir, "test")
