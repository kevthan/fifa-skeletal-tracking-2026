#!/bin/bash

set -euo pipefail

if [ "$#" -ne 2 ]; then
	echo "Usage: $0 /path/to/preds /path/to/output_videos"
	exit 1
fi

PREDS_DIR="$1"
OUTPUT_PATH="$2"

if [ ! -d "$PREDS_DIR" ]; then
	echo "Predictions folder not found: $PREDS_DIR"
	exit 1
fi

mkdir -p "$OUTPUT_PATH"

found_npz=0

for pred_file in "$PREDS_DIR"/*.npz; do
	if [ ! -f "$pred_file" ]; then
		continue
	fi

	found_npz=1
	seq_name="$(basename "$pred_file" .npz)"
	pred_dir="$(dirname "$pred_file")"

	# Resolve calibration directory relative to each prediction file.
	calib_dir="${pred_dir}/calibration"
	if [ ! -d "$calib_dir" ]; then
		calib_dir="$(dirname "$pred_dir")/calibration"
	fi

	if [ ! -d "$calib_dir" ]; then
		echo "[skip] Calibration directory not found for $pred_file"
		echo "       Tried: ${pred_dir}/calibration and $(dirname "$pred_dir")/calibration"
		continue
	fi

	echo "[run] sequence=${seq_name}"
	echo "      pred=${pred_file}"
	echo "      calib=${calib_dir}"

	if ! python visualize.py \
		--sequence "$seq_name" \
		-p "$pred_file" \
		--calibration_dir "$calib_dir" \
		--output_path "$OUTPUT_PATH" \
		--headless; then
		echo "[fail] visualize.py failed for ${seq_name}"
	fi
done

if [ "$found_npz" -eq 0 ]; then
	echo "No .npz files found in: $PREDS_DIR"
	exit 1
fi

echo "Done. Videos written to: $OUTPUT_PATH"
