#!/bin/bash

# ensure correct usage
if [ "$#" -ne 1 ] || { [ "$1" != "val" ] && [ "$1" != "test" ] && [ "$1" != "full" ]; }; then
    echo "Usage: $0 [split: val, test, full]"
    exit 1
fi

SPLIT=$1

SUBMISSION_NAME="041226_no_ft_translation_no_smoothen"
OUTPUT_DIR="/Users/kevin/Projects/fifa-challenge-2026/experiments/${SUBMISSION_NAME}/"
PRED_PATH="${OUTPUT_DIR}/${SUBMISSION_NAME}_${SPLIT}.npz"

# create predictions in .npz format
python main.py -s data/sequences_${SPLIT}.txt -o ${PRED_PATH} -c

# prepare the submission zip file
python prepare_submission.py \
--predictions-path ${PRED_PATH} \
--output-dir ${OUTPUT_DIR} \
--split $SPLIT
