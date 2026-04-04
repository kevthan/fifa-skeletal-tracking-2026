#!/bin/bash

# ensure correct usage
if [ "$#" -ne 1 ] || { [ "$1" != "val" ] && [ "$1" != "test" ] && [ "$1" != "full" ]; }; then
    echo "Usage: $0 [split: val, test, full]"
    exit 1
fi

SPLIT=$1

OUTPUT_DIR="out_040326_ft_translation_avg_15joints"

# create predictions in .npz format
python main.py -s data/sequences_${SPLIT}.txt -o ${OUTPUT_DIR}/submission_${SPLIT}.npz -c

# prepare the submission zip file
python prepare_submission.py --split $SPLIT --output-dir ${OUTPUT_DIR}
