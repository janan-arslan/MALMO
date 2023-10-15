#!/bin/bash

#SBATCH --cpus-per-task=1
#SBATCH --mem=400MB
#SBATCH --time=300:00:00
#SBATCH --chdir=.
#SBATCH --output=./bash-log/%A_%a.txt
#SBATCH --error=./bash-log/%A_%a.txt
#SBATCH --job-name=bv-segmentation
#SBATCH --array=1-40

## Set the input, model, and output paths. Adjust these accordingly.
INPUT_DIR="/path/to/input/directory"
MODEL_PATH="/path/to/model/file"
OUTPUT_DIR="/path/to/output/directory"

## Run the Python script
python 2d_patch_level_segmentation_v2_hpc.py -input $INPUT_DIR -model $MODEL_PATH -output $OUTPUT_DIR 
