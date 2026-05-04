#!/bin/bash
#
# Script for merging Open-VCLIP fine-tuned checkpoints.
#
# This script wraps model_merging.py. By default, it saves only the merged
# delta/task vector, matching the workflow used for zero-shot evaluation with
# TEST.MERGE_WITH_BASE=True.
#
# 1. Save only the merged delta/task vector (default behavior).
#    Use this when you want to store the merged update separately and apply it
#    later during evaluation with TEST.MERGE_WITH_BASE=True.
#    The output file will be named:
#      <MERGE_MODE>_delta.pyth
#
# 2. Optional: save full merged checkpoints for one or more alpha values.
#    This workflow is provided as commented code at the bottom of the script.
#    To use it, comment the active --save_delta command and uncomment the
#    ALPHAS block.
#    Use it when you want ready-to-load checkpoints where the merged delta has
#    already been added back to the base model:
#      merged_checkpoint = base_checkpoint + alpha * merged_delta
#    The output file will be named:
#      <MERGE_MODE>_alpha<alpha>.pyth
#
# Edit the paths below before running this script. The checkpoint paths can point
# either directly to .pyth files or to training output directories containing a
# checkpoints/checkpoint_epoch_*.pyth file.

set -euo pipefail

# ====== PARAMS TO EDIT ======
ROOT=/path/to/robust-ovar/repository

BASE_CKPT_FILE=/path/to/openvclip/checkpoints/openvclip-b16/swa_2_22.pth
OUTPUT_DIR=/path/to/merged/checkpoints/output

# Merging method. Supported values: mean, iso-c, tsv-m
MERGE_MODE=mean

# Device used for merging computations. Use cpu for portability, or cuda if
# enough GPU memory is available.
DEVICE=cpu

# Source checkpoints to merge.
SOURCE_CKPTS=(
  /path/to/source/checkpoints/source_dataset_1/checkpoint_epoch_00022.pyth
  /path/to/source/checkpoints/source_dataset_2/checkpoint_epoch_00022.pyth
  /path/to/source/checkpoints/source_dataset_3/checkpoint_epoch_00022.pyth
)

# ============================

cd "$ROOT"
mkdir -p "$OUTPUT_DIR"

python -W ignore -u model_merging.py \
  --ckpts "${SOURCE_CKPTS[@]}" \
  --mode "$MERGE_MODE" \
  --base "$BASE_CKPT_FILE" \
  --save_delta \
  --output_dir "$OUTPUT_DIR" \
  --device "$DEVICE"

# Optional: save full merged checkpoints for specific alpha values instead of
# only the delta. Uncomment this block and comment the command above if needed.
#
# ALPHAS=(0.5 1.0 1.5)
#
# for ALPHA in "${ALPHAS[@]}"; do
#   python -W ignore -u model_merging.py \
#     --ckpts "${SOURCE_CKPTS[@]}" \
#     --mode "$MERGE_MODE" \
#     --base "$BASE_CKPT_FILE" \
#     --alpha "$ALPHA" \
#     --output_dir "$OUTPUT_DIR" \
#     --device "$DEVICE"
# done
