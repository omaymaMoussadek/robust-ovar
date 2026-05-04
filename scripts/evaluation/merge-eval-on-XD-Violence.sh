#!/bin/bash
#
# Evaluation script for a merged checkpoint on XD-Violence.
#
# LOAD_CKPT_FILE should point to a merged delta checkpoint produced by
# scripts/merging/merging.sh, for example:
#   /path/to/merged/checkpoints/output/mean_delta.pyth
#
# BASE_CKPT_FILE should point to the Open-VCLIP base checkpoint used to compute
# the task vectors, for example:
#   /path/to/openvclip/checkpoints/openvclip-b16/swa_2_22.pth
#
# The evaluated model is reconstructed at test time as:
#   model = BASE_CKPT_FILE + MERGE_ALPHA * LOAD_CKPT_FILE
#
# Change MERGE_ALPHA to evaluate a different scaling coefficient. To evaluate a
# ViT-L/14 merged delta, also replace the config with:
#   configs/Kinetics/TemporalCLIP_vitl14_8x16_STAdapter.yaml

# ====== PARAMS TO EDIT ======
DATASET_NAME=XD-Violence
NUM_CLASSES=7
ROOT=/path/to/robust-ovar/repository
DATA=/path/to/datasets/root
CKPT=/path/to/checkpoints/root
LOG_DIR=/path/to/evaluation/logs/merge/xd-violence

LOAD_CKPT_FILE=/path/to/merged/checkpoints/output/mean_delta.pyth
BASE_CKPT_FILE=/path/to/openvclip/checkpoints/openvclip-b16/swa_2_22.pth
MERGE_ALPHA=1.0

INDEX2CLS_FILE=$DATA/$DATASET_NAME/annotations/XD-Violence-index2cls.json
PATCHING_RATIO=0.0
CLIP_ORI_PATH=/path/to/clip/cache/ViT-B-16.pt
# ============================

export PYTHONPATH="$ROOT/slowfast:$PYTHONPATH"
cd "$ROOT"

ALPHA_TAG=${MERGE_ALPHA//./p}
LOG_PREFIX="eval_${DATASET_NAME}_merge_alpha${ALPHA_TAG}"
mkdir -p "$LOG_DIR"

python -W ignore -u tools/run_net.py \
    --cfg configs/Kinetics/TemporalCLIP_vitb16_8x16_STAdapter.yaml \
    --opts DATA.PATH_TO_DATA_DIR $DATA/$DATASET_NAME/annotations \
    DATA.PATH_PREFIX $DATA/$DATASET_NAME/videos \
    DATA.PATH_LABEL_SEPARATOR , \
    DATA.INDEX_LABEL_MAPPING_FILE $INDEX2CLS_FILE \
    TRAIN.ENABLE False \
    TEST.BATCH_SIZE 2 \
    NUM_GPUS 1 \
    DATA_LOADER.NUM_WORKERS 2 \
    DATA_LOADER.PIN_MEMORY False \
    DATA.DECODING_BACKEND "pyav" \
    MODEL.NUM_CLASSES $NUM_CLASSES \
    TEST.CUSTOM_LOAD True \
    TEST.CUSTOM_LOAD_FILE $LOAD_CKPT_FILE \
    TEST.MERGE_WITH_BASE True \
    TEST.MERGE_BASE_FILE $BASE_CKPT_FILE \
    TEST.MERGE_ALPHA $MERGE_ALPHA \
    TEST.SAVE_RESULTS_PATH temp.pyth \
    TEST.NUM_ENSEMBLE_VIEWS 3 \
    TEST.NUM_SPATIAL_CROPS 1 \
    TEST.PATCHING_MODEL True \
    TEST.PATCHING_RATIO $PATCHING_RATIO \
    TEST.CLIP_ORI_PATH $CLIP_ORI_PATH \
    > "$LOG_DIR/${LOG_PREFIX}.out" \
    2> "$LOG_DIR/${LOG_PREFIX}.err"
