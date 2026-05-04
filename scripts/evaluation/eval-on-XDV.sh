#!/bin/bash
#
# Evaluation script for a model checkpoint specified by LOAD_CKPT_FILE.
#
# If LOAD_CKPT_FILE is set to one of the Open-VCLIP base checkpoints:
#   /path/to/openvclip/checkpoints/openvclip-b16/swa_2_22.pth
#   /path/to/openvclip/checkpoints/openvclip-l14/swa_2_22.pth
# this script performs zero-shot evaluation on XD-Violence.
#
# To evaluate a fine-tuned or merged checkpoint, replace LOAD_CKPT_FILE
# with the checkpoint path to test.
#
# If you evaluate a ViT-L/14 checkpoint, also replace:
#   configs/Kinetics/TemporalCLIP_vitb16_8x16_STAdapter.yaml
# with:
#   configs/Kinetics/TemporalCLIP_vitl14_8x16_STAdapter.yaml
#
# Make sure DATA, INDEX2CLS_FILE, NUM_CLASSES, and CLIP_ORI_PATH match
# your local dataset layout and checkpoint backbone.

# ====== PARAMS TO EDIT ======
DATASET_NAME=XD-Violence
NUM_CLASSES=7
ROOT=/path/to/robust-ovar/repository
DATA=/path/to/datasets/root
CKPT=/path/to/checkpoints/root
LOG_DIR=/path/to/evaluation/logs/vitb16/standard_h_xdviolence

LOAD_CKPT_FILE=/path/to/finetuned/checkpoints/checkpoint_epoch_00022.pyth
INDEX2CLS_FILE=$DATA/$DATASET_NAME/annotations/XD-Violence-index2cls.json
PATCHING_RATIO=0.0
CLIP_ORI_PATH=/path/to/clip/cache/ViT-B-16.pt
# ============================

export PYTHONPATH="$ROOT/slowfast:$PYTHONPATH"
cd $ROOT

LOG_PREFIX="eval_${DATASET_NAME}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
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
    DATA.DECODING_BACKEND "pyav" \
    MODEL.NUM_CLASSES $NUM_CLASSES \
    TEST.CUSTOM_LOAD True \
    TEST.CUSTOM_LOAD_FILE $LOAD_CKPT_FILE \
    TEST.SAVE_RESULTS_PATH temp.pyth \
    TEST.NUM_ENSEMBLE_VIEWS 3 \
    TEST.NUM_SPATIAL_CROPS 1 \
    TEST.PATCHING_MODEL True \
    TEST.PATCHING_RATIO $PATCHING_RATIO \
    TEST.CLIP_ORI_PATH $CLIP_ORI_PATH \
    > "$LOG_DIR/${LOG_PREFIX}.out" \
    2> "$LOG_DIR/${LOG_PREFIX}.err"
