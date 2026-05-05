#!/bin/bash
#
# Fine-tuning script for Open-VCLIP on UCF101.
#
# This script fine-tunes a model initialized from LOAD_CKPT_FILE and saves the
# resulting checkpoint under OUTPUT_SUBDIR.
#
# Usage:
# 1. Edit the placeholders in the "PARAMS TO EDIT" section.
# 2. Run the script with:
#      bash scripts/training/ft-UCF101-reg0.sh
# 3. For ViT-L/14, also replace the config file, LOAD_CKPT_FILE, and
#    TRAIN.CLIP_ORI_PATH with ViT-L/14-compatible paths.
#
# Batch size depends on GPU memory, backbone, number of frames/views, and video
# resolution. Lower TRAIN.BATCH_SIZE and TEST.BATCH_SIZE if you get OOM errors.
#
# ====== PARAMS TO EDIT ======
DATASETNAME=ucf101
NUM_CLASSES=101
ROOT=/path/to/robust-ovar/repository
DATA=/path/to/datasets/root
CKPT=/path/to/checkpoints/root
LOAD_CKPT_FILE=/path/to/openvclip/checkpoints/openvclip-b16/swa_2_22.pth

ANNOTATION_DIR=$DATA/$DATASETNAME/annotations
VIDEO_DIR=$DATA/$DATASETNAME/videos
INDEX2CLS_FILE=$ANNOTATION_DIR/ucf101-index2cls.json
OUTPUT_SUBDIR=finetuning_zeroshot/ft-ovclip-standard-ucf101
PATCHING_RATIO=0.0
# ============================

export PYTHONPATH="$ROOT/slowfast:$PYTHONPATH"

cd "$ROOT"

python -W ignore -u tools/run_net.py \
  --cfg configs/Kinetics/TemporalCLIP_vitb16_8x16_STAdapter.yaml \
  --opts \
  DATA.PATH_TO_DATA_DIR $ANNOTATION_DIR \
  DATA.PATH_PREFIX $VIDEO_DIR \
  DATA.PATH_LABEL_SEPARATOR , \
  DATA.INDEX_LABEL_MAPPING_FILE $INDEX2CLS_FILE \
  TRAIN.ENABLE True \
  TRAIN.AUTO_RESUME False \
  OUTPUT_DIR $CKPT/$OUTPUT_SUBDIR \
  TRAIN.BATCH_SIZE 2 \
  TEST.BATCH_SIZE 2 \
  TEST.NUM_ENSEMBLE_VIEWS 3 \
  TEST.NUM_SPATIAL_CROPS 1 \
  NUM_GPUS 1 \
  SOLVER.MAX_EPOCH 22 \
  SOLVER.WARMUP_EPOCHS 2.0 \
  SOLVER.BASE_LR 3.33e-6 \
  SOLVER.WARMUP_START_LR 3.33e-8 \
  SOLVER.COSINE_END_LR 3.33e-8 \
  TRAIN.MIXED_PRECISION True \
  DATA.DECODING_BACKEND "pyav" \
  MODEL.NUM_CLASSES $NUM_CLASSES \
  MODEL.TEMPORAL_MODELING_TYPE 'expand_temporal_view' \
  MIXUP.ENABLE False \
  AUG.ENABLE False \
  AUG.NUM_SAMPLE 1 \
  TRAIN.EVAL_PERIOD 1 \
  TRAIN.CHECKPOINT_PERIOD 22 \
  MODEL.LOSS_FUNC soft_cross_entropy \
  TRAIN.LINEAR_CONNECT_CLIMB False \
  TRAIN.CLIP_ORI_PATH /path/to/clip/cache/ViT-B-16.pt \
  TRAIN.LINEAR_CONNECT_LOSS_RATIO 0.5 \
  TRAIN.LINEAR_CONNECT_SAMPLE_L 0.0 \
  TRAIN.LINEAR_CONNECT_SAMPLE_R 0.6 \
  TRAIN.CUSTOM_LOAD True \
  TRAIN.CUSTOM_LOAD_FILE $LOAD_CKPT_FILE \
