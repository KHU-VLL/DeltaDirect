#!/bin/bash
# LLaVA-Video-7B + DeltaDirect on MoDirect-Inst
MODEL=lmms-lab/LLaVA-Video-7B-Qwen2
DATA_YAML=scripts/train/configs/data/modirect_inst.yaml
VIDEO_FOLDER=data/MoDirect/MoDirect-Inst

LR=1e-5
EPOCHS=1
FRAMES=8
NUM_GPUS=8
BATCH_SIZE=9
GRAD_ACCUM=2
DS_CONFIG=scripts/deepspeed/zero2.json

MM_TUNABLE="mm_mlp_adapter,mm_language_model"
LORA_ENABLE=true
LORA_R=64
LORA_ALPHA=128
LORA_DROPOUT=0.05

USE_DELTA_DIRECT=true
DELTA_DIRECT_LAMBDA=1.0

VERSION=qwen_1_5
ADD_TIME_INSTRUCTION=false

EXP_NAME="llava-video-7b-qwen2_deltadirect_modirect-inst_lora-r${LORA_R}_f${FRAMES}_ep${EPOCHS}_lr${LR}_bs${BATCH_SIZE}_ga${GRAD_ACCUM}"
WORK_DIR="work_dirs/${EXP_NAME}"
WANDB_PROJECT="DeltaDirect"
