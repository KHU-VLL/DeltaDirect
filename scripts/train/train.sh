#!/bin/bash
# Usage: bash scripts/train/train.sh scripts/train/configs/llava_video_7b_deltadirect.sh

set -euo pipefail

CONFIG_FILE="${1:?Usage: $0 <config.sh>}"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: config file not found: $CONFIG_FILE"
    exit 1
fi
source "$CONFIG_FILE"

REQUIRED_VARS=(
    EXP_NAME MODEL DATA_YAML VIDEO_FOLDER
    LR EPOCHS FRAMES NUM_GPUS BATCH_SIZE GRAD_ACCUM
    MM_TUNABLE DS_CONFIG VERSION ADD_TIME_INSTRUCTION USE_DELTA_DIRECT
)
for var in "${REQUIRED_VARS[@]}"; do
    if [[ -z "${!var:-}" ]]; then
        echo "ERROR: required variable '$var' is not set in $CONFIG_FILE"
        exit 1
    fi
done

EFFECTIVE_BATCH=$((NUM_GPUS * BATCH_SIZE * GRAD_ACCUM))
WORK_DIR="${WORK_DIR:-work_dirs/${EXP_NAME}}"
mkdir -p "${WORK_DIR}"
cp "$CONFIG_FILE" "${WORK_DIR}/"

export WANDB_PROJECT="${WANDB_PROJECT:-DeltaDirect}"
export WANDB_RUN_NAME="${EXP_NAME}"

echo "================= ${EXP_NAME} ================="
echo "  Model:        ${MODEL}"
echo "  Data:         ${DATA_YAML}"
echo "  DeltaDirect:  ${USE_DELTA_DIRECT}"
echo "  LR: ${LR} | epochs: ${EPOCHS} | frames: ${FRAMES} | effective batch size: ${EFFECTIVE_BATCH}"
echo "  Work dir:     ${WORK_DIR}"
echo "==============================================="

CMD=(
    deepspeed
    --num_nodes 1
    --num_gpus "${NUM_GPUS}"
    --master_port "${MASTER_PORT:-51310}"
    llava/train/train_mem.py
    --deepspeed "${DS_CONFIG}"
    --model_name_or_path "${MODEL}"
    --version "${VERSION}"
    --data_path "${DATA_YAML}"
    --image_folder "${IMAGE_FOLDER:-./}"
    --video_folder "${VIDEO_FOLDER}"
    --mm_tunable_parts="${MM_TUNABLE}"
    --vision_tower "${VISION_TOWER:-google/siglip-so400m-patch14-384}"
    --mm_projector_type "${PROJECTOR_TYPE:-mlp2x_gelu}"
    --mm_vision_select_layer -2
    --mm_use_im_start_end False
    --mm_use_im_patch_token False
    --group_by_modality_length True
    --image_aspect_ratio anyres_max_9
    --image_grid_pinpoints "(1x1),...,(6x6)"
    --mm_patch_merge_type spatial_unpad
    --mm_spatial_pool_mode "${POOL_MODE:-bilinear}"
    --mm_spatial_pool_stride "${POOL_STRIDE:-2}"
    --mm_newline_position "${MM_NEWLINE_POSITION:-grid}"
    --bf16 True
    --run_name "${EXP_NAME}"
    --output_dir "${WORK_DIR}"
    --num_train_epochs "${EPOCHS}"
    --per_device_train_batch_size "${BATCH_SIZE}"
    --per_device_eval_batch_size 1
    --gradient_accumulation_steps "${GRAD_ACCUM}"
    --evaluation_strategy no
    --save_strategy "${SAVE_STRATEGY:-steps}"
    --save_steps "${SAVE_STEPS:-700}"
    --save_total_limit "${SAVE_TOTAL_LIMIT:-4}"
    --learning_rate "${LR}"
    --weight_decay 0.
    --warmup_ratio "${WARMUP_RATIO:-0.03}"
    --lr_scheduler_type cosine
    --logging_steps 1
    --tf32 True
    --model_max_length 32768
    --gradient_checkpointing True
    --dataloader_num_workers "${NUM_WORKERS:-4}"
    --lazy_preprocess True
    --report_to "${REPORT_TO:-wandb}"
    --dataloader_drop_last True
    --frames_upbound "${FRAMES}"
    --add_time_instruction "${ADD_TIME_INSTRUCTION}"
    --force_sample True
)

# LoRA
if [[ "${LORA_ENABLE:-false}" == "true" ]]; then
    CMD+=(
        --lora_enable True
        --lora_r "${LORA_R:-64}"
        --lora_alpha "${LORA_ALPHA:-128}"
        --lora_dropout "${LORA_DROPOUT:-0.05}"
    )
fi

# DeltaDirect
if [[ "${USE_DELTA_DIRECT}" == "true" ]]; then
    CMD+=(
        --use_delta_direct True
        --delta_direct_lambda "${DELTA_DIRECT_LAMBDA:-1.0}"
    )
fi

if [[ -n "${EXTRA_ARGS:-}" ]]; then
    CMD+=(${EXTRA_ARGS})
fi

"${CMD[@]}" 2>&1 | tee -a "${WORK_DIR}/train.log"
