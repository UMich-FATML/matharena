#!/usr/bin/env bash
#SBATCH --job-name=qwen35-arxiv-sft
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=train/logs/%x-%j.out
#SBATCH --error=train/logs/%x-%j.err

set -euo pipefail
ulimit -c 0 || true
cd "${SLURM_SUBMIT_DIR}"

set -a
source train/configs/grpo_qwen35_2b.env
if [[ -f train/configs/local.env ]]; then
  source train/configs/local.env
fi
set +a

: "${SFT_HF_MODEL_REPO_ID:?Set SFT_HF_MODEL_REPO_ID to the HF model repo to upload}"

export HF_HOME="${HF_HOME:-${SCRATCH}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export EDF_IMAGESTORE="${EDF_IMAGESTORE:-${SCRATCH}/.edf_imagestore}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN}}"
fi

export SFT_BASE_MODEL="${SFT_BASE_MODEL:-${MODEL_PATH:-Qwen/Qwen3.5-2B}}"
export SFT_OUTPUTS_DATASET_ID="${SFT_OUTPUTS_DATASET_ID:-MathArena/arxivmath-training_outputs}"
export SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-${SCRATCH}/checkpoints/qwen35-2b-arxivmath-sft}"
export SFT_RESUME_FROM_CHECKPOINT="${SFT_RESUME_FROM_CHECKPOINT:-}"
export SFT_PRIVATE="${SFT_PRIVATE:-true}"
export SFT_ONLY_CORRECT="${SFT_ONLY_CORRECT:-true}"
export SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-64000}"
export SFT_MAX_SAMPLES="${SFT_MAX_SAMPLES:-}"
export SFT_ENABLE_THINKING="${SFT_ENABLE_THINKING:-${ENABLE_THINKING:-true}}"
export SFT_NPROC_PER_NODE="${SFT_NPROC_PER_NODE:-${SLURM_GPUS_PER_TASK:-4}}"
export SFT_PER_DEVICE_TRAIN_BATCH_SIZE="${SFT_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export SFT_GRADIENT_ACCUMULATION_STEPS="${SFT_GRADIENT_ACCUMULATION_STEPS:-4}"
export SFT_LEARNING_RATE="${SFT_LEARNING_RATE:-1e-5}"
export SFT_NUM_TRAIN_EPOCHS="${SFT_NUM_TRAIN_EPOCHS:-10}"
export SFT_MAX_STEPS="${SFT_MAX_STEPS:--1}"
export SFT_VAL_FRACTION="${SFT_VAL_FRACTION:-0}"
export SFT_REPORT_TO="${SFT_REPORT_TO:-none}"
export SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-50}"
export SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-100}"
export SFT_UPLOAD_CHECKPOINTS="${SFT_UPLOAD_CHECKPOINTS:-true}"
export SFT_ASYNC_CHECKPOINT_UPLOAD="${SFT_ASYNC_CHECKPOINT_UPLOAD:-true}"
export SFT_FSDP="${SFT_FSDP:-}"
export SFT_LOSS_CHUNK_SIZE="${SFT_LOSS_CHUNK_SIZE:-16384}"
export SFT_OPTIM="${SFT_OPTIM:-adamw_torch_fused}"
export SFT_GRADIENT_CHECKPOINTING="${SFT_GRADIENT_CHECKPOINTING:-true}"
export SFT_GRADIENT_CHECKPOINTING_USE_REENTRANT="${SFT_GRADIENT_CHECKPOINTING_USE_REENTRANT:-false}"
export SFT_FREEZE_UNUSED_CHUNKED_PARAMETERS="${SFT_FREEZE_UNUSED_CHUNKED_PARAMETERS:-true}"
export SFT_DDP_FIND_UNUSED_PARAMETERS="${SFT_DDP_FIND_UNUSED_PARAMETERS:-true}"
export SFT_DDP_BROADCAST_BUFFERS="${SFT_DDP_BROADCAST_BUFFERS:-false}"
export SFT_ATTENTION_IMPL="${SFT_ATTENTION_IMPL:-flash_attention_2}"
export SFT_DATALOADER_NUM_WORKERS="${SFT_DATALOADER_NUM_WORKERS:-2}"
export SFT_SEED="${SFT_SEED:-42}"

mkdir -p "${HF_HOME}" "${EDF_IMAGESTORE}" "${SFT_OUTPUT_DIR}" train/logs

srun \
  --mpi=pmix \
  --network=disable_rdzv_get \
  --environment=./train/env/cscs-verl.toml \
  bash -lc '
    set -euo pipefail
    ulimit -c 0 || true
    source "${VENV_DIR}/bin/activate"

    IFS=",: " read -r -a SFT_OUTPUTS_DATASET_IDS <<< "${SFT_OUTPUTS_DATASET_ID}"
    if (( ${#SFT_OUTPUTS_DATASET_IDS[@]} == 0 )); then
      echo "SFT_OUTPUTS_DATASET_ID did not contain any dataset ids." >&2
      exit 1
    fi

    args=(
      --outputs-dataset-id "${SFT_OUTPUTS_DATASET_IDS[@]}"
      --outputs-dataset-split "${SFT_OUTPUTS_DATASET_SPLIT:-train}"
      --base-model "${SFT_BASE_MODEL}"
      --output-dir "${SFT_OUTPUT_DIR}"
      --hub-model-id "${SFT_HF_MODEL_REPO_ID}"
      --private "${SFT_PRIVATE}"
      --only-correct "${SFT_ONLY_CORRECT}"
      --max-length "${SFT_MAX_LENGTH}"
      --enable-thinking "${SFT_ENABLE_THINKING}"
      --per-device-train-batch-size "${SFT_PER_DEVICE_TRAIN_BATCH_SIZE}"
      --gradient-accumulation-steps "${SFT_GRADIENT_ACCUMULATION_STEPS}"
      --learning-rate "${SFT_LEARNING_RATE}"
      --num-train-epochs "${SFT_NUM_TRAIN_EPOCHS}"
      --max-steps "${SFT_MAX_STEPS}"
      --val-fraction "${SFT_VAL_FRACTION}"
      --report-to "${SFT_REPORT_TO}"
      --save-steps "${SFT_SAVE_STEPS}"
      --eval-steps "${SFT_EVAL_STEPS}"
      --upload-checkpoints "${SFT_UPLOAD_CHECKPOINTS}"
      --async-checkpoint-upload "${SFT_ASYNC_CHECKPOINT_UPLOAD}"
      --fsdp "${SFT_FSDP}"
      --loss-chunk-size "${SFT_LOSS_CHUNK_SIZE}"
      --optim "${SFT_OPTIM}"
      --gradient-checkpointing "${SFT_GRADIENT_CHECKPOINTING}"
      --gradient-checkpointing-use-reentrant "${SFT_GRADIENT_CHECKPOINTING_USE_REENTRANT}"
      --freeze-unused-chunked-parameters "${SFT_FREEZE_UNUSED_CHUNKED_PARAMETERS}"
      --ddp-find-unused-parameters "${SFT_DDP_FIND_UNUSED_PARAMETERS}"
      --ddp-broadcast-buffers "${SFT_DDP_BROADCAST_BUFFERS}"
      --attn-implementation "${SFT_ATTENTION_IMPL}"
      --dataloader-num-workers "${SFT_DATALOADER_NUM_WORKERS}"
      --seed "${SFT_SEED}"
    )
    if [[ -n "${SFT_FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP:-}" ]]; then
      args+=(--fsdp-transformer-layer-cls-to-wrap "${SFT_FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP}")
    fi
    if [[ -n "${SFT_FSDP_MIN_NUM_PARAMS:-}" ]]; then
      args+=(--fsdp-min-num-params "${SFT_FSDP_MIN_NUM_PARAMS}")
    fi
    if [[ -n "${SFT_MAX_SAMPLES}" ]]; then
      args+=(--max-samples "${SFT_MAX_SAMPLES}")
    fi
    if [[ -n "${SFT_RESUME_FROM_CHECKPOINT}" ]]; then
      args+=(--resume-from-checkpoint "${SFT_RESUME_FROM_CHECKPOINT}")
    fi

    echo "SFT long-context run: datasets=${SFT_OUTPUTS_DATASET_IDS[*]} max_length=${SFT_MAX_LENGTH} nproc_per_node=${SFT_NPROC_PER_NODE} fsdp=${SFT_FSDP:-ddp} loss_chunk_size=${SFT_LOSS_CHUNK_SIZE} grad_accum=${SFT_GRADIENT_ACCUMULATION_STEPS} val_fraction=${SFT_VAL_FRACTION} save_steps=${SFT_SAVE_STEPS} resume_from_checkpoint=${SFT_RESUME_FROM_CHECKPOINT:-none} upload_checkpoints=${SFT_UPLOAD_CHECKPOINTS} async_upload=${SFT_ASYNC_CHECKPOINT_UPLOAD} freeze_unused=${SFT_FREEZE_UNUSED_CHUNKED_PARAMETERS} ddp_find_unused=${SFT_DDP_FIND_UNUSED_PARAMETERS} ddp_broadcast_buffers=${SFT_DDP_BROADCAST_BUFFERS}"
    python -m torch.distributed.run \
      --standalone \
      --nnodes=1 \
      --nproc_per_node="${SFT_NPROC_PER_NODE}" \
      train/scripts/sft_from_outputs.py "${args[@]}"
  '
