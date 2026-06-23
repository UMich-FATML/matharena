#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0 || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${TRAIN_ROOT}/configs/grpo_qwen35_2b.env}"

if [[ "${SOURCE_CONFIG:-true}" == "true" && -f "${CONFIG_FILE}" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${CONFIG_FILE}"
  set +a
fi
if [[ -f "${TRAIN_ROOT}/configs/local.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${TRAIN_ROOT}/configs/local.env"
  set +a
fi

: "${SCRATCH:?SCRATCH must be set on CSCS}"

export HF_HOME="${HF_HOME:-${SCRATCH}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
export TORCH_CUDNN_V8_API_DISABLED="${TORCH_CUDNN_V8_API_DISABLED:-1}"
unset VLLM_USE_V1
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
  echo "Unsetting PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}; vLLM memory pool is incompatible with expandable_segments." >&2
  unset PYTORCH_CUDA_ALLOC_CONF
fi
VLLM_ATTENTION_BACKEND_VALUE="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
unset VLLM_VERSION VLLM_FLASH_ATTN_SRC_DIR VLLM_ATTENTION_BACKEND
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-${SLURM_JOB_ID:-manual}}"
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO="${RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO:-0}"
export PYTHONPATH="${TRAIN_ROOT}/../src:${PYTHONPATH:-}"
unset HIP_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN}}"
fi

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-2B}"
DATA_DIR="${DATA_DIR:-${SCRATCH}/matharena_arxivmath_verl}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH}/checkpoints/matharena-qwen35-2b-grpo-thinking16k}"
REWARD_FN="${REWARD_FN:-${TRAIN_ROOT}/scripts/arxivmath_reward.py}"
export OUTPUT_DIR
export HF_MODEL_REPO_ID="${HF_MODEL_REPO_ID:-MathArena/qwen3.5-2b-arxivmath-grpo}"
export HF_EXPORT_DIR="${HF_EXPORT_DIR:-${SCRATCH}/hf_exports/qwen35-2b-arxivmath-grpo-thinking16k}"
export HF_MODEL_PRIVATE="${HF_MODEL_PRIVATE:-true}"

mkdir -p "${HF_HOME}" "${RAY_TMPDIR}" "${OUTPUT_DIR}"

if [[ ! -f "${DATA_DIR}/train.parquet" || ! -f "${DATA_DIR}/val.parquet" ]]; then
  echo "Missing veRL parquet files in ${DATA_DIR}." >&2
  echo "Run: sbatch train/slurm/01_prepare_arxivmath_data.sbatch" >&2
  exit 1
fi

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
ROLLOUT_N="${ROLLOUT_N:-4}"
TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-4}"
TRAINER_NNODES="${TRAINER_NNODES:-${SLURM_JOB_NUM_NODES:-1}}"
ENABLE_THINKING="${ENABLE_THINKING:-true}"
MAX_PROMPT_LENGTH_VALUE="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH_VALUE="${MAX_RESPONSE_LENGTH:-16384}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH_VALUE + MAX_RESPONSE_LENGTH_VALUE))}"
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-1}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-${ROLLOUT_MAX_MODEL_LEN}}"
LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-${ROLLOUT_MAX_MODEL_LEN}}"
ENABLE_ACTIVATION_OFFLOAD="${ENABLE_ACTIVATION_OFFLOAD:-false}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
RESUME_MODE="${RESUME_MODE:-disable}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-null}"
if (( TRAINER_N_GPUS_PER_NODE < 1 )); then
  echo "TRAINER_N_GPUS_PER_NODE must be positive, got ${TRAINER_N_GPUS_PER_NODE}" >&2
  exit 1
fi
if (( TRAINER_NNODES < 1 )); then
  echo "TRAINER_NNODES must be positive, got ${TRAINER_NNODES}" >&2
  exit 1
fi
TOTAL_TRAINER_GPUS=$((TRAINER_NNODES * TRAINER_N_GPUS_PER_NODE))
ROLLOUT_BATCH_SIZE=$((TRAIN_BATCH_SIZE * ROLLOUT_N))
if (( ROLLOUT_BATCH_SIZE % TOTAL_TRAINER_GPUS != 0 )); then
  echo "TRAIN_BATCH_SIZE * ROLLOUT_N must divide across GPUs: ${ROLLOUT_BATCH_SIZE} % ${TOTAL_TRAINER_GPUS} != 0" >&2
  exit 1
fi
PER_RANK_ACTOR_BATCH=$((ROLLOUT_BATCH_SIZE / TOTAL_TRAINER_GPUS))
SAFE_PPO_MINI_BATCH_SIZE="${TRAIN_BATCH_SIZE}"
if [[ -n "${PPO_MINI_BATCH_SIZE_OVERRIDE:-}" ]]; then
  PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE_OVERRIDE}"
elif [[ -n "${PPO_MINI_BATCH_SIZE:-}" && "${PPO_MINI_BATCH_SIZE}" != "${SAFE_PPO_MINI_BATCH_SIZE}" ]]; then
  echo "Ignoring PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE}; using safe prompt-level value ${SAFE_PPO_MINI_BATCH_SIZE}. Set PPO_MINI_BATCH_SIZE_OVERRIDE to override intentionally." >&2
  PPO_MINI_BATCH_SIZE="${SAFE_PPO_MINI_BATCH_SIZE}"
else
  PPO_MINI_BATCH_SIZE="${SAFE_PPO_MINI_BATCH_SIZE}"
fi
PPO_MINI_BATCH_EXPANDED=$((PPO_MINI_BATCH_SIZE * ROLLOUT_N))
if (( PPO_MINI_BATCH_SIZE < 1 || TRAIN_BATCH_SIZE % PPO_MINI_BATCH_SIZE != 0 )); then
  echo "PPO mini-batch must divide the prompt batch: ${TRAIN_BATCH_SIZE} % ${PPO_MINI_BATCH_SIZE} != 0" >&2
  exit 1
fi
if (( PPO_MINI_BATCH_EXPANDED % TOTAL_TRAINER_GPUS != 0 )); then
  echo "PPO mini-batch after rollout expansion must divide across GPUs: ${PPO_MINI_BATCH_EXPANDED} % ${TOTAL_TRAINER_GPUS} != 0" >&2
  exit 1
fi
PPO_MINI_BATCH_PER_GPU=$((PPO_MINI_BATCH_EXPANDED / TOTAL_TRAINER_GPUS))
if (( PPO_MINI_BATCH_PER_GPU < 1 || PER_RANK_ACTOR_BATCH % PPO_MINI_BATCH_PER_GPU != 0 )); then
  echo "PPO mini-batch per GPU must divide the per-rank actor batch: ${PER_RANK_ACTOR_BATCH} % ${PPO_MINI_BATCH_PER_GPU} != 0" >&2
  exit 1
fi
export TRAIN_BATCH_SIZE VAL_BATCH_SIZE ROLLOUT_N TRAINER_NNODES TRAINER_N_GPUS_PER_NODE PPO_MINI_BATCH_SIZE

echo "GRPO geometry: train_batch_size=${TRAIN_BATCH_SIZE} rollout_n=${ROLLOUT_N} rollout_batch_size=${ROLLOUT_BATCH_SIZE} nnodes=${TRAINER_NNODES} n_gpus_per_node=${TRAINER_N_GPUS_PER_NODE} total_gpus=${TOTAL_TRAINER_GPUS} per_rank_actor_batch=${PER_RANK_ACTOR_BATCH} ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} ppo_mini_batch_expanded=${PPO_MINI_BATCH_EXPANDED} ppo_mini_batch_per_gpu=${PPO_MINI_BATCH_PER_GPU} ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
echo "Qwen thinking mode: enable_thinking=${ENABLE_THINKING} max_response_length=${MAX_RESPONSE_LENGTH_VALUE} rollout_max_model_len=${ROLLOUT_MAX_MODEL_LEN} max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-18432} max_num_seqs=${MAX_NUM_SEQS:-16} val_before_train=${VAL_BEFORE_TRAIN} resume_mode=${RESUME_MODE}"
echo "Actor memory: ulysses_sequence_parallel_size=${ULYSSES_SEQUENCE_PARALLEL_SIZE} ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU} activation_offload=${ENABLE_ACTIVATION_OFFLOAD} param_offload=${ACTOR_PARAM_OFFLOAD:-false} optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD:-false} pytorch_cuda_alloc_conf=${PYTORCH_CUDA_ALLOC_CONF:-unset}"
echo "Actor/ref model dtype: ${ACTOR_MODEL_DTYPE:-bf16}"

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_penalty=low_var_kl \
  data.train_files="${DATA_DIR}/train.parquet" \
  data.val_files="${DATA_DIR}/val.parquet" \
  data.prompt_key=prompt \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.train_max_samples="${TRAIN_MAX_SAMPLES:--1}" \
  data.val_max_samples="${VAL_MAX_SAMPLES:--1}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH_VALUE}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH_VALUE}" \
  data.filter_overlong_prompts=True \
  data.truncation=left \
  data.trust_remote_code=True \
  ++data.apply_chat_template_kwargs.enable_thinking="${ENABLE_THINKING}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  ++actor_rollout_ref.model.enable_activation_offload="${ENABLE_ACTIVATION_OFFLOAD}" \
  actor_rollout_ref.actor.optim.lr="${LR:-1e-6}" \
  actor_rollout_ref.actor.optim.lr_warmup_steps="${LR_WARMUP_STEPS:-10}" \
  actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
  actor_rollout_ref.actor.optim.weight_decay="${WEIGHT_DECAY:-0.1}" \
  actor_rollout_ref.actor.optim.clip_grad="${CLIP_GRAD:-1.0}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-64}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF:-0.001}" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  ++actor_rollout_ref.actor.ulysses_sequence_parallel_size="${ULYSSES_SEQUENCE_PARALLEL_SIZE}" \
  actor_rollout_ref.actor.fsdp_config.model_dtype="${ACTOR_MODEL_DTYPE:-bf16}" \
  actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD:-false}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD:-false}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.50}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE:-1.0}" \
  actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P:-0.95}" \
  actor_rollout_ref.rollout.top_k="${ROLLOUT_TOP_K:-20}" \
  actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH_VALUE}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-18432}" \
  actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS:-16}" \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.enforce_eager=True \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend="${VLLM_ATTENTION_BACKEND_VALUE}" \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMPERATURE:-1.0}" \
  actor_rollout_ref.rollout.val_kwargs.top_p="${VAL_TOP_P:-0.95}" \
  actor_rollout_ref.rollout.val_kwargs.top_k="${VAL_TOP_K:-20}" \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}" \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}" \
  ++actor_rollout_ref.ref.ulysses_sequence_parallel_size="${ULYSSES_SEQUENCE_PARALLEL_SIZE}" \
  actor_rollout_ref.ref.fsdp_config.model_dtype="${ACTOR_MODEL_DTYPE:-bf16}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  reward_model.enable=False \
  reward_model.reward_manager=naive \
  custom_reward_function.path="${REWARD_FN}" \
  custom_reward_function.name=compute_score \
  trainer.critic_warmup=0 \
  trainer.logger="${LOGGER_BACKENDS:-['console']}" \
  trainer.project_name="${PROJECT_NAME:-matharena-arxivmath-grpo}" \
  trainer.experiment_name="${EXPERIMENT_NAME:-qwen35-2b-grpo-thinking16k-gh200}" \
  trainer.n_gpus_per_node="${TRAINER_N_GPUS_PER_NODE}" \
  trainer.nnodes="${TRAINER_NNODES}" \
  trainer.save_freq="${SAVE_FREQ:-10}" \
  trainer.test_freq="${TEST_FREQ:-5}" \
  trainer.total_epochs="${TOTAL_EPOCHS:-5}" \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.resume_mode="${RESUME_MODE}" \
  trainer.resume_from_path="${RESUME_FROM_PATH}" \
  trainer.log_val_generations="${LOG_VAL_GENERATIONS:-8}" \
  trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
  "$@"

if [[ "${UPLOAD_MODEL_TO_HF:-true}" == "true" ]]; then
  bash "${TRAIN_ROOT}/scripts/export_and_upload_hf_model.sh"
fi
