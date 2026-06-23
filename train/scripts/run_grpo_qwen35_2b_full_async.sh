#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0 || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${TRAIN_ROOT}/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${TRAIN_ROOT}/configs/grpo_qwen35_2b.env}"

if [[ "${SOURCE_CONFIG:-true}" == "true" ]]; then
  if [[ -f "${CONFIG_FILE}" ]]; then
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
  echo "For actor backward OOMs, prefer ENABLE_ACTIVATION_OFFLOAD=true or lower MAX_RESPONSE_LENGTH." >&2
  unset PYTORCH_CUDA_ALLOC_CONF
fi
VLLM_ATTENTION_BACKEND_VALUE="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
unset VLLM_VERSION VLLM_FLASH_ATTN_SRC_DIR VLLM_ATTENTION_BACKEND
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-${SLURM_JOB_ID:-manual}}"
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO="${RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO:-0}"
export VERL_SRC="${VERL_SRC:-${SCRATCH}/src/verl}"
export PYTHONPATH="${REPO_ROOT}/src:${VERL_SRC}:${PYTHONPATH:-}"
unset HIP_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN}}"
fi

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-2B}"
DATA_DIR="${DATA_DIR:-${SCRATCH}/matharena_arxivmath_verl}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH}/checkpoints/matharena-qwen35-2b-grpo-thinking16k-fullasync}"
REWARD_FN="${REWARD_FN:-${TRAIN_ROOT}/scripts/arxivmath_reward.py}"
export OUTPUT_DIR
export HF_MODEL_REPO_ID="${HF_MODEL_REPO_ID:-MathArena/qwen3.5-2b-arxivmath-grpo}"
export HF_EXPORT_DIR="${HF_EXPORT_DIR:-${SCRATCH}/hf_exports/qwen35-2b-arxivmath-grpo-thinking16k-fullasync}"
export HF_MODEL_PRIVATE="${HF_MODEL_PRIVATE:-true}"

mkdir -p "${HF_HOME}" "${RAY_TMPDIR}" "${OUTPUT_DIR}"

if [[ "${FULL_ASYNC_GEN_BATCH_SIZE:-1}" != "1" ]]; then
  cat >&2 <<EOF
FULL_ASYNC_GEN_BATCH_SIZE=${FULL_ASYNC_GEN_BATCH_SIZE} is invalid for this veRL fully-async rollouter.
verl.experimental.fully_async_policy.fully_async_rollouter asserts data.gen_batch_size == 1.
Use rollout concurrency knobs instead: MAX_NUM_SEQS, MAX_NUM_BATCHED_TOKENS,
FULL_ASYNC_REQUIRE_BATCHES, FULL_ASYNC_STALENESS_THRESHOLD, and rollout nodes/replicas.
EOF
  exit 2
fi

if [[ ! -f "${DATA_DIR}/train.parquet" || ! -f "${DATA_DIR}/val.parquet" ]]; then
  echo "Missing veRL parquet files in ${DATA_DIR}." >&2
  echo "Run: sbatch train/slurm/01_prepare_arxivmath_data.sbatch" >&2
  exit 1
fi

FULL_ASYNC_ENTRYPOINT="$(
  python - "${FULL_ASYNC_ENTRYPOINT:-}" <<'PY'
import importlib.util
import sys

candidates = [arg for arg in sys.argv[1:] if arg]
candidates += [
    "recipe.fully_async_policy.fully_async_main",
    "verl.experimental.fully_async_policy.fully_async_main",
]
for name in candidates:
    try:
        spec = importlib.util.find_spec(name)
    except Exception:
        spec = None
    if spec is not None:
        print(name)
        raise SystemExit(0)
raise SystemExit(1)
PY
)" || {
  cat >&2 <<EOF
Could not find a veRL full-async entrypoint.
Tried:
  recipe.fully_async_policy.fully_async_main
  verl.experimental.fully_async_policy.fully_async_main

If recipe.* is missing, rerun setup after initializing veRL recipes:
  git -C "\${VERL_SRC:-${SCRATCH}/src/verl}" submodule update --init --recursive recipe
EOF
  exit 1
}

TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-4}"
FULL_ASYNC_TRAINER_NNODES="${FULL_ASYNC_TRAINER_NNODES:-1}"
FULL_ASYNC_ROLLOUT_NNODES="${FULL_ASYNC_ROLLOUT_NNODES:-1}"
FULL_ASYNC_ROLLOUT_N_GPUS_PER_NODE="${FULL_ASYNC_ROLLOUT_N_GPUS_PER_NODE:-${TRAINER_N_GPUS_PER_NODE}}"
FULL_ASYNC_GEN_BATCH_SIZE="${FULL_ASYNC_GEN_BATCH_SIZE:-1}"
FULL_ASYNC_TRAIN_BATCH_SIZE="${FULL_ASYNC_TRAIN_BATCH_SIZE:-0}"
FULL_ASYNC_REQUIRE_BATCHES="${FULL_ASYNC_REQUIRE_BATCHES:-1}"
FULL_ASYNC_STALENESS_THRESHOLD="${FULL_ASYNC_STALENESS_THRESHOLD:-0.5}"
FULL_ASYNC_PARTIAL_ROLLOUT="${FULL_ASYNC_PARTIAL_ROLLOUT:-true}"
FULL_ASYNC_USE_TRAINER_DO_VALIDATE="${FULL_ASYNC_USE_TRAINER_DO_VALIDATE:-true}"
FULL_ASYNC_ACTOR_STRATEGY="${FULL_ASYNC_ACTOR_STRATEGY:-fsdp}"
FULL_ASYNC_ROLLOUT_MODE="${FULL_ASYNC_ROLLOUT_MODE:-async}"
FULL_ASYNC_ROLLOUT_NAME="${FULL_ASYNC_ROLLOUT_NAME:-vllm}"
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-true}"
RETURN_MULTI_MODAL_INPUTS="${RETURN_MULTI_MODAL_INPUTS:-false}"
FULL_ASYNC_CHECKPOINT_ENGINE_BACKEND="${FULL_ASYNC_CHECKPOINT_ENGINE_BACKEND:-nccl}"
FULL_ASYNC_CHECKPOINT_ENGINE_MODULE="${FULL_ASYNC_CHECKPOINT_ENGINE_MODULE:-verl.checkpoint_engine.nccl_checkpoint_engine}"

ROLLOUT_N="${ROLLOUT_N:-4}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
TRAIN_BATCH_SIZE_REFERENCE="${FULL_ASYNC_REFERENCE_TRAIN_BATCH_SIZE:-$((16 * FULL_ASYNC_TRAINER_NNODES))}"
FULL_ASYNC_TRIGGER_PARAMETER_SYNC_STEP="${FULL_ASYNC_TRIGGER_PARAMETER_SYNC_STEP:-$((TRAIN_BATCH_SIZE_REFERENCE / (FULL_ASYNC_REQUIRE_BATCHES * PPO_MINI_BATCH_SIZE)))}"
if (( FULL_ASYNC_TRIGGER_PARAMETER_SYNC_STEP < 1 )); then
  FULL_ASYNC_TRIGGER_PARAMETER_SYNC_STEP=1
fi

MAX_PROMPT_LENGTH_VALUE="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH_VALUE="${MAX_RESPONSE_LENGTH:-16384}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH_VALUE + MAX_RESPONSE_LENGTH_VALUE))}"
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-1}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-${ROLLOUT_MAX_MODEL_LEN}}"
LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-${ROLLOUT_MAX_MODEL_LEN}}"
if (( ULYSSES_SEQUENCE_PARALLEL_SIZE > 1 )); then
  cat >&2 <<EOF
ULYSSES_SEQUENCE_PARALLEL_SIZE=${ULYSSES_SEQUENCE_PARALLEL_SIZE} is unsafe for this veRL full-async Qwen job.
The ref/old-log-prob path applies temperature after remove-padding, but with Ulysses SP the logits and temperature metadata have different token counts.
The first trainer step fails with a ${ULYSSES_SEQUENCE_PARALLEL_SIZE}x tensor-size mismatch in verl/workers/engine/fsdp/transformer_impl.py.
Use ULYSSES_SEQUENCE_PARALLEL_SIZE=1; if memory is tight, reduce MAX_RESPONSE_LENGTH or enable offload instead.
EOF
  exit 2
fi
if (( PPO_MAX_TOKEN_LEN_PER_GPU < ROLLOUT_MAX_MODEL_LEN )); then
  echo "Raising PPO_MAX_TOKEN_LEN_PER_GPU from ${PPO_MAX_TOKEN_LEN_PER_GPU} to ${ROLLOUT_MAX_MODEL_LEN} to cover max_prompt_length + max_response_length." >&2
  PPO_MAX_TOKEN_LEN_PER_GPU="${ROLLOUT_MAX_MODEL_LEN}"
fi
if (( LOG_PROB_MAX_TOKEN_LEN_PER_GPU < ROLLOUT_MAX_MODEL_LEN )); then
  echo "Raising LOG_PROB_MAX_TOKEN_LEN_PER_GPU from ${LOG_PROB_MAX_TOKEN_LEN_PER_GPU} to ${ROLLOUT_MAX_MODEL_LEN} to cover max_prompt_length + max_response_length." >&2
  LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${ROLLOUT_MAX_MODEL_LEN}"
fi
ENABLE_ACTIVATION_OFFLOAD="${ENABLE_ACTIVATION_OFFLOAD:-false}"
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-false}"
ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-false}"
REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-false}"
AUTO_ENABLE_LONG_CONTEXT_MEMORY_SAFETY="${AUTO_ENABLE_LONG_CONTEXT_MEMORY_SAFETY:-true}"
if (( MAX_RESPONSE_LENGTH_VALUE >= 32768 )) && [[ "${AUTO_ENABLE_LONG_CONTEXT_MEMORY_SAFETY,,}" == "true" ]]; then
  if [[ "${ENABLE_ACTIVATION_OFFLOAD,,}" == "false" ]]; then
    echo "Enabling activation offload for MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH_VALUE}; 32k actor backward does not reliably fit without Ulysses SP." >&2
    ENABLE_ACTIVATION_OFFLOAD=true
  fi
  if [[ "${ACTOR_PARAM_OFFLOAD,,}" == "false" ]]; then
    echo "Enabling actor parameter offload for MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH_VALUE}." >&2
    ACTOR_PARAM_OFFLOAD=true
  fi
  if [[ "${ACTOR_OPTIMIZER_OFFLOAD,,}" == "false" ]]; then
    echo "Enabling actor optimizer offload for MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH_VALUE}." >&2
    ACTOR_OPTIMIZER_OFFLOAD=true
  fi
  if [[ "${REF_PARAM_OFFLOAD,,}" == "false" ]]; then
    echo "Enabling reference parameter offload for MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH_VALUE}." >&2
    REF_PARAM_OFFLOAD=true
  fi
  if [[ "${ROLLOUT_ENFORCE_EAGER,,}" == "false" ]]; then
    echo "Forcing ROLLOUT_ENFORCE_EAGER=true for MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH_VALUE} to avoid vLLM CUDA graph memory on shared GPUs." >&2
    ROLLOUT_ENFORCE_EAGER=true
  fi
  if [[ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ]]; then
    export PYTORCH_CUDA_ALLOC_CONF="${LONG_CONTEXT_PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"
    echo "Setting PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF} for long-context actor backward fragmentation." >&2
  fi
  echo "Set AUTO_ENABLE_LONG_CONTEXT_MEMORY_SAFETY=false to keep the explicit memory settings unchanged." >&2
fi
ENABLE_THINKING="${ENABLE_THINKING:-true}"
RESUME_MODE="${RESUME_MODE:-disable}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-null}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"

if [[ -z "${FULL_ASYNC_TOTAL_ROLLOUT_STEPS:-}" ]]; then
  FULL_ASYNC_TOTAL_ROLLOUT_STEPS="$(
    TRAIN_PARQUET="${DATA_DIR}/train.parquet" python - <<'PY'
import os
import pyarrow.parquet as pq

rows = pq.ParquetFile(os.environ["TRAIN_PARQUET"]).metadata.num_rows
max_samples = int(os.environ.get("TRAIN_MAX_SAMPLES", "-1"))
if max_samples > 0:
    rows = min(rows, max_samples)
epochs = int(os.environ.get("TOTAL_EPOCHS", "5"))
print(rows * epochs)
PY
  )"
fi

TRAINER_GPUS=$((FULL_ASYNC_TRAINER_NNODES * TRAINER_N_GPUS_PER_NODE))
ROLLOUT_GPUS=$((FULL_ASYNC_ROLLOUT_NNODES * FULL_ASYNC_ROLLOUT_N_GPUS_PER_NODE))
TOTAL_ASYNC_GPUS=$((TRAINER_GPUS + ROLLOUT_GPUS))

echo "Full-async entrypoint: ${FULL_ASYNC_ENTRYPOINT}"
echo "Full-async resources: trainer_nodes=${FULL_ASYNC_TRAINER_NNODES} rollout_nodes=${FULL_ASYNC_ROLLOUT_NNODES} trainer_gpus=${TRAINER_GPUS} rollout_gpus=${ROLLOUT_GPUS} total_gpus=${TOTAL_ASYNC_GPUS}"
echo "Full-async policy: rollout_n=${ROLLOUT_N} ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} require_batches=${FULL_ASYNC_REQUIRE_BATCHES} trigger_parameter_sync_step=${FULL_ASYNC_TRIGGER_PARAMETER_SYNC_STEP} staleness_threshold=${FULL_ASYNC_STALENESS_THRESHOLD} partial_rollout=${FULL_ASYNC_PARTIAL_ROLLOUT} total_rollout_steps=${FULL_ASYNC_TOTAL_ROLLOUT_STEPS}"
echo "Qwen thinking mode: enable_thinking=${ENABLE_THINKING} max_response_length=${MAX_RESPONSE_LENGTH_VALUE} rollout_max_model_len=${ROLLOUT_MAX_MODEL_LEN} max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-18432} max_num_seqs=${MAX_NUM_SEQS:-16} rollout_enforce_eager=${ROLLOUT_ENFORCE_EAGER}"
echo "Input modality: return_multi_modal_inputs=${RETURN_MULTI_MODAL_INPUTS}"
echo "Actor memory: ulysses_sequence_parallel_size=${ULYSSES_SEQUENCE_PARALLEL_SIZE} ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU} activation_offload=${ENABLE_ACTIVATION_OFFLOAD} actor_param_offload=${ACTOR_PARAM_OFFLOAD} actor_optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD} ref_param_offload=${REF_PARAM_OFFLOAD} pytorch_cuda_alloc_conf=${PYTORCH_CUDA_ALLOC_CONF:-unset}"
echo "Actor/ref model dtype: ${ACTOR_MODEL_DTYPE:-bf16}"
echo "Checkpoint engine: backend=${FULL_ASYNC_CHECKPOINT_ENGINE_BACKEND} module=${FULL_ASYNC_CHECKPOINT_ENGINE_MODULE}"
CHECKPOINT_ENGINE_BACKEND="${FULL_ASYNC_CHECKPOINT_ENGINE_BACKEND}" CHECKPOINT_ENGINE_MODULE="${FULL_ASYNC_CHECKPOINT_ENGINE_MODULE}" python - <<'PY'
import importlib
import os

backend = os.environ["CHECKPOINT_ENGINE_BACKEND"]
module = os.environ.get("CHECKPOINT_ENGINE_MODULE")
if module:
    importlib.import_module(module)
from verl.checkpoint_engine.base import CheckpointEngineRegistry
CheckpointEngineRegistry.get(backend)
print(f"checkpoint_engine_{backend}=ok")
PY

python -m "${FULL_ASYNC_ENTRYPOINT}" \
  ++train_batch_size="${FULL_ASYNC_TRAIN_BATCH_SIZE}" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_penalty=low_var_kl \
  ++algorithm.rollout_correction.bypass_mode=True \
  data.train_files="${DATA_DIR}/train.parquet" \
  data.val_files="${DATA_DIR}/val.parquet" \
  data.prompt_key=prompt \
  data.train_batch_size=0 \
  data.gen_batch_size="${FULL_ASYNC_GEN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE:-16}" \
  data.train_max_samples="${TRAIN_MAX_SAMPLES:--1}" \
  data.val_max_samples="${VAL_MAX_SAMPLES:--1}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH_VALUE}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH_VALUE}" \
  data.filter_overlong_prompts=True \
  data.truncation=left \
  data.trust_remote_code=True \
  data.return_raw_chat=True \
  data.return_multi_modal_inputs="${RETURN_MULTI_MODAL_INPUTS}" \
  ++data.apply_chat_template_kwargs.enable_thinking="${ENABLE_THINKING}" \
  actor_rollout_ref.hybrid_engine=False \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  ++actor_rollout_ref.model.enable_activation_offload="${ENABLE_ACTIVATION_OFFLOAD}" \
  actor_rollout_ref.actor.strategy="${FULL_ASYNC_ACTOR_STRATEGY}" \
  actor_rollout_ref.actor.optim.lr="${LR:-1e-6}" \
  actor_rollout_ref.actor.optim.lr_warmup_steps="${LR_WARMUP_STEPS:-10}" \
  actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
  actor_rollout_ref.actor.optim.weight_decay="${WEIGHT_DECAY:-0.1}" \
  actor_rollout_ref.actor.optim.clip_grad="${CLIP_GRAD:-1.0}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}" \
  ++actor_rollout_ref.actor.use_rollout_log_probs=True \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF:-0.001}" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  ++actor_rollout_ref.actor.ulysses_sequence_parallel_size="${ULYSSES_SEQUENCE_PARALLEL_SIZE}" \
  actor_rollout_ref.actor.fsdp_config.model_dtype="${ACTOR_MODEL_DTYPE:-bf16}" \
  actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}" \
  actor_rollout_ref.rollout.name="${FULL_ASYNC_ROLLOUT_NAME}" \
  actor_rollout_ref.rollout.mode="${FULL_ASYNC_ROLLOUT_MODE}" \
  actor_rollout_ref.rollout.checkpoint_engine.backend="${FULL_ASYNC_CHECKPOINT_ENGINE_BACKEND}" \
  actor_rollout_ref.rollout.checkpoint_engine.custom_backend_module="${FULL_ASYNC_CHECKPOINT_ENGINE_MODULE}" \
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
  actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER}" \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend="${VLLM_ATTENTION_BACKEND_VALUE}" \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=True \
  ++actor_rollout_ref.rollout.calculate_log_probs=True \
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
  actor_rollout_ref.ref.fsdp_config.model_dtype="${ACTOR_MODEL_DTYPE:-bf16}" \
  ++actor_rollout_ref.ref.ulysses_sequence_parallel_size="${ULYSSES_SEQUENCE_PARALLEL_SIZE}" \
  actor_rollout_ref.ref.fsdp_config.param_offload="${REF_PARAM_OFFLOAD}" \
  ++critic.strategy="${FULL_ASYNC_ACTOR_STRATEGY}" \
  reward_model.enable=False \
  reward_model.reward_manager=naive \
  custom_reward_function.path="${REWARD_FN}" \
  custom_reward_function.name=compute_score \
  rollout.nnodes="${FULL_ASYNC_ROLLOUT_NNODES}" \
  rollout.n_gpus_per_node="${FULL_ASYNC_ROLLOUT_N_GPUS_PER_NODE}" \
  rollout.total_rollout_steps="${FULL_ASYNC_TOTAL_ROLLOUT_STEPS}" \
  ++rollout.test_freq="${FULL_ASYNC_ROLLOUT_TEST_FREQ:-${TEST_FREQ:-10}}" \
  async_training.require_batches="${FULL_ASYNC_REQUIRE_BATCHES}" \
  async_training.staleness_threshold="${FULL_ASYNC_STALENESS_THRESHOLD}" \
  async_training.trigger_parameter_sync_step="${FULL_ASYNC_TRIGGER_PARAMETER_SYNC_STEP}" \
  async_training.partial_rollout="${FULL_ASYNC_PARTIAL_ROLLOUT}" \
  async_training.use_trainer_do_validate="${FULL_ASYNC_USE_TRAINER_DO_VALIDATE}" \
  trainer.critic_warmup=0 \
  trainer.logger="${LOGGER_BACKENDS:-['console']}" \
  trainer.project_name="${PROJECT_NAME:-matharena-arxivmath-grpo}" \
  trainer.experiment_name="${EXPERIMENT_NAME:-qwen35-2b-grpo-thinking16k-fullasync-gh200}" \
  trainer.n_gpus_per_node="${TRAINER_N_GPUS_PER_NODE}" \
  trainer.nnodes="${FULL_ASYNC_TRAINER_NNODES}" \
  trainer.save_freq="${SAVE_FREQ:-10}" \
  trainer.test_freq="${TEST_FREQ:-10}" \
  trainer.total_epochs="${TOTAL_EPOCHS:-5}" \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.resume_mode="${RESUME_MODE}" \
  trainer.resume_from_path="${RESUME_FROM_PATH}" \
  trainer.log_val_generations="${LOG_VAL_GENERATIONS:-4}" \
  trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
  "$@"

if [[ "${UPLOAD_MODEL_TO_HF:-true}" == "true" ]]; then
  bash "${TRAIN_ROOT}/scripts/export_and_upload_hf_model.sh"
fi
