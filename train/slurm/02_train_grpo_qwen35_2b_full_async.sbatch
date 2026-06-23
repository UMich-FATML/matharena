#!/usr/bin/env bash
#SBATCH --job-name=qwen35-arxiv-grpo-fa
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=72
#SBATCH --time=12:00:00
#SBATCH --output=train/logs/%x-%j.out
#SBATCH --error=train/logs/%x-%j.err

set -euo pipefail
ulimit -c 0 || true
cd "${SLURM_SUBMIT_DIR}"

source_config_preserving_env_overrides() {
  local config_file="${1:?config file required}"
  local local_file="${2:-}"
  local -a names=()
  local name
  for file in "${config_file}" "${local_file}"; do
    [[ -n "${file}" && -f "${file}" ]] || continue
    while IFS= read -r name; do
      names+=("${name}")
    done < <(sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' "${file}")
  done

  declare -A was_set=()
  declare -A saved_value=()
  for name in "${names[@]}"; do
    if [[ ${!name+x} ]]; then
      was_set["${name}"]=1
      saved_value["${name}"]="${!name}"
    fi
  done

  set -a
  source "${config_file}"
  if [[ -n "${local_file}" && -f "${local_file}" ]]; then
    source "${local_file}"
  fi
  set +a

  for name in "${!was_set[@]}"; do
    export "${name}=${saved_value[${name}]}"
  done
}

source_config_preserving_env_overrides train/configs/grpo_qwen35_2b.env train/configs/local.env
source train/scripts/slurm_ray_cluster.sh

export HF_HOME="${SCRATCH}/huggingface"
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN}}"
fi
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-${SLURM_JOB_ID}}"
export EDF_IMAGESTORE="${EDF_IMAGESTORE:-${SCRATCH}/.edf_imagestore}"
export TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-4}"
export FULL_ASYNC_ROLLOUT_N_GPUS_PER_NODE="${FULL_ASYNC_ROLLOUT_N_GPUS_PER_NODE:-${TRAINER_N_GPUS_PER_NODE}}"

default_trainer_nodes=$((SLURM_JOB_NUM_NODES / 2))
if (( default_trainer_nodes < 1 )); then
  default_trainer_nodes=1
fi
export FULL_ASYNC_TRAINER_NNODES="${FULL_ASYNC_TRAINER_NODES:-${default_trainer_nodes}}"
export FULL_ASYNC_ROLLOUT_NNODES="${FULL_ASYNC_ROLLOUT_NODES:-$((SLURM_JOB_NUM_NODES - FULL_ASYNC_TRAINER_NNODES))}"
export FULL_ASYNC_TOTAL_NNODES=$((FULL_ASYNC_TRAINER_NNODES + FULL_ASYNC_ROLLOUT_NNODES))
if (( FULL_ASYNC_ROLLOUT_NNODES < 1 || FULL_ASYNC_TOTAL_NNODES != SLURM_JOB_NUM_NODES )); then
  echo "Full async needs trainer + rollout nodes to equal allocated nodes, with at least one rollout node." >&2
  echo "Got trainer=${FULL_ASYNC_TRAINER_NNODES}, rollout=${FULL_ASYNC_ROLLOUT_NNODES}, allocated=${SLURM_JOB_NUM_NODES}." >&2
  echo "Override with FULL_ASYNC_TRAINER_NODES and FULL_ASYNC_ROLLOUT_NODES." >&2
  exit 1
fi

export EXPERIMENT_NAME="${FULL_ASYNC_EXPERIMENT_NAME:-qwen35-2b-grpo-thinking16k-fullasync-gh200}"
export OUTPUT_DIR="${FULL_ASYNC_OUTPUT_DIR:-${SCRATCH}/checkpoints/matharena-qwen35-2b-grpo-thinking16k-fullasync}"
export HF_EXPORT_DIR="${FULL_ASYNC_HF_EXPORT_DIR:-${SCRATCH}/hf_exports/qwen35-2b-arxivmath-grpo-thinking16k-fullasync}"
if [[ -n "${TRAIN_PPO_MINI_BATCH_SIZE:-}" ]]; then
  export PPO_MINI_BATCH_SIZE_OVERRIDE="${TRAIN_PPO_MINI_BATCH_SIZE}"
fi

mkdir -p "${HF_HOME}" "${EDF_IMAGESTORE}" "${OUTPUT_DIR}" train/logs
echo "Full async split: trainer_nodes=${FULL_ASYNC_TRAINER_NNODES} rollout_nodes=${FULL_ASYNC_ROLLOUT_NNODES}"

slurm_run_with_ray_cluster "${FULL_ASYNC_TOTAL_NNODES}" "${TRAINER_N_GPUS_PER_NODE}" "full-async Ray cluster" '
  set -euo pipefail
  ulimit -c 0 || true
  source "${VENV_DIR}/bin/activate"
  export RAY_ADDRESS="${RAY_ADDRESS}"
  SOURCE_CONFIG=false bash train/scripts/run_grpo_qwen35_2b_full_async.sh
'
