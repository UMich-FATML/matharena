#!/usr/bin/env bash
#SBATCH --job-name=qwen35-arxiv-grpo-mn
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

set -a
source train/configs/grpo_qwen35_2b.env
if [[ -f train/configs/local.env ]]; then
  source train/configs/local.env
fi
set +a
source train/scripts/slurm_ray_cluster.sh

export HF_HOME="${SCRATCH}/huggingface"
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN}}"
fi
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-${SLURM_JOB_ID}}"
export EDF_IMAGESTORE="${EDF_IMAGESTORE:-${SCRATCH}/.edf_imagestore}"
export TRAINER_NNODES="${MULTINODE_TRAINER_NNODES:-${SLURM_JOB_NUM_NODES}}"
export TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-4}"
export TRAIN_BATCH_SIZE="${MULTINODE_TRAIN_BATCH_SIZE:-$((16 * TRAINER_NNODES))}"
export VAL_BATCH_SIZE="${MULTINODE_VAL_BATCH_SIZE:-$((16 * TRAINER_NNODES))}"
if [[ -n "${TRAIN_PPO_MINI_BATCH_SIZE:-}" ]]; then
  export PPO_MINI_BATCH_SIZE_OVERRIDE="${TRAIN_PPO_MINI_BATCH_SIZE}"
fi

mkdir -p "${HF_HOME}" "${EDF_IMAGESTORE}" "${OUTPUT_DIR}" train/logs
echo "Multi-node GRPO batch: train_batch_size=${TRAIN_BATCH_SIZE} val_batch_size=${VAL_BATCH_SIZE}"

slurm_run_with_ray_cluster "${TRAINER_NNODES}" "${TRAINER_N_GPUS_PER_NODE}" "GRPO Ray cluster" '
  set -euo pipefail
  ulimit -c 0 || true
  source "${VENV_DIR}/bin/activate"
  export RAY_ADDRESS="${RAY_ADDRESS}"
  SOURCE_CONFIG=false bash train/scripts/run_grpo_qwen35_2b.sh
'
