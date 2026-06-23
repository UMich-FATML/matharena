#!/usr/bin/env bash
#SBATCH --job-name=prep-arxivmath-verl
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
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

export HF_HOME="${SCRATCH}/huggingface"
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN}}"
fi
export EDF_IMAGESTORE="${EDF_IMAGESTORE:-${SCRATCH}/.edf_imagestore}"
mkdir -p "${HF_HOME}" "${EDF_IMAGESTORE}" "${DATA_DIR}" train/logs

srun \
  --mpi=pmix \
  --network=disable_rdzv_get \
  --environment=./train/env/cscs-verl.toml \
  bash -lc '
    set -euo pipefail
    ulimit -c 0 || true
    if [[ -n "${HF_TOKEN:-}" ]]; then
      export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN}}"
    fi
    source "${VENV_DIR}/bin/activate"
    python train/scripts/prepare_arxivmath_verl.py \
      --dataset-id "${DATASET_ID}" \
      --output-dir "${DATA_DIR}" \
      --val-size 64
  '
