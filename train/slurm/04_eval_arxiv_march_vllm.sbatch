#!/usr/bin/env bash
#SBATCH --job-name=eval-vllm
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=72
#SBATCH --time=12:00:00
#SBATCH --output=train/logs/%x-%j.out
#SBATCH --error=train/logs/%x-%j.err

set -euo pipefail
ulimit -c 0 || true
cd "${SLURM_SUBMIT_DIR}"
export EVAL_COMP="${EVAL_COMP:-${1:-arxiv/march}}"

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
mkdir -p "${HF_HOME}" "${EDF_IMAGESTORE}" train/logs outputs

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
    bash train/scripts/eval_arxiv_march_vllm.sh
  '
