#!/usr/bin/env bash
#SBATCH --job-name=setup-verl-qwen35
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=train/logs/%x-%j.out
#SBATCH --error=train/logs/%x-%j.err

set -euo pipefail
ulimit -c 0 || true
cd "${SLURM_SUBMIT_DIR}"

export HF_HOME="${SCRATCH}/huggingface"
export EDF_IMAGESTORE="${EDF_IMAGESTORE:-${SCRATCH}/.edf_imagestore}"
export VENV_DIR="${VENV_DIR:-${SCRATCH}/venvs/verl-qwen35}"
export RECREATE_VENV="${RECREATE_VENV:-true}"
export VERL_SRC="${VERL_SRC:-${SCRATCH}/src/verl}"
export VERL_REF="${VERL_REF:-main}"
export VLLM_TORCH_BACKEND="${VLLM_TORCH_BACKEND:-auto}"
export ALLOW_PYPI_VLLM_UPGRADE="${ALLOW_PYPI_VLLM_UPGRADE:-false}"
export EXPECTED_VLLM_VERSION="${EXPECTED_VLLM_VERSION:-0.19.0}"

mkdir -p \
  "${HF_HOME}" \
  "${EDF_IMAGESTORE}" \
  "$(dirname "${VENV_DIR}")" \
  "$(dirname "${VERL_SRC}")" \
  train/logs

srun \
  --mpi=pmix \
  --network=disable_rdzv_get \
  --environment=./train/env/cscs-verl.toml \
  bash -lc '
    set -euo pipefail
    ulimit -c 0 || true
    export HF_HOME="${SCRATCH}/huggingface"
    export TOKENIZERS_PARALLELISM=false
    unset SETUPTOOLS_USE_DISTUTILS
    unset PIP_CONSTRAINT PIP_BUILD_CONSTRAINT PIP_USE_FEATURE
    export PIP_NO_INPUT=1
    unset VLLM_VERSION VLLM_FLASH_ATTN_SRC_DIR VLLM_ATTENTION_BACKEND
    export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"
    echo "EXPECTED_VLLM_VERSION=${EXPECTED_VLLM_VERSION:-}"

    if [[ "${RECREATE_VENV}" == "true" && -d "${VENV_DIR}" ]]; then
      backup="${VENV_DIR}.bak.$(date +%Y%m%d%H%M%S)"
      echo "Moving existing venv to ${backup}"
      mv "${VENV_DIR}" "${backup}"
    fi

    python -m venv --system-site-packages "${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
    python -m pip install --upgrade \
      "pip<26" \
      "setuptools>=77,<80" \
      wheel \
      "packaging<=25.0" \
      ninja \
      cmake \
      uv \
      setuptools-scm \
      jinja2 \
      pybind11

    # The NVIDIA image can expose a system huggingface_hub without pip RECORD
    # metadata. Shadow it in the venv rather than asking pip to uninstall it.
    python -m pip install --upgrade --ignore-installed --no-deps \
      "huggingface_hub>=0.34.0,<1.0" \
      "optree>=0.16.0,<1"

    if [[ -e "${VERL_SRC}" ]] && ! git -C "${VERL_SRC}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      backup="${VERL_SRC}.bak.$(date +%Y%m%d%H%M%S)"
      echo "Moving invalid VERL_SRC checkout to ${backup}"
      mv "${VERL_SRC}" "${backup}"
    fi
    if [[ ! -e "${VERL_SRC}" ]]; then
      git clone https://github.com/verl-project/verl.git "${VERL_SRC}"
    fi
    git -C "${VERL_SRC}" fetch --tags origin
    git -C "${VERL_SRC}" checkout "${VERL_REF}"
    if [[ "${VERL_REF}" == "main" ]]; then
      git -C "${VERL_SRC}" pull --ff-only origin main
    fi
    git -C "${VERL_SRC}" submodule update --init --recursive recipe || true

    python - <<PY
import torch
print("container_torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
try:
    import vllm
except Exception as exc:
    raise SystemExit(f"The container must provide vLLM; import failed before setup: {exc}")
print("container_vllm", vllm.__version__)
expected = "${EXPECTED_VLLM_VERSION}"
actual_base = vllm.__version__.split("+", 1)[0]
if expected and actual_base != expected:
    raise SystemExit(f"Expected container vLLM {expected}, got {vllm.__version__}")
PY

    python - <<PY
import setuptools
import setuptools.build_meta
print("setuptools", setuptools.__version__, setuptools.__file__)
print("setuptools_build_meta", "ok")
PY

    python -m pip install --no-build-isolation \
      "numpy<2.0.0" \
      "accelerate" \
      "datasets" \
      "peft" \
      "transformers>=5.5.3,<6" \
      "hf-transfer" \
      "pyarrow>=15.0.0" \
      "pandas" \
      "huggingface_hub>=1.5,<2" \
      "wandb" \
      "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
      "torchdata" \
      "ray[default]>=2.48.0" \
      "cupy-cuda13x>=13.6,<15" \
      "codetiming" \
      "hydra-core==1.4.0.dev1" \
      "omegaconf==2.4.0.dev4" \
      "pylatexenc" \
      "qwen-vl-utils" \
      "mathruler" \
      "nvidia-ml-py>=12.560.30" \
      "fastapi[standard]>=0.115.0" \
      "optree>=0.16.0,<1" \
      "pydantic>=2.9" \
      "grpcio>=1.62.1" \
      "regex>=2024.11.6" \
      "loguru>=0.7.3" \
      "sympy>=1.13.1" \
      "antlr4-python3-runtime==4.11.1"

    bash "${SLURM_SUBMIT_DIR}/train/scripts/install_eval_deps.sh"

    if [[ "${ALLOW_PYPI_VLLM_UPGRADE}" == "true" ]]; then
      uv pip install --reinstall-package vllm --upgrade vllm --torch-backend="${VLLM_TORCH_BACKEND}"
    fi

    cd "${VERL_SRC}"
    python -m pip install --no-deps -e .

    cd "${SLURM_SUBMIT_DIR}"
    python -m pip install --no-deps -e .

    python - <<PY
import subprocess
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
from omegaconf import OmegaConf
assert OmegaConf.create({"ok": True}).ok is True
import verl
print("verl", getattr(verl, "__version__", "editable"))
print("verl_commit", subprocess.check_output(["git", "-C", "${VERL_SRC}", "rev-parse", "--short", "HEAD"]).decode().strip())
import transformers
import datasets
import multiprocess.queues
import ray.dag
from verl.checkpoint_engine.base import CheckpointEngineRegistry
from verl.checkpoint_engine.nccl_checkpoint_engine import NCCLCheckpointEngine
assert CheckpointEngineRegistry.get("nccl") is NCCLCheckpointEngine
print("verl_nccl_checkpoint_engine", "ok")
from transformers import AutoConfig
qwen_config = AutoConfig.from_pretrained("Qwen/Qwen3.5-2B", trust_remote_code=True)
assert qwen_config.model_type == "qwen3_5"
print("transformers", transformers.__version__, transformers.__file__, "qwen3_5", "ok")
print("datasets", datasets.__version__)
print("multiprocess_queues", "ok")
print("ray_dag", "ok")
try:
    import vllm
    print("vllm", vllm.__version__)
    expected = "${EXPECTED_VLLM_VERSION}"
    actual_base = vllm.__version__.split("+", 1)[0]
    if expected and actual_base != expected:
        raise SystemExit(f"Expected vLLM {expected}, got {vllm.__version__}")
except Exception as exc:
    raise SystemExit(f"vLLM import failed: {exc}")
from matharena.parser import check_answers, parse_answer
half, _ = parse_answer(r"\frac{1}{2}")
decimal, _ = parse_answer("0.5")
assert check_answers(half, decimal)
print("matharena_parser", "ok")
PY
  '
