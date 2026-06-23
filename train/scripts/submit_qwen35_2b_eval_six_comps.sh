#!/usr/bin/env bash
set -euo pipefail

SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-a0163}"
EVAL_SLURM_SCRIPT="${EVAL_SLURM_SCRIPT:-train/slurm/04_eval_arxiv_march_vllm.sbatch}"

MODELS=(
  # "qwen/qwen3.5_2b|Qwen/Qwen3.5-2B|qwen/qwen3.5-2b"
  "qwen/qwen3.6_35b|Qwen/Qwen3.6-35B-A3B|Qwen/Qwen3.6-35B-A3B"
  # "qwen/qwen3.5_2b_sft|MathArena/qwen3.5-2b-arxivmath-sft|MathArena/qwen3.5-2b-arxivmath-sft"
  "qwen/qwen3.5_2b_brokenarxiv_sft|MathArena/qwen3.5-2b-brokenarxiv-sft|MathArena/qwen3.5-2b-brokenarxiv-sft"
  # "qwen/qwen3.5_2b_arxivmath_brokenarxiv_sft|MathArena/qwen3.5-2b-arxivmath-brokenarxiv-sft|MathArena/qwen3.5-2b-arxivmath-brokenarxiv-sft"
)

COMPS=(
  "arxiv/april|4"
  "arxiv/may|4"
  "arxiv_false/april_original|2"
  "arxiv_false/april_disprove|2"
  "arxiv_false/may_original|2"
  "arxiv_false/may_disprove|2"
)

for model_spec in "${MODELS[@]}"; do
  IFS="|" read -r model_config model_path served_model_name <<< "${model_spec}"
  for comp_spec in "${COMPS[@]}"; do
    IFS="|" read -r comp eval_n <<< "${comp_spec}"
    echo "Submitting ${model_config} on ${comp} with n=${eval_n}"
    sbatch --account="${SBATCH_ACCOUNT}" \
      --export=ALL,EVAL_MODEL_PATH="${model_path}",EVAL_SERVED_MODEL_NAME="${served_model_name}",EVAL_MODEL_CONFIG="${model_config}",EVAL_N="${eval_n}" \
      "${EVAL_SLURM_SCRIPT}" "${comp}"
  done
done
