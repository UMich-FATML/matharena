#!/usr/bin/env bash
set -euo pipefail

SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-a0163}"
EVAL_SLURM_SCRIPT="${EVAL_SLURM_SCRIPT:-train/slurm/04_eval_arxiv_march_vllm.sbatch}"
APEX_COMP="${APEX_COMP:-apex/shortlist_2025}"
APEX_EVAL_N="${APEX_EVAL_N:-4}"

MODELS=(
  "qwen/qwen3.5_2b|Qwen/Qwen3.5-2B|qwen/qwen3.5-2b"
  "qwen/qwen3.5_2b_sft|MathArena/qwen3.5-2b-arxivmath-sft|MathArena/qwen3.5-2b-arxivmath-sft"
  "qwen/qwen3.5_2b_brokenarxiv_sft|MathArena/qwen3.5-2b-brokenarxiv-sft|MathArena/qwen3.5-2b-brokenarxiv-sft"
  "qwen/qwen3.5_2b_arxivmath_brokenarxiv_sft|MathArena/qwen3.5-2b-arxivmath-brokenarxiv-sft|MathArena/qwen3.5-2b-arxivmath-brokenarxiv-sft"
)

for model_spec in "${MODELS[@]}"; do
  IFS="|" read -r model_config model_path served_model_name <<< "${model_spec}"
  echo "Submitting ${model_config} on ${APEX_COMP} with n=${APEX_EVAL_N}"
  sbatch --account="${SBATCH_ACCOUNT}" \
    --export=ALL,EVAL_MODEL_PATH="${model_path}",EVAL_SERVED_MODEL_NAME="${served_model_name}",EVAL_MODEL_CONFIG="${model_config}",EVAL_N="${APEX_EVAL_N}" \
    "${EVAL_SLURM_SCRIPT}" "${APEX_COMP}"
done
