#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0 || true

: "${SCRATCH:?SCRATCH must be set on CSCS}"
: "${OUTPUT_DIR:?OUTPUT_DIR must point at the veRL checkpoint directory}"
: "${HF_MODEL_REPO_ID:?HF_MODEL_REPO_ID must be set, e.g. MathArena/qwen3.5-2b-arxivmath-grpo}"

HF_EXPORT_DIR="${HF_EXPORT_DIR:-${SCRATCH}/hf_exports/${EXPERIMENT_NAME:-qwen35-2b-grpo-gh200}}"
MERGE_BACKEND="${MERGE_BACKEND:-fsdp}"

find_actor_checkpoint() {
  local step_file="${OUTPUT_DIR}/latest_checkpointed_iteration.txt"
  if [[ -n "${CHECKPOINT_STEP:-}" ]]; then
    for prefix in global_step global_steps; do
      local candidate="${OUTPUT_DIR}/${prefix}_${CHECKPOINT_STEP}/actor"
      if [[ -d "${candidate}" ]]; then
        printf "%s\n" "${candidate}"
        return 0
      fi
    done
  fi

  if [[ -f "${step_file}" ]]; then
    local step
    step="$(tr -d '[:space:]' < "${step_file}")"
    for prefix in global_step global_steps; do
      local candidate="${OUTPUT_DIR}/${prefix}_${step}/actor"
      if [[ -d "${candidate}" ]]; then
        printf "%s\n" "${candidate}"
        return 0
      fi
    done
  fi

  find "${OUTPUT_DIR}" -maxdepth 2 -type d -path "*/actor" | sort -V | tail -n 1
}

ACTOR_CHECKPOINT_DIR="$(find_actor_checkpoint)"
if [[ -z "${ACTOR_CHECKPOINT_DIR}" || ! -d "${ACTOR_CHECKPOINT_DIR}" ]]; then
  echo "Could not find an actor checkpoint under ${OUTPUT_DIR}" >&2
  exit 1
fi

mkdir -p "${HF_EXPORT_DIR}"

UPLOAD_ARGS=()
if [[ "${HF_MODEL_PRIVATE:-false}" == "true" ]]; then
  UPLOAD_ARGS+=(--private)
fi

echo "Merging ${MERGE_BACKEND} checkpoint:"
echo "  ${ACTOR_CHECKPOINT_DIR}"
echo "Export target:"
echo "  ${HF_EXPORT_DIR}"
echo "Hugging Face repo:"
echo "  ${HF_MODEL_REPO_ID}"

python -m verl.model_merger merge \
  --backend "${MERGE_BACKEND}" \
  --local_dir "${ACTOR_CHECKPOINT_DIR}" \
  --target_dir "${HF_EXPORT_DIR}" \
  --hf_upload_path "${HF_MODEL_REPO_ID}" \
  "${UPLOAD_ARGS[@]}"
