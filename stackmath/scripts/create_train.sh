#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATHARENA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${MATHARENA_ROOT}/.." && pwd)"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  source "${REPO_ROOT}/.env"
  set +a
fi

CRAWL_ROOT="${CRAWL_ROOT:-${REPO_ROOT}/stackexchange_data}"
PAPER_ROOT="${PAPER_ROOT:-${REPO_ROOT}/data/stackmath-training}"
SITES="${SITES:-math.stackexchange.com mathoverflow.net}"
MIN_QUESTION_SCORE="${MIN_QUESTION_SCORE:-3}"
MIN_ANSWER_SCORE="${MIN_ANSWER_SCORE:-3}"
LIMIT_PER_SITE="${LIMIT_PER_SITE:-20}"
INCLUDE_CLOSED="${INCLUDE_CLOSED:-false}"
CREATE_QUERIES_MODEL_CONFIG="${CREATE_QUERIES_MODEL_CONFIG:-openai/gpt-56-sol-xhigh}"
VERIFY_QUERIES_MODEL_CONFIG="${VERIFY_QUERIES_MODEL_CONFIG:-openai/gpt-56-sol}"
FULLTEXT_REVIEW_MODEL_CONFIG="${FULLTEXT_REVIEW_MODEL_CONFIG:-anthropic/opus_48}"

read -r -a SITE_VALUES <<< "${SITES}"
if [[ "${#SITE_VALUES[@]}" -eq 0 ]]; then
  echo "SITES must contain at least one site" >&2
  exit 2
fi

INCLUDE_CLOSED_ARGS=()
case "${INCLUDE_CLOSED,,}" in
  1|true|yes) INCLUDE_CLOSED_ARGS=(--include-closed) ;;
  0|false|no) ;;
  *)
    echo "INCLUDE_CLOSED must be one of: true, false, 1, 0, yes, no" >&2
    exit 2
    ;;
esac

export PYTHONPATH="${MATHARENA_ROOT}:${MATHARENA_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
PIXI=(pixi run --manifest-path "${REPO_ROOT}/pixi.toml")

cd "${MATHARENA_ROOT}"

"${PIXI[@]}" python stackmath/scripts/train/ingest_stackexchange_crawl.py \
  --crawl-root "${CRAWL_ROOT}" \
  --paper-root "${PAPER_ROOT}" \
  --site "${SITE_VALUES[@]}" \
  --min-question-score "${MIN_QUESTION_SCORE}" \
  --min-answer-score "${MIN_ANSWER_SCORE}" \
  --limit-per-site "${LIMIT_PER_SITE}" \
  "${INCLUDE_CLOSED_ARGS[@]}"

"${PIXI[@]}" python stackmath/scripts/shared/create_queries.py \
  --model-config "${CREATE_QUERIES_MODEL_CONFIG}" \
  --paper-root "${PAPER_ROOT}" \
  --fulltext

"${PIXI[@]}" python stackmath/scripts/shared/verify_queries.py \
  --model-config "${VERIFY_QUERIES_MODEL_CONFIG}" \
  --paper-root "${PAPER_ROOT}"

"${PIXI[@]}" python stackmath/scripts/shared/fulltext_review.py \
  --model-config "${FULLTEXT_REVIEW_MODEL_CONFIG}" \
  --paper-root "${PAPER_ROOT}" \
  --full-text-source local
