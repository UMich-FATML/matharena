#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATHARENA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${MATHARENA_ROOT}/.." && pwd)"

export PYTHONPATH="${MATHARENA_ROOT}:${MATHARENA_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_CONFIG="${MODEL_CONFIG:-openai/gpt-54-high}"
CREATE_QUERIES_MODEL_CONFIG="${CREATE_QUERIES_MODEL_CONFIG:-${MODEL_CONFIG}}"
VERIFY_QUERIES_MODEL_CONFIG="${VERIFY_QUERIES_MODEL_CONFIG:-${MODEL_CONFIG}}"
FULLTEXT_REVIEW_MODEL_CONFIG="${FULLTEXT_REVIEW_MODEL_CONFIG:-${MODEL_CONFIG}}"
PAPER_ROOT="${PAPER_ROOT:-arxivmath/train_co_nt}"
CRAWL_ROOT="${CRAWL_ROOT:-../arxiv_papers_data}"

PIXI=(pixi --manifest-path "${REPO_ROOT}/pixi.toml" run)
LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

cd "${MATHARENA_ROOT}"

"${PIXI[@]}" python arxivmath/scripts/train/extract_co_nt_from_crawl.py \
  --crawl-root "${CRAWL_ROOT}" \
  --paper-root "${PAPER_ROOT}" \
  "${LIMIT_ARGS[@]}"

"${PIXI[@]}" python arxivmath/scripts/shared/create_queries.py \
  --model-config "${CREATE_QUERIES_MODEL_CONFIG}" \
  --paper-root "${PAPER_ROOT}" \
  --prompt arxivmath/prompts/arxiv/query_fulltext.md \
  --full-text-source local \
  "${LIMIT_ARGS[@]}"

"${PIXI[@]}" python arxivmath/scripts/shared/verify_queries.py \
  --model-config "${VERIFY_QUERIES_MODEL_CONFIG}" \
  --paper-root "${PAPER_ROOT}" \
  "${LIMIT_ARGS[@]}"

"${PIXI[@]}" python arxivmath/scripts/shared/fulltext_review.py \
  --model-config "${FULLTEXT_REVIEW_MODEL_CONFIG}" \
  --paper-root "${PAPER_ROOT}" \
  --full-text-source local \
  "${LIMIT_ARGS[@]}"
