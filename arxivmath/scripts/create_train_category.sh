#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 <primary-category> [<primary-category> ...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATHARENA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${MATHARENA_ROOT}/.." && pwd)"

export PYTHONPATH="${MATHARENA_ROOT}:${MATHARENA_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_CONFIG="${MODEL_CONFIG:-openai/gpt-56-sol-xhigh}"
CREATE_QUERIES_MODEL_CONFIG="${CREATE_QUERIES_MODEL_CONFIG:-${MODEL_CONFIG}}"
VERIFY_QUERIES_MODEL_CONFIG="${VERIFY_QUERIES_MODEL_CONFIG:-${MODEL_CONFIG}}"
FULLTEXT_REVIEW_MODEL_CONFIG="${FULLTEXT_REVIEW_MODEL_CONFIG:-anthropic/opus_47_high}"
PAPER_ROOT="${PAPER_ROOT:-arxivmath/train_category}"
CRAWL_ROOT="${CRAWL_ROOT:-../arxiv_papers_data}"
LIMIT="${LIMIT:-200}"
MIN_CITATIONS="${MIN_CITATIONS:-10}"

PIXI=(pixi run --manifest-path "${REPO_ROOT}/pixi.toml")
LIMIT_ARGS=(--limit "${LIMIT}")
PRIMARY_CATEGORY_ARGS=()
for primary_category in "$@"; do
  PRIMARY_CATEGORY_ARGS+=(--primary-category "${primary_category}")
done

cd "${MATHARENA_ROOT}"

"${PIXI[@]}" python arxivmath/scripts/train/ingest_arxiv_crawl.py \
  --crawl-root "${CRAWL_ROOT}" \
  --paper-root "${PAPER_ROOT}" \
  --min-citations "${MIN_CITATIONS}" \
  "${PRIMARY_CATEGORY_ARGS[@]}" \
  "${LIMIT_ARGS[@]}"

"${PIXI[@]}" python arxivmath/scripts/shared/create_queries.py \
  --model-config "${CREATE_QUERIES_MODEL_CONFIG}" \
  --paper-root "${PAPER_ROOT}" \
  --prompt arxivmath/prompts/arxiv/fulltext_query.md \
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
