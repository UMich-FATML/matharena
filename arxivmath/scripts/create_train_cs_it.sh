#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_ROOT="${PAPER_ROOT:-arxivmath/train_cs_it}" exec "${SCRIPT_DIR}/create_train_category.sh" cs.IT
