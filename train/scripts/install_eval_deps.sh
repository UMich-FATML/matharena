#!/usr/bin/env bash
set -euo pipefail

unset PIP_CONSTRAINT PIP_BUILD_CONSTRAINT PIP_USE_FEATURE
export PIP_NO_INPUT=1

python -m pip install --no-build-isolation \
  "anthropic>=0.84.0" \
  "aristotlelib>=1.0.1" \
  "axiom-axle>=1.1.0" \
  "docker>=7.1.0" \
  "modal>=1.0.0" \
  "openai>=1.102.0" \
  "Pillow>=11.0.0" \
  "PyMuPDF>=1.26.7" \
  "python-dotenv>=1.1.1" \
  "pyyaml>=6.0" \
  "requests>=2.32.3" \
  "thefuzz>=0.22.1" \
  "together>=1.3.14" \
  "tqdm>=4.67.0" \
  "zstandard>=0.23.0"

python - <<'PY'
import anthropic
import aristotlelib
import axle
import docker
import fitz
import modal
import openai
from PIL import Image
import requests
import together
import yaml
import zstandard

print("matharena_eval_deps", "ok")
PY
