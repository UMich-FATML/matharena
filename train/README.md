# Qwen3.5-2B GRPO on CSCS

Run from the repo root on a CSCS login node.

## Setup

```bash
export HF_HOME=$SCRATCH/huggingface
printf 'HF_TOKEN=%q\n' 'hf_...' > train/configs/local.env
```

`local.env` is for server-local secrets and is ignored by git. Use a token with read access for the private dataset and write access for model upload.

If needed, add `--account=a0163` to every `sbatch`.

```bash
sbatch --account=a0163 train/slurm/00_setup_verl_env.sbatch
```

This is the only install/setup entrypoint. It creates `$SCRATCH/venvs/verl-qwen35` inside the CSCS container and installs veRL, MathArena, evaluator dependencies, and the NVIDIA-container-compatible vLLM stack.
It recreates the venv by default; rerun the same job if the env drifts. Use `--export=ALL,RECREATE_VENV=false` only if you intentionally want an in-place update.
It pins ANTLR 4.11.1 because the MathArena parser needs it for SymPy LaTeX fallback.
It installs Transformers v5 in the venv because Qwen3.5 support was added after the v4.57.x line.
The default NVIDIA 26.04 vLLM image provides an ABI-matched vLLM 0.19.0/FlashInfer stack for CUDA 13.2 and is the recommended image for Qwen3.6. NVIDIA build suffixes such as `0.19.0+...nv26.04...` are expected.
Do not upgrade vLLM from PyPI unless you are also rebuilding or replacing the matching FlashInfer JIT cache; set `ALLOW_PYPI_VLLM_UPGRADE=true` only for that explicit experiment.
vLLM attention is forced to `FLASH_ATTN` through the vLLM engine argument to avoid mixing upstream `flashinfer` with the NVIDIA image's `flashinfer-jit-cache`.
Qwen3.5 is a multimodal model even for text prompts. Rollout uses vLLM's documented `language_model_only` mode to skip the vision encoder for this text-only GRPO run.
Training defaults `TORCH_CUDNN_V8_API_DISABLED=1` because NVIDIA 26.04 can fail in the Qwen3.5/Qwen-VL vision patch `Conv3d` path with `GET was unable to find an engine`; set it to `0` only when testing the raw 26.04 cuDNN frontend path.
Clean setup does not require a separate repair step. If an old in-place venv has shadowed or half-installed packages, rerun setup with the default `RECREATE_VENV=true`.

## Train

```bash
sbatch --account=a0163 train/slurm/01_prepare_arxivmath_data.sbatch
sbatch --account=a0163 train/slurm/02_train_grpo_qwen35_2b_multinode.sbatch
```

The multi-node script defaults to 2 nodes and scales the prompt batch to `16 * nodes`. Override node count and batch size like this:

```bash
sbatch --account=a0163 --nodes=4 --export=ALL,MULTINODE_TRAIN_BATCH_SIZE=64,MULTINODE_VAL_BATCH_SIZE=64 train/slurm/02_train_grpo_qwen35_2b_multinode.sbatch
```

Experimental full-async training uses separate trainer and rollout node pools inside one Ray cluster:

```bash
sbatch --account=a0163 train/slurm/02_train_grpo_qwen35_2b_full_async.sbatch
```

If the job reports that no full-async entrypoint was found, rerun `00_setup_verl_env.sbatch` so the veRL recipe submodule is initialized.

The full-async script defaults to a balanced split of the allocated nodes. For 4 nodes, this example uses 2 trainer nodes and 2 rollout nodes:

```bash
sbatch --account=a0163 --nodes=4 --export=ALL,FULL_ASYNC_TRAINER_NODES=2,FULL_ASYNC_ROLLOUT_NODES=2 train/slurm/02_train_grpo_qwen35_2b_full_async.sbatch
```

Logs:

```bash
tail -f train/logs/<job-name>-<job-id>.out
```

## W&B

One run, no key on disk:

```bash
export WANDB_API_KEY='...'
sbatch --account=a0163 --export=ALL,WANDB_API_KEY,LOGGER_BACKENDS="['console','wandb']" train/slurm/02_train_grpo_qwen35_2b.sbatch
```

Or put the key in ignored local config:

```bash
cat >> train/configs/local.env <<'EOF'
WANDB_API_KEY=...
LOGGER_BACKENDS="['console','wandb']"
EOF
sbatch --account=a0163 train/slurm/02_train_grpo_qwen35_2b.sbatch
```

## Upload

Production training uploads privately by default:

```bash
HF_MODEL_REPO_ID=MathArena/qwen3.5-2b-arxivmath-grpo
HF_EXPORT_DIR=$SCRATCH/hf_exports/qwen35-2b-arxivmath-grpo-thinking16k
HF_MODEL_PRIVATE=true
```

Disable upload:

```bash
sbatch --account=a0163 --export=ALL,UPLOAD_MODEL_TO_HF=false train/slurm/02_train_grpo_qwen35_2b.sbatch
```

Make upload public:

```bash
sbatch --account=a0163 --export=ALL,HF_MODEL_PRIVATE=false train/slurm/02_train_grpo_qwen35_2b.sbatch
```

Smoke runs never upload.

## SFT

Run a full-parameter SFT pass on the correct responses from `MathArena/arxivmath-training_outputs`:

```bash
sbatch --account=a0163 \
  --export=ALL,SFT_HF_MODEL_REPO_ID=MathArena/qwen3.5-2b-arxivmath-sft \
  train/slurm/05_sft_qwen35_2b_from_outputs.sbatch
```

Every saved checkpoint is uploaded to the Hub by default.

Useful overrides:

```bash
SFT_ONLY_CORRECT=true
SFT_MAX_SAMPLES=256
SFT_MAX_LENGTH=80000
SFT_VAL_FRACTION=0.05
SFT_SAVE_STEPS=50
SFT_RESUME_FROM_CHECKPOINT=${SCRATCH}/checkpoints/qwen35-2b-arxivmath-sft/checkpoint-STEP
SFT_UPLOAD_CHECKPOINTS=true
SFT_ASYNC_CHECKPOINT_UPLOAD=true
SFT_LOSS_CHUNK_SIZE=16384
```

Set `SFT_RESUME_FROM_CHECKPOINT=latest` to resume from the latest checkpoint in `SFT_OUTPUT_DIR`, or set it to a specific local `checkpoint-STEP` directory.

## Eval

Evaluate the exported model on `arxiv/march`:

```bash
sbatch --account=a0163 train/slurm/04_eval_arxiv_march_vllm.sbatch
```

Evaluate the base model:

```bash
sbatch --account=a0163 --export=ALL,EVAL_MODEL_PATH=Qwen/Qwen3.5-2B,EVAL_SERVED_MODEL_NAME=qwen/qwen3.5-2b,EVAL_MODEL_CONFIG=qwen/qwen3.5_2b train/slurm/04_eval_arxiv_march_vllm.sbatch
```

Evaluate on the arxivmath training data:

```bash
sbatch --account=a0163 train/slurm/04_eval_arxiv_march_vllm.sbatch arxiv/training
```