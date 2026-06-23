#!/usr/bin/env python3
"""Upload an existing SFT checkpoint directory to a Hugging Face model repo."""

import argparse
from pathlib import Path


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def upload_checkpoint(checkpoint_dir: Path, repo_id: str, private: bool) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(checkpoint_dir),
        commit_message=f"Upload MathArena SFT checkpoint {checkpoint_dir.name}",
        ignore_patterns=[
            "optimizer.pt",
            "scheduler.pt",
            "rng_state*.pth",
            "scaler.pt",
            "trainer_state.json",
            "training_args.bin",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--hub-model-id", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--private", type=str_to_bool, default=True)
    parser.add_argument("--save-tokenizer", type=str_to_bool, default=True)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise SystemExit(f"Checkpoint directory does not exist: {checkpoint_dir}")

    if args.save_tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        tokenizer.save_pretrained(checkpoint_dir)

    upload_checkpoint(checkpoint_dir, args.hub_model_id, args.private)


if __name__ == "__main__":
    main()
