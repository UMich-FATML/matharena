#!/usr/bin/env python3
"""Convert MathArena arxivmath Hugging Face rows into veRL parquet files."""

import argparse
import os
from typing import Any, Dict

from datasets import Dataset, load_dataset


DEFAULT_INSTRUCTION = (
    "You are given a difficult question. Your task is to solve the problem.\n"
    "Put the final answer you find within \\\\boxed{}.\n"
)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def make_record(example: Dict[str, Any], idx: int, split: str, instruction: str) -> Dict[str, Any]:
    question = clean_text(example.get("question"))
    answer = clean_text(example.get("answer"))
    if not question or not answer:
        raise ValueError(f"Row {idx} is missing question or answer.")

    prompt = f"{instruction}\n\n{question}"
    return {
        "data_source": "matharena/arxivmath",
        "prompt": [{"role": "user", "content": prompt}],
        "ability": "research_math",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "split": split,
            "index": idx,
            "paper_id": clean_text(example.get("paper_id")),
            "title": clean_text(example.get("title")),
            "question": question,
            "answer": answer,
        },
    }


def map_split(dataset: Dataset, split: str, instruction: str) -> Dataset:
    return dataset.map(
        lambda example, idx: make_record(example, idx, split, instruction),
        with_indices=True,
        remove_columns=dataset.column_names,
    )


def get_hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default="MathArena/training-arxivmath")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    load_kwargs = {}
    hf_token = get_hf_token()
    if hf_token:
        load_kwargs["token"] = hf_token
    try:
        dataset = load_dataset(args.dataset_id, split=args.hf_split, **load_kwargs)
    except Exception as exc:
        if exc.__class__.__name__ == "DatasetNotFoundError":
            raise RuntimeError(
                f"Could not access Hugging Face dataset {args.dataset_id!r}. "
                "For a private dataset, set HF_TOKEN in train/configs/local.env "
                "or submit with --export=ALL,HF_TOKEN=hf_..."
            ) from exc
        raise
    if args.max_rows is not None:
        dataset = dataset.select(range(min(args.max_rows, len(dataset))))
    if len(dataset) < 2:
        raise ValueError("Need at least two rows to create train and validation parquet files.")

    val_size = min(args.val_size, max(1, len(dataset) // 5))
    split = dataset.train_test_split(test_size=val_size, seed=args.seed, shuffle=True)

    train_dataset = map_split(split["train"], "train", args.instruction)
    val_dataset = map_split(split["test"], "validation", args.instruction)

    train_path = os.path.join(args.output_dir, "train.parquet")
    val_path = os.path.join(args.output_dir, "val.parquet")
    train_dataset.to_parquet(train_path)
    val_dataset.to_parquet(val_path)

    print(f"Wrote {len(train_dataset)} train rows to {train_path}")
    print(f"Wrote {len(val_dataset)} validation rows to {val_path}")


if __name__ == "__main__":
    main()
