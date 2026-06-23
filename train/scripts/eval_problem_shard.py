#!/usr/bin/env python3
"""Print 1-based problem IDs assigned to an eval shard."""

import argparse
from pathlib import Path

import yaml


def problem_ids_for_shard(n_problems: int, shard_index: int, num_shards: int) -> list[int]:
    if n_problems < 0:
        raise ValueError("n_problems must be non-negative")
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    return [problem_id for problem_id in range(1, n_problems + 1) if (problem_id - 1) % num_shards == shard_index]


def load_n_problems(comp: str, configs_dir: Path) -> int:
    config_path = configs_dir / f"{comp}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Competition config not found: {config_path}")
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    n_problems = config.get("n_problems")
    if not isinstance(n_problems, int):
        raise ValueError(f"Competition config {config_path} has invalid n_problems={n_problems!r}")
    return n_problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comp", required=True, help="Competition name, e.g. arxiv/training")
    parser.add_argument("--shard-index", required=True, type=int, help="Zero-based shard index")
    parser.add_argument("--num-shards", required=True, type=int, help="Total number of shards")
    parser.add_argument("--configs-dir", default="configs/competitions", type=Path)
    args = parser.parse_args()

    n_problems = load_n_problems(args.comp, args.configs_dir)
    for problem_id in problem_ids_for_shard(n_problems, args.shard_index, args.num_shards):
        print(problem_id)


if __name__ == "__main__":
    main()
