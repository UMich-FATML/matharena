#!/usr/bin/env python3
"""MathArena parser reward for arxivmath GRPO runs in veRL."""

from typing import Any, Optional

from matharena.parser import check_answers, extract_answer, parse_answer


def compute_score(
    data_source: str = "",
    solution_str: str = "",
    ground_truth: str = "",
    extra_info: Optional[dict] = None,
    **_: Any,
) -> float:
    del data_source, extra_info
    gold_answer_is_list = "," in str(ground_truth)
    model_answer, _warning = extract_answer(
        solution_str,
        strict_parsing=False,
        parse=True,
        list_answer=gold_answer_is_list,
    )
    typed_gold_answer, _warning = parse_answer(str(ground_truth), list_answer=gold_answer_is_list)
    return 1.0 if check_answers(model_answer, typed_gold_answer) else 0.0
