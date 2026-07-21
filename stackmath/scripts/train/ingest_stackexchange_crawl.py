#!/usr/bin/env python3
import argparse
import json
import math
import os
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pyarrow.parquet as pq


DEFAULT_SITES = ("math.stackexchange.com", "mathoverflow.net")
DEFAULT_COLUMNS = (
    "Id",
    "PostTypeId",
    "AcceptedAnswerId",
    "CreationDate",
    "Score",
    "ViewCount",
    "Body",
    "OwnerUserId",
    "Title",
    "Tags",
    "AnswerCount",
    "CommentCount",
    "ContentLicense",
    "ClosedDate",
    "CommunityOwnedDate",
    "site",
    "dump_answer_count",
    "answer_scores",
    "accepted_answer_body",
    "answer_bodies",
    "url",
)


@dataclass
class ExtractionSummary:
    files_inspected: int = 0
    files_skipped_by_site_statistics: int = 0
    row_groups_inspected: int = 0
    inspected: int = 0
    selected: int = 0
    written: int = 0
    skipped_wrong_site: int = 0
    skipped_site_quota_reached: int = 0
    skipped_missing_question_id: int = 0
    skipped_missing_question_body: int = 0
    skipped_closed: int = 0
    skipped_low_question_score: int = 0
    skipped_no_answers: int = 0
    skipped_malformed_answers: int = 0
    skipped_low_answer_score: int = 0
    skipped_missing_accepted_answer: int = 0
    skipped_accepted_answer_not_in_answers: int = 0
    skipped_ambiguous_accepted_answer: int = 0
    written_by_site: dict[str, int] = field(default_factory=dict)


def safe_dir_name(site: str, question_id: str) -> str:
    return f"{quote(site, safe='.-_')}_{quote(question_id, safe='.-_')}"


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def _clean_scalar(value: Any) -> Any:
    if _is_missing(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _clean_string(value: Any) -> str:
    value = _clean_scalar(value)
    return "" if value is None else str(value).strip()


def _raw_string(value: Any) -> str:
    value = _clean_scalar(value)
    return "" if value is None else str(value)


def _clean_int(value: Any) -> int | None:
    value = _clean_scalar(value)
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _list_value(value: Any) -> list[Any] | None:
    value = _clean_scalar(value)
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return None
    return list(value)


def _parse_tags(value: Any) -> list[str]:
    raw_tags = _clean_string(value)
    if not raw_tags:
        return []
    return [tag.strip() for tag in raw_tags.split("|") if tag.strip()]


def _normalize_sites(sites: Collection[str]) -> tuple[str, ...]:
    if isinstance(sites, str):
        sites = [sites]
    normalized = []
    for site in sites:
        cleaned = str(site).strip()
        if not cleaned:
            raise ValueError("site names must be nonempty")
        if cleaned not in normalized:
            normalized.append(cleaned)
    return tuple(normalized)


def _chunk_paths(crawl_root: Path) -> list[Path]:
    return sorted(crawl_root.glob("questions_part_*.parquet"))


def _site_column_index(parquet_file: pq.ParquetFile) -> int | None:
    try:
        return parquet_file.schema.names.index("site")
    except ValueError:
        return None


def _stat_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _parquet_may_contain_sites(parquet_file: pq.ParquetFile, sites: Collection[str]) -> bool:
    if not sites:
        return False
    site_column_index = _site_column_index(parquet_file)
    if site_column_index is None:
        return True
    for row_group_index in range(parquet_file.num_row_groups):
        statistics = parquet_file.metadata.row_group(row_group_index).column(site_column_index).statistics
        if statistics is None or not statistics.has_min_max:
            return True
        minimum = _stat_string(statistics.min)
        maximum = _stat_string(statistics.max)
        if any(minimum <= site <= maximum for site in sites):
            return True
    return False


def _read_columns(parquet_file: pq.ParquetFile) -> list[str]:
    available = set(parquet_file.schema_arrow.names)
    return [column for column in DEFAULT_COLUMNS if column in available]


def _validated_answers(row: dict[str, Any]) -> tuple[list[tuple[str, int]], str] | tuple[None, str]:
    bodies = _list_value(row.get("answer_bodies"))
    scores = _list_value(row.get("answer_scores"))
    if bodies is None or scores is None:
        return None, "malformed"
    if not bodies and not scores:
        return None, "none"
    if not bodies or not scores or len(bodies) != len(scores):
        return None, "malformed"

    answers = []
    for body_value, score_value in zip(bodies, scores, strict=True):
        body = _raw_string(body_value)
        score = _clean_int(score_value)
        if not body.strip() or score is None:
            return None, "malformed"
        answers.append((body, score))
    return answers, "ok"


def _render_full_text(
    title: str,
    question_body: str,
    answers: Sequence[tuple[str, int]],
    accepted_answer_body: str,
) -> tuple[str, list[tuple[str, int, bool]]]:
    ordered_answers = sorted(
        ((body, score, body == accepted_answer_body) for body, score in answers),
        key=lambda answer: answer[1],
        reverse=True,
    )
    sections = [f"# {title}" if title else "# Untitled question", "", "## Question", "", question_body]
    sections.extend(["", "## Answers"])
    for index, (body, score, accepted) in enumerate(ordered_answers, start=1):
        accepted_label = ", accepted" if accepted else ""
        sections.extend(["", f"### Answer {index} (score: {score}{accepted_label})", "", body])
    return "\n".join(sections).rstrip() + "\n", ordered_answers


def _metadata_from_row(
    row: dict[str, Any],
    site: str,
    question_id: str,
    question_body: str,
    ordered_answers: Sequence[tuple[str, int, bool]],
) -> dict[str, Any]:
    tags = _parse_tags(row.get("Tags"))
    source_id = f"{site}:{question_id}"
    url = _clean_string(row.get("url")) or f"https://{site}/questions/{question_id}"
    accepted_position = next(index for index, answer in enumerate(ordered_answers, start=1) if answer[2])
    content_license = _clean_string(row.get("ContentLicense"))
    return {
        "id": source_id,
        "source": "stackexchange",
        "source_id": source_id,
        "site": site,
        "question_id": question_id,
        "url": url,
        "created": _clean_string(row.get("CreationDate")),
        "updated": "",
        "title": _clean_string(row.get("Title")),
        "body": question_body,
        "abstract": question_body,
        "tags": tags,
        "categories": tags,
        "primary_category": tags[0] if tags else "",
        "authors": [],
        "question_owner_user_id": _clean_string(row.get("OwnerUserId")),
        "question_score": _clean_int(row.get("Score")),
        "view_count": _clean_int(row.get("ViewCount")),
        "answer_count": _clean_int(row.get("AnswerCount")),
        "dump_answer_count": _clean_int(row.get("dump_answer_count")),
        "comment_count": _clean_int(row.get("CommentCount")),
        "answer_scores": [score for _, score, _ in ordered_answers],
        "accepted_answer_id": _clean_string(row.get("AcceptedAnswerId")),
        "accepted_answer_position": accepted_position,
        "content_license": content_license,
        "license": content_license,
        "closed_date": _clean_string(row.get("ClosedDate")),
        "community_owned_date": _clean_string(row.get("CommunityOwnedDate")),
        "post_type_id": _clean_string(row.get("PostTypeId")),
        "raw_tags": _clean_string(row.get("Tags")),
    }


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def extract_from_crawl(
    crawl_root: str | os.PathLike[str],
    paper_root: str | os.PathLike[str],
    *,
    sites: Collection[str] = DEFAULT_SITES,
    min_question_score: int = 3,
    min_answer_score: int = 3,
    limit_per_site: int | None = None,
    include_closed: bool = False,
) -> ExtractionSummary:
    crawl_root = Path(crawl_root)
    paper_root = Path(paper_root)
    requested_sites = _normalize_sites(sites)
    if limit_per_site is not None and limit_per_site < 0:
        raise ValueError("limit_per_site must be nonnegative or None")

    summary = ExtractionSummary(written_by_site={site: 0 for site in requested_sites})
    if not requested_sites or limit_per_site == 0:
        return summary
    requested_site_set = set(requested_sites)
    pending_sites = set(requested_sites)

    for chunk_path in _chunk_paths(crawl_root):
        if limit_per_site is not None and not pending_sites:
            break
        summary.files_inspected += 1
        parquet_file = pq.ParquetFile(chunk_path)
        sites_needed = pending_sites if limit_per_site is not None else requested_site_set
        if not _parquet_may_contain_sites(parquet_file, sites_needed):
            summary.files_skipped_by_site_statistics += 1
            continue

        columns = _read_columns(parquet_file)
        for row_group_index in range(parquet_file.num_row_groups):
            summary.row_groups_inspected += 1
            table = parquet_file.read_row_group(row_group_index, columns=columns)
            for row in table.to_pylist():
                summary.inspected += 1
                site = _clean_string(row.get("site"))
                if site not in requested_site_set:
                    summary.skipped_wrong_site += 1
                    continue
                if limit_per_site is not None and summary.written_by_site[site] >= limit_per_site:
                    summary.skipped_site_quota_reached += 1
                    continue

                question_id = _clean_string(row.get("Id"))
                if not question_id:
                    summary.skipped_missing_question_id += 1
                    continue
                question_body = _raw_string(row.get("Body"))
                if not question_body.strip():
                    summary.skipped_missing_question_body += 1
                    continue
                if not include_closed and _clean_string(row.get("ClosedDate")):
                    summary.skipped_closed += 1
                    continue
                question_score = _clean_int(row.get("Score"))
                if question_score is None or question_score < min_question_score:
                    summary.skipped_low_question_score += 1
                    continue

                answers, answer_status = _validated_answers(row)
                if answer_status == "none":
                    summary.skipped_no_answers += 1
                    continue
                if answer_status != "ok" or answers is None:
                    summary.skipped_malformed_answers += 1
                    continue
                if max(score for _, score in answers) < min_answer_score:
                    summary.skipped_low_answer_score += 1
                    continue

                accepted_answer_id = _clean_string(row.get("AcceptedAnswerId"))
                accepted_answer_body = _raw_string(row.get("accepted_answer_body"))
                if not accepted_answer_id or not accepted_answer_body.strip():
                    summary.skipped_missing_accepted_answer += 1
                    continue
                accepted_matches = sum(body == accepted_answer_body for body, _ in answers)
                if accepted_matches == 0:
                    summary.skipped_accepted_answer_not_in_answers += 1
                    continue
                if accepted_matches > 1:
                    summary.skipped_ambiguous_accepted_answer += 1
                    continue

                summary.selected += 1
                full_text, ordered_answers = _render_full_text(
                    _clean_string(row.get("Title")),
                    question_body,
                    answers,
                    accepted_answer_body,
                )
                thread_dir = paper_root / safe_dir_name(site, question_id)
                thread_dir.mkdir(parents=True, exist_ok=True)
                _write_json(
                    thread_dir / "metadata.json",
                    _metadata_from_row(row, site, question_id, question_body, ordered_answers),
                )
                _write_text(thread_dir / "full_text.md", full_text)
                summary.written += 1
                summary.written_by_site[site] += 1
                if limit_per_site is not None and summary.written_by_site[site] >= limit_per_site:
                    pending_sites.discard(site)
                    if not pending_sites:
                        return summary
    return summary


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected a nonnegative integer")
    return parsed


def _create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract accepted-answer Stack Exchange discussions into a MathArena paper root."
    )
    parser.add_argument(
        "--crawl-root",
        default="../stackexchange_data",
        help="Root containing questions_part_*.parquet files.",
    )
    parser.add_argument(
        "--paper-root",
        default="stackmath/paper",
        help="Output directory containing one folder per Stack Exchange thread.",
    )
    parser.add_argument(
        "--site",
        dest="sites",
        nargs="+",
        default=list(DEFAULT_SITES),
        help="One or more Stack Exchange sites to include.",
    )
    parser.add_argument("--min-question-score", type=int, default=3)
    parser.add_argument("--min-answer-score", type=int, default=3)
    parser.add_argument(
        "--limit-per-site",
        type=_nonnegative_int,
        default=None,
        help="Optional maximum number of written threads for each requested site.",
    )
    parser.add_argument(
        "--include-closed",
        action="store_true",
        help="Include closed questions; closed questions are excluded by default.",
    )
    return parser


def main() -> None:
    args = _create_argument_parser().parse_args()
    summary = extract_from_crawl(
        args.crawl_root,
        args.paper_root,
        sites=args.sites,
        min_question_score=args.min_question_score,
        min_answer_score=args.min_answer_score,
        limit_per_site=args.limit_per_site,
        include_closed=args.include_closed,
    )
    print(
        f"Extracted Stack Exchange discussions for sites={','.join(args.sites)} "
        f"min_question_score={args.min_question_score} min_answer_score={args.min_answer_score} "
        f"include_closed={args.include_closed}: files_inspected={summary.files_inspected}, "
        f"files_skipped_by_site_statistics={summary.files_skipped_by_site_statistics}, "
        f"row_groups_inspected={summary.row_groups_inspected}, inspected={summary.inspected}, "
        f"selected={summary.selected}, written={summary.written}, "
        f"written_by_site={json.dumps(summary.written_by_site, sort_keys=True)}, "
        f"skipped_wrong_site={summary.skipped_wrong_site}, "
        f"skipped_site_quota_reached={summary.skipped_site_quota_reached}, "
        f"skipped_missing_question_id={summary.skipped_missing_question_id}, "
        f"skipped_missing_question_body={summary.skipped_missing_question_body}, "
        f"skipped_closed={summary.skipped_closed}, "
        f"skipped_low_question_score={summary.skipped_low_question_score}, "
        f"skipped_no_answers={summary.skipped_no_answers}, "
        f"skipped_malformed_answers={summary.skipped_malformed_answers}, "
        f"skipped_low_answer_score={summary.skipped_low_answer_score}, "
        f"skipped_missing_accepted_answer={summary.skipped_missing_accepted_answer}, "
        f"skipped_accepted_answer_not_in_answers={summary.skipped_accepted_answer_not_in_answers}, "
        f"skipped_ambiguous_accepted_answer={summary.skipped_ambiguous_accepted_answer}"
    )


if __name__ == "__main__":
    main()
