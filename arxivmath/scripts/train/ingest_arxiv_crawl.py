#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


DEFAULT_COLUMNS = [
    "ext_arxiv",
    "primary_category",
    "title",
    "abstract",
    "authors",
    "categories",
    "license",
    "doi",
    "journal-ref",
    "update_date",
    "latest_version",
    "citationCount",
    "content_json",
    "cited_arxiv_ids",
]
DEFAULT_PRIMARY_CATEGORIES = {"math.CO", "math.NT"}


@dataclass
class ExtractionSummary:
    inspected: int = 0
    selected: int = 0
    written: int = 0
    skipped_missing_arxiv_id: int = 0
    skipped_low_citation: int = 0
    skipped_wrong_category: int = 0
    skipped_outside_posted_month_range: int = 0
    skipped_missing_text: int = 0


@dataclass(frozen=True)
class _SelectionFilter:
    """A row predicate and the bookkeeping needed when it rejects a row."""

    predicate: Callable[[dict[str, Any]], bool]
    required_columns: tuple[str, ...]
    rejection_counter: str


def safe_dir_name(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_")


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(math.isnan(value))
    except TypeError:
        return False


def _clean_scalar(value: Any) -> Any:
    if _is_missing(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _clean_string(value: Any) -> str:
    value = _clean_scalar(value)
    if value is None:
        return ""
    return str(value).strip()


def _split_categories(value: Any) -> list[str]:
    value = _clean_scalar(value)
    if value is None:
        return []
    if isinstance(value, str):
        return [category for category in value.split() if category]
    if isinstance(value, (list, tuple)):
        return [str(category).strip() for category in value if str(category).strip()]
    return []


def _format_authors(value: Any) -> list[dict[str, str]]:
    value = _clean_scalar(value)
    if value is None:
        return []
    if isinstance(value, str):
        names = [name.strip() for name in value.split(",") if name.strip()]
        return [{"forenames": "", "keyname": name} for name in names]
    if isinstance(value, (list, tuple)):
        authors = []
        for author in value:
            if isinstance(author, dict):
                keyname = _clean_string(author.get("keyname") or author.get("name"))
                forenames = _clean_string(author.get("forenames"))
                if keyname or forenames:
                    authors.append({"forenames": forenames, "keyname": keyname})
            else:
                name = _clean_string(author)
                if name:
                    authors.append({"forenames": "", "keyname": name})
        return authors
    return []


def _list_strings(value: Any) -> list[str]:
    value = _clean_scalar(value)
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _extract_full_text(content_json: Any) -> str:
    content_json = _clean_scalar(content_json)
    if not content_json:
        return ""
    if isinstance(content_json, str):
        try:
            content = json.loads(content_json)
        except json.JSONDecodeError:
            return ""
    elif isinstance(content_json, dict):
        content = content_json
    else:
        return ""
    text = content.get("text") if isinstance(content, dict) else None
    return text.strip() if isinstance(text, str) else ""


def _metadata_from_row(row: dict[str, Any], arxiv_id: str) -> dict[str, Any]:
    return {
        "id": arxiv_id,
        "created": "",
        "updated": _clean_string(row.get("update_date")),
        "title": _clean_string(row.get("title")),
        "abstract": _clean_string(row.get("abstract")),
        "categories": _split_categories(row.get("categories")),
        "primary_category": _clean_string(row.get("primary_category")),
        "comments": "",
        "journal_ref": _clean_string(row.get("journal-ref")),
        "doi": _clean_string(row.get("doi")),
        "license": _clean_string(row.get("license")),
        "authors": _format_authors(row.get("authors")),
        "latest_version": _clean_string(row.get("latest_version")),
        "citationCount": _clean_scalar(row.get("citationCount")),
        "cited_arxiv_ids": _list_strings(row.get("cited_arxiv_ids")),
    }


def _write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _chunk_paths(crawl_root: Path) -> list[Path]:
    return sorted(crawl_root.glob("chunk_*.parquet"))


def _read_columns(
    parquet_file: pq.ParquetFile,
    selection_filters: Sequence[_SelectionFilter],
) -> list[str]:
    available = set(parquet_file.schema_arrow.names)
    requested = dict.fromkeys(DEFAULT_COLUMNS)
    for selection_filter in selection_filters:
        requested.update(dict.fromkeys(selection_filter.required_columns))
    return [column for column in requested if column in available]


def _matches_primary_category(row: dict[str, Any], primary_categories: set[str]) -> bool:
    return _clean_string(row.get("primary_category")) in primary_categories


def _meets_minimum_citations(row: dict[str, Any], min_citations: float) -> bool:
    citation_count = _clean_scalar(row.get("citationCount"))
    return citation_count is not None and float(citation_count) >= min_citations


def _validate_posted_month(value: str) -> str:
    if not re.fullmatch(r"\d{4}-(?:0[1-9]|1[0-2])", value):
        raise ValueError(f"invalid posted month {value!r}; expected YYYY-MM")
    return value


def _posted_month_from_arxiv_id(arxiv_id: str) -> str | None:
    arxiv_id = arxiv_id.removeprefix("arXiv:")
    match = re.fullmatch(r"(?:[^/]+/)?(\d{2})(\d{2})(?:\.\d{4,5}|\d{3})(?:v\d+)?", arxiv_id)
    if match is None:
        return None
    short_year, month = (int(part) for part in match.groups())
    if not 1 <= month <= 12:
        return None
    year = 1900 + short_year if short_year >= 91 else 2000 + short_year
    return f"{year:04d}-{month:02d}"


def _within_posted_month_range(
    row: dict[str, Any],
    posted_from_month: str | None,
    posted_until_month: str | None,
) -> bool:
    posted_month = _posted_month_from_arxiv_id(_clean_string(row.get("ext_arxiv")))
    if posted_month is None:
        return False
    if posted_from_month is not None and posted_month < posted_from_month:
        return False
    return posted_until_month is None or posted_month <= posted_until_month


def _selection_filters(
    primary_categories: set[str],
    min_citations: float,
    posted_from_month: str | None = None,
    posted_until_month: str | None = None,
) -> tuple[_SelectionFilter, ...]:
    """Build the ordered row filters used by the crawl ingester.

    Add future selection rules here, together with the parquet columns they
    require and the ExtractionSummary counter used for rejected rows.
    """
    filters = [
        _SelectionFilter(
            predicate=lambda row: _matches_primary_category(row, primary_categories),
            required_columns=("primary_category",),
            rejection_counter="skipped_wrong_category",
        ),
        _SelectionFilter(
            predicate=lambda row: _meets_minimum_citations(row, min_citations),
            required_columns=("citationCount",),
            rejection_counter="skipped_low_citation",
        ),
    ]
    if posted_from_month is not None or posted_until_month is not None:
        filters.append(
            _SelectionFilter(
                predicate=lambda row: _within_posted_month_range(
                    row,
                    posted_from_month,
                    posted_until_month,
                ),
                required_columns=("ext_arxiv",),
                rejection_counter="skipped_outside_posted_month_range",
            )
        )
    return tuple(filters)


def _passes_selection_filters(
    row: dict[str, Any],
    selection_filters: Sequence[_SelectionFilter],
    summary: ExtractionSummary,
) -> bool:
    for selection_filter in selection_filters:
        if selection_filter.predicate(row):
            continue
        rejection_count = getattr(summary, selection_filter.rejection_counter)
        setattr(summary, selection_filter.rejection_counter, rejection_count + 1)
        return False
    return True


def extract_from_crawl(
    crawl_root: str | os.PathLike[str],
    paper_root: str | os.PathLike[str],
    *,
    limit: int | None = None,
    primary_categories: list[str] | tuple[str, ...] | set[str] | None = None,
    min_citations: float = 10,
    posted_from_month: str | None = None,
    posted_until_month: str | None = None,
) -> ExtractionSummary:
    crawl_root = Path(crawl_root)
    paper_root = Path(paper_root)
    if posted_from_month is not None:
        posted_from_month = _validate_posted_month(posted_from_month)
    if posted_until_month is not None:
        posted_until_month = _validate_posted_month(posted_until_month)
    if (
        posted_from_month is not None
        and posted_until_month is not None
        and posted_from_month > posted_until_month
    ):
        raise ValueError("posted_from_month must not be after posted_until_month")
    primary_category_set = set(primary_categories or DEFAULT_PRIMARY_CATEGORIES)
    selection_filters = _selection_filters(
        primary_category_set,
        min_citations,
        posted_from_month,
        posted_until_month,
    )
    summary = ExtractionSummary()

    for chunk_path in _chunk_paths(crawl_root):
        parquet_file = pq.ParquetFile(chunk_path)
        columns = _read_columns(parquet_file, selection_filters)
        for row_group_idx in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(row_group_idx, columns=columns)
            for row in table.to_pylist():
                summary.inspected += 1
                arxiv_id = _clean_string(row.get("ext_arxiv"))
                if not arxiv_id:
                    summary.skipped_missing_arxiv_id += 1
                    continue
                if not _passes_selection_filters(row, selection_filters, summary):
                    continue

                summary.selected += 1
                full_text = _extract_full_text(row.get("content_json"))
                if not full_text:
                    summary.skipped_missing_text += 1
                    continue

                paper_dir = paper_root / safe_dir_name(arxiv_id)
                paper_dir.mkdir(parents=True, exist_ok=True)
                _write_json(paper_dir / "metadata.json", _metadata_from_row(row, arxiv_id))
                _write_text(paper_dir / "full_text.md", full_text)
                summary.written += 1
                if limit is not None and summary.written >= limit:
                    return summary
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract cited arXiv papers from the local parquet crawl into a MathArena paper root."
    )
    parser.add_argument("--crawl-root", default="../arxiv_papers_data", help="Root containing chunk_*.parquet files.")
    parser.add_argument("--paper-root", default="arxivmath/train_co_nt", help="Output paper-root directory.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on written papers.")
    parser.add_argument(
        "--primary-category",
        action="append",
        dest="primary_categories",
        default=None,
        help="Primary arXiv category to include. May be repeated. Defaults to math.CO and math.NT.",
    )
    parser.add_argument(
        "--min-citations",
        type=float,
        default=10,
        help="Minimum citationCount required for extraction.",
    )
    parser.add_argument(
        "--from-month",
        dest="posted_from_month",
        type=_validate_posted_month,
        default=None,
        help="Earliest initial posting month to include, in YYYY-MM format.",
    )
    parser.add_argument(
        "--until-month",
        dest="posted_until_month",
        type=_validate_posted_month,
        default=None,
        help="Latest initial posting month to include, in YYYY-MM format.",
    )
    args = parser.parse_args()

    categories = args.primary_categories or sorted(DEFAULT_PRIMARY_CATEGORIES)
    summary = extract_from_crawl(
        args.crawl_root,
        args.paper_root,
        limit=args.limit,
        primary_categories=categories,
        min_citations=args.min_citations,
        posted_from_month=args.posted_from_month,
        posted_until_month=args.posted_until_month,
    )
    posted_month_range = f"{args.posted_from_month or '*'}..{args.posted_until_month or '*'}"
    print(
        f"Extracted crawl papers for primary_categories={','.join(categories)} "
        f"min_citations={args.min_citations} posted_month_range={posted_month_range}: "
        f"inspected={summary.inspected}, selected={summary.selected}, written={summary.written}, "
        f"skipped_missing_arxiv_id={summary.skipped_missing_arxiv_id}, "
        f"skipped_wrong_category={summary.skipped_wrong_category}, "
        f"skipped_low_citation={summary.skipped_low_citation}, "
        f"skipped_outside_posted_month_range={summary.skipped_outside_posted_month_range}, "
        f"skipped_missing_text={summary.skipped_missing_text}"
    )


if __name__ == "__main__":
    main()
