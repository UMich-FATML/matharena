#!/usr/bin/env python3
import argparse
import json
import math
import os
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
    skipped_missing_text: int = 0


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


def _read_columns(parquet_file: pq.ParquetFile) -> list[str]:
    available = set(parquet_file.schema_arrow.names)
    return [column for column in DEFAULT_COLUMNS if column in available]


def extract_from_crawl(
    crawl_root: str | os.PathLike[str],
    paper_root: str | os.PathLike[str],
    *,
    limit: int | None = None,
    primary_categories: list[str] | tuple[str, ...] | set[str] | None = None,
    min_citations: float = 10,
) -> ExtractionSummary:
    crawl_root = Path(crawl_root)
    paper_root = Path(paper_root)
    primary_category_set = set(primary_categories or DEFAULT_PRIMARY_CATEGORIES)
    summary = ExtractionSummary()

    for chunk_path in _chunk_paths(crawl_root):
        parquet_file = pq.ParquetFile(chunk_path)
        columns = _read_columns(parquet_file)
        for row_group_idx in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(row_group_idx, columns=columns)
            for row in table.to_pylist():
                summary.inspected += 1
                arxiv_id = _clean_string(row.get("ext_arxiv"))
                if not arxiv_id:
                    summary.skipped_missing_arxiv_id += 1
                    continue
                if _clean_string(row.get("primary_category")) not in primary_category_set:
                    summary.skipped_wrong_category += 1
                    continue
                citation_count = _clean_scalar(row.get("citationCount"))
                if citation_count is None or float(citation_count) < min_citations:
                    summary.skipped_low_citation += 1
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
    args = parser.parse_args()

    categories = args.primary_categories or sorted(DEFAULT_PRIMARY_CATEGORIES)
    summary = extract_from_crawl(
        args.crawl_root,
        args.paper_root,
        limit=args.limit,
        primary_categories=categories,
        min_citations=args.min_citations,
    )
    print(
        f"Extracted crawl papers for primary_categories={','.join(categories)} "
        f"min_citations={args.min_citations}: "
        f"inspected={summary.inspected}, selected={summary.selected}, written={summary.written}, "
        f"skipped_missing_arxiv_id={summary.skipped_missing_arxiv_id}, "
        f"skipped_wrong_category={summary.skipped_wrong_category}, "
        f"skipped_low_citation={summary.skipped_low_citation}, "
        f"skipped_missing_text={summary.skipped_missing_text}"
    )


if __name__ == "__main__":
    main()
