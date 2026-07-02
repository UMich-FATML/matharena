import json
from pathlib import Path

import pandas as pd
import pytest


def _write_chunk(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_parquet(path, index=False)


def _paper_dir(root: Path, paper_id: str) -> Path:
    return root / paper_id.replace("/", "_")


def test_extract_co_nt_from_crawl_filters_and_writes_raw_paper_root(tmp_path):
    from arxivmath.scripts.train.extract_co_nt_from_crawl import extract_from_crawl

    crawl_root = tmp_path / "crawl"
    paper_root = tmp_path / "paper_root"
    crawl_root.mkdir()
    _write_chunk(
        crawl_root / "chunk_001.parquet",
        [
            {
                "ext_arxiv": "2401.00001",
                "primary_category": "math.CO",
                "title": "Kept combinatorics",
                "abstract": "A useful result.",
                "authors": "Ada Lovelace, Emmy Noether",
                "categories": "math.CO math.NT",
                "license": "http://creativecommons.org/licenses/by/4.0/",
                "doi": "10.1000/example",
                "journal-ref": "Example Journal",
                "update_date": "2026-01-02",
                "latest_version": "v2",
                "citationCount": 10.0,
                "content_json": json.dumps({"text": "Full combinatorics text"}),
                "cited_arxiv_ids": ["2401.00000"],
            },
            {
                "ext_arxiv": "2401.00002",
                "primary_category": "math.NT",
                "title": "Low citation number theory",
                "abstract": "Too low.",
                "authors": "Carl Gauss",
                "categories": "math.NT",
                "citationCount": 9.0,
                "content_json": json.dumps({"text": "Low citation text"}),
                "cited_arxiv_ids": [],
            },
            {
                "ext_arxiv": "2401.00003",
                "primary_category": "math.AG",
                "title": "Secondary combinatorics only",
                "abstract": "Not primary CO.",
                "authors": "Sofia Kovalevskaya",
                "categories": "math.AG math.CO",
                "citationCount": 99.0,
                "content_json": json.dumps({"text": "Wrong primary category text"}),
                "cited_arxiv_ids": [],
            },
        ],
    )

    summary = extract_from_crawl(crawl_root, paper_root)

    assert summary.selected == 1
    assert summary.written == 1
    kept_dir = _paper_dir(paper_root, "2401.00001")
    assert (kept_dir / "full_text.md").read_text(encoding="utf-8") == "Full combinatorics text\n"
    metadata = json.loads((kept_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["id"] == "2401.00001"
    assert metadata["primary_category"] == "math.CO"
    assert metadata["categories"] == ["math.CO", "math.NT"]
    assert metadata["citationCount"] == 10.0
    assert metadata["journal_ref"] == "Example Journal"
    assert metadata["authors"] == [
        {"forenames": "", "keyname": "Ada Lovelace"},
        {"forenames": "", "keyname": "Emmy Noether"},
    ]
    assert not _paper_dir(paper_root, "2401.00002").exists()
    assert not _paper_dir(paper_root, "2401.00003").exists()


def test_extract_co_nt_from_crawl_requires_full_text(tmp_path):
    from arxivmath.scripts.train.extract_co_nt_from_crawl import extract_from_crawl

    crawl_root = tmp_path / "crawl"
    paper_root = tmp_path / "paper_root"
    crawl_root.mkdir()
    _write_chunk(
        crawl_root / "chunk_001.parquet",
        [
            {
                "ext_arxiv": "2401.00004",
                "primary_category": "math.NT",
                "title": "No full text",
                "abstract": "Missing text.",
                "authors": "Srinivasa Ramanujan",
                "categories": "math.NT",
                "citationCount": 10.0,
                "content_json": json.dumps({"source": {}}),
            }
        ],
    )

    summary = extract_from_crawl(crawl_root, paper_root)

    assert summary.selected == 1
    assert summary.skipped_missing_text == 1
    assert summary.written == 0
    assert not _paper_dir(paper_root, "2401.00004").exists()


def test_fulltext_review_local_source_reads_full_text_without_ocr(tmp_path, monkeypatch):
    from arxivmath.scripts.shared import fulltext_review

    paper_root = tmp_path / "paper_root"
    paper_dir = _paper_dir(paper_root, "2401.00005")
    paper_dir.mkdir(parents=True)
    (paper_dir / "full_text.md").write_text("Local crawl text\n", encoding="utf-8")

    def fail_ocr(*args, **kwargs):
        raise AssertionError("OCR should not be used for local full-text source")

    monkeypatch.setattr(fulltext_review, "ensure_ocr_batch", fail_ocr)

    assert fulltext_review.load_full_texts(paper_root, ["2401.00005"], source="local") == {
        "2401.00005": "Local crawl text\n"
    }


def test_fulltext_review_local_source_errors_when_full_text_missing(tmp_path):
    from arxivmath.scripts.shared import fulltext_review

    paper_root = tmp_path / "paper_root"
    _paper_dir(paper_root, "2401.00006").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="full_text.md"):
        fulltext_review.load_full_texts(paper_root, ["2401.00006"], source="local")
