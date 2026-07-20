import json
import sys
from pathlib import Path

import pandas as pd
import pytest


def _write_chunk(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_parquet(path, index=False)


def _paper_dir(root: Path, paper_id: str) -> Path:
    return root / paper_id.replace("/", "_")


def test_ingest_arxiv_crawl_filters_and_writes_raw_paper_root(tmp_path):
    from arxivmath.scripts.train.ingest_arxiv_crawl import extract_from_crawl

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

    summary = extract_from_crawl(crawl_root, paper_root, primary_categories=["math.CO", "math.NT"])

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


def test_extract_from_crawl_accepts_requested_primary_categories(tmp_path):
    from arxivmath.scripts.train.ingest_arxiv_crawl import extract_from_crawl

    crawl_root = tmp_path / "crawl"
    paper_root = tmp_path / "paper_root"
    crawl_root.mkdir()
    _write_chunk(
        crawl_root / "chunk_001.parquet",
        [
            {
                "ext_arxiv": "2401.01001",
                "primary_category": "math.AP",
                "title": "Kept PDEs",
                "abstract": "A useful PDE result.",
                "authors": "Olga Ladyzhenskaya",
                "categories": "math.AP math.NA",
                "citationCount": 10.0,
                "content_json": json.dumps({"text": "Full PDE text"}),
            },
            {
                "ext_arxiv": "2401.01002",
                "primary_category": "cs.IT",
                "title": "Kept information theory",
                "abstract": "A useful coding result.",
                "authors": "Claude Shannon",
                "categories": "cs.IT math.IT",
                "citationCount": 11.0,
                "content_json": json.dumps({"text": "Full information theory text"}),
            },
            {
                "ext_arxiv": "2401.01003",
                "primary_category": "math.NT",
                "title": "Wrong primary category",
                "abstract": "Not requested.",
                "authors": "Carl Gauss",
                "categories": "math.NT math.AP",
                "citationCount": 99.0,
                "content_json": json.dumps({"text": "Wrong category text"}),
            },
        ],
    )

    summary = extract_from_crawl(
        crawl_root,
        paper_root,
        primary_categories=["math.AP", "cs.IT"],
        min_citations=10,
    )

    assert summary.selected == 2
    assert summary.written == 2
    assert summary.skipped_wrong_category == 1
    assert json.loads((_paper_dir(paper_root, "2401.01001") / "metadata.json").read_text(encoding="utf-8"))[
        "primary_category"
    ] == "math.AP"
    assert json.loads((_paper_dir(paper_root, "2401.01002") / "metadata.json").read_text(encoding="utf-8"))[
        "primary_category"
    ] == "cs.IT"
    assert not _paper_dir(paper_root, "2401.01003").exists()


def test_extract_from_crawl_omits_category_filter_when_categories_are_none(tmp_path):
    from arxivmath.scripts.train.ingest_arxiv_crawl import extract_from_crawl

    crawl_root = tmp_path / "crawl"
    paper_root = tmp_path / "paper_root"
    crawl_root.mkdir()
    _write_chunk(
        crawl_root / "chunk_001.parquet",
        [
            {
                "ext_arxiv": "invalid-but-present",
                "primary_category": "math.AG",
                "citationCount": 10,
                "content_json": json.dumps({"text": "Full algebraic geometry text"}),
            }
        ],
    )

    summary = extract_from_crawl(crawl_root, paper_root, primary_categories=None)

    assert summary.written == 1
    assert summary.skipped_wrong_category == 0


def test_extract_from_crawl_empty_category_collection_selects_nothing(tmp_path):
    from arxivmath.scripts.train.ingest_arxiv_crawl import extract_from_crawl

    crawl_root = tmp_path / "crawl"
    paper_root = tmp_path / "paper_root"
    crawl_root.mkdir()
    _write_chunk(
        crawl_root / "chunk_001.parquet",
        [
            {
                "ext_arxiv": "2401.01004",
                "primary_category": "math.AG",
                "citationCount": 10,
                "content_json": json.dumps({"text": "Full algebraic geometry text"}),
            }
        ],
    )

    summary = extract_from_crawl(crawl_root, paper_root, primary_categories=[])

    assert summary.written == 0
    assert summary.skipped_wrong_category == 1


def test_extract_from_crawl_supports_custom_min_citations(tmp_path):
    from arxivmath.scripts.train.ingest_arxiv_crawl import extract_from_crawl

    crawl_root = tmp_path / "crawl"
    paper_root = tmp_path / "paper_root"
    crawl_root.mkdir()
    _write_chunk(
        crawl_root / "chunk_001.parquet",
        [
            {
                "ext_arxiv": "2401.02001",
                "primary_category": "math.AP",
                "title": "Below custom citation threshold",
                "abstract": "Too low.",
                "authors": "Maryam Mirzakhani",
                "categories": "math.AP",
                "citationCount": 10.0,
                "content_json": json.dumps({"text": "Low citation text"}),
            },
            {
                "ext_arxiv": "2401.02002",
                "primary_category": "math.AP",
                "title": "Meets custom citation threshold",
                "abstract": "High enough.",
                "authors": "Karen Uhlenbeck",
                "categories": "math.AP",
                "citationCount": 12.0,
                "content_json": json.dumps({"text": "High citation text"}),
            },
        ],
    )

    summary = extract_from_crawl(crawl_root, paper_root, primary_categories=["math.AP"], min_citations=12)

    assert summary.selected == 1
    assert summary.written == 1
    assert summary.skipped_low_citation == 1
    assert not _paper_dir(paper_root, "2401.02001").exists()
    assert _paper_dir(paper_root, "2401.02002").exists()


@pytest.mark.parametrize(
    ("arxiv_id", "expected"),
    [
        ("2401.01234", "2024-01"),
        ("2401.01234v2", "2024-01"),
        ("arXiv:0704.0001", "2007-04"),
        ("math/0301001", "2003-01"),
        ("acc-phys/9609003", "1996-09"),
        ("not-an-arxiv-id", None),
        ("2413.01234", None),
    ],
)
def test_posted_month_is_derived_from_modern_and_legacy_arxiv_ids(arxiv_id, expected):
    from arxivmath.scripts.train.ingest_arxiv_crawl import _posted_month_from_arxiv_id

    assert _posted_month_from_arxiv_id(arxiv_id) == expected


def test_extract_from_crawl_filters_by_requested_posted_months(tmp_path):
    from arxivmath.scripts.train.ingest_arxiv_crawl import extract_from_crawl

    crawl_root = tmp_path / "crawl"
    paper_root = tmp_path / "paper_root"
    crawl_root.mkdir()
    rows = []
    for arxiv_id in ("2312.00001", "2401.00001", "2403.00001", "2404.00001", "invalid"):
        rows.append(
            {
                "ext_arxiv": arxiv_id,
                "primary_category": "math.AP",
                "citationCount": 10,
                "content_json": json.dumps({"text": f"Full text for {arxiv_id}"}),
            }
        )
    _write_chunk(crawl_root / "chunk_001.parquet", rows)

    summary = extract_from_crawl(
        crawl_root,
        paper_root,
        primary_categories=["math.AP"],
        months=["2024-01", "2024-03", "2024-03"],
    )

    assert summary.selected == 2
    assert summary.written == 2
    assert summary.skipped_unselected_posted_month == 3
    assert _paper_dir(paper_root, "2401.00001").exists()
    assert _paper_dir(paper_root, "2403.00001").exists()
    assert not _paper_dir(paper_root, "2312.00001").exists()
    assert not _paper_dir(paper_root, "2404.00001").exists()


@pytest.mark.parametrize("month", ["2024-1", "2024-13"])
def test_extract_from_crawl_validates_requested_months(tmp_path, month):
    from arxivmath.scripts.train.ingest_arxiv_crawl import extract_from_crawl

    with pytest.raises(ValueError, match="expected YYYY-MM"):
        extract_from_crawl(
            tmp_path / "crawl",
            tmp_path / "paper_root",
            months=["2024-01", month],
        )


def test_extract_from_crawl_empty_month_collection_selects_nothing(tmp_path):
    from arxivmath.scripts.train.ingest_arxiv_crawl import extract_from_crawl

    crawl_root = tmp_path / "crawl"
    paper_root = tmp_path / "paper_root"
    crawl_root.mkdir()
    _write_chunk(
        crawl_root / "chunk_001.parquet",
        [
            {
                "ext_arxiv": "2401.00001",
                "primary_category": "math.AP",
                "citationCount": 10,
                "content_json": json.dumps({"text": "Full text"}),
            }
        ],
    )

    summary = extract_from_crawl(crawl_root, paper_root, months=[])

    assert summary.written == 0
    assert summary.skipped_unselected_posted_month == 1


def test_ingest_argument_parser_uses_cli_only_category_defaults():
    from arxivmath.scripts.train.ingest_arxiv_crawl import _create_argument_parser

    parser = _create_argument_parser()

    assert parser.parse_args([]).primary_categories == ["math.CO", "math.NT"]
    explicit = parser.parse_args(["--primary-category", "math.AP", "cs.IT"])
    assert explicit.primary_categories == ["math.AP", "cs.IT"]


def test_selection_filters_stop_after_first_rejection():
    from arxivmath.scripts.train.ingest_arxiv_crawl import (
        ExtractionSummary,
        _passes_selection_filters,
        _selection_filters,
    )

    summary = ExtractionSummary()
    filters = _selection_filters({"math.AP"}, min_citations=10)

    accepted = _passes_selection_filters(
        {"primary_category": "math.NT", "citationCount": 1},
        filters,
        summary,
    )

    assert not accepted
    assert summary.skipped_wrong_category == 1
    assert summary.skipped_low_citation == 0


def test_selection_filters_accept_row_matching_every_filter():
    from arxivmath.scripts.train.ingest_arxiv_crawl import (
        ExtractionSummary,
        _passes_selection_filters,
        _selection_filters,
    )

    summary = ExtractionSummary()
    filters = _selection_filters({"math.AP"}, min_citations=10)

    accepted = _passes_selection_filters(
        {"primary_category": "math.AP", "citationCount": 10},
        filters,
        summary,
    )

    assert accepted
    assert summary.skipped_wrong_category == 0
    assert summary.skipped_low_citation == 0


def test_ingest_arxiv_crawl_requires_full_text(tmp_path):
    from arxivmath.scripts.train.ingest_arxiv_crawl import extract_from_crawl

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


def test_create_queries_fulltext_mode_injects_full_text(tmp_path, monkeypatch):
    from arxivmath.scripts.shared import create_queries

    paper_root = tmp_path / "paper_root"
    paper_dir = _paper_dir(paper_root, "2401.00007")
    paper_dir.mkdir(parents=True)
    (paper_dir / "metadata.json").write_text(
        json.dumps({"title": "Full-text title", "abstract": "Abstract only."}),
        encoding="utf-8",
    )
    (paper_dir / "full_text.md").write_text("Full paper body with theorem details.\n", encoding="utf-8")
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("Title={title}\nAbstract={abstract}\nFull={full_text}", encoding="utf-8")
    model_config_path = tmp_path / "model.yaml"
    model_config_path.write_text("model: fake-model\napi: fake\n", encoding="utf-8")

    captured_queries = []

    class FakeClient:
        def __init__(self, **kwargs):
            assert kwargs["model"] == "fake-model"

        def run_queries(self, queries):
            captured_queries.extend(queries)
            yield 0, [{"role": "assistant", "content": '{"keep": false}'}], {"cost": 0.0}

    monkeypatch.setattr(create_queries, "APIClient", FakeClient)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "create_queries.py",
            "--model-config",
            str(model_config_path),
            "--paper-root",
            str(paper_root),
            "--prompt",
            str(prompt_path),
            "--fulltext",
        ],
    )

    create_queries.main()

    assert len(captured_queries) == 1
    prompt = captured_queries[0][0]["content"]
    assert "Title=Full-text title" in prompt
    assert "Abstract=Abstract only." in prompt
    assert "Full=Full paper body with theorem details.\n" in prompt


def test_create_queries_fulltext_mode_errors_when_full_text_missing(tmp_path, monkeypatch):
    from arxivmath.scripts.shared import create_queries

    paper_root = tmp_path / "paper_root"
    paper_dir = _paper_dir(paper_root, "2401.00008")
    paper_dir.mkdir(parents=True)
    (paper_dir / "metadata.json").write_text(
        json.dumps({"title": "Missing full text", "abstract": "Abstract only."}),
        encoding="utf-8",
    )
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("Full={full_text}", encoding="utf-8")
    model_config_path = tmp_path / "model.yaml"
    model_config_path.write_text("model: fake-model\napi: fake\n", encoding="utf-8")

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def run_queries(self, queries):
            raise AssertionError("run_queries should not be called when full_text.md is missing")

    monkeypatch.setattr(create_queries, "APIClient", FakeClient)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "create_queries.py",
            "--model-config",
            str(model_config_path),
            "--paper-root",
            str(paper_root),
            "--prompt",
            str(prompt_path),
            "--fulltext",
        ],
    )

    with pytest.raises(FileNotFoundError, match="full_text.md"):
        create_queries.main()


@pytest.mark.parametrize(
    ("mode_args", "expected_prompt"),
    [
        ([], "arxivmath/prompts/arxiv/query.md"),
        (["--fulltext"], "arxivmath/prompts/arxiv/fulltext_query.md"),
        (["--false"], "arxivmath/prompts/broken/false.md"),
        (["--lean"], "arxivmath/prompts/lean/extract_lean_abstract.md"),
    ],
)
def test_create_queries_modes_select_default_prompt(monkeypatch, mode_args, expected_prompt):
    from arxivmath.scripts.shared import create_queries

    loaded_prompts = []
    monkeypatch.setattr(create_queries, "load_prompt_template", loaded_prompts.append)
    monkeypatch.setattr(create_queries, "resolve_model_config_path", lambda path: path)
    monkeypatch.setattr(create_queries, "load_model_config", lambda path: {"model": "fake-model"})
    monkeypatch.setattr(create_queries, "APIClient", lambda **kwargs: object())
    monkeypatch.setattr(create_queries, "list_paper_ids", lambda paper_root: [])
    monkeypatch.setattr(sys, "argv", ["create_queries.py", "--model-config", "fake.yaml", *mode_args])

    create_queries.main()

    assert loaded_prompts == [expected_prompt]


def test_create_queries_standard_mode_does_not_supply_full_text(monkeypatch):
    from arxivmath.scripts.shared import create_queries

    formatted_values = []

    class RecordingTemplate:
        def format(self, **values):
            formatted_values.append(values)
            return "formatted prompt"

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def run_queries(self, queries):
            yield 0, [{"role": "assistant", "content": '{"keep": false}'}], {"cost": 0.0}

    monkeypatch.setattr(create_queries, "load_prompt_template", lambda path: RecordingTemplate())
    monkeypatch.setattr(create_queries, "resolve_model_config_path", lambda path: path)
    monkeypatch.setattr(create_queries, "load_model_config", lambda path: {"model": "fake-model"})
    monkeypatch.setattr(create_queries, "APIClient", FakeClient)
    monkeypatch.setattr(create_queries, "list_paper_ids", lambda paper_root: ["2401.00009"])
    monkeypatch.setattr(create_queries, "load_annotation", lambda *args: {})
    monkeypatch.setattr(
        create_queries,
        "load_metadata",
        lambda *args: {"title": " Title ", "abstract": " Abstract "},
    )
    monkeypatch.setattr(create_queries, "save_annotation", lambda *args: None)
    monkeypatch.setattr(sys, "argv", ["create_queries.py", "--model-config", "fake.yaml"])

    create_queries.main()

    assert formatted_values == [{"title": "Title", "abstract": "Abstract"}]


@pytest.mark.parametrize("other_mode", ["--false", "--lean"])
def test_create_queries_fulltext_mode_is_mutually_exclusive(monkeypatch, other_mode):
    from arxivmath.scripts.shared import create_queries

    monkeypatch.setattr(
        sys,
        "argv",
        ["create_queries.py", "--model-config", "fake.yaml", "--fulltext", other_mode],
    )

    with pytest.raises(SystemExit, match="2"):
        create_queries.main()
