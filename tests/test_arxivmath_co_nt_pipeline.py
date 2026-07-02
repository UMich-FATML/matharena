import json
import os
import subprocess
import sys
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


def test_create_queries_local_full_text_source_injects_full_text(tmp_path, monkeypatch):
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
            "--full-text-source",
            "local",
        ],
    )

    create_queries.main()

    assert len(captured_queries) == 1
    prompt = captured_queries[0][0]["content"]
    assert "Title=Full-text title" in prompt
    assert "Abstract=Abstract only." in prompt
    assert "Full=Full paper body with theorem details.\n" in prompt


def test_create_queries_local_full_text_source_errors_when_full_text_missing(tmp_path, monkeypatch):
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
            "--full-text-source",
            "local",
        ],
    )

    with pytest.raises(FileNotFoundError, match="full_text.md"):
        create_queries.main()


def test_create_train_co_nt_script_supports_fulltext_model_override(tmp_path):
    matharena_root = Path(__file__).resolve().parents[1]
    script_path = matharena_root / "arxivmath" / "scripts" / "create_train_co_nt.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    pixi_log = tmp_path / "pixi_calls.log"
    fake_pixi = fake_bin / "pixi"
    fake_pixi.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >> "$PIXI_CALL_LOG"\n',
        encoding="utf-8",
    )
    fake_pixi.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    env["PIXI_CALL_LOG"] = str(pixi_log)
    env["LIMIT"] = "20"
    env["MODEL_CONFIG"] = "openai/gpt-54-high"
    env["FULLTEXT_REVIEW_MODEL_CONFIG"] = "anthropic/opus_47_high"

    subprocess.run(["bash", str(script_path)], cwd=matharena_root, env=env, check=True)

    calls = pixi_log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 4
    assert "extract_co_nt_from_crawl.py" in calls[0]
    assert "--limit 20" in calls[0]
    assert "create_queries.py" in calls[1]
    assert "--model-config openai/gpt-54-high" in calls[1]
    assert "--limit 20" in calls[1]
    assert "verify_queries.py" in calls[2]
    assert "--model-config openai/gpt-54-high" in calls[2]
    assert "--limit 20" in calls[2]
    assert "fulltext_review.py" in calls[3]
    assert "--model-config anthropic/opus_47_high" in calls[3]
    assert "--full-text-source local" in calls[3]
    assert "--limit 20" in calls[3]
