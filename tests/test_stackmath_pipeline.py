import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


MATHARENA_ROOT = Path(__file__).resolve().parents[1]
QUERY_SAFEGUARD = (
    "   Accepted-answer status and community scores are evidence, not authority. Consider the mathematical "
    "content of the entire discussion and reject the discussion thread if its answers leave the claimed result "
    "unresolved or give incompatible conclusions."
)
REVIEW_SAFEGUARD = (
    "Accepted-answer status and community scores are evidence, not authority. Consider the mathematical content "
    "of the entire discussion and discard the question if the answers leave the claimed result unresolved or give "
    "incompatible conclusions."
)


def _adapt_prompt(source: str, replacements: list[tuple[str, str]], anchor: str, safeguard: str) -> str:
    for old, new in replacements:
        source = source.replace(old, new)
    assert source.count(anchor) == 1
    return source.replace(anchor, f"{anchor}\n\n{safeguard}")


def _write_part(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_parquet(path, index=False)


def _row(**overrides) -> dict:
    row = {
        "Id": "123",
        "PostTypeId": "1",
        "AcceptedAnswerId": "9001",
        "CreationDate": "2024-01-02T03:04:05.000",
        "Score": 3,
        "ViewCount": 42,
        "Body": "<p>Raw <strong>question</strong> body.</p>",
        "OwnerUserId": "77",
        "Title": "A mathematical question",
        "Tags": "|algebra|number-theory|",
        "AnswerCount": 2,
        "CommentCount": 4,
        "ContentLicense": "CC BY-SA 4.0",
        "ClosedDate": None,
        "CommunityOwnedDate": None,
        "ParentId": None,
        "site": "math.stackexchange.com",
        "tags_list": None,
        "dump_answer_count": 2,
        "answer_scores": [5, 2],
        "accepted_answer_body": "<p>Accepted answer.</p>",
        "best_answer_body": "<p>Higher-scored answer.</p>",
        "answer_bodies": ["<p>Higher-scored answer.</p>", "<p>Accepted answer.</p>"],
        "url": "https://math.stackexchange.com/questions/123/example",
    }
    row.update(overrides)
    return row


def _thread_dir(root: Path, site: str, question_id: str) -> Path:
    from stackmath.scripts.train.ingest_stackexchange_crawl import safe_dir_name

    return root / safe_dir_name(site, question_id)


def test_ingest_cli_defaults_and_list_valued_sites():
    from stackmath.scripts.train.ingest_stackexchange_crawl import DEFAULT_SITES, _create_argument_parser

    parser = _create_argument_parser()
    defaults = parser.parse_args([])

    assert defaults.sites == list(DEFAULT_SITES)
    assert defaults.min_question_score == 3
    assert defaults.min_answer_score == 3
    assert defaults.limit_per_site is None
    assert defaults.include_closed is False
    assert parser.parse_args(["--site", "math.stackexchange.com", "mathoverflow.net"]).sites == [
        "math.stackexchange.com",
        "mathoverflow.net",
    ]


def test_ingest_filters_and_writes_complete_discussions(tmp_path):
    from stackmath.scripts.train.ingest_stackexchange_crawl import extract_from_crawl

    crawl_root = tmp_path / "crawl"
    paper_root = tmp_path / "paper_root"
    crawl_root.mkdir()
    rows = [
        _row(),
        _row(
            site="mathoverflow.net",
            Id="123",
            AcceptedAnswerId="9002",
            accepted_answer_body="<p>MO accepted.</p>",
            answer_bodies=["<p>MO accepted.</p>"],
            answer_scores=[3],
            AnswerCount=1,
            dump_answer_count=1,
            Tags="|combinatorics|",
            url="https://mathoverflow.net/questions/123/example",
        ),
        _row(Id="low-question", Score=2),
        _row(Id="closed", ClosedDate="2024-01-03T00:00:00.000"),
        _row(Id="low-answer", answer_scores=[2, 1]),
        _row(Id="no-answers", answer_scores=[], answer_bodies=[], AcceptedAnswerId=None, accepted_answer_body=None),
        _row(Id="no-accepted", AcceptedAnswerId=None, accepted_answer_body=None),
    ]
    _write_part(crawl_root / "questions_part_001.parquet", rows)

    summary = extract_from_crawl(crawl_root, paper_root)

    assert summary.written == 2
    assert summary.written_by_site == {"math.stackexchange.com": 1, "mathoverflow.net": 1}
    assert summary.skipped_low_question_score == 1
    assert summary.skipped_closed == 1
    assert summary.skipped_low_answer_score == 1
    assert summary.skipped_no_answers == 1
    assert summary.skipped_missing_accepted_answer == 1

    mse_dir = _thread_dir(paper_root, "math.stackexchange.com", "123")
    mo_dir = _thread_dir(paper_root, "mathoverflow.net", "123")
    assert mse_dir != mo_dir
    assert mse_dir.is_dir() and mo_dir.is_dir()

    metadata = json.loads((mse_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["id"] == "math.stackexchange.com:123"
    assert metadata["site"] == "math.stackexchange.com"
    assert metadata["question_id"] == "123"
    assert metadata["body"] == "<p>Raw <strong>question</strong> body.</p>"
    assert metadata["abstract"] == metadata["body"]
    assert metadata["tags"] == ["algebra", "number-theory"]
    assert metadata["categories"] == metadata["tags"]
    assert metadata["primary_category"] == "algebra"
    assert metadata["answer_scores"] == [5, 2]
    assert metadata["accepted_answer_position"] == 2
    assert metadata["content_license"] == "CC BY-SA 4.0"
    assert metadata["url"] == "https://math.stackexchange.com/questions/123/example"

    full_text = (mse_dir / "full_text.md").read_text(encoding="utf-8")
    assert "<p>Raw <strong>question</strong> body.</p>" in full_text
    assert full_text.index("### Answer 1 (score: 5)") < full_text.index("### Answer 2 (score: 2, accepted)")
    assert "<p>Higher-scored answer.</p>" in full_text
    assert "<p>Accepted answer.</p>" in full_text


def test_ingest_applies_per_site_quotas_after_validation(tmp_path):
    from stackmath.scripts.train.ingest_stackexchange_crawl import extract_from_crawl

    crawl_root = tmp_path / "crawl"
    paper_root = tmp_path / "paper_root"
    crawl_root.mkdir()
    _write_part(
        crawl_root / "questions_part_001.parquet",
        [
            _row(Id="invalid-first", Score=0),
            _row(Id="mse-first"),
            _row(Id="mse-second"),
            _row(
                site="mathoverflow.net",
                Id="mo-first",
                url="https://mathoverflow.net/questions/mo-first/example",
            ),
            _row(
                site="mathoverflow.net",
                Id="mo-second",
                url="https://mathoverflow.net/questions/mo-second/example",
            ),
        ],
    )

    summary = extract_from_crawl(crawl_root, paper_root, limit_per_site=1)

    assert summary.written == 2
    assert summary.written_by_site == {"math.stackexchange.com": 1, "mathoverflow.net": 1}
    assert summary.skipped_low_question_score == 1
    assert _thread_dir(paper_root, "math.stackexchange.com", "mse-first").is_dir()
    assert not _thread_dir(paper_root, "math.stackexchange.com", "mse-second").exists()
    assert _thread_dir(paper_root, "mathoverflow.net", "mo-first").is_dir()
    assert not _thread_dir(paper_root, "mathoverflow.net", "mo-second").exists()


def test_ingest_closed_questions_are_opt_in(tmp_path):
    from stackmath.scripts.train.ingest_stackexchange_crawl import extract_from_crawl

    crawl_root = tmp_path / "crawl"
    crawl_root.mkdir()
    _write_part(crawl_root / "questions_part_001.parquet", [_row(Id="closed", ClosedDate="2024-01-03")])

    excluded = extract_from_crawl(crawl_root, tmp_path / "excluded")
    included = extract_from_crawl(crawl_root, tmp_path / "included", include_closed=True)

    assert excluded.written == 0
    assert excluded.skipped_closed == 1
    assert included.written == 1


def test_ingest_rejects_malformed_or_unmatchable_accepted_answers(tmp_path):
    from stackmath.scripts.train.ingest_stackexchange_crawl import extract_from_crawl

    crawl_root = tmp_path / "crawl"
    crawl_root.mkdir()
    _write_part(
        crawl_root / "questions_part_001.parquet",
        [
            _row(Id="misaligned", answer_scores=[5], answer_bodies=["a", "b"]),
            _row(Id="missing-match", accepted_answer_body="not present"),
            _row(
                Id="ambiguous-match",
                accepted_answer_body="duplicate",
                answer_bodies=["duplicate", "duplicate"],
                answer_scores=[5, 4],
            ),
        ],
    )

    summary = extract_from_crawl(crawl_root, tmp_path / "paper_root")

    assert summary.written == 0
    assert summary.skipped_malformed_answers == 1
    assert summary.skipped_accepted_answer_not_in_answers == 1
    assert summary.skipped_ambiguous_accepted_answer == 1


def test_ingest_prunes_files_using_site_statistics(tmp_path):
    from stackmath.scripts.train.ingest_stackexchange_crawl import extract_from_crawl

    crawl_root = tmp_path / "crawl"
    crawl_root.mkdir()
    _write_part(
        crawl_root / "questions_part_001.parquet",
        [_row(site="superuser.com", Id="not-math", url="https://superuser.com/questions/not-math")],
    )

    summary = extract_from_crawl(crawl_root, tmp_path / "paper_root", sites=["math.stackexchange.com"])

    assert summary.files_inspected == 1
    assert summary.files_skipped_by_site_statistics == 1
    assert summary.row_groups_inspected == 0
    assert summary.inspected == 0


def test_stackmath_shared_stages_select_stackmath_prompts(monkeypatch):
    from stackmath.scripts.shared import create_queries, fulltext_review, verify_queries

    cases = [
        (
            create_queries,
            ["create_queries.py", "--model-config", "fake", "--fulltext"],
            "stackmath/prompts/stackexchange/fulltext_query.md",
        ),
        (
            verify_queries,
            ["verify_queries.py", "--model-config", "fake"],
            "stackmath/prompts/stackexchange/verify.md",
        ),
        (
            fulltext_review,
            ["fulltext_review.py", "--model-config", "fake", "--full-text-source", "local"],
            "stackmath/prompts/stackexchange/fulltext_review.md",
        ),
    ]

    for module, argv, expected_prompt in cases:
        loaded_prompts = []
        monkeypatch.setattr(module, "load_prompt_template", loaded_prompts.append)
        monkeypatch.setattr(module, "resolve_model_config_path", lambda path: path)
        monkeypatch.setattr(module, "load_model_config", lambda path: {"model": "fake-model"})
        monkeypatch.setattr(module, "APIClient", lambda **kwargs: object())
        monkeypatch.setattr(module, "list_paper_ids", lambda paper_root: [])
        monkeypatch.setattr(sys, "argv", argv)

        module.main()

        assert loaded_prompts == [expected_prompt]


def test_stackmath_prompts_are_minimal_arxivmath_adaptations():
    arxiv_prompt_root = MATHARENA_ROOT / "arxivmath" / "prompts" / "arxiv"
    stack_prompt_root = MATHARENA_ROOT / "stackmath" / "prompts" / "stackexchange"

    assert (stack_prompt_root / "verify.md").read_text(encoding="utf-8") == (
        arxiv_prompt_root / "verify.md"
    ).read_text(encoding="utf-8")

    arxiv_query = (arxiv_prompt_root / "fulltext_query.md").read_text(encoding="utf-8")
    expected_query = _adapt_prompt(
        arxiv_query,
        [
            ("research papers", "Stack Exchange discussion threads"),
            ("paper or abstract", "discussion thread or original question"),
            ("a research paper", "a Stack Exchange discussion thread"),
            ("original abstract or paper", "original question or discussion thread"),
            ("papers", "discussion threads"),
            ("paper", "discussion thread"),
            ("authors", "participants"),
            ("in this work", "in this discussion"),
            ("# Abstract", "# Original question"),
        ],
        "   The answer must be derivable *directly and unambiguously* from the provided full discussion thread "
        "text, without requiring external references.",
        QUERY_SAFEGUARD,
    )
    assert (stack_prompt_root / "fulltext_query.md").read_text(encoding="utf-8") == expected_query

    arxiv_review = (arxiv_prompt_root / "fulltext_review.md").read_text(encoding="utf-8")
    expected_review = _adapt_prompt(
        arxiv_review,
        [
            ("a research paper", "a Stack Exchange discussion thread"),
            ("paper's", "discussion thread's"),
            ("full paper", "full discussion thread"),
            ("full text", "full discussion thread"),
            ("paper", "discussion thread"),
            ("authors", "participants"),
            ("abstract", "original question"),
            ("in this work", "in this discussion"),
        ],
        "- Keep the question if it is already accurate and central.",
        REVIEW_SAFEGUARD,
    )
    assert (stack_prompt_root / "fulltext_review.md").read_text(encoding="utf-8") == expected_review


def test_stackmath_generation_injects_the_archived_discussion(tmp_path, monkeypatch):
    from stackmath.scripts.shared import create_queries

    paper_root = tmp_path / "paper_root"
    thread_dir = paper_root / "math.stackexchange.com_123"
    thread_dir.mkdir(parents=True)
    (thread_dir / "metadata.json").write_text(
        json.dumps({"title": "Thread title", "abstract": "Original body", "license": "CC BY-SA 4.0"}),
        encoding="utf-8",
    )
    (thread_dir / "full_text.md").write_text("Question and all answers\n", encoding="utf-8")
    model_config = tmp_path / "model.yaml"
    model_config.write_text("model: fake-model\napi: fake\n", encoding="utf-8")
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
            str(model_config),
            "--paper-root",
            str(paper_root),
            "--fulltext",
        ],
    )

    create_queries.main()

    prompt = captured_queries[0][0]["content"]
    assert "Thread title" in prompt
    assert "Question and all answers\n" in prompt
    annotation = json.loads((thread_dir / "llm_annotation.json").read_text(encoding="utf-8"))
    assert annotation["keep"] is False


def test_stackmath_verification_and_review_update_annotations(tmp_path, monkeypatch):
    from stackmath.scripts.shared import fulltext_review, verify_queries

    paper_root = tmp_path / "paper_root"
    thread_dir = paper_root / "math.stackexchange.com_123"
    thread_dir.mkdir(parents=True)
    (thread_dir / "metadata.json").write_text(
        json.dumps({"title": "Thread title", "abstract": "Original body", "authors": []}),
        encoding="utf-8",
    )
    (thread_dir / "full_text.md").write_text("Complete archived discussion\n", encoding="utf-8")
    (thread_dir / "llm_annotation.json").write_text(
        json.dumps({"keep": True, "question": "Original standalone question?", "answer": "2"}),
        encoding="utf-8",
    )
    model_config = tmp_path / "model.yaml"
    model_config.write_text("model: fake-model\napi: fake\n", encoding="utf-8")

    class VerifyClient:
        def __init__(self, **kwargs):
            pass

        def run_queries(self, queries):
            assert "Original standalone question?" in queries[0][0]["content"]
            yield 0, [{"role": "assistant", "content": '{"keep": true}'}], {"cost": 0.0}

    monkeypatch.setattr(verify_queries, "APIClient", VerifyClient)
    monkeypatch.setattr(
        sys,
        "argv",
        ["verify_queries.py", "--model-config", str(model_config), "--paper-root", str(paper_root)],
    )
    verify_queries.main()

    verified = json.loads((thread_dir / "llm_annotation.json").read_text(encoding="utf-8"))
    assert verified["keep"] is True
    assert verified["verification"]["keep"] is True

    class ReviewClient:
        def __init__(self, **kwargs):
            pass

        def run_queries(self, queries):
            prompt = queries[0][0]["content"]
            assert "Original standalone question?" in prompt
            assert "Complete archived discussion\n" in prompt
            response = {
                "action": "edit",
                "question": "Original standalone question, under the archived assumption?",
                "rationale": "A necessary assumption was omitted.",
            }
            yield 0, [{"role": "assistant", "content": json.dumps(response)}], {"cost": 0.0}

    monkeypatch.setattr(fulltext_review, "APIClient", ReviewClient)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fulltext_review.py",
            "--model-config",
            str(model_config),
            "--paper-root",
            str(paper_root),
            "--full-text-source",
            "local",
        ],
    )
    fulltext_review.main()

    reviewed = json.loads((thread_dir / "llm_annotation.json").read_text(encoding="utf-8"))
    assert reviewed["keep"] is True
    assert reviewed["question"] == "Original standalone question, under the archived assumption?"
    assert reviewed["answer"] == "2"
    assert reviewed["review"]["status"] == "keep"
    assert reviewed["full_text_review"]["action"] == "edit"


def test_runner_contract_and_bash_syntax():
    repo_root = Path(__file__).resolve().parents[2]
    runner = repo_root / "matharena" / "stackmath" / "scripts" / "create_train.sh"
    subprocess.run(["bash", "-n", str(runner)], check=True)
    text = runner.read_text(encoding="utf-8")

    assert "math.stackexchange.com mathoverflow.net" in text
    assert "openai/gpt-56-sol-xhigh" in text
    assert "openai/gpt-56-sol" in text
    assert "anthropic/opus_48" in text
    assert "--site \"${SITE_VALUES[@]}\"" in text
    assert "--limit-per-site \"${LIMIT_PER_SITE}\"" in text
    assert "--fulltext" in text
    assert "--full-text-source local" in text
    assert text.count("--limit-per-site") == 1
    assert " --limit " not in text
