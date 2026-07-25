import json
import sys
from pathlib import Path

import pytest


def test_annotation_save_preserves_checkpoint_when_replacement_write_fails(tmp_path, monkeypatch):
    from matharena import arxivbench_utils

    paper_root = tmp_path / "papers"
    paper_dir = paper_root / "2401.00001"
    paper_dir.mkdir(parents=True)
    annotation_path = paper_dir / "metadata_lean_fulltext.json"
    original = {"keep": True, "statement": "Existing checkpoint"}
    annotation_path.write_text(json.dumps(original), encoding="utf-8")

    def fail_after_partial_write(data, destination, **kwargs):
        destination.write('{"keep":')
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(arxivbench_utils.json, "dump", fail_after_partial_write)

    with pytest.raises(OSError, match="No space left on device"):
        arxivbench_utils.save_annotation(
            paper_root,
            "2401.00001",
            {"keep": False},
            "metadata_lean_fulltext.json",
        )

    assert json.loads(annotation_path.read_text(encoding="utf-8")) == original
    assert list(paper_dir.glob(".metadata_lean_fulltext.json.*.tmp")) == []


def test_lean_fulltext_annotation_retries_until_statement_and_proof_exist():
    from arxivmath.scripts.shared.create_queries import needs_annotation

    assert needs_annotation({"keep": True, "statement": "A theorem"}, lean_mode=True, require_proof=True)
    assert not needs_annotation(
        {"keep": True, "statement": "A theorem", "proof": "Its proof."},
        lean_mode=True,
        require_proof=True,
    )


def test_fulltext_candidate_verification_renders_paper_and_proof():
    from arxivmath.scripts.shared.verify_queries import render_prompt

    rendered = render_prompt(
        "{title}|{abstract}|{statement}|{proof}|{full_text}",
        {
            "_metadata": {"title": "T", "abstract": "A"},
            "statement": "S",
            "proof": "P",
        },
        lean_mode=True,
        full_text="FULL",
    )

    assert rendered == "T|A|S|P|FULL"


def test_statement_only_validator_rejects_proofs_helpers_and_imports():
    from arxivmath.scripts.lean.formalize_statements import needs_formalization, validate_statement_only

    assert validate_statement_only("theorem good : True := by sorry") == []
    assert validate_statement_only("theorem proved : True := by trivial")
    assert validate_statement_only("def helper := 1\ntheorem bad : True := by sorry")
    assert validate_statement_only("import Mathlib\ntheorem bad : True := by sorry")
    assert needs_formalization(
        {"keep": True, "statement": "A theorem", "verification": {"keep": True}},
        required_keep_key="verification",
    )
    assert not needs_formalization(
        {"keep": True, "statement": "A theorem", "verification": {"raw": "malformed"}},
        required_keep_key="verification",
    )


def test_formalizer_uses_shared_tool_ceiling_and_one_final_compile(tmp_path, monkeypatch):
    from arxivmath.scripts.lean import formalize_statements

    paper_root = tmp_path / "papers"
    paper_dir = paper_root / "2401.00001"
    paper_dir.mkdir(parents=True)
    (paper_dir / "metadata.json").write_text("{}", encoding="utf-8")
    (paper_dir / "metadata_lean_fulltext.json").write_text(
        json.dumps({"keep": True, "statement": "True is true.", "proof": "Trivial."}),
        encoding="utf-8",
    )
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run_queries(self, queries):
            assert "Trivial." in queries[0][0]["content"]
            yield 0, [{"role": "assistant", "content": "theorem generated : True := by sorry"}], {"cost": 0}

    compile_calls = []

    def fake_compile(code):
        compile_calls.append(code)
        return {"okay": True, "errors": [], "warnings": ["declaration uses 'sorry'"], "infos": []}

    monkeypatch.setattr(formalize_statements, "APIClient", FakeClient)
    monkeypatch.setattr(formalize_statements, "get_lean_feedback_dict", fake_compile)
    matharena_root = Path(formalize_statements.__file__).resolve().parents[3]
    prompt_path = matharena_root / "arxivmath/prompts/lean/formalize_fulltext.md"
    model_path = matharena_root / "configs/models/openai/gpt-56-sol-xhigh.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "formalize_statements.py",
            "--model-config",
            str(model_path),
            "--paper-root",
            str(paper_root),
            "--prompt",
            str(prompt_path),
            "--annotation-filename",
            "metadata_lean_fulltext.json",
        ],
    )

    formalize_statements.main()

    assert captured["max_tool_calls"] == 24
    assert compile_calls == ["theorem generated : True := by sorry"]
    annotation = json.loads((paper_dir / "metadata_lean_fulltext.json").read_text(encoding="utf-8"))
    assert annotation["formalization_compilation"]["okay"] is True


def test_strict_review_retries_malformed_records():
    from arxivmath.scripts.shared.fulltext_review import (
        has_explicit_decision,
        has_explicit_keep,
        should_review,
    )

    malformed = {"keep": True, "hidden_condition": {"parsed": {"rationale": "unclear"}}}
    assert should_review(
        malformed,
        key="hidden_condition",
        lean_mode=True,
        require_explicit_decision=True,
    )
    assert has_explicit_decision({"parsed": {"action": "keep"}}, "hidden_condition")
    assert has_explicit_decision({"parsed": {"keep": False}}, "solid_authors")
    assert has_explicit_keep({"keep": True})
    assert has_explicit_keep({"parsed": {"keep": "true"}})
    assert has_explicit_keep({"parsed": {"action": "keep"}})
    assert not has_explicit_keep({"parsed": {"keep": False}})
    assert not has_explicit_keep({"parsed": {"rationale": "missing decision"}})


def test_strict_review_requires_explicit_cumulative_prerequisite_passes():
    from arxivmath.scripts.shared.fulltext_review import should_review

    annotation = {
        "keep": True,
        "semantic_verification": {"keep": True},
        "solid_authors": {"parsed": {"keep": True}},
    }
    required = ["semantic_verification", "solid_authors"]

    assert should_review(annotation, key="hidden_condition", required_keep_keys=required)
    annotation["solid_authors"] = {"parsed": {"keep": False}}
    assert not should_review(annotation, key="hidden_condition", required_keep_keys=required)
    annotation["solid_authors"] = {"parsed": {"rationale": "missing decision"}}
    assert not should_review(annotation, key="hidden_condition", required_keep_keys=required)
    del annotation["semantic_verification"]
    assert not should_review(annotation, key="hidden_condition", required_keep_keys=required)


def test_extract_json_ignores_math_braces_before_fenced_decision():
    from matharena.arxivbench_utils import extract_json

    response = """The admissible set is {t | t > 0}, so the statement aligns.
```json
{"keep": true, "rationale": "faithful"}
```"""

    assert extract_json(response) == {"keep": True, "rationale": "faithful"}


def test_extract_json_tries_later_balanced_fragments():
    from matharena.arxivbench_utils import extract_json

    assert extract_json('The set {x | x > 0} is relevant. Decision: {"keep": false}') == {"keep": False}


def test_extract_json_repairs_trailing_commas_outside_strings():
    from matharena.arxivbench_utils import extract_json

    response = """```json
{
  "keep": true,
  "rationale": "A comma before }, is text, not syntax.",
}
```"""

    assert extract_json(response) == {
        "keep": True,
        "rationale": "A comma before }, is text, not syntax.",
    }


def test_cached_verification_recovers_newly_parseable_raw_without_another_query():
    from arxivmath.scripts.shared.verify_queries import recover_cached_verification

    annotation = {
        "keep": True,
        "semantic_verification": {
            "raw": 'The set is {x | x > 0}.\n```json\n{"keep": false, "rationale": "mismatch"}\n```',
        },
    }

    assert recover_cached_verification(annotation, "semantic_verification")
    assert annotation["keep"] is False
    assert annotation["keep_original"] is True
    assert annotation["semantic_verification"]["keep"] is False
    assert annotation["semantic_verification"]["parsed"]["rationale"] == "mismatch"
    assert not recover_cached_verification(annotation, "semantic_verification")

    rejected_downstream = {
        "keep": False,
        "semantic_verification": {
            "raw": '```json\n{"keep": true, "rationale": "faithful"}\n```',
        },
    }
    assert not recover_cached_verification(rejected_downstream, "semantic_verification")
    assert rejected_downstream["keep"] is False
