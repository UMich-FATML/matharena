import json
import sys
from pathlib import Path


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
    from arxivmath.scripts.lean.formalize_statements import validate_statement_only

    assert validate_statement_only("theorem good : True := by sorry") == []
    assert validate_statement_only("theorem proved : True := by trivial")
    assert validate_statement_only("def helper := 1\ntheorem bad : True := by sorry")
    assert validate_statement_only("import Mathlib\ntheorem bad : True := by sorry")


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
    from arxivmath.scripts.shared.fulltext_review import has_explicit_decision, should_review

    malformed = {"keep": True, "hidden_condition": {"parsed": {"rationale": "unclear"}}}
    assert should_review(
        malformed,
        key="hidden_condition",
        lean_mode=True,
        require_explicit_decision=True,
    )
    assert has_explicit_decision({"parsed": {"action": "keep"}}, "hidden_condition")
    assert has_explicit_decision({"parsed": {"keep": False}}, "solid_authors")
