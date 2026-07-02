#!/usr/bin/env python3
import argparse
import os
import re
import signal

from dotenv import find_dotenv, load_dotenv

from matharena.api_client import APIClient
from matharena.arxivbench_utils import (
    extract_json,
    list_paper_ids,
    load_annotation,
    load_metadata,
    load_model_config,
    load_prompt_template,
    resolve_model_config_path,
    save_annotation,
)
from matharena.parser import WarningType, check_answers, extract_answer, parse_answer
from matharena.utils import normalize_conversation


FINAL_ANNOTATION_FILENAME = "llm_annotation.json"
FALSE_ANNOTATION_FILENAME = "llm_metadata_false.json"
LEAN_ANNOTATION_FILENAME = "metadata_lean_abstract.json"
LEAN_DEFAULT_PROMPT = "arxivmath/prompts/lean/extract_lean_abstract.md"
ARXIV_NONEXCLUSIVE_LICENSE_URL = "http://arxiv.org/licenses/nonexclusive-distrib/1.0/"
UNSAFE_ARXIV_ANSWER_RE = re.compile(
    r"\\(?:aleph|beth|cap|circ|cup|in|infty|land|lor|mathbb|mathcal|mathbf|mathrm|neg|oplus|operatorname|otimes|prod|sqcup|sum|text|tilde|vee|wedge)(?=[^A-Za-z]|$)|[\u222a\u2229\u2228\u2227\u2297\u2295\u221e\u00ac]"
)
ANSWER_NUMBER_RE = re.compile(r"(?<![A-Za-z])-?\d+(?![A-Za-z])")
ANSWER_VARIABLE_RE = re.compile(r"(?<!\\)(?<![A-Za-z])([A-Za-z])(?![A-Za-z])")


def needs_annotation(annotation, overwrite=False, false_mode=False, lean_mode=False):
    if overwrite:
        return True
    if lean_mode:
        keep = annotation.get("keep")
        if keep is None:
            return True
        if keep is True and not annotation.get("statement"):
            return True
        return False
    if false_mode:
        if annotation.get("keep") is None:
            return True
        if annotation.get("keep") is not True:
            return False
        return not (
            annotation.get("original_statement")
            and annotation.get("perturbed_statement")
            and annotation.get("falsity_explanation")
        )
    keep = annotation.get("keep")
    if keep is None:
        return True
    if keep is True and (not annotation.get("question") or not annotation.get("answer")):
        return True
    return False

def coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes"}:
            return True
        if lowered in {"false", "no"}:
            return False
    return None


def should_skip_license(metadata, skip_arxiv_license=False):
    if not skip_arxiv_license:
        return False
    return (metadata.get("license") or "").strip() == ARXIV_NONEXCLUSIVE_LICENSE_URL


def load_local_full_text(paper_root, paper_id):
    full_text_path = os.path.join(paper_root, paper_id, "full_text.md")
    if not os.path.isfile(full_text_path):
        raise FileNotFoundError(f"Missing local full_text.md for question generation: {full_text_path}")
    with open(full_text_path, "r", encoding="utf-8") as f:
        return f.read()


def _arxiv_answer_mutations(answer):
    for match in ANSWER_NUMBER_RE.finditer(answer):
        yield answer[: match.start()] + str(int(match.group(0)) + 1) + answer[match.end() :]
    match = ANSWER_VARIABLE_RE.search(answer)
    if match:
        replacement = "y" if match.group(1) == "x" else "x"
        yield answer[: match.start()] + replacement + answer[match.end() :]


def _arxiv_parser_reject(reason, **extra):
    return {"keep": False, "reason": reason, **extra}


def _raise_parser_safety_timeout(signum, frame):
    raise TimeoutError


def check_arxiv_answer_parser_safety(answer):
    """Return whether a normal arxivmath answer is safe for parser-based grading."""
    answer = (answer or "").strip()
    if not answer:
        return _arxiv_parser_reject("empty_answer")

    if UNSAFE_ARXIV_ANSWER_RE.search(answer):
        return _arxiv_parser_reject("unsupported_answer_notation")
    if (":" in answer and ("\\{" in answer or "{" in answer)) or "\\mid" in answer or "\\;|" in answer:
        return _arxiv_parser_reject("set_builder_answer")

    list_answer = "," in answer
    try:
        parsed, warning = parse_answer(answer, list_answer=list_answer)
        extracted, extract_warning = extract_answer(
            rf"\boxed{{{answer}}}", strict_parsing=False, parse=True, list_answer=list_answer
        )
    except Exception as exc:
        return _arxiv_parser_reject("parse_exception", exception=repr(exc))

    if parsed is None or warning >= WarningType.MAJOR:
        return _arxiv_parser_reject("unparseable_answer", warning=warning.name)
    if extracted is None or extract_warning >= WarningType.MAJOR or not check_answers(extracted, parsed):
        return _arxiv_parser_reject("exact_self_grade_failed", warning=extract_warning.name)

    for mutated in _arxiv_answer_mutations(answer):
        try:
            mutated_parsed, mutated_warning = parse_answer(mutated, list_answer=list_answer)
        except Exception:
            continue
        if mutated_parsed is not None and mutated_warning < WarningType.MAJOR and check_answers(mutated_parsed, parsed):
            return _arxiv_parser_reject("parser_insensitive_to_answer_mutation", mutated_answer=mutated)

    return {"keep": True, "reason": "parser_safe", "warning": warning.name, "parsed_answer": str(parsed)}


def main():
    load_dotenv(find_dotenv(usecwd=True))
    parser = argparse.ArgumentParser(description="Generate paper annotations from abstracts using an LLM.")
    parser.add_argument("--model-config", required=True, help="Path under ../configs/models (e.g. openai/gpt-5-mini).")
    parser.add_argument("--paper-root", default="arxivmath/paper", help="Root directory containing paper folders.")
    parser.add_argument("--prompt", default=None, help="Prompt template path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of papers to query.")
    parser.add_argument("--max-papers", type=int, default=None, help="Optional limit on paper ids to inspect.")
    parser.add_argument(
        "--full-text-source",
        choices=["none", "local"],
        default="none",
        help="Optional source for {full_text} prompt injection. 'local' reads full_text.md from each paper directory.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--false", action="store_true", help="Use the false-statement pipeline.")
    mode_group.add_argument("--lean", action="store_true", help="Use the Lean abstract-candidate pipeline.")
    parser.add_argument("--annotation-filename", default=None, help="Annotation filename to read/write.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing annotations.")
    parser.add_argument(
        "--skip-arxiv-license",
        action="store_true",
        help="Skip papers using arXiv's non-exclusive distribution-only license.",
    )
    args = parser.parse_args()

    if args.lean:
        prompt_path = args.prompt or LEAN_DEFAULT_PROMPT
        annotation_filename = args.annotation_filename or LEAN_ANNOTATION_FILENAME
    else:
        prompt_path = args.prompt or ("arxivmath/prompts/broken/false.md" if args.false else "arxivmath/prompts/arxiv/query.md")
        annotation_filename = args.annotation_filename or (FALSE_ANNOTATION_FILENAME if args.false else FINAL_ANNOTATION_FILENAME)
    prompt_template = load_prompt_template(prompt_path)
    model_config_path = resolve_model_config_path(args.model_config)
    model_config = load_model_config(model_config_path)
    model_name = model_config["model"]
    client = APIClient(**model_config)

    paper_ids = list_paper_ids(args.paper_root)
    if args.max_papers:
        paper_ids = paper_ids[:args.max_papers]
    queries = []
    query_paper_ids = []
    for paper_id in paper_ids:
        annotation = load_annotation(args.paper_root, paper_id, annotation_filename)
        if not needs_annotation(
            annotation,
            overwrite=args.overwrite,
            false_mode=args.false,
            lean_mode=args.lean,
        ):
            continue
        metadata = load_metadata(args.paper_root, paper_id)
        if should_skip_license(metadata, skip_arxiv_license=args.skip_arxiv_license):
            continue
        full_text = ""
        if args.full_text_source == "local":
            full_text = load_local_full_text(args.paper_root, paper_id)
        prompt = prompt_template.format(
            title=(metadata.get("title") or "").strip(),
            abstract=(metadata.get("abstract") or "").strip(),
            full_text=full_text,
        )
        queries.append([{"role": "user", "content": prompt}])
        query_paper_ids.append(paper_id)
        if args.limit and len(queries) >= args.limit:
            break
    
    if not queries:
        return

    total_cost = 0.0
    kept_ids = []
    for idx, conversation, cost in client.run_queries(queries):
        conversation = normalize_conversation(conversation)
        paper_id = query_paper_ids[idx]
        metadata = load_metadata(args.paper_root, paper_id)
        response = ""
        if conversation and isinstance(conversation[-1], dict):
            response = conversation[-1].get("content", "") or ""
        parsed = extract_json(response)
        annotation = {
            "model": model_name,
            "raw": response,
            "cost": cost.get("cost", 0.0),
        }
        if args.lean:
            annotation.update(
                {
                    "title": metadata.get("title") or "",
                    "abstract": metadata.get("abstract") or "",
                    "source_mode": "abstract",
                }
            )
        keep_value = None
        if isinstance(parsed, dict):
            keep_value = coerce_bool(parsed.get("keep"))
            annotation["parsed"] = parsed
            if args.lean:
                if parsed.get("statement"):
                    annotation["statement"] = str(parsed["statement"]).strip()
                if parsed.get("rationale"):
                    annotation["rationale"] = str(parsed["rationale"]).strip()
            elif args.false:
                if parsed.get("original_statement"):
                    annotation["original_statement"] = str(parsed["original_statement"]).strip()
                if parsed.get("perturbed_statement"):
                    annotation["perturbed_statement"] = str(parsed["perturbed_statement"]).strip()
                if parsed.get("why_false_given_original"):
                    annotation["falsity_explanation"] = str(parsed["why_false_given_original"]).strip()
            else:
                if parsed.get("question"):
                    annotation["question"] = str(parsed["question"]).strip()
                if parsed.get("answer"):
                    annotation["answer"] = str(parsed["answer"]).strip()
                if keep_value is True:
                    old_handler = signal.signal(signal.SIGALRM, _raise_parser_safety_timeout)
                    signal.alarm(30)
                    try:
                        parser_safety = check_arxiv_answer_parser_safety(annotation.get("answer", ""))
                    except TimeoutError:
                        parser_safety = _arxiv_parser_reject("parser_safety_timeout", timeout_seconds=30)
                    finally:
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, old_handler)
                    annotation["parser_safety_check"] = parser_safety
                    if parser_safety.get("keep") is not True:
                        keep_value = False
        if keep_value is not None:
            annotation["keep"] = keep_value
            if keep_value is True:
                kept_ids.append(paper_id)
        save_annotation(args.paper_root, paper_id, annotation, annotation_filename)
        total_cost += annotation["cost"]

    print(f"Completed {len(queries)} queries. Total cost: ${total_cost:.6f}")
    print(f"Kept {len(kept_ids)} papers: {', '.join(kept_ids)}")


if __name__ == "__main__":
    main()
