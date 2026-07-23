#!/usr/bin/env python3
import argparse
import os
from datetime import datetime

from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm

from matharena.api_client import APIClient
from matharena.arxivbench_utils import (
    ensure_ocr_batch,
    extract_json,
    get_latest_fields,
    get_latest_pair,
    list_paper_ids,
    load_annotation,
    load_metadata,
    load_model_config,
    load_prompt_template,
    resolve_model_config_path,
    save_annotation,
)
from matharena.utils import normalize_conversation


FINAL_ANNOTATION_FILENAME = "llm_annotation.json"
FALSE_ANNOTATION_FILENAME = "llm_metadata_false.json"
LEAN_ANNOTATION_FILENAME = "metadata_lean_abstract.json"


def load_full_texts(paper_root, paper_ids, source="ocr", redo=False):
    if source == "ocr":
        return ensure_ocr_batch(paper_ids, redo=redo)
    if source != "local":
        raise ValueError(f"Unsupported full-text source: {source}")

    full_texts = {}
    missing = []
    for paper_id in paper_ids:
        full_text_path = os.path.join(paper_root, paper_id, "full_text.md")
        if not os.path.isfile(full_text_path):
            missing.append(full_text_path)
            continue
        with open(full_text_path, "r", encoding="utf-8") as f:
            full_texts[paper_id] = f.read()
    if missing:
        raise FileNotFoundError(
            "Missing local full_text.md files for full-text review: "
            + ", ".join(missing[:10])
            + (" ..." if len(missing) > 10 else "")
        )
    return full_texts


def coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    return None


def has_explicit_decision(record, key):
    if not isinstance(record, dict):
        return False
    parsed = record.get("parsed")
    if not isinstance(parsed, dict):
        return False
    if key == "solid_authors":
        return coerce_bool(parsed.get("keep")) is not None
    return str(parsed.get("action", "")).strip().lower() in {"keep", "discard"}


def has_explicit_keep(record):
    if not isinstance(record, dict):
        return False
    keep_value = coerce_bool(record.get("keep"))
    if keep_value is not None:
        return keep_value
    parsed = record.get("parsed")
    if not isinstance(parsed, dict):
        return False
    keep_value = coerce_bool(parsed.get("keep"))
    if keep_value is not None:
        return keep_value
    return str(parsed.get("action", "")).strip().lower() == "keep"


def should_review(
    annotation,
    overwrite=False,
    key="full_text_review",
    lean_mode=False,
    require_explicit_decision=False,
    required_keep_keys=(),
):
    if annotation.get("keep") is not True:
        return False
    if any(not has_explicit_keep(annotation.get(required_key)) for required_key in required_keep_keys):
        return False
    if not overwrite and key in annotation:
        if not require_explicit_decision or has_explicit_decision(annotation.get(key), key):
            return False
    review = annotation.get("review") or {}
    if lean_mode:
        return not review or review.get("status") == "keep"
    if review and review.get("status") != "keep":
        return False
    return True


def main():
    load_dotenv(find_dotenv(usecwd=True))
    parser = argparse.ArgumentParser(description="Re-check kept arXiv questions against full paper OCR.")
    parser.add_argument("--model-config", required=True, help="Path under ../configs/models (e.g. openai/gpt-5-mini).")
    parser.add_argument("--paper-root", default="arxivmath/paper", help="Root directory containing paper folders.")
    parser.add_argument("--prompt", default=None, help="Prompt template path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of papers to process.")
    parser.add_argument("--max-papers", type=int, default=None, help="Optional limit on paper ids to inspect.")
    parser.add_argument("--redo-ocr", action="store_true", help="Force OCR even if cached markdown exists.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing full-text review results.")
    parser.add_argument("--key", default="full_text_review", help="Annotation key to store the review under.")
    parser.add_argument("--enable-web-search", action="store_true", help="Enable web search for additional context.")
    parser.add_argument("--skip-ocr", action="store_true", help="Skip OCR and full text injection.")
    parser.add_argument(
        "--full-text-source",
        choices=["ocr", "local"],
        default="ocr",
        help="Source for full text when not using --skip-ocr. 'local' reads full_text.md from each paper directory.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--false", action="store_true", help="Use the false-statement pipeline.")
    mode_group.add_argument("--lean", action="store_true", help="Use the Lean abstract-candidate pipeline.")
    parser.add_argument("--annotation-filename", default=None, help="Annotation filename to read/write.")
    parser.add_argument(
        "--require-explicit-decision",
        action="store_true",
        help="Retry malformed outputs and mutate keep only after an explicit keep/discard decision.",
    )
    parser.add_argument(
        "--require-keep-key",
        action="append",
        default=[],
        metavar="KEY",
        help="Only review annotations with an explicit keep decision under KEY. Repeat for cumulative prerequisites.",
    )
    args = parser.parse_args()

    prompt_path = args.prompt or (
        "arxivmath/prompts/broken/false_fulltext_review.md"
        if args.false
        else "arxivmath/prompts/arxiv/fulltext_review.md"
    )
    annotation_filename = args.annotation_filename or (
        LEAN_ANNOTATION_FILENAME
        if args.lean
        else FALSE_ANNOTATION_FILENAME
        if args.false
        else FINAL_ANNOTATION_FILENAME
    )
    prompt_template = load_prompt_template(prompt_path)
    model_config_path = resolve_model_config_path(args.model_config)
    model_config = load_model_config(model_config_path)
    model_name = model_config["model"]
    if args.enable_web_search:
        if model_config.get("api") == "google":
            model_config["tools"] = [(None, {"google_search": {}})]
            model_config["use_gdm_tools"] = True
            model_config["max_tool_calls"] = 50
        else:
            model_config["tools"] = [(None, {"type": "web_search"})]
    client = APIClient(**model_config)

    discarded = []
    updated = []
    kept = []
    malformed = []
    total_cost = 0.0

    paper_ids = list_paper_ids(args.paper_root)
    if args.max_papers:
        paper_ids = paper_ids[:args.max_papers]
    review_ids = []
    for paper_id in paper_ids:
        annotation = load_annotation(args.paper_root, paper_id, annotation_filename)
        if should_review(
            annotation,
            overwrite=args.overwrite,
            key=args.key,
            lean_mode=args.lean,
            require_explicit_decision=args.require_explicit_decision,
            required_keep_keys=args.require_keep_key,
        ):
            review_ids.append(paper_id)

    query_inputs = []
    for paper_id in tqdm(review_ids):
        annotation = load_annotation(args.paper_root, paper_id, annotation_filename)
        question = answer = original_statement = perturbed_statement = falsity_explanation = statement = formalized_statement = ""
        if args.key != "solid_authors":
            if args.false:
                latest = get_latest_fields(
                    annotation,
                    ["original_statement", "perturbed_statement", "falsity_explanation"],
                )
                if not latest:
                    continue
                original_statement, perturbed_statement, falsity_explanation = latest
            elif args.lean:
                statement = (annotation.get("statement") or "").strip()
                formalized_statement = (annotation.get("formalized_statement") or "").strip()
                if not statement:
                    continue
                question = statement
                original_statement = statement
            else:
                latest = get_latest_pair(annotation)
                if not latest:
                    continue
                question, answer = latest
        metadata = load_metadata(args.paper_root, paper_id)
        query_inputs.append(
            (
                paper_id,
                metadata,
                question,
                answer,
                original_statement,
                formalized_statement,
                perturbed_statement,
                falsity_explanation,
                statement,
            )
        )

        if args.limit and len(query_inputs) >= args.limit:
            break

    if not query_inputs:
        print("No papers need review.")
        return

    full_texts = {}
    if not args.skip_ocr and args.key != "solid_authors":
        full_texts = load_full_texts(
            args.paper_root,
            [paper_id for paper_id, *_ in query_inputs],
            source=args.full_text_source,
            redo=args.redo_ocr,
        )
        missing_text_ids = {paper_id for paper_id, *_ in query_inputs if paper_id not in full_texts}
        for paper_id in sorted(missing_text_ids):
            annotation = load_annotation(args.paper_root, paper_id, annotation_filename)
            annotation["keep"] = False
            save_annotation(args.paper_root, paper_id, annotation, annotation_filename)
            discarded.append(paper_id)
        query_inputs = [query_input for query_input in query_inputs if query_input[0] not in missing_text_ids]
        if not query_inputs:
            return

    queries = []
    query_paper_ids = []
    for (
        paper_id,
        metadata,
        question,
        answer,
        original_statement,
        formalized_statement,
        perturbed_statement,
        falsity_explanation,
        statement,
    ) in query_inputs:
        prompt = prompt_template.format(
            question=question,
            answer=answer,
            original_statement=original_statement,
            formalized_statement=formalized_statement,
            perturbed_statement=perturbed_statement,
            falsity_explanation=falsity_explanation,
            statement=statement,
            full_text=full_texts.get(paper_id, ""),
            title=metadata.get("title") or "",
            authors=", ".join([f"{author['forenames']} {author['keyname']}" for author in metadata.get("authors", [])]),
            abstract=metadata.get("abstract") or "",
        )
        queries.append([{"role": "user", "content": prompt}])
        query_paper_ids.append(paper_id)

    for idx, conversation, cost in client.run_queries(queries):
        conversation = normalize_conversation(conversation)
        if idx >= len(query_paper_ids):
            continue
        paper_id = query_paper_ids[idx]
        annotation = load_annotation(args.paper_root, paper_id, annotation_filename)
        response = ""
        if conversation and isinstance(conversation[-1], dict):
            response = conversation[-1].get("content", "") or ""
        parsed = extract_json(response)
        action = None
        keep_value = None
        if isinstance(parsed, dict):
            action = str(parsed.get("action", "")).strip().lower() or None
            edited_question = parsed.get("question")
            edited_original = parsed.get("original_statement")
            edited_perturbed = parsed.get("perturbed_statement")
            edited_falsity = parsed.get("falsity_explanation")
            keep_value = coerce_bool(parsed.get("keep"))

        review_record = {
            "model": model_name,
            "raw": response,
            "cost": cost.get("cost", 0.0),
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        if isinstance(parsed, dict):
            review_record["parsed"] = parsed
            if "rationale" in parsed:
                review_record["rationale"] = parsed.get("rationale")
        if action:
            review_record["action"] = action

        explicit_decision = has_explicit_decision(review_record, args.key)
        if args.require_explicit_decision and not explicit_decision:
            annotation[args.key] = review_record
            save_annotation(args.paper_root, paper_id, annotation, annotation_filename)
            total_cost += review_record["cost"]
            malformed.append(paper_id)
            continue

        review = annotation.get("review") or {}
        if action == "discard" or (action is None and '"action": "discard"' in response) or keep_value is False:
            review["status"] = "discard"
            review["updated_at"] = review_record["updated_at"]
            annotation["keep"] = False
            discarded.append(paper_id)
        elif action == "edit":
            if args.false:
                for field, value in [
                    ("original_statement", edited_original),
                    ("perturbed_statement", edited_perturbed),
                    ("falsity_explanation", edited_falsity),
                ]:
                    if value and str(value).strip():
                        review[field] = str(value).strip()
                        annotation[field] = review[field]
                review["updated_at"] = review_record["updated_at"]
                review["status"] = "keep"
                annotation["keep"] = True
                updated.append(paper_id)
            elif args.lean:
                edited_statement = parsed.get("statement") if isinstance(parsed, dict) else None
                if edited_statement and str(edited_statement).strip():
                    review["statement"] = str(edited_statement).strip()
                    annotation["statement"] = review["statement"]
                    review["updated_at"] = review_record["updated_at"]
                    review["status"] = "keep"
                    annotation["keep"] = True
                    updated.append(paper_id)
                else:
                    kept.append(paper_id)
            elif edited_question and str(edited_question).strip():
                review["question"] = str(edited_question).strip()
                annotation["question"] = review["question"]
                review["updated_at"] = review_record["updated_at"]
                review["status"] = "keep"
                annotation["keep"] = True
                updated.append(paper_id)
            else:
                kept.append(paper_id)
        else:
            annotation["keep"] = True
            kept.append(paper_id)

        annotation["review"] = review
        annotation[args.key] = review_record
        save_annotation(args.paper_root, paper_id, annotation, annotation_filename)
        total_cost += review_record["cost"]

    print(f"Full-text review complete. Total cost: ${total_cost:.6f}")
    print(f"Discarded ({len(discarded)}): {', '.join(discarded)}")
    print(f"Updated ({len(updated)}): {', '.join(updated)}")
    print(f"Kept ({len(kept)}): {', '.join(kept)}")
    print(f"Malformed and left pending ({len(malformed)}): {', '.join(malformed)}")


if __name__ == "__main__":
    main()
