#!/usr/bin/env python3
import argparse
import re
from datetime import datetime, timezone

from matharena.api_client import APIClient
from matharena.arxivbench_utils import (
    list_paper_ids,
    load_annotation,
    load_model_config,
    load_prompt_template,
    resolve_model_config_path,
    save_annotation,
)
from matharena.tools.lean_execution import (
    format_lean_feedback,
    get_lean_feedback_dict,
    lean_explore_search,
    loogle,
    verify_lean,
)
from matharena.utils import normalize_conversation


ANNOTATION_FILENAME = "metadata_lean_abstract.json"
DEFAULT_PROMPT = "arxivmath/prompts/lean/formalize.md"
LEAN_CODE_BLOCK_RE = re.compile(r"```(?:lean)?\s*(.*?)```", re.DOTALL)
LEAN_DECL_RE = re.compile(
    r"(?m)^\s*(theorem|lemma|def|definition|abbrev|opaque|axiom|constant|instance|example|inductive|structure|class)\b"
)
STATEMENT_ONLY_RE = re.compile(r"\A\s*theorem\b[\s\S]*:=\s*by\s+sorry\s*\Z")


def validate_statement_only(code):
    """Return structural errors for the statement-only training-label contract."""
    code = (code or "").strip()
    errors = []
    declarations = LEAN_DECL_RE.findall(code)
    if declarations != ["theorem"]:
        errors.append("Output must contain exactly one theorem declaration and no other declarations.")
    if re.search(r"(?m)^\s*import\b", code):
        errors.append("Output must not contain imports.")
    if len(re.findall(r"\bsorry\b", code)) != 1 or not STATEMENT_ONLY_RE.fullmatch(code):
        errors.append("The theorem must end exactly with `:= by sorry` and contain no proof or surrounding commands.")
    return errors

def needs_formalization(annotation, overwrite=False, required_keep_key=None):
    if annotation.get("keep") is not True:
        return False
    if required_keep_key and (annotation.get(required_keep_key) or {}).get("keep") is not True:
        return False
    if not annotation.get("statement"):
        return False
    if overwrite:
        return True
    return not annotation.get("formalized_statement")

def extract_lean_code(response):
    text = (response or "").strip()
    if not text:
        return ""
    match = LEAN_CODE_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    # GPT models on the Responses API sometimes return bare Lean code instead of a fenced block.
    if ":= by" in text or re.search(r"^(open|namespace|section|variable|noncomputable|def|lemma|theorem|example)\b", text):
        return text
    return response # returning the entire response as a fallback

def build_tool_specs():
    return [
        (
            verify_lean,
            {
                "type": "function",
                "function": {
                    "name": "verify_lean",
                    "description": "Compile Lean 4 code with Mathlib and return compiler feedback. Do not include imports.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Lean code for the statement only, typically ending with := by sorry.",
                            }
                        },
                        "required": ["code"],
                    },
                },
            },
        ),
        (
            loogle,
            {
                "type": "function",
                "function": {
                    "name": "loogle",
                    "description": "Search Mathlib declarations by exact name or a small type pattern with `_` holes.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The loogle query to run."},
                            "max_results": {"type": "integer", "description": "Maximum number of hits to return."},
                        },
                        "required": ["query"],
                    },
                },
            },
        ),
        (
            lean_explore_search,
            {
                "type": "function",
                "function": {
                    "name": "leanfinder",
                    "description": "Search Lean libraries with LeanExplore using natural-language or fuzzy semantic queries.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query to run."},
                            "max_results": {"type": "integer", "description": "Maximum number of hits to return."},
                        },
                        "required": ["query"],
                    },
                },
            },
        ),
    ]


def main():
    parser = argparse.ArgumentParser(description="Formalize kept Lean benchmark statements with an LLM and Lean tool feedback.")
    parser.add_argument("--model-config", required=True, help="Path under ../configs/models (e.g. openai/gpt-5-mini).")
    parser.add_argument("--paper-root", default="arxivmath/paper", help="Root directory containing paper folders.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt template path.")
    parser.add_argument("--annotation-filename", default=ANNOTATION_FILENAME, help="Annotation filename to read/write.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of papers to query.")
    parser.add_argument("--max-papers", type=int, default=None, help="Optional limit on paper ids to inspect.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing formalizations.")
    parser.add_argument(
        "--require-keep-key",
        default=None,
        metavar="KEY",
        help="Only formalize annotations with an explicit keep decision under KEY.",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=24,
        help="Shared maximum number of calls across verify_lean, loogle, and leanfinder per paper.",
    )
    args = parser.parse_args()

    prompt_template = load_prompt_template(args.prompt)
    model_config = load_model_config(resolve_model_config_path(args.model_config))
    model_name = model_config["model"]
    client_args = model_config.copy()
    client_args["tools"] = build_tool_specs()
    client_args["max_tool_calls"] = args.max_tool_calls
    client = APIClient(**client_args)

    paper_ids = list_paper_ids(args.paper_root)
    if args.max_papers:
        paper_ids = paper_ids[:args.max_papers]

    queries = []
    query_paper_ids = []
    for paper_id in paper_ids:
        annotation = load_annotation(args.paper_root, paper_id, args.annotation_filename)
        if not needs_formalization(
            annotation,
            overwrite=args.overwrite,
            required_keep_key=args.require_keep_key,
        ):
            continue
        prompt = prompt_template.format(
            statement=annotation.get("statement", "").strip(),
            proof=annotation.get("proof", "").strip(),
        )
        queries.append([{"role": "user", "content": prompt}])
        query_paper_ids.append(paper_id)
        if args.limit and len(queries) >= args.limit:
            break

    if not queries:
        print("No kept Lean statements need formalization.")
        return

    total_cost = 0.0
    compiled_ids = []
    failed_ids = []
    for idx, conversation, cost in client.run_queries(queries):
        conversation = normalize_conversation(conversation)
        paper_id = query_paper_ids[idx]
        annotation = load_annotation(args.paper_root, paper_id, args.annotation_filename)
        response = ""
        if conversation and isinstance(conversation[-1], dict):
            response = conversation[-1].get("content", "") or ""

        candidate = extract_lean_code(response)
        structural_errors = validate_statement_only(candidate)
        if candidate and not structural_errors:
            compilation = get_lean_feedback_dict(candidate)
        else:
            compilation = {
                "okay": False,
                "errors": structural_errors or ["No Lean theorem statement was returned."],
                "warnings": [],
                "infos": [],
            }
        compilation["structural_errors"] = structural_errors
        compile_feedback = format_lean_feedback(compilation)
        compiles = compilation["okay"] and not compilation["errors"]

        annotation["formalization_model"] = model_name
        annotation["formalization_raw"] = response
        annotation["formalized_statement"] = candidate
        annotation["formalization_feedback"] = compile_feedback
        annotation["formalization_compilation"] = compilation
        annotation["formalization_cost"] = cost.get("cost", 0.0)
        annotation["formalization_updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        if compiles:
            annotation["keep"] = True
            compiled_ids.append(paper_id)
        else:
            annotation["keep"] = False
            failed_ids.append(paper_id)

        save_annotation(args.paper_root, paper_id, annotation, args.annotation_filename)
        total_cost += annotation["formalization_cost"]

    print(f"Completed {len(queries)} formalization queries. Total cost: ${total_cost:.6f}")
    print(f"Compiled with sorries: {len(compiled_ids)} papers: {', '.join(compiled_ids)}")
    print(f"Failed final Lean check: {len(failed_ids)} papers: {', '.join(failed_ids)}")


if __name__ == "__main__":
    main()
