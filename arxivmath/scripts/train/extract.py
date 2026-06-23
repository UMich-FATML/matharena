#!/usr/bin/env python3
import argparse
from typing import Dict, List

from datasets import Dataset, Features, Sequence, Value

from matharena.arxivbench_utils import (
    get_latest_fields,
    get_latest_pair,
    list_paper_ids,
    load_annotation,
    load_metadata,
)


ARXIV_ANNOTATION_FILENAME = "llm_annotation.json"
FALSE_ANNOTATION_FILENAME = "llm_metadata_false.json"
ARXIV_NONEXCLUSIVE_LICENSE_URL = "http://arxiv.org/licenses/nonexclusive-distrib/1.0/"
BASE_FEATURES = Features(
    {
        "paper_id": Value("string"),
        "title": Value("string"),
        "authors": Sequence(Value("string")),
        "abstract": Value("string"),
        "license": Value("string"),
        "created": Value("string"),
        "updated": Value("string"),
        "categories": Sequence(Value("string")),
        "comments": Value("string"),
        "journal_ref": Value("string"),
        "doi": Value("string"),
    }
)
ARXIV_FEATURES = BASE_FEATURES.copy()
ARXIV_FEATURES.update(
    {
        "question": Value("string"),
        "answer": Value("string"),
    }
)
FALSE_FEATURES = BASE_FEATURES.copy()
FALSE_FEATURES.update(
    {
        "original_statement": Value("string"),
        "perturbed_statement": Value("string"),
        "falsity_explanation": Value("string"),
    }
)


def format_authors(metadata: Dict) -> List[str]:
    author_names = []
    for author in metadata.get("authors") or []:
        forenames = (author.get("forenames") or "").strip()
        keyname = (author.get("keyname") or "").strip()
        full_name = " ".join(part for part in [forenames, keyname] if part)
        if full_name:
            author_names.append(full_name)
    return author_names


def base_row(paper_id: str, metadata: Dict) -> Dict:
    return {
        "paper_id": paper_id,
        "title": metadata.get("title") or "",
        "authors": format_authors(metadata),
        "abstract": metadata.get("abstract") or "",
        "license": metadata.get("license") or "",
        "created": metadata.get("created") or "",
        "updated": metadata.get("updated") or "",
        "categories": metadata.get("categories") or [],
        "comments": metadata.get("comments") or "",
        "journal_ref": metadata.get("journal_ref") or "",
        "doi": metadata.get("doi") or "",
    }


def should_skip_license(metadata: Dict, skip_arxiv_license: bool = False) -> bool:
    if not skip_arxiv_license:
        return False
    return (metadata.get("license") or "").strip() == ARXIV_NONEXCLUSIVE_LICENSE_URL


def is_accepted_arxiv(annotation: Dict) -> bool:
    if not annotation:
        return False
    review = annotation.get("review") or {}
    if review and review.get("status") != "keep":
        return False
    if annotation.get("keep") is not True:
        return False
    verification = annotation.get("verification") or {}
    return verification.get("keep") is True


def is_accepted_false(annotation: Dict) -> bool:
    if not annotation:
        return False
    review = annotation.get("review") or {}
    if review and review.get("status") != "keep":
        return False
    if annotation.get("keep") is not True:
        return False
    verification = annotation.get("verification") or {}
    if verification.get("keep") is not True:
        return False
    prior_work = annotation.get("prior_work_filter") or {}
    if prior_work:
        parsed = prior_work.get("parsed") or {}
        if parsed.get("action") != "keep":
            return False
    solid_authors = annotation.get("solid_authors") or {}
    if solid_authors:
        parsed = solid_authors.get("parsed") or {}
        if parsed.get("keep") is not True:
            return False
    return True


def build_arxiv_rows(paper_root: str, skip_arxiv_license: bool = False) -> List[Dict]:
    rows = []
    for paper_id in list_paper_ids(paper_root):
        annotation = load_annotation(paper_root, paper_id, ARXIV_ANNOTATION_FILENAME)
        if not is_accepted_arxiv(annotation):
            continue
        pair = get_latest_pair(annotation)
        if not pair:
            continue
        question, answer = pair
        metadata = load_metadata(paper_root, paper_id)
        if should_skip_license(metadata, skip_arxiv_license=skip_arxiv_license):
            continue
        row = base_row(paper_id, metadata)
        row.update(
            {
                "question": question,
                "answer": answer,
            }
        )
        rows.append(row)
    return rows


def build_false_rows(paper_root: str, skip_arxiv_license: bool = False) -> List[Dict]:
    rows = []
    for paper_id in list_paper_ids(paper_root):
        annotation = load_annotation(paper_root, paper_id, FALSE_ANNOTATION_FILENAME)
        if not is_accepted_false(annotation):
            continue
        fields = get_latest_fields(
            annotation,
            ["original_statement", "perturbed_statement", "falsity_explanation"],
        )
        if not fields:
            continue
        original_statement, perturbed_statement, falsity_explanation = fields
        metadata = load_metadata(paper_root, paper_id)
        if should_skip_license(metadata, skip_arxiv_license=skip_arxiv_license):
            continue
        row = base_row(paper_id, metadata)
        row.update(
            {
                "original_statement": original_statement,
                "perturbed_statement": perturbed_statement,
                "falsity_explanation": falsity_explanation,
            }
        )
        rows.append(row)
    return rows


def dataset_from_rows(rows: List[Dict], features: Features) -> Dataset:
    return Dataset.from_list(rows, features=features)


def main():
    parser = argparse.ArgumentParser(
        description="Upload accepted arXiv benchmark training data to Hugging Face Hub."
    )
    parser.add_argument("--org", default="MathArena", help="Hugging Face organization or user name.")
    parser.add_argument("--repo-name", required=True, help="Hugging Face dataset repo name.")
    parser.add_argument("--paper-root", default="arxivmath/paper_train", help="Root directory containing paper folders.")
    parser.add_argument(
        "--false",
        dest="false_mode",
        action="store_true",
        help="Upload accepted false-theorem rows instead of arxivmath question/answer rows.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make the dataset public. Default behavior uploads it privately.",
    )
    parser.add_argument(
        "--skip-arxiv-license",
        action="store_true",
        help="Skip papers using arXiv's non-exclusive distribution-only license.",
    )
    args = parser.parse_args()

    dataset_name = "arxiv_false" if args.false_mode else "arxivmath"
    if args.false_mode:
        rows = build_false_rows(args.paper_root, skip_arxiv_license=args.skip_arxiv_license)
        features = FALSE_FEATURES
    else:
        rows = build_arxiv_rows(args.paper_root, skip_arxiv_license=args.skip_arxiv_license)
        features = ARXIV_FEATURES

    if not rows:
        raise ValueError(f"No accepted {dataset_name} rows found to upload.")

    dataset = dataset_from_rows(rows, features)
    dataset.push_to_hub(
        f"{args.org}/{args.repo_name}",
        private=not args.public,
    )
    print(
        "Uploaded dataset "
        f"{args.org}/{args.repo_name} with "
        f"{len(rows)} {dataset_name} rows."
    )


if __name__ == "__main__":
    main()
