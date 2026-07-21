# StackMath training data generation

StackMath adapts the arxivmath training pipeline to accepted-answer discussions from the local Stack Exchange parquet archive. It runs four resume-safe stages: ingestion, question generation, standalone verification, and review against the complete archived discussion.

## Prerequisites

Run commands from the top-level `1T-math` repository. Initialize the `matharena` submodule and pixi environment, and make the Stack Exchange archive available at `stackexchange_data/` or set `CRAWL_ROOT` to another directory containing `questions_part_*.parquet`.

The default OpenAI stages use the Codex login configured by the repository model files. The default reviewer requires `ANTHROPIC_API_KEY`; it may be placed in the top-level `.env` file. Generated data and source crawl files are local artifacts and should remain outside Git.

## Run the pipeline

```bash
bash matharena/stackmath/scripts/create_train.sh
```

The default run writes up to 20 qualifying records from each of `math.stackexchange.com` and `mathoverflow.net` under `data/stackmath-training/`. Each record contains:

- `metadata.json`, including source URL, site, question and accepted-answer IDs, scores, tags, dates, and CC license;
- `full_text.md`, containing the original question and all archived answers in descending score order, with the accepted answer marked;
- `llm_annotation.json`, added and updated by the three model stages.

All source HTML is retained verbatim inside the Markdown discussion wrapper. The archive does not contain comment bodies or answer-author identities, so neither is represented as discussion text.

Stages skip complete annotation records. Re-running the same command resumes a partial run while ingestion refreshes only `metadata.json` and `full_text.md`.

## Overrides

The runner accepts these environment variables:

- `CRAWL_ROOT` and `PAPER_ROOT`;
- `SITES`, as a whitespace-separated list;
- `MIN_QUESTION_SCORE` and `MIN_ANSWER_SCORE`, both defaulting to `3`;
- `LIMIT_PER_SITE`, defaulting to `20`;
- `INCLUDE_CLOSED`, defaulting to `false`;
- `CREATE_QUERIES_MODEL_CONFIG`, defaulting to `openai/gpt-56-sol-xhigh`;
- `VERIFY_QUERIES_MODEL_CONFIG`, defaulting to `openai/gpt-56-sol`;
- `FULLTEXT_REVIEW_MODEL_CONFIG`, defaulting to `anthropic/opus_48`.

For example:

```bash
SITES="math.stackexchange.com" LIMIT_PER_SITE=100 \
  bash matharena/stackmath/scripts/create_train.sh
```

Ingestion requires an open question by default, question score at least 3, at least one answer, maximum answer score at least 3, and a nonempty accepted answer that can be matched uniquely to the archived answer array. The accepted answer itself does not need score 3. Low-scored answers remain in qualifying discussions so the reviewer sees the complete archived answer set.

## Run stages directly

Set the package path once, then use the top-level pixi manifest:

```bash
export PYTHONPATH="$PWD/matharena:$PWD/matharena/src${PYTHONPATH:+:$PYTHONPATH}"

pixi run --manifest-path pixi.toml python \
  matharena/stackmath/scripts/train/ingest_stackexchange_crawl.py \
  --crawl-root stackexchange_data \
  --paper-root data/stackmath-training \
  --site math.stackexchange.com mathoverflow.net \
  --min-question-score 3 --min-answer-score 3 --limit-per-site 20

cd matharena
pixi run --manifest-path ../pixi.toml python stackmath/scripts/shared/create_queries.py \
  --model-config openai/gpt-56-sol-xhigh --paper-root ../data/stackmath-training --fulltext
pixi run --manifest-path ../pixi.toml python stackmath/scripts/shared/verify_queries.py \
  --model-config openai/gpt-56-sol --paper-root ../data/stackmath-training
pixi run --manifest-path ../pixi.toml python stackmath/scripts/shared/fulltext_review.py \
  --model-config anthropic/opus_48 --paper-root ../data/stackmath-training --full-text-source local
```

The archived posts use Creative Commons licenses recorded per record. Preserve `url` and `content_license` from `metadata.json` in any downstream export so attribution and license handling remain possible.
