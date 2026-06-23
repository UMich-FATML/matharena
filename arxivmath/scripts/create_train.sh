uv run python arxivmath/scripts/shared/create_queries.py --model-config gemini/gemini-31-pro-insait --skip-arxiv-license --limit 45000 --paper-root arxivmath/paper_train
uv run python arxivmath/scripts/shared/verify_queries.py --model-config gemini/gemini-31-pro-insait --paper-root arxivmath/paper_train
uv run python arxivmath/scripts/shared/fulltext_review.py --model-config gemini/gemini-31-pro-insait --paper-root arxivmath/paper_train
# uv run python arxivmath/scripts/shared/fulltext_review.py --model-config gemini/gemini-31-pro-insait --key solid_authors --prompt arxivmath/prompts/shared/solid_authors.md --enable-web-search --skip-ocr

uv run python arxivmath/scripts/shared/create_queries.py --false --model-config gemini/gemini-31-pro-insait --skip-arxiv-license --paper-root arxivmath/paper_train --limit 45000
uv run python arxivmath/scripts/shared/verify_queries.py --false --model-config gemini/gemini-31-pro-insait --paper-root arxivmath/paper_train
uv run python arxivmath/scripts/shared/fulltext_review.py --false --model-config gemini/gemini-31-pro-insait --paper-root arxivmath/paper_train
# uv run python arxivmath/scripts/shared/fulltext_review.py --false --model-config gemini/gemini-31-pro-medium --key solid_authors --prompt arxivmath/prompts/shared/solid_authors.md --enable-web-search --skip-ocr
