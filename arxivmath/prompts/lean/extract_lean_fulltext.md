# Task

Select exactly one theorem from the supplied arXiv paper for a Lean 4 statement-formalization training example. Use the paper's full text, not merely its title or abstract.

Keep a candidate only when:
- the paper explicitly states the theorem and supplies a proof or enough supporting context to explain it;
- the natural-language statement is precise, self-contained, and faithful to the paper;
- the statement can plausibly be expressed using Lean 4 and Mathlib without substantial paper-specific infrastructure;
- all definitions and hypotheses needed to understand the claim can be included in the extracted statement;
- it is a mathematical theorem rather than an empirical, heuristic, or expository claim.

Prefer a central, nontrivial result. Do not strengthen or weaken the theorem, invent missing hypotheses, or silently replace paper-specific notions with familiar ones. Copy neither TeX boilerplate nor citation prose. The proof field should be a concise natural-language proof or supporting context grounded in the paper, sufficient to help a formalizer understand the intended claim; it need not reproduce every proof detail.

Respond only with JSON:

```json
{{
  "keep": true,
  "statement": "self-contained natural-language theorem statement",
  "proof": "informal proof or supporting context from the paper",
  "rationale": "short selection rationale"
}}
```

If no suitable candidate exists, respond with:

```json
{{"keep": false, "rationale": "short explanation"}}
```

# Title
{title}

# Abstract
{abstract}

# Paper Full Text
{full_text}
