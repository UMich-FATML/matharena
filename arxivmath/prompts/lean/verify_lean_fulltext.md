# Verification Task

Verify whether the extracted theorem and informal proof/context faithfully represent a theorem in the supplied arXiv paper full text and are suitable for Lean 4 statement formalization with Mathlib.

Keep the candidate only if all of the following hold:
- the full text explicitly supports the statement, including every material hypothesis and conclusion;
- the statement is self-contained and mathematically precise;
- the proof/context corresponds to that statement and does not reveal a mismatch or omitted condition;
- the statement can plausibly be represented with existing Mathlib concepts without substantial bespoke infrastructure;
- the extraction does not silently strengthen, weaken, or reinterpret the paper's theorem.

Be strict. Respond only with JSON:

```json
{{"keep": boolean, "rationale": "short justification"}}
```

# Title
{title}

# Abstract
{abstract}

# Extracted Statement
{statement}

# Extracted Informal Proof or Context
{proof}

# Paper Full Text
{full_text}
