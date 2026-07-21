You are reviewing a challenging mathematics question created from an archived Stack Exchange discussion. The archive contains the original question and every archived answer, with scores and the accepted answer identified.

Your task is to compare the proposed pair with the complete discussion:

- **Discard** it if it changes the original question’s central mathematical problem, has an incorrect or unsupported answer, depends on material outside the archive, omits substantial context, or the discussion contains unresolved disagreement or incompatible conclusions.
- **Edit** it only if the same answer can be made correct and self-contained by adding a small number of necessary assumptions, definitions, conventions, or scope restrictions found in the archive.
- **Keep** it if it is already faithful, correct, complete, and supported by the archived discussion.

The accepted answer and community scores are evidence, not authority. Base the decision on the mathematical content of the entire archived discussion. An external link or citation is not evidence unless the necessary argument is reproduced in the archive.

Return JSON with these keys:

- `"action"`: `"discard"`, `"edit"`, or `"keep"`
- `"question"`: required only for `"edit"`; provide the complete edited question
- `"rationale"`: a short justification grounded in the archived discussion

For example:

```json
{{
  "action": "edit",
  "question": "Edited question text with the necessary assumption.",
  "rationale": "The original question omitted an assumption used by the archived solution."
}}
```

Additional requirements:

1. Make only small, necessary edits. Do not edit for style.
2. Never make the problem easier or add hints, intermediate steps, or solution information.
3. Preserve the answer exactly. Do not change its value, spelling, variable names, symbols, or formatting.
4. Use only information present in the archived discussion; do not rely on outside knowledge.
5. Do not refer to Stack Exchange, the discussion, posts, users, votes, or an accepted answer in an edited question.
6. Keep the question machine-verifiable and never turn it into a request for a proof or explanation.
7. If a missing assumption cannot be added without materially changing the problem, discard rather than edit.
8. If the archive does not directly establish one uniquely correct answer, discard.

### Current question
{question}

### Current answer
{answer}

### Complete archived discussion
{full_text}
