# Verification Task

You are verifying a proposed question-answer pair for a challenging mathematics benchmark. This stage checks the pair as a standalone benchmark item; the source discussion is intentionally unavailable.

Return `"keep": true` only if every criterion below holds:

1. The question is fully self-contained: every variable, term, convention, assumption, and requested quantity is defined precisely enough to answer without outside context.
2. The question has exactly one objective answer and is not yes/no, multiple-choice, subjective, or a request to prove, explain, justify, or derive something.
3. The question and answer do not refer to Stack Exchange, a discussion, a post, a user, votes, an accepted answer, or any other unavailable source.
4. The answer is either a single numerical value or a pure LaTeX mathematical expression containing no English words.
5. The answer is nontrivial: reject answers equal to (0) or (1), and reject an answer that merely repeats the variable requested by the question. Small nontrivial variations such as (n+1) are allowed.
6. The answer format is suitable for rule-based comparison and does not require prose or semantic interpretation.

If any criterion fails or the interpretation depends on a convention that the question does not specify, return `"keep": false`.

## Output Format

Respond only with a JSON object:

```json
{{
  "keep": boolean
}}
```

# Proposed Question
{question}

# Proposed Answer
{answer}
