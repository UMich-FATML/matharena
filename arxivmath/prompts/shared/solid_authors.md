# Task
Your job is to determine whether at least one author of the given scientific paper has a solid publication record in the relevant field.
In particular, you need to verify if at least one author is an expert in the field related to the paper, i.e., is a PhD student or has a higher degree in the field of study.

## Output Format

Respond **only** with a JSON object:

```json
{{
  "keep": boolean,
  "rationale": "brief justification for the decision"
}}
```

If you could not confirm any author satisfies the criteria, output `"keep": false`.
Otherwise, output `"keep": true`.
Always include a concise rationale. For `"keep": true`, name the author and the evidence that they have a solid publication record or appropriate academic standing in the relevant field. For `"keep": false`, briefly state what you checked and why it was insufficient.

---

# Paper Title
{title}

# Paper Abstract
{abstract}

# Paper Authors
{authors}
