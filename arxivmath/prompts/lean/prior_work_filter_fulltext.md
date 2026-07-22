# Prior-Work Review

Determine whether the extracted theorem is a contribution of this paper rather than a theorem already established, or straightforwardly implied, by cited prior work. Use the full paper text, including its introduction, related-work discussion, theorem attribution, and proof context.

Discard if the paper attributes the theorem or a materially equivalent result to prior work, describes it as an immediate/standard consequence of known work, or leaves meaningful uncertainty. Otherwise keep it.

Respond only with JSON:

```json
{{"action": "keep" | "discard", "rationale": "justification grounded in the paper"}}
```

### Extracted theorem
{original_statement}

### Paper full text
{full_text}
