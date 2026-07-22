# AI-Assistance Review

Inspect the supplied paper full text. Discard the item if the paper says that any generative-AI or LLM system was used to write, edit, translate, generate, prove, check, or otherwise produce the paper or its mathematical content. This includes acknowledgements or disclosures involving systems such as ChatGPT, OpenAI, Claude, Anthropic, Gemini, Llama, or comparable tools. A reference that merely studies AI is not by itself evidence of AI-assisted authorship.

Also discard if every author explicitly identifies as an independent researcher without institutional affiliation. When the evidence is ambiguous, discard.

Respond only with JSON:

```json
{{"action": "keep" | "discard", "rationale": "justification grounded in the paper"}}
```

### Extracted theorem
{question}

### Paper full text
{full_text}
