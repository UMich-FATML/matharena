# Task Description

You are constructing evaluation questions for a benchmark on **challenging mathematics**, spanning contest, undergraduate, graduate, and research-level material. The benchmark measures whether an LLM can solve a precise mathematical problem without access to the source discussion.

You will be given the title and complete archived text of a Stack Exchange discussion: the original question followed by all archived answers, with answer scores and the accepted answer identified. Determine whether the **central mathematical problem posed by the original question** can be faithfully reframed as a **single, precise, objectively verifiable question** with a **unique, deterministic answer**.

If such a question can be formed, return it with its answer. Otherwise reject the discussion. Most discussions should be rejected: acceptance and community scores are evidence, not proof that an answer is correct, complete, or unambiguous.

## Criteria for an Acceptable Question-Answer Pair

Keep a discussion only if all of the following hold:

1. **Direct support from the archive**
   The answer must follow directly and unambiguously from the archived question and answers. Reject discussions whose resolution depends on an external link, image, citation, computation, or omitted context.

2. **Faithful central problem**
   The generated question must target the main mathematical problem asked by the original poster. Reframing it to be self-contained is allowed, but do not invent a different problem or use incidental facts from an answer.

3. **Challenging and nontrivial**
   The resulting problem must require meaningful mathematical reasoning. Reject questions that are routine, directly reveal the answer, or become easy after making them self-contained.

4. **Resolved and objectively correct**
   The discussion must establish exactly one answer. Inspect the entire discussion, not just the accepted answer. Reject unresolved disagreements, incompatible answers, conjectures, open problems, and conclusions that are merely asserted.

5. **Unambiguous and self-contained**
   Define every variable, convention, assumption, and quantity needed by a reader who has never seen the source. Do not refer to Stack Exchange, the thread, users, votes, an accepted answer, or phrases such as “as discussed above.”

6. **Answer format constraint**
   The answer must be either a single numerical value or a pure LaTeX mathematical expression. It must contain no English words.

   Prefer parser-friendly objects such as integers, rational numbers, radicals, polynomials, rational functions, elementary expressions, finite tuples, or intervals. Reject answers requiring notation-heavy structures or expressions that are difficult to grade mechanically, including:

   - unevaluated sums or products;
   - set-builder notation or geometric loci;
   - unions, intersections, joins, tensors, coproducts, composition, or logical formulas;
   - named structures or notation classes requiring semantic interpretation.

7. **Question type restriction**
   The question must not be yes/no, multiple-choice, or a request to prove, justify, derive, or explain something.

8. **Machine-verifiable**
   The answer must be extractable and comparable as a string or parsed mathematical expression.

9. **Exact bounds only**
   If the discussion establishes only a loose bound or inequality, reject it. A bound is usable only when the archived discussion establishes it as exact or tight and the resulting answer is unique.

10. **All context included without hints**
    Include every necessary definition and assumption, but do not copy solution steps, intermediate results, or hints into the question. The problem may be long when that is necessary for precision.

## Output Format

Respond only with a JSON object:

```json
{{
  "keep": boolean,
  "question": string,
  "answer": string
}}
```

If no valid pair can be formed, output:

```json
{{
  "keep": false
}}
```

Do not include text outside the JSON object.

# Original discussion title
{title}

# Complete archived discussion
{full_text}
