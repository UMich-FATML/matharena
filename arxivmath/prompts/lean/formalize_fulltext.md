# Task

Formalize the supplied theorem as exactly one Lean 4 theorem statement using Mathlib. The informal proof/context is evidence about the intended meaning; do not formalize or reproduce the proof.

Requirements:
- Return exactly one `theorem` declaration ending exactly in `:= by sorry`.
- Do not return imports, namespaces, helper declarations, explanations, markdown fences, or proof tactics after `sorry`.
- Use existing Mathlib definitions for standard concepts; do not introduce placeholder predicates, new axioms, constants, opaque declarations, or paper-local surrogates merely to compile.
- Preserve every material hypothesis and conclusion. Do not weaken the theorem or add assumptions not justified by the natural-language statement.
- If no faithful Mathlib formalization is possible, return `No suitable candidate for formalization found.`

# Tools

You may make at most 24 calls total across these tools:
- `verify_lean`: compile a proposed statement and inspect diagnostics;
- `loogle`: find Mathlib declarations by name or type pattern;
- `leanfinder`: semantically search Lean/Mathlib declarations.

# Natural-Language Theorem
{statement}

# Informal Proof or Supporting Context
{proof}
