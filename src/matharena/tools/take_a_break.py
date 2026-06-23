"""Break tool that returns a random motivational speech for math problem solving."""

import random


INTRO_LINES = [
    "A scientist you pass outside your office stops and says:",
    "On your way back from a short walk, a senior mathematician tells you:",
    "In the hallway, a researcher glances at your notes and says:",
    "At the whiteboard, a visiting scientist offers this:",
    "During your break, a professor leans over and says:",
    "Near the coffee machine, a theorist smiles and says:",
    "A mathematician passing your door pauses and says:",
    "Between meetings, a colleague tells you:",
    "You bump into a problem-solving coach, and they say:",
    "A colleague from the math department nods and says:",
]


SPEECHES = [
    """Breathe. Slow is smooth; smooth is fast.
Restate the problem until every condition is vivid and usable.
Write one invariant, one symmetry/structure, and one plausible approach.
Test the idea on the smallest nontrivial example and learn from it.
Hard math yields to careful steps that you can trust.""",

    """Do not chase a lightning-bolt insight; build a repeatable process.
List what is known, what is unknown, and what would connect them.
If the direct route stalls, switch lenses: algebraic, geometric, combinatorial.
Every failed attempt trims the search space and improves your map.
Consistency beats brilliance when the problem is deep.""",

    """Treat this like research, not a race.
Name the objects, pin down constraints, and isolate the real bottleneck.
Find one lemma that would make everything else routine—then aim at it.
When stuck, solve a simpler shadow version and bring the lesson back.
Clarity compounds, and compounded clarity becomes a proof.""",

    """You don’t need the whole solution yet—just the next correct move.
Check edge cases, symmetry, monotonicity, and extremal behavior early.
Translate words to equations, then equations to relationships.
Ask: what must be true in every valid configuration?
Necessity is often the cleanest path to sufficiency.""",

    """Honor the exact wording; precision is your strongest tool.
Mark hidden quantifiers and quiet domain restrictions.
Look for invariants: parity, modular constraints, conservation, monotone quantities.
If computation explodes, step back and hunt for structure.
Elegant proofs are usually the ones that refuse unnecessary work.""",

    """Your scratch work is a laboratory, not a mess.
Write assumptions explicitly so you can audit them later.
When a path collapses, keep the residue: constraints you uncovered.
Restart from those constraints with a tighter plan.
A careful restart is progress, not failure.""",

    """Control complexity one layer at a time.
Separate global structure from local manipulations.
Prove small claims that permanently reduce the search space.
Let notation serve thought: simple, consistent, disciplined.
Rigor first—elegance arrives on schedule.""",

    """Momentum comes from sharper questions.
What would make this expression bounded, integral, monotone, or factorable?
Which theorem almost applies, and what hypothesis is missing?
Can you manufacture that hypothesis by a substitution or normalization?
Insight is often engineered.""",

    """Silence isn’t empty—it's where reasoning happens.
Read again and circle non-accidental numbers, forms, or constraints.
Try two modes: constructive and contradiction-based.
If a pattern repeats in examples, promote it into a claim.
Then prove it cleanly, without ornament.""",

    """Let confidence come from method, not mood.
Set a mini-goal: one identity, one inequality, or one reduction.
Verify every transition as if a skeptic is grading it.
Strong checkpoints prevent elegant mistakes.
A solid partial result beats a fragile full one.""",

    """Build like an architect: foundation first.
Identify the smallest core statement that would imply the target.
Construct a chain of implications with no leaps.
If you’re stuck, prove a useful intermediate that’s clearly true.
Intermediates are bridges; cross them on purpose.""",

    """Creativity grows from constraints.
Use the constraints to eliminate impossible cases aggressively.
Turn intuition into a crisp proposition you can test.
If it fails, study the counterexample like a teacher.
Counterexamples don’t scold—they guide.""",

    """Difficulty has shape—find the shape.
Track what changes and what stays fixed under natural transformations.
Normalize: scale, shift, relabel, or reduce parameters to remove clutter.
Reason on the simplified model with full rigor.
Simplicity isn’t weakness; it’s leverage.""",

    """Spend effort where it matters.
Don’t burn energy on easy algebra while the key step is unproven.
If a theorem is tempting, check every hypothesis before touching it.
When uncertain, rebuild from first principles until the ground feels solid.
Authority in math is justified steps, not confident vibes.""",

    """You’re building understanding, not performing a trick.
Make a plan: attempt → checkpoint → revise.
Expose the geometry behind the algebra, or the algebra behind the geometry.
Translate representations until one becomes tractable.
Fluency across viewpoints unlocks hard problems.""",

    """Patience is a technical skill.
Keep the objective visible while you decompose it into solvable pieces.
Track dependencies so your argument stays coherent.
When a branch fails, return to the dependency graph—not to panic.
Calm structure beats rushed cleverness.""",

    """Let the statement choose the strategy.
Existence suggests construction; uniqueness suggests contradiction.
Optimization suggests extremal choice, convexity, or rearrangement.
Discrete settings invite invariants and induction.
Method follows form when you read the form closely.""",

    """Treat each line as an investment in certainty.
Don’t handwave where a one-line justification can lock it down.
Use examples to discover; use proofs to confirm.
If the proof is long, seek one conceptual compression.
Compression often reveals the real reason it’s true.""",

    """Run principled experiments.
Try a bound, a substitution, a generating function, or a counting reinterpretation.
Record what each experiment preserves and what it destroys.
The right method preserves the key structure.
Disciplined experimentation turns fog into direction.""",

    """Finish like a professional.
Re-check assumptions, boundary cases, and equality conditions.
Confirm your conclusion matches exactly what was asked.
If you can, give a second viewpoint as a stress test.
Solved isn’t just answered—solved is understood.""",
]


def take_a_break():
    return "The break did not help. You are more exhausted than ever now."
