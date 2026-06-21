# xai_paper/ — document map and rules

Two living documents, two jobs. Keep them separate so we stop going in loops.

## `xai_paper_plan.md` — the STORYLINE (and Nature structure)
**Purpose:** how to structure the paper so the storyline is sound. This is what we
**write the paper from.** It complies with the Nature Portfolio structure
(Abstract → Introduction → Results → Discussion → Methods → End matter).

**Contains:** the thesis and framing; the subject & scope; the narrative throughline;
the contributions; the Results *arc* (one line per phase — *what* each shows, not
*how*); the discussion arc; the figure/display-item story; journal choice.

**Does NOT contain:** experimental setups, method lists, scoring tables, the oracle,
the T3 procedure, the compute plan, or the substrate audit. Those live in
`experiment_design.md`.

## `experiment_design.md` — the EXPERIMENTS
**Purpose:** the full list of experiments and exactly how each is measured.

**Contains:** the correctness criterion; the ground-truth oracle; how we obtain T3;
the substrate-feasibility audit; and for **each of Phases A–D** a table

> | Analysis | Finding (what the method outputs) | Measured score |

with the named methods (+ citations) as rows, followed by the ideal explanation /
when-right / best-case for that phase; then the expected-pass/fail method matrix and
the compute plan.

**Phase E (synthesis)** is deferred — we design it once results are in; **semantics
(T3) will be central there.** Do not detail it in advance.

## `paper/` — the LaTeX build (`sn-jnl`, Nature Portfolio).
## `document_check.md` — submission-compliance checklist.

## Rule of thumb
- "How do we *say* it?" → `xai_paper_plan.md`.
- "What do we *run*, and how do we *score* it?" → `experiment_design.md`.
- A storyline change edits the plan **only**; a new/changed experiment edits the
  design **only**. If a change touches both, say so explicitly and edit both in one go.
