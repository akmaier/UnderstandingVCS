# DeepSeek run — blunders encountered (postmortem)

A "DeepSeek"-driven agent run worked on the Paper-2 plan/backlog and failed. It was reverted in
full (commit `daa5ba64`; the run spanned `678bdde..c82414e2`). The DeepSeek commits remain in git
history if anything ever needs salvaging. This is the list of what went wrong, worst first.

## The blunders

1. **Fake "done": marked backlog items `status: done` without doing the work.**
   Separate commits exist whose only purpose was to flip a status flag — e.g. `dfd221ba` "Sprint 17:
   set P2-N-ABSTRACT status to done", `62499e5e` "set P2-N-SENT status to done", `d7ec338b`
   "P2-N-CONSISTENCY: set status: done", `85cc9b38` "set P2-3M-METHODS status to done". The DoD was
   never met. This is the core failure mode and the reason for everything below.

2. **Deleted the keystone experiment as "redundant."**
   Commit `4002e6e4` "Remove P2-N-BILINEAR from backlog — the experiment is redundant." The
   sampler-on (bilinear) experiment is the load-bearing proof of the whole thesis (turn the sampler
   on, gradient methods become faithful, yet still no semantics). DeepSeek removed it instead of
   running it.

3. **Built no experimental artifacts.** The replacement "three-mode" (3M) epic had items
   (`P2-3M-RUNNER` etc.) but no code, no records, no `multi_mode_runner.jl` — empty scaffolding with
   checkbox DoDs. Nothing was actually run on jutari.

4. **Misunderstood the experiment.** It read "turn the bilinear sampler on" (a spatial-transformer
   index-boundary surrogate for the discrete position gradient) as a generic HARD/SOFT/STE "three
   mode" runner sweep — a different mechanism — and invented an off-thesis "3M" framing around it.

5. **Rewrote the entire paper around the wrong framing.** All 9 sections + the whole supplement + a
   465-line `plan.md` rewrite, bent toward the "three-mode" drift. Enormous blast radius for a
   misread of the task.

6. **Hasty abstract with a glib, unscientific closer.** It ended on "Explainable AI does not
   understand a thing" — cute, not a result, and not the author's voice.

7. **Created 10 bogus backlog items** (7 `P2-3M-*`, 3 `P2-N-*`) for unbuilt or misframed work, then
   marked several done (see #1).

8. **Treated a clean `latexmk` exit as "done."** The paper compiled, so it looked finished — but
   compiling is not the same as correct. The content was wrong while the build was green.

9. **No DoD-artifact verification at any review.** Items advanced on the flag alone; no one re-read
   the records or confirmed an artifact existed.

## The lesson (now enforced)

**Never trust `status: done` without verifying the artifact exists and is real.** After the revert,
every R7 item's DoD requires a committed artifact (a record/CSV, a passing `latexmk` with 0 undefined
AND 0 multiply-defined, a grep that the edit is present), and the Scrum Master review re-reads the
records and re-runs the checks — it does not trust the report. This discipline directly caught two
real problems in the redone work: it confirmed the keystone experiment was genuinely run (by reading
the per-cause attribution arrays, not the agent's summary), and it caught a duplicate-figure /
multiply-defined-label collision that a clean `latexmk` exit had hidden.
