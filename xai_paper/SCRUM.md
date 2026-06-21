# SCRUM.md — agentic SCRUM for the XAI paper program (reusable for P2–P5)

How multiple subagents build a paper in parallel, off a shared backlog, committing to
`main`, without stepping on each other. **This file is generic** — it governs P2
(`xai_2_interpretability/`) first and is reused verbatim for P3/P4/P5. Each paper supplies
its own `SPEC.md` and `backlog/`; the *process* below does not change.

> Read order for a new agent: `general_paper_plan.md` → this file → your paper's
> `SPEC.md` → your paper's `backlog/` (board + your assigned item files).

---

## 0. The one rule that makes parallelism safe

**Within a sprint, no two items share a file.** Every backlog item declares an exact
**file-scope** (the paths it may create/modify). The Scrum Master admits items into a
sprint only if their file-scopes are pairwise **disjoint**. Therefore parallel developer
agents never write the same file, and concurrent pushes to `main` rebase cleanly.

**Shared files are written only at a barrier** (Sprint Review / Planning), by the Scrum
Master, never by parallel developers. Shared files are: the backlog **board**, the SPEC,
the LaTeX **master** (`paper/main.tex`), and any **aggregate** result/leaderboard file.
(Per-section LaTeX is split so writers get disjoint files — see §6.)

If two needed items would touch the same file, they go in **different sprints**, or one
is split. That conflict is resolved in Review/Planning, not at runtime.

---

## 1. Roles

- **Product Owner (PO)** — the human (Andreas) + the lead assistant. Owns `SPEC.md` and
  priorities. Approves sprint goals. Only the PO edits `SPEC.md`.
- **Scrum Master / Orchestrator (SM)** — one coordinating step (the lead assistant in
  the main session) that runs the **barriers**: Planning (select + assign items, verify
  disjoint file-scopes), and Review (verify Definition of Done, integrate shared files,
  update the board, refine the backlog). The SM does *not* run experiments.
- **Developer agents** — the parallel subagents. Each is handed a disjoint set of items,
  works in its **own git worktree**, and commits/pushes its increment to `main`.

## 2. Artifacts

- **`SPEC.md`** (per paper) — the full specification: epics → tasks, the source of truth
  the backlog is built from. PO-owned.
- **`backlog/`** (per paper):
  - `backlog/BOARD.md` — the sprint board: every item, its status, sprint, owner slot,
    file-scope, deps. **SM-only writes.**
  - `backlog/<ID>.md` — one file per item (schema in §3). A developer updates **only the
    `status` of items assigned to it**; the item body is set at Planning.
  - `backlog/TEMPLATE.md` — the item template.
- **`sprints/sprint-<N>.md`** (per paper) — the Planning note (goal, item→agent
  assignment, the disjointness check) and the Review note (what passed, what's carried
  over, new items). SM-owned.
- **The Increment** — committed, pushed work on `main` that meets the Definition of Done.

## 3. Backlog item schema (`backlog/<ID>.md`)

```markdown
---
id: P2-E3-A2            # paper-epic-task
title: Phase-A lesion sweep on Space Invaders
epic: E3 (Phase A)
status: todo            # todo | in-sprint | in-progress | review | done | blocked
sprint: 2              # set at Planning; empty until then
owner: agent-1         # agent slot, set at Planning
where: cluster         # local | cluster
depends_on: [P2-E1-ORACLE, P2-E0-HARNESS]
file_scope:            # the conflict key — exact paths this item may create/modify
  - tools/xai_study/phaseA_kording/lesion_sweep.py
  - tools/xai_study/phaseA_kording/out/A2_space_invaders.*
estimate: M            # S | M | L
spec_ref: SPEC.md#e3-phase-a
---

## Goal
<one paragraph: what to produce>

## Definition of Done
- [ ] runs to completion (command given)
- [ ] result written to the declared out/ path in the agreed schema (SPEC §results)
- [ ] no regression to Paper-1 gates if any shared code touched (it must not be)
- [ ] committed + pushed to main; primary pulled ff-only
- [ ] item status set to `done`

## Notes / handoff
<anything the Review needs>
```

## 4. The sprint lifecycle

```
Planning (SM)  ──►  Parallel Execution (Developer agents)  ──►  Review (SM)  ──►  Retro/Next Planning
   barrier                     no barrier                          barrier
```

### 4a. Sprint Planning (SM, at a barrier)
1. Pick the sprint **goal** from `SPEC.md` (respect `depends_on`: an item enters a sprint
   only when all its deps are `done`).
2. Select candidate items; **verify file-scopes are pairwise disjoint** (the §0 rule). If
   two clash, drop one to a later sprint or split it.
3. Assign each item to an **agent slot** (`owner`), balancing load and keeping each
   slot's items independent. Tag `local`/`cluster`; **serialize heavy cluster jobs** that
   contend for the same nodes/GPU (see §7).
4. Write `sprints/sprint-<N>.md` (goal, assignment table, the disjointness check) and set
   each selected item's `sprint`, `owner`, `status: in-sprint` on the board + item files.
5. Commit the plan to `main`.

### 4b. Parallel Execution (each Developer agent)
The protocol every developer follows (also in §5):
1. Read `general_paper_plan.md`, `SCRUM.md`, the paper `SPEC.md`, and **your assigned
   item files** (the SM gives you their IDs).
2. For each assigned item, in order of its deps: work **only inside its `file_scope`**,
   in your **own worktree**.
3. Meet the **Definition of Done** (run it; produce results in the agreed schema).
4. `git fetch origin && git rebase origin/main` → commit → `git push origin HEAD:main`
   (rebase-before-push is mandatory; disjoint scopes ⇒ clean). Then
   `git -C <primary> pull --ff-only`.
5. Set the item `status: done` in its own `backlog/<ID>.md` (you own that file this
   sprint) and push.
6. Do **not** touch the board, the SPEC, `paper/main.tex`, or aggregate files — those are
   the SM's at Review.

### 4c. Sprint Review (SM, at a barrier)
1. For each item: verify the Definition of Done (re-run gates / inspect the committed
   increment). Mark `done` or bounce to `blocked`/carry-over with a reason.
2. Do the **integration writes** that need shared files: update `backlog/BOARD.md`;
   aggregate results into the leaderboard; `\input` new LaTeX sections into
   `paper/main.tex`; rebuild the paper if writing happened.
3. **Refine the backlog**: add items discovered during the sprint; re-estimate; re-order.
4. Write the Review section of `sprints/sprint-<N>.md` (done / carried / new / risks).
5. Commit; this is the synchronization point. Then run Planning for sprint N+1.

### 4d. Retrospective (lightweight)
One or two lines in the Review note: what slowed parallelism, any file-scope clashes that
should have been caught at Planning, cluster contention to avoid next time.

## 5. Developer agent protocol (copy/paste contract)

> You are a developer agent in sprint N for paper `<P>`. Read
> `xai_paper/general_paper_plan.md`, `xai_paper/SCRUM.md`, `xai_paper/<P>/SPEC.md`, and
> your assigned items `xai_paper/<P>/backlog/<IDs>.md`. Work ONLY inside each item's
> `file_scope`. Do the work, meet its Definition of Done, then from your worktree:
> `git fetch origin && git rebase origin/main`, commit, `git push origin HEAD:main`,
> `git -C <primary> pull --ff-only`, and set the item `status: done`. Do NOT edit the
> board, SPEC, `paper/main.tex`, or any aggregate file. If blocked or you discover a
> file-scope clash, stop, set `status: blocked` with a note, and report — do not work
> outside your scope. Return a short summary: items done, result paths, anything Review
> needs.

## 6. Parallel paper-writing without conflicts

Split `paper/main.tex` into per-section files and `\input` them:
`paper/sections/00_abstract.tex`, `01_intro.tex`, `02_related.tex`, `03_methods.tex`,
`04_results_A.tex`, `05_results_B.tex`, `06_results_C.tex`, `07_results_compare.tex`,
`08_discussion.tex`, `09_endmatter.tex`. Each writing item owns **one** section file
(disjoint). `paper/main.tex` (the master with `\input`s) and `references.bib` are
**SM-only at Review** (the SM merges new bib keys + adds any new `\input`). Figures: each
figure is its own `.py`/`.pdf` pair under `paper/figures/` (disjoint per item).

## 7. Local vs cluster, and the Paper-1 gotchas (carry into every sprint)

- `where: local` → runs on this machine (M1 Max): the oracle, pilots, small sweeps,
  figures, writing.
- `where: cluster` → LME Slurm (`ssh maier@…`, repo on `/cluster/maier`,
  `tools/cluster/*.sbatch`, jaxtari GPU venv `jax[cuda12]`): full lesion/occlusion
  sweeps, SAE training, batched gradients.
- **Always `git fetch && git rebase origin/main` before push** (a background jaxtari
  agent may push concurrently); **always `git -C <primary> pull --ff-only` after push.**
- **jutari before jaxtari** (jaxtari ~205× slower) — never let a jaxtari item block a
  deliverable; tag jaxtari work as background/cluster.
- **Heavy gates run alone** (pytest xdist can hang at 0% CPU under load) — don't schedule
  two heavy local jobs in the same sprint slot.
- **One SSH session per wakeup** on the cluster; never `sudo`; never bypass host-key
  changes.
- **Never regress the Paper-1 bit-exactness gates** — no P2 item may modify
  jutari/jaxtari core; if an experiment needs a hook, it goes in `tools/xai_study/`, not
  the emulator. (If an emulator hook is truly required, that's a PO decision, its own
  item, gated on the 64/64 sweeps.)

## 8. Definition of Done (global)
An item is `done` only when: it runs to completion from a documented command; its outputs
are in the agreed results schema (SPEC §results) at the declared paths; nothing outside
its `file_scope` changed; Paper-1 gates are untouched/green; it is committed and pushed to
`main` and the primary is pulled; and its `status` is `done`. Writing items add: the
section compiles within `paper/` and stays within the venue's length budget
(`document_check.md`).

## 9. How a sprint is actually launched (orchestration)

A sprint's parallel execution is run as **one Workflow** (the `Workflow` tool):
`parallel(items.map(item => () => agent(developerPrompt(item), {isolation: 'worktree'})))`
— each developer agent gets the §5 contract + its item file; `isolation: 'worktree'`
gives each its own checkout so file writes never collide and pushes serialize via
rebase. The Workflow returns when all developers finish = the synchronization barrier;
the SM then runs Review (§4c) in the main session. Sprint 0 (backlog build) and Reviews
are SM steps, not fan-outs. (A sprint can also be run by spawning the developer agents
with the `Agent` tool and waiting for all to finish — same contract.)

## 10. Sprint 0 is special: build the backlog
The **first** SCRUM action for a paper is **Sprint 0 — Backlog Construction**: an SM/agent
reads `SPEC.md` and **emits one `backlog/<ID>.md` per task** (schema §3), fills
`backlog/BOARD.md`, and assigns each item a tentative sprint + file-scope, verifying the
§0 disjointness per sprint. Sprint 0 also bootstraps shared infra (the harness skeleton,
state recorder, results schema, cluster templates) so later sprints have disjoint,
runnable scopes. Only after Sprint 0 do execution sprints begin.

## 11. Reuse for P3/P4/P5
Copy nothing. Each paper folder already has `plan.md` + `experiment_design.md`; add its
own `SPEC.md` + `backlog/` + `sprints/` and run this same process. The shared oracle/T3/
triad live in `xai_2_interpretability/experiment_design.md` §0–§3 — P3/P4/P5 items depend
on P2 having produced the oracle code (a cross-paper `depends_on`).
