# xai_paper/ — a multi-paper research program (start here)

This folder is **not one paper**. It is a program of papers (P2–P5) built on the
differentiable Atari VCS from Paper 1. If you are an agent or collaborator picking this
up: **read [`general_paper_plan.md`](general_paper_plan.md) first** — it is the map
(the vision, the shared assets, the per-paper theses and venues, the sequencing).

## Layout
- **[`general_paper_plan.md`](general_paper_plan.md)** — the program map. Read first.
- **[`SCRUM.md`](SCRUM.md)** — the **reusable agentic SCRUM process**: how multiple
  subagents build a paper in parallel off a shared backlog, commit to `main`, and
  synchronize at sprint barriers without write-conflicts. Governs P2 first; reused for
  P3/P4/P5.
- **[`STYLE.md`](STYLE.md)** — Andreas Maier's **writing-style guide** (analysed from three
  of his papers; Paper 1 excluded). Mandatory for every paper-drafting item; reused P2–P5.
- **One subfolder per paper, named `xai_N_topic/`:**
  - **[`xai_2_interpretability/`](xai_2_interpretability/)** — P2 (active): the
    interpretability ground-truth benchmark — Phases **A** (neuroscience) **+ B**
    (attribution/XAI) **+ C** (mechanistic interp). → *Nature Machine Intelligence*.
    **This folder also defines the shared oracle / T3 / correctness triad** that the
    other papers reuse.
  - **[`xai_3_psychology/`](xai_3_psychology/)** — P3: psychology-of-a-known-machine —
    Phase **D** (behavioral probing of the game's own logic). → *PNAS*.
  - **[`xai_4_recovery/`](xai_4_recovery/)** — P4: software reverse engineering /
    design recovery — Phase **E**. → *ICSE/FSE*.
  - **[`xai_5_agents/`](xai_5_agents/)** — P5 (outline): the learned-agent capstone
    (`emulator ∘ agent`, partial ground truth). → *NMI/Nature*.

## The two documents inside every paper folder (do not blur them)
- **`plan.md` — the STORYLINE + venue structure.** How to tell the story so it is
  sound; what we *write the paper from*. Thesis, framing, contributions, the Results
  *arc* (one line per phase — *what* it shows, not *how*), Discussion, the venue's
  section structure, figures. **No** experimental setups, method lists, or scoring
  tables.
- **`experiment_design.md` — the EXPERIMENTS.** The full list and how each is scored.
  Each phase leads with a table `| Analysis | Finding (what the method outputs) |
  Measured score | Needs T3? |` whose rows are the named methods (+ citations), then
  ideal / when-right / best-case. Also the method matrix and compute plan. (P2's also
  holds the shared §0 triad, §1 oracle, §2 T3 procedure, §3 substrate audit; P3/P4
  reference those rather than repeat them.)
- Plus **`document_check.md`** (venue compliance) and **`paper/`** (LaTeX).
- **Execution layer** (added for P2; reused per paper): **`SPEC.md`** — the full work
  breakdown (epics → tasks with file-scopes/deps/where/DoD) that the SCRUM backlog is built
  from; **`backlog/`** — one file per work item + the sprint `BOARD.md`; **`sprints/`** —
  per-sprint planning/review notes. The agentic process over these is [`SCRUM.md`](SCRUM.md).
  *Mapping:* `plan.md`/`experiment_design.md` = **what & why**; `SPEC.md`/`backlog/` =
  **work items & state**; `SCRUM.md` = **how agents execute them**.

**Rule of thumb:** "How do we *say* it?" → `plan.md`. "What do we *run*, and how do we
*score* it?" → `experiment_design.md`. A storyline change edits the plan only; a
new/changed experiment edits the design only; if a change truly touches both, say so
and edit both in one go.

## Invariants every paper inherits (see general_paper_plan.md for detail)
- **Subject = the VCS itself**, never learned agents — until P5 (the deliberate hard case).
- **Ground truth in tiers:** T1 causal (exact) · T2 hardware (exact) · T3 game-concept
  (partial; verified by intervention). T3 is needed only for *semantic-level* metrics.
- **"Right" = Faithful ∧ Sufficient ∧ Minimal**; "interpretable" = ground-truth recovery
  (our operationalization; Barbiero 2025 is contrast, not the source of this definition).
- **Necessary-condition screen:** a method that fails here — perfect ground truth, no
  learning — should not be trusted on a neural net. The benchmark is a screen.
- **Shared oracle/T3/triad live in `xai_2_interpretability/experiment_design.md` §0–§3.**

## Experiment harness
Code scaffolding (specs + pilot entry points) is in `tools/xai_study/` (phaseA–E +
ground_truth); its README maps each phase to its paper.
