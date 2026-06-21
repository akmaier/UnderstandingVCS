# General paper plan — the XAI research program on the differentiable Atari VCS

This folder holds a **multi-paper program**, not one paper. Paper 1 (done, AAAI-27)
delivered the platform; Papers 2–5 use it. Each paper lives in its own subfolder
`xai_N_topic/`. **Read this file first**, then the subfolder you're working in. The
per-folder rules are in [`README.md`](README.md).

## The vision (one program, many chapters)
The Atari VCS is a complex system that is **fully known, bit-exact, and
differentiable** (Paper 1). That makes it a universal **model organism for the science
of understanding opaque systems**: for any output we can compute the *true* causal
structure, so we can **score interpretability methods against ground truth** instead of
eyeballing them. The community uses several *traditions* to understand opaque systems —
neuroscience/mechanistic, attribution/XAI, mechanistic interpretability, behavioral/
psychology, and software reverse engineering — and none can be *validated* in the wild.
We validate them all, on one system, one tradition at a time.

**The spine that every paper inherits:**
- **Subject = the VCS itself** (chip + program + game logic), *not* learned agents —
  until P5, which is the deliberate hard case.
- **Ground truth in tiers:** **T1** causal (exact, all 64 games) · **T2** hardware
  semantics & data-flow (exact) · **T3** game-concept labels (partial; OCAtari/AtariARI
  + verified by intervention).
- **"Right" = F ∧ S ∧ M** — Faithful (claimed causes = true causes), Sufficient
  (predicts held-out interventions on the exact re-run), Minimal & at the right level.
- **"Interpretable" = ground-truth recovery** — *our* operationalization (Barbiero et
  al. 2025 is cited as related/contrast — their notion is inference-equivariance, which
  is different; do not attribute "ground-truth recovery" to them).
- **The necessary-condition screen** (the answer to "why does a 1977 chip predict
  anything about neural nets?"): a method that fails *here* — with perfect ground truth
  and no learning — has no business being trusted on a neural network. The benchmark is
  a *screen*, not a sufficiency predictor.

## Shared assets (built once, reused)
The **ground-truth oracle**, the **T3 procedure**, and the **F∧S∧M triad** are defined
in **`xai_2_interpretability/experiment_design.md` §0–§3** and **reused** by P3/P4/P5
(they reference it rather than re-deriving). The LaTeX/template lives per paper (each
venue differs). Experiment harness scaffolding is in `tools/xai_study/` (phase→paper
map in its README).

## The papers

| Paper (folder) | Thesis | Phases | Primary venue | Why that venue |
|---|---|---|---|---|
| **P2** `xai_2_interpretability/` | *Do our interpretability methods recover what is true?* — a ground-truth faithfulness benchmark for the interpretability toolkit | **A** (neuroscience baseline) **+ B** (attribution/XAI) **+ C** (mechanistic interp) | **Nature Machine Intelligence** (write to *Nature*'s bar; shared template, decide at submission) | NMI covers interpretability/XAI, AI↔neuroscience, benchmarks, methods-critique; the broad reframe+benchmark story fits Article/Analysis. |
| **P3** `xai_3_psychology/` | *The psychology of a known machine* — the first ground-truthed verdict on behavioral / psychology-of-AI probing | **D** (behavioral probing of the game's own decision logic) | **PNAS** (alt: NMI; *Cognition*/CogSci) | Binz & Schulz 2023 and Shiffrin & Mitchell 2023 — the exact debate D answers — are PNAS. |
| **P4** `xai_4_recovery/` | *Recovering the lost specification* — a non-circular ground-truth benchmark for software reverse engineering / design recovery | **E** (semantic / design recovery) | **ICSE / FSE / ASE** (alt: USENIX Security/NDSS for the binary slice; TSE/EMSE journal) | Daikon, concept assignment, active automata learning, decompilation are SE/PL/security work; SE lacks a non-circular RE ground truth. |
| **P5** `xai_5_agents/` | *From known software to learned policy* — apply the validated toolkit to DQN agents, where ground truth is only **partial** (the hard case) | the toolkit ∘ learned agents (`emulator ∘ agent`) | **Nature Machine Intelligence / Nature** | The capstone: tests whether the VCS-validated methodology transfers to neural policies; the original "interpret real RL agents" goal. |

**A stays inside P2** as the calibration baseline (alone it largely re-runs Jonas &
Kording; as the "even with unlimited noiseless data, classical methods score low-F"
contrast it sets up B/C). If ever standalone: *PLOS Comp Biol* / *eLife*.

## Phase → paper map
- **A, B, C → P2**  ·  **D → P3**  ·  **E → P4**  ·  **agents (emulator ∘ agent) → P5**

## Sequencing & dependencies
1. **P2 first** — it builds the shared oracle + T3 + triad + the benchmark artifact.
2. **P3 and P4 in parallel after P2** — independent communities/reviewers; both cite P2
   for the oracle. P4 can move fast (off-the-shelf tools: Daikon, an L* library, Ghidra).
3. **P5 last** — needs the toolkit proven on ground truth first.

## Folder & naming convention
- One subfolder per paper: **`xai_N_topic/`** (N = paper number).
- Inside each: **`plan.md`** (storyline + venue structure — what we write from),
  **`experiment_design.md`** (the experiments; per-phase `Analysis | Finding | Measured
  score` tables), **`document_check.md`** (venue compliance), **`paper/`** (LaTeX).
- The plan-vs-experiment_design split rule (don't blur them) is in `README.md`.

## Status
- **P1:** complete (AAAI-27, arXiv package built).
- **P2:** planned in depth (this is the active folder).
- **P3, P4:** carved plans + experiment tables present; await P2 results / go-ahead.
- **P5:** outline only (designed after P2–P4).

## Housekeeping note
Verify Paper-1 `STATUS.md`/`RESULTS.md` reflect the final **64/64** bit-exactness and
the **bounded conformance window** before any paper cites the substrate — those files
are stale relative to the Paper-1 manuscript.
