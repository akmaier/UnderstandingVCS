# P2 — Full specification (the source for the backlog)

What must be **built, run, and written** to deliver Paper 2 (interpretability ground-truth
benchmark: Phases A+B+C → *Nature Machine Intelligence*). The *why / what-counts-as-right*
is in [`plan.md`](plan.md) and [`experiment_design.md`](experiment_design.md); this file is
the **work breakdown** the SCRUM backlog is generated from (see [`../SCRUM.md`](../SCRUM.md)
§10). Each task lists a tentative **file-scope** (the conflict key), **deps**, **where**
(local/cluster), and **Definition of Done (DoD)**.

> Conventions: subject = the VCS (no agents). Code lives in `tools/xai_study/`; results in
> each phase's `out/`; the paper in `paper/`. **ROMs:
`xitari/games/Atari-2600-VCS-ROM-Collection/ROMS/` (local, gitignored — use in place).**
Results schema in §R. Game set in §G.
> Reuse Paper-1 gates; never modify the emulator core (SCRUM §7).

## G. Game set (decide in E0, fix for the paper)
- **Pilot game:** Space Invaders (Phase A) / Pong (oracle, B, C running example).
- **Core set (headline):** ~6 games with verified T3 (e.g., Pong, Breakout, Space
  Invaders, Seaquest, Ms. Pac-Man, Q*Bert) — used for all of A/B/C with full scoring.
- **Breadth set (generality):** the remaining ALE games for T1/T2-only metrics (no T3
  needed), reported in Supplementary.

## R. Results schema (decide in E0, then frozen)
Every experiment writes a self-describing record: `out/<phase>/<exp>_<game>[_<state>].json`
with `{paper, phase, method, game, frame/state, target_output, metric_name, value,
ci/stderr, n, seed, where, commit, oracle_ref, timestamp}` + any arrays as sibling `.npz`.
A tiny `results_index.csv` (append-only, **SM-merged at Review**) lists every record. This
makes the leaderboard (E6) a pure read.

---

## E0 — Infrastructure & harness bootstrap  *(Sprint 0–1; mostly local)*
Turns `tools/xai_study/` from specs into a runnable harness with disjoint per-experiment
files.
- **E0-1 Harness skeleton & results schema.** A small `tools/xai_study/common/` lib:
  game loading, deterministic replay-to-state, the results-record writer (§R), seeds.
  *scope:* `tools/xai_study/common/**`. *where:* local. *DoD:* importable; round-trips a
  result record; unit-tested.
- **E0-2 State/trajectory recorder.** Dump per-step RAM/registers/TIA-RIOT/opcode/
  framebuffer for a ROM+input-trace (reuses Paper-1 dumps). *scope:*
  `tools/xai_study/common/record_state.*`. *deps:* E0-1. *where:* local. *DoD:* produces a
  trajectory tensor for SI within the conformance horizon; documented.
- **E0-3 Cluster job templates.** `tools/cluster/xai_*.sbatch` for array sweeps + GPU
  batches (mirror Paper-1 pattern). *scope:* `tools/cluster/xai_*.sbatch`,
  `tools/xai_study/common/cluster.md`. *where:* local (authoring). *DoD:* a dry-run job
  script validated; documents `/cluster/maier`, jaxtari GPU venv, one-SSH rule.
- **E0-4 Game-set + T3 coverage audit.** Fix §G; list which games have verified T3.
  *scope:* `tools/xai_study/common/game_set.md`, `.../game_set.json`. *deps:* E2-1. *DoD:*
  core/breadth sets fixed; T3 coverage per game recorded.
- **E0-5 Backlog construction (Sprint 0).** Generate `backlog/<ID>.md` for every task here
  + fill `backlog/BOARD.md`. *scope:* `backlog/**`. *DoD:* every SPEC task has an item with
  file-scope/deps/where/DoD; per-sprint disjointness verified (SCRUM §0).

## E1 — Ground-truth oracle  *(Sprint 1; local, the foundation)*
The object every method is scored against (`experiment_design.md` §1).
- **E1-1 Intervention oracle (exact).** Δ`y`(u) by occlude/clamp/replace/resample + exact
  re-run, batched via SOFT-STE. *scope:* `tools/xai_study/ground_truth/oracle_intervene.*`,
  `.../out/oracle_*`. *deps:* E0-1,E0-2. *where:* local→cluster for scale. *DoD:* exact
  causal map for the Pong score/ball-pixel; bit-exact re-run asserted.
- **E1-2 Gradient oracle (content path).** ∂`y`/∂u + IG via the differentiable substrate;
  **content-path only**, with the index/position caveat from `experiment_design.md` §1.
  *scope:* `.../ground_truth/oracle_grad.*`. *deps:* E1-1. *DoD:* gradient map for a content
  output; documented zero-on-index behavior.
- **E1-3 Cross-check + validation.** Intervention↔gradient correlation; flag non-smooth
  points; the (1)↔(2) report. *scope:* `.../ground_truth/oracle_xcheck.*`,
  `.../out/oracle_xcheck_*`. *deps:* E1-1,E1-2. *DoD:* correlation reported on Pong; the
  oracle is declared the reference for A/B/C.

## E2 — T3 acquisition  *(Sprint 1–2; local)*
Game-concept labels, verified by intervention (`experiment_design.md` §2).
- **E2-1 Import OCAtari/AtariARI candidates.** *scope:*
  `tools/xai_study/t3/import_labels.*`, `.../t3/out/candidates_*`. *where:* local. *DoD:*
  candidate RAM→concept maps for the core set.
- **E2-2 Verify-by-intervention + offset-correct.** Perturb byte → object moves on the
  exact framebuffer; upgrade to verified-causal; record the verified-rate. *scope:*
  `.../t3/verify_labels.*`, `.../t3/out/verified_*`. *deps:* E2-1,E1-1. *DoD:* verified T3
  for the core set + a per-game verified-rate.
- **E2-3 Discover (correlation + sweep).** New labels where no source exists. *scope:*
  `.../t3/discover_labels.*`, `.../t3/out/discovered_*`. *deps:* E2-2. *where:* cluster
  (sweeps). *DoD:* extended coverage logged.

## E3 — Phase A: neuroscience battery  *(Sprints 2–3; local pilot, cluster full)*
One item per analysis (`experiment_design.md` §4); each its own runner + `out/` → disjoint.
- **E3-0 Phase-A pilot** (A2+A3+A7 on SI, scored). *scope:* `phaseA_kording/pilot_si.*`,
  `.../out/pilotA_*`. *deps:* E0,E1. *where:* local. *DoD:* pilot scores produced.
- **E3-1..E3-8** A1 connectomics · A2 lesions · A3 tuning · A4 correlations · A5 LFP · A6
  Granger · A7 dim-reduction · A8 whole-state — each on the core set (+ breadth for
  T1/T2 metrics). *scope (each):* `phaseA_kording/<A#>_*.py`, `.../out/<A#>_*`. *deps:*
  E3-0. *where:* lesion/sweep = cluster; analyses = local. *DoD (each):* the metric in
  `experiment_design.md` §4 computed + recorded (§R).
- **E3-9 (optional) Visual6502 transistor track** for A1/A2 head-to-head. *scope:*
  `phaseA_kording/visual6502/**`. *PO-gated after the pilot* (novelty risk per plan).

## E4 — Phase B: attribution / XAI  *(Sprints 2–4; cluster for sweeps)*
One item per method (`experiment_design.md` §5); each its own runner + `out/`.
- **E4-0 Phase-B pilot** (IG vs oracle, one output, one game). *scope:*
  `phaseB_attribution/pilot_ig_vs_oracle.*`, `.../out/pilotB_*`. *deps:* E1. *where:* local.
- **E4-1..E4-N** vanilla gradient · Grad×Input/DeepLIFT · Guided BP · SmoothGrad · **IG** ·
  Expected Gradients · occlusion · extremal perturbation · RISE · LIME · KernelSHAP/Shapley
  · on-distribution counterfactual · **N/A audit** (Grad-CAM/attention/VIPER do-not-apply
  writeup). *scope (each):* `phaseB_attribution/<method>.py`, `.../out/<method>_*`. *deps:*
  E4-0. *where:* oracle sweeps = cluster (GPU), gradients = cluster (GPU). *DoD (each):* corr
  + deletion/insertion AUC on the true VCS + precision@k recorded; T3-object metrics where
  applicable.

## E5 — Phase C: mechanistic interpretability  *(Sprints 2–4; cluster for SAE/sweeps)*
One item per method (`experiment_design.md` §6).
- **E5-0 Phase-C pilot** (activation patching + 1 SAE on VCS state). *scope:*
  `phaseC_mechanistic/pilot_patch_sae.*`, `.../out/pilotC_*`. *deps:* E1,E0-2. *where:* local.
- **E5-1..E5-N** activation patching/causal tracing · interchange/DAS · attribution+edge
  patching · path patching · ACDC · SAEs · NMF/PCA · causal scrubbing · probing+control
  tasks · logit/tuned lens. *scope (each):* `phaseC_mechanistic/<method>.py`,
  `.../out/<method>_*`. *deps:* E5-0. *where:* SAE training = cluster (GPU), patching sweeps
  = cluster. *DoD (each):* recovered-vs-exact-patch / feature↔variable / scrubbing metric
  recorded; T3 for the semantic checks.

## E6 — Cross-tradition comparison + benchmark artifact  *(Sprint 5; local; SM-integrated)*
- **E6-1 Leaderboard.** Read `results_index.csv`; place A/B/C methods on the shared
  faithfulness-vs-plausibility axes; the discrepancy table. *scope:*
  `tools/xai_study/compare/leaderboard.*`, `.../out/leaderboard.*`. *deps:* E3,E4,E5. *DoD:*
  one comparable F∧S(∧M-where-available) score per method.
- **E6-2 Benchmark artifact.** Package tasks + oracle + metrics as a reusable benchmark
  (with ROM-not-redistributed handling). *scope:* `tools/xai_study/benchmark/**`. *deps:*
  E6-1. *DoD:* a third party can run one method end-to-end and get a score.
- **E6-3 Faithful-method demonstration.** Show a causal/gradient method near-ceiling vs a
  popular one near-chance (the headline). *scope:* `compare/faithful_demo.*`. *deps:* E6-1.

## E7 — Figures  *(Sprint 5; local; one figure per item)*
Each figure = its own `.py` + `.pdf` (disjoint). Figures (plan.md): (1) platform & oracle;
(2) A–C on the shared axes (headline); (3) Kording battery scored; (4) attribution vs
mech-interp; (5) VCS↔NN failure-mode / representativeness map; (6) failure taxonomy.
*scope (each):* `paper/figures/fig_<k>.py`, `paper/figures/fig_<k>.pdf`. *deps:* E6 (for
data-driven figures), none for schematic ones. *DoD:* vector PDF, colour-blind-safe,
self-contained legend.

## E8 — Nature paper draft (writing)  *(Sprint 6; local; one section per item)*
Split `paper/main.tex` into `paper/sections/*.tex` (SCRUM §6) so writers are disjoint.
- **E8-0 Skeleton.** SM converts `paper/` to `main.tex` + `\input` of empty section files.
  **All drafting follows [`../STYLE.md`](../STYLE.md)** — Andreas Maier's voice, already
  analysed from his three papers in `papers/` (the Gentle-Introduction, Known-Operator, and
  hybrid-ML review; **Paper 1 excluded**). *scope:* `paper/main.tex`, `paper/sections/*.tex`
  (created empty). *SM task.*
- **E8-1..E8-9** abstract · intro (gap→opportunity→move→payoff + representativeness +
  prior ground-truth benchmarks) · related work · methods (emulator, oracle, metrics) ·
  results A · results B · results C · results-comparison · discussion + end-matter. *scope
  (each):* one `paper/sections/<NN>_*.tex`. *deps:* E6,E7 (results/figures) for results
  sections; E1–E2 for methods. *DoD (each):* section drafted **in Maier's voice per
  [`../STYLE.md`](../STYLE.md) (run its §6 self-check)**, cites real refs (verify), compiles
  via the master, within the venue length budget (`document_check.md`).

## E9 — Reproducibility & submission prep  *(Sprint 7; local; SM-led)*
- **E9-1 document_check pass** (`document_check.md` fully satisfied: statements, Reporting
  Summary, limits). *scope:* `document_check.md`, `paper/sections/09_*`. 
- **E9-2 Reference verification** (no-hallucination pass on the new bib). *scope:*
  `paper/references.bib`.
- **E9-3 Reproducibility bundle** (seeds, configs, benchmark; ROMs via hashes/AutoROM).
  *scope:* `tools/xai_study/benchmark/REPRO.md`.
- **E9-4 Final build + page/limit gate.** *scope:* `paper/` (SM). *DoD:* clean
  pdflatex+bibtex, within caps, all statements present.

---

## Sprint roadmap (dependency-ordered; details/mechanics in `../SCRUM.md`)
- **Sprint 0** — E0-5 backlog build + E0-1/E0-3 infra skeleton. *(SM + 1–2 agents.)*
- **Sprint 1** — E1 oracle (E1-1/2/3) ∥ E0-2 recorder ∥ E2-1 import ∥ pilots E3-0/E4-0/E5-0.
  Barrier: oracle validated, pilots green, scope decided.
- **Sprints 2–4** — full experiments, fanned out: E3 (A1–A8) ∥ E4 (methods) ∥ E5 (methods)
  ∥ E2-2/E2-3. Each method/analysis is its own item (disjoint files); cluster sweeps tagged
  and serialized where they contend.
- **Sprint 5** — E6 comparison + benchmark + E7 figures (each figure its own item).
- **Sprint 6** — E8 paper draft (each section its own item; parallel writing).
- **Sprint 7** — E9 reproducibility + document_check + final build.

## Definition of Done (paper-level)
All E0–E7 results in `results_index.csv`; the leaderboard + headline figure produced; the
Nature draft compiles within caps with all required statements; the benchmark artifact
released; no Paper-1 gate regressed. Then PO review → submission.
