# P4 — Paper plan (storyline & venue structure)

> **Role of this file:** storyline + venue structure. Experiments → `experiment_
> design.md`. Program map → [`../general_paper_plan.md`](../general_paper_plan.md).
> **Shared oracle / T3 / triad → [`../xai_2_interpretability/experiment_design.md`](../xai_2_interpretability/experiment_design.md)
> §0–§3 (reference, don't repeat).** Status: carved plan; awaits P2 results.

## P4 in one line
*Recovering the lost specification.* The thing we study is **software**; understanding it
means recovering its **documentation/design**, not just describing its behavior. We turn
the **software-reverse-engineering** toolkit on the VCS ROMs and **score the recovered
documentation against the truth we hold** — the first *non-circular* ground-truth
benchmark for design recovery. This is **Phase E** of the program.

## Metadata
- **Working title:** *Recovering the lost specification: a ground-truth benchmark for
  software reverse engineering.*
- **Authors (provisional):** A. Maier, S. Bayer, P. Krauss (+ an SE/PL collaborator).
- **Target:** **ICSE / FSE / ASE** (top SE venues; conference-driven). Alternatives:
  USENIX Security / NDSS (for the binary-analysis slice); *TSE* / *EMSE* (journal).

## The framing (the bar)
The IEEE definition of *software* — **programs + procedures + documentation + data**
(IEEE Std 610.12-1990) — sets the bar: a complete understanding must contain the
**documentation/design**, not only the behavior. A deployed ROM is software with its
documentation *stripped*, so recovering it is, by definition, **reverse engineering /
design recovery** (Chikofsky & Cross 1990). Their definition is the key: design recovery
must *"add domain knowledge, external information, and deduction to the observations …
beyond those obtainable from the system itself"* — i.e. recovery is **provably under-
determined without an external anchor** (the SE echo of Locatello et al. 2019's
impossibility result). Our anchor: behavioral observation (the P3 bridge) + causal
intervention against the bit-exact framebuffer.

## The critique that motivates the benchmark
Today's "solutions" do not *recover* semantics — they **re-author** it:
- **AtariARI** hand-reads commented disassemblies; **OCAtari** hand-codes object
  extractors + the render-offset rules; **JAXAtari** fully reimplements the games.
These are *forward* engineering from an assumed design; and using a reimplementation as
the ground truth is **circular**. We hold the artifact **and** an independent bit-exact
ground truth, so we can score *genuine recovery* — the contribution SE currently lacks.

## Subject & scope
The VCS ROMs (real shipped software). Recovery targets: the **data dictionary**
(RAM→concept/type), **routines/concepts**, the **behavioral specification**, and a
**redocumentation**. Scored against the true design we hold (disassembly + T1/T2 + T3).
No learned agents.

## Contributions
1. The **first non-circular ground-truth benchmark for software reverse engineering /
   design recovery** — artifact + independent truth, on real shipped software.
2. A measured **recovery rate** toward the IEEE documentation bar, per recovery target.
3. Evidence on **what reverse engineering can and cannot recover** — structure/behavior
   (high) vs intent/names (low, AI-hard) — and the role of the behavioral anchor (P3).
4. A reusable harness (invariant mining, active automata learning, decompilation,
   variable/concept recovery) wired to the substrate as an exact membership oracle.

## Honesty guards (from review)
- **Active automata learning is NOT "exactly" learned.** Membership queries are exact;
  **equivalence queries are not** — for a 2^1000-state machine the bit-exact reference
  gives a *high-coverage conformance-testing* (PAC-style) equivalence oracle, not an
  exact one, and only over a chosen finite **abstraction**. Report the residual
  equivalence gap as a measured quantity. (See `experiment_design.md`.)
- **The "recovery %" denominator must be defined concretely** — the intent/naming layer
  is partly unrecoverable, so define what fraction of *what* counts as recovered, or the
  headline number is rhetorical.

---

# Paper structure (SE conference)
- **Abstract; Introduction:** software = documentation+programs (IEEE); the stripped-ROM
  problem; reimplementation is circular; we score genuine recovery.
- **Background / Related work:** reverse engineering & design recovery (Chikofsky &
  Cross); program comprehension / concept assignment (Biggerstaff); spec mining (Ammons),
  invariant detection (Daikon/Ernst), model learning (Angluin L*, Vaandrager); binary
  type/var recovery (TIE, Howard, REWARDS) + neural (DEBIN, DIRE, DIRTY); decompilation
  (Ghidra; LLM4Decompile; Pearce). Position vs AtariARI/OCAtari/JAXAtari.
- **Approach:** the VCS as a ground-truth RE testbed; the shared oracle (cite P2 §1); the
  exact membership oracle + conformance EQ oracle.
- **Evaluation:** the Phase-E experiments (`experiment_design.md`) — recovery rate per
  target, with the equivalence gap and the recovery-% denominator made explicit.
- **Discussion / Threats to validity:** under-determination; horizon; representativeness.
- **Artifact** (SE venues value artifact evaluation): the benchmark + harness.

## Open questions
- Conference (ICSE/FSE/ASE) vs security (USENIX/NDSS, if the binary-analysis slice
  leads) vs journal (TSE/EMSE). Decide by which sub-result is strongest after pilots.
- Whether to scope to a few well-understood games (clean data dictionary) for the
  headline recovery-rate, then breadth in the artifact.
