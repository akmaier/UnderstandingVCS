
> **STATUS (2026-07-17): all items applied.** Judgment calls: **B7** handled via the body F/S/M definition ("Faithful and Sufficient and Minimal") rather than the caption; **B14** resolved by dropping the "Phase~E" label (not inventing a Phase~D) and noting Papers 3-5 are unpublished companions; **S2** handled contextually ("to the byte and pixel"); **S18** glossed once in the Supplement, main text uses normal citations; **M5** (symbol note) skipped as clutter for a Nature paper; **M9** already adequate (caption names blue=VCS / vermillion=NN); **M12** ("by construction") kept as standard idiom.
# JARGON_AUDIT — fresh-reader stumble report for Paper 2

**Provenance.** Generated 2026-07-17 by a fresh-reader audit: a sub-agent read Paper 2 in the
compiled PDF order (Abstract → Intro → Results A–C → Across-traditions → Discussion →
Methods-last → End matter → Supplement) while deliberately **not** reading `README.md` or
Paper 1, simulating a smart ML/XAI reader with zero project knowledge. Calibration example
from the authors: "the gap excludes zero" (shorthand for "the 95% bootstrap CI of the gap
excludes zero") surprises a new reader.

**How to use.** Each item has a checkbox — tick it when addressed (fixed, or consciously
kept as-is). Severity: BLOCKS understanding > SLOWS the reader > MINOR.

**Structural root cause.** Because **Methods is the last numbered section**, most core
machinery (regimes, F/S/M, oracle formalism, plausibility proxy, tiers, the sampler,
Prop. zero-gradient) is **used in the Abstract and all Results sections before it is
defined**. That one fact generates the majority of the worst stumbles.

---

## Top 10 to fix first

1. [x] **"regime" / "position regime" / "content regime"** — the primary results axis; used from the Abstract onward, defined only in Methods (last section).
2. [x] **F / S / M as bare symbols; "sufficiency S", "minimality M"** — numbers like `S=0.13`, `M=0.08`, `S=−0.46` appear in Results with no scale, sign convention, or definition until Methods. A *negative* sufficiency is uninterpretable on first encounter.
3. [x] **"the gap excludes zero"** — CI-significance shorthand; first used in the Abstract *before* its expansion four lines later; recurs ~14 times.
4. [x] **"plausibility proxy"** — a headline axis; that it is a rubric score, *not* a human study, is stated only in Methods/S7. Without it the "danger zone" is unreadable.
5. [x] **"bilinear sampler"** — the keystone experiment hinges on it; what it *is* is explained only in Methods.
6. [x] **T3 labels / Tier 1–3** — carry the semantic-verification argument in Results B/C; tiers defined only in Methods.
7. [x] **"naive gradient is provably zero" / "hard forward map" / Prop. zero-gradient** — a load-bearing "provably" referenced from the Abstract on; the proposition lives only in Methods.
8. [x] **TIA and RIOT** — named repeatedly, **never expanded anywhere** (verified). Also "6502".
9. [x] **"Paper 1" / "Maier et al." / "companion paper" + "Paper 3/4/5" + "Phase E"** — an invisible five-paper program; **"Phase E" appears with no Phase D anywhere**.
10. [x] **`do(u)` notation** — Pearl's do-operator with no gloss; `do(u:=v')→Δy(u)` is opaque outside causal inference.

---

## SEVERITY: BLOCKS UNDERSTANDING
(ordered by first appearance)

- [x] **B1. "the gap excludes zero"** — `00_abstract.tex`: "($0.71$ versus $0.38$; the gap excludes zero)". First use precedes the "95 percent bootstrap interval … excludes zero" expansion. FIX: expand on first use ("the 95% bootstrap confidence interval of the gap excludes zero, i.e., the difference is significant"), then abbreviate.
- [x] **B2. "position regime" / "content regime"** — `00_abstract.tex` onward. The five-regime output taxonomy (content / position-index / event / score / pixel) is enumerated only in Methods §oracle. FIX: one-clause gloss at first use, or move the regime list forward.
- [x] **B3. F / S / M bare symbols; negative S** — `04_results_A.tex` "low sufficiency ($S = 0.13$)"; `06_results_C.tex` "S$=−0.46$". Scales (S∈[−1,1], F,M∈[0,1]) unknown until Methods. FIX: at first Results use, "sufficiency S (held-out predictive correlation, in [−1,1])", "minimality M (…, in [0,1])".
- [x] **B4. "plausibility proxy"** — `01_intro.tex` Fig. 1 caption onward. FIX: "a documented per-method-tradition plausibility *proxy* (a fixed rubric score, not a human-subjects measurement)".
- [x] **B5. "content-path gradient" / "content path"** — Fig. 1 caption uses it as a core capability; defined only in Methods. FIX: gloss ("the differentiable path through register/color/graphics values, as opposed to the discrete position read").
- [x] **B6. `do(u)` / `do(u:=v')→Δy(u)`** — Fig. 1 caption, Methods. FIX: "the intervention operator do(u) (set cause u to a value and re-run)".
- [x] **B7. "correctness triad" / `right = F∧S∧M`** — Fig. 1 caption before any definition; "∧" as boolean AND over thresholded scores not obvious. FIX: expand once ("Faithful AND Sufficient AND Minimal, each a thresholded score").
- [x] **B8. "bilinear sampler"** — Abstract/Intro/Compare name it; mechanism only in Methods. FIX: "(a differentiable sub-pixel interpolation that turns the discrete sprite-position read into a smooth function, so a gradient can flow)".
- [x] **B9. "Prop. zero" / "hard forward map" / "naive gradient"** — asserted repeatedly before Methods states the proposition. FIX: one-sentence statement of the result at first Results-B use; gloss "naive gradient (gradient of the exact/hard forward map, before any smoothing surrogate)".
- [x] **B10. T3 labels / three-tier ground truth** — `05_results_B.tex` "external T3 labels (AtariARI, OCAtari)"; tiers defined only in Methods §tiers. FIX: first use: "external semantic labels (our Tier-3 ground truth: game-concept names imported from AtariARI/OCAtari; §Methods)".
- [x] **B11. TIA / RIOT / 6502 never expanded** — `04`/`05`/`03`. FIX: expand once in Methods §substrate: "the 6502 CPU, the TIA (Television Interface Adaptor, graphics/sound), and the RIOT (RAM-I/O-Timer)".
- [x] **B12. "off-/on-manifold"** — `05_results_B.tex` load-bearing; defined only in Methods. FIX: gloss ("off-manifold: a byte value the running program would never itself write").
- [x] **B13. "IOI style"** — `06_results_C.tex`; IOI (Indirect Object Identification) never expanded. FIX: "the single-edge style of the IOI (indirect-object-identification) circuit study".
- [x] **B14. Papers 1/3/4/5, "companion paper", "Phase E" (no Phase D exists)** — Intro + Discussion reference an invisible five-paper program; the phase alphabet jumps A,B,C → E. FIX: one line describing Paper 1 at first cite; state Papers 3–5 are unpublished companions not relied on here; fix or rename "Phase E".
- [x] **B15. "validated conformance window" / "horizon"** — Discussion (read before Methods) uses it without the bound. FIX: add the concrete "~30-frame RAM / 60-frame screen window certified by Paper 1".

---

## SEVERITY: SLOWS THE READER
(ordered by first appearance)

- [x] **S1.** "VCS" unexpanded in the Abstract (expanded only in Intro). FIX: "Atari 2600 VCS (Video Computer System)".
- [x] **S2.** "bit-exact" never explicitly defined. FIX: one clause ("identical to the original hardware down to every bit of RAM and pixel").
- [x] **S3.** "family-mean gap" / method "family" — which methods form the two families is only tabulated later. FIX: define both families with 1–2 example methods at first use.
- [x] **S4.** "positive control" / "oracle-as-method". FIX: "the oracle scored as if it were a method (a positive control that must hit 1.0)".
- [x] **S5.** "jutari"/"jaxtari" — why two? FIX: "two interchangeable, cross-validated implementations (jutari in Julia, jaxtari in JAX)".
- [x] **S6.** "substrate" as a standing noun. FIX: introduce once ("the differentiable VCS port, our 'substrate'").
- [x] **S7.** Dense neuroscience-battery vocabulary (spike-word, weak-pairwise/strong-global, LFP analogue, beam clock, register-transfer level, vsync). FIX: brief glosses on the most opaque.
- [x] **S8.** "FSM composite" reads as *finite state machine* (it is the F/S/M triad); "recon-error" unexpanded. FIX: "the reconstruction-error / (F,S,M)-triad composite".
- [x] **S9.** "matched fraction" / "matched-component fraction" never crisply defined. FIX: "the share of known game variables the method's components correctly align to".
- [x] **S10.** "the six core games" used before the roster is listed (only in Results B). FIX: list at first mention or forward-reference.
- [x] **S11.** The 6 / 42 / 54 / 64 / 1773 / 30 / 31 count thicket — funnel explained only in Supplement S3. FIX: one early sentence: "64 emulated games → 54 with semantic labels → 42 scorable; six are 'core' running examples."
- [x] **S12.** "$M\times$ the compute" collides with minimality M (`05_results_B.tex`). FIX: use $K\times$ ("for $K$ baseline draws").
- [x] **S13.** "IG" used without "(IG)" introduction. FIX: "Integrated Gradients (IG)" at first mention.
- [x] **S14.** Shapley/LIME internals (Σφ completeness, coalitions, surrogate R²) unglossed. FIX: one phrase each or a footnote.
- [x] **S15.** "cell 84 … within the window" — "window" silently means the analysis horizon, not a screen region. FIX: "within the 15-frame analysis horizon".
- [x] **S16.** Phase-C method table lists DAS / ACDC / logit-tuned lens etc. before any are introduced (DAS expanded later; ACDC only in §Across-traditions). FIX: expand acronyms at first (table) use + 3–5-word descriptions.
- [x] **S17.** "danger zone" definition trails the first plausibility-axis uses. FIX: cross-reference the definition from the first mention.
- [x] **S18.** "committed record" / `leaderboard.json` / "§R records" / "E6-1" / `faithful_demo` — insider artifact names. FIX: state once these are files in the released repo ("committed" = version-controlled); keep sprint IDs out of reader-facing text.
- [x] **S19.** "straight-through estimator (STE)" + "temperature" unglossed. FIX: "a surrogate gradient whose forward pass still returns the exact discrete value".
- [x] **S20.** "color clocks", "strobe write", "position-counter reset", "race the beam" — VCS timing model behind the zero-gradient proposition. FIX: two-sentence beam-timing gloss ("color clock = one pixel-time of the scanning beam; strobe write = a write that latches a sprite's horizontal position").
- [x] **S21.** "calibration split" vs "held-out split" for sufficiency. FIX: "we fit the effect claim on one set of interventions and test it on a disjoint set."
- [x] **S22.** "deletion/insertion area-under-curve" undescribed. FIX: "progressively deleting/inserting the top-attributed inputs and measuring output change".
- [x] **S23.** "algorithmic level" / "implementational level" (Marr) used in Intro/Methods, explained only in the Discussion. FIX: gloss at first use ("what function is computed, in Marr's sense").

---

## SEVERITY: MINOR

- [x] **M1.** Faithfulness scale unstated in the Abstract. FIX: "(0 = chance, 1 = oracle ceiling)".
- [x] **M2.** "saliency map" — 3-word gloss for truly general readers.
- [x] **M3.** PCA never expanded (NMF is).
- [x] **M4.** Bare "SAE" never introduced as "(SAE)".
- [x] **M5.** Symbol stack ($F_1$, ρ, P@k, $e_{\mathrm{idx}}$, one-hot, Jacobian) — consider a symbol note.
- [x] **M6.** "no-op" — informal CS slang.
- [x] **M7.** Unlabeled "[n=11]" / "[n=14]" (method counts per family).
- [x] **M8.** Table "Family" labels ("black-box", "interv.") do not exactly match the prose's two-family framing ("causal and intervention" vs "gradient and correlational").
- [x] **M9.** Color names as data encoding ("blue … vermillion") in the map figure caption.
- [x] **M10.** "IEEE 610.12" — add "(the IEEE software-engineering glossary)".
- [x] **M11.** Scattered low-level terms (disassembly, opcode, framebuffer, strobe, latch).
- [x] **M12.** "by construction" used very frequently.

---

## Cross-cutting recommendation

The single highest-leverage fix is the **"used before defined" problem created by placing
Methods last.** Nearly every BLOCKS item (B2–B10, B15) is a concept the Results assume but
Methods defines. Either

- **(a)** add a short **"Definitions at a glance" box after the Introduction** covering
  *regime, F/S/M, the intervention oracle, the plausibility proxy, T1–T3 tiers, the bilinear
  sampler, and the 64→54→42 game funnel*, or
- **(b)** add a one-clause gloss at the first Results use of each term, cross-referencing the
  full Methods definition.

Option (a) alone converts most first-pass blockers into mere slow-downs.

---

**Verified facts from the audit:** TIA, RIOT, PCA, and IOI are never expanded in any section
body. "Phase E" is referenced in the Discussion with no "Phase D" anywhere. "regime" first
appears in the Abstract but is enumerated only in Methods §oracle. The "excludes zero /
crossed zero" shorthand recurs ~14 times.

**Files reviewed:** `paper/sections/{00_abstract,01_intro,04_results_A,05_results_B,06_results_C,07_results_compare,08_discussion,03_methods,09_endmatter}.tex`
and `paper/supplement/*.tex` (in compiled reading order; `02_related.tex` is a stub).
