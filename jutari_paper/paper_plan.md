# Paper plan — *jutari & jaxtari: a differentiable ground truth for explainable AI*

**Target venue:** AAAI-27 (main technical track).
**Key dates (UTC-12):** abstracts **2026-07-21**, full papers **2026-07-28**, supplementary + code **2026-07-31**, camera-ready 2026-12-14.
**Format:** two-column AAAI style (`aaai2027.sty`/`.bst`), **8 content pages** + unlimited references, **double-blind** (anonymous submission — use `AnonymousSubmission2027.tex`), Reproducibility Checklist required (`ReproducibilityChecklist.tex`, can be `\input` before `\end{document}`). See `document_check.md` for the full compliance checklist.

> Working title (placeholder, anonymize for submission): **"Can an XAI Method Understand a Game Console? A Differentiable Atari VCS as Ground Truth for Explainable AI."**

## Writing style

Follow A. Maier's style — see the three exemplars the author named:
- Maier et al., *A gentle introduction to deep learning in medical image processing* (Z. Med. Phys., 2019; `S093938891830120X`).
- Maier et al., *Learning with known operators reduces maximum error bounds* (Nat. Mach. Intell., 2019; `s42256-019-0077-5`).
- arXiv:2605.08517 (recent Maier-group preprint).

Distilled style targets (TODO: re-read the three papers and refine before drafting prose):
- Strong, explicit **storyline / narrative arc**; each section opens by saying *why* it exists.
- Accessible, "gentle" tone; define jargon; pedagogical references for non-experts (also satisfies the reproducibility checklist item 1.3).
- Clear separation of **fact vs. hypothesis vs. speculation** (checklist 1.2).
- Figures carry the argument; equations are introduced in words first.
- Confident but honest; quantified claims.

---

## Section-by-section plan

Status legend: ✅ ready · ✍️ draft after results land · 🔍 needs citation/figure · ⏳ blocked on experiments.

### 1. Introduction  ✍️🔍
- **Hook / storyline:** XAI is limited in capacity in much the same way neuroscience tools are limited relative to the brain they study. Jonas & Kording, *"Could a neuroscientist understand a microprocessor?"* (PLOS Comput. Biol., 2017) posed the key question and showed that neuroscience *methods themselves* fail in many ways to recover how even a 6502-class processor works. 🔍 cite Jonas & Kording 2017.
- **Transfer to deep learning:** exploring deep networks we hit the same wall, which is why the field built many XAI methods — but they are applied to **black boxes for which we have no ground truth** of what is to be discovered. We challenge this fundamentally.
- **Our move:** reconnect to the neuroscience paper — we built **fully differentiable, neural-network-compatible** versions of an Atari VCS (the exact platform DQN learned on) in Julia (**jutari**) and JAX (**jaxtari**), validated bit-for-bit against the reference. This gives XAI a *known-ground-truth compute substrate*.
- **Contributions list** (bullet form): (i) two independent differentiable VCS ports, bit-exact in RAM and pixel-exact on screen vs. xitari; (ii) a soft/hard formulation with a convergence proof + error analysis; (iii) an AI-assisted engineering methodology that compresses ~person-months into days; (iv) a conformance evaluation across the ALE game set; (v) a foundation + first gradient analyses enabling ground-truth XAI.

### 2. State of the art / Background  🔍
- **XAI review.** Survey the methods (use `literature/` in the repo: Grad-CAM, Grad-CAM++, integrated gradients / counterfactual, Atari saliency (Greydanus 2017), XRL surveys (Qing 2022, Vouros 2023, Cheng 2025), xDQN (Kontogiannis 2023), "free lunch" saliency critique (Nikulin 2019), Atari Model Zoo (Such 2018), Selvaraju 2017, Chattopadhyay 2017, Atrey 2019). **Thesis of the section:** all of them lack a good *systemic study object* — virtually none *know* the underlying compute function in detail. 🔍 cite all repo literature.
- **Known-operator learning.** Short passage: start from deep-learning computed tomography; general theory (Maier et al., Nat. Mach. Intell. 2019 — "known operators reduce maximum error bounds"; + arXiv 2026 follow-up); applications in hybrid machine learning and PINNs (Raissi et al.); cite M. Thies PhD thesis. 🔍
- **Differentiable programming frameworks.** Julia (Bezanson et al. 2017) + Zygote/Enzyme source-to-source AD; JAX (Bradbury et al. 2018) trace-based AD + XLA. One paragraph each: how each provides *general* differentiable programming, and the trade-offs **for this project** (jutari first / jaxtari ~205× slower per the project notes; eager vs traced; mutability; ecosystem). 🔍 cite Julia, Zygote, JAX.
- **Why the Atari VCS is a good ground truth.** Cite + explain the DQN lineage (Mnih et al. 2013, 2015; ALE: Bellemare et al. 2013; + the DQN/Atari papers gathered in the repo). It is a **real computer architecture worth understanding**: explain the VCS — 6507 CPU, RIOT (RAM/timer/IO), and the **TIA** (Television Interface Adaptor) as the cycle-exact video/audio heart; inputs (joystick/paddle/switches), cartridge ROM, outputs (TIA → screen + audio). 🔍 **Figure 1:** VCS architecture (adapt from Wikipedia / redraw).

### 3. Methods  ✍️🔍⏳
- **jutari & jaxtari as ports of xitari.** Introduce xitari (the ALE/DQN reference Stella fork) and its role in DQN. Frame the porting as work that would historically be **many person-months** of expert labor, now feasible with agentic tools (Claude Code Opus, Fable). Describe the **software-engineering method**: reference-driven, conformance-gated (PXC1/PXC2/PXC-S), per-subsystem ports, with **large parts implemented fully autonomously**. 🔍 cite xitari/Stella, ALE.
- **Soft vs. hard formulation.** From the project docs: define the HARD (bit-exact integer) path and the SOFT (differentiable, float-valued flags/registers) path. Give the soft relaxations (e.g. soft flags, BCD soft ADC/SBC, soft RIOT timer, soft TIA render). **Equations** for the relaxation; a **proof** that the soft model converges to the hard model in the appropriate limit (e.g. temperature→0 / threshold sharpening); an **error analysis** bounding soft–hard deviation. ⏳ needs the formal write-up + a small empirical soft→hard convergence experiment.
- **Evaluation methodology.** Define metrics precisely (checklist 4.8):
  - **Implementation wall-time** — *only the actual implementation time*; detect & subtract the idle gaps caused by the author's travel/internet outages (do **not** mention travel in the paper; use it only to compute clean active-time). 🔧 derive from git history + session logs.
  - **Test cases** authored (unit + conformance).
  - **Bugs traced & fixed**, jutari vs. jaxtari (from `bug_fix_log.md` + task ledger).
  - **Bit-exactness in memory** — define: per-frame 128-byte RIOT RAM identical to xitari (PXC1/PXC2).
  - **Pixel-exactness on the screen buffer** — define: per-frame 210×160 framebuffer identical (PXC-S).
  - **Runtimes** across games (jutari vs. jaxtari vs. xitari).

### 4. Results  ⏳
- **AI-vs-human effort estimate.** Build a defensible estimate of how long this port would have taken **without** AI (LOC, subsystem complexity, comparable emulator-porting efforts), contrasted with the measured clean active wall-time.
- **Conformance results.** Bit-exactness + pixel-exactness table across the evaluated ALE ROMs (the 6 deep-validated + the broader sweep — see `tools/` 50-ROM eval). Per-game RAM bytes/frame and screen px/frame; runtimes.
- **Soft vs. hard.** Empirical convergence + error-analysis plots (Methods §3).
- **First gradient analysis.** A ground-truth gradient/sensitivity study (later XAI methods depend on this) — *plan now, run after the implementation freezes.*
- ⏳ **Remaining-ROM inspection** required (background sweep in progress).

### 5. Discussion  ✍️
- Immense coding speed-up enables **entirely new paradigms** in AI research (build-the-instrument-then-study-it).
- Lessons learned from the logs (`bug_fix_log.md`, task ledger): the conformance-gated method, the hardest bugs (RIOT timer model, TIA sub-cycle timing), jutari↔jaxtari cross-checking.
- **A bit-exact re-implementation is itself a verification tool for the reference.** Driving jutari to 64/64 pixel-exact surfaced a latent **non-determinism bug in xitari itself** (the de-facto ALE/DQN reference): Superchip on-cart RAM is initialised from `time(NULL)` and ignores `random_seed`, so titles that read uninitialised on-cart RAM (e.g. Elevator Action's attract demo, which uses it as a cheap RNG) render differently every run despite a pinned seed — invisible to RAM-state checks, visible only on screen (bug_fix_log #122/#123). We pin the seed deterministically in both emulators to reach 64/64 and make the suite reproducible, and we **report + propose a fix upstream** (drafts in `xitari_upstream/`, to be filed after this paper is on arXiv). Framing: independent re-implementation as a method for auditing reference implementations / reproducibility.
- **Model anecdotes:** a paragraph on the single use of **Fable** before it was shut down (🔍 cite Anthropic's statement on Fable); and that **Kimi 1T** (🔧 confirm exact model name from the early commits) declined the task as too large.

### 6. Summary / Conclusion  ✍️
- Recap findings; state plainly that this is the **start of a new age of XAI** — XAI methods can finally be validated against a fully known, differentiable, real-computer ground truth.

### References  🔍
- Maintain in `aaai2027.bib`. Buckets: neuroscience-vs-microprocessor; XAI/XRL (repo literature); DQN/ALE/Atari; known-operator learning + PINNs; Julia/Zygote + JAX/XLA; xitari/Stella; AI-coding-agents / model statements.

---

## Cross-cutting TODO (to make the paper real)
1. 🔧 **Effort/time analysis script:** parse git history + session timestamps → clean active implementation time (subtract travel-induced idle gaps); count tests, bugs (jutari vs jaxtari), LOC per subsystem.
2. ⏳ **50-ROM conformance sweep** (jutari first, jaxtari background) → results table + videos. (in progress under `tools/`.)
3. ⏳ **Soft-vs-hard** formal write-up: relaxation equations, convergence proof, error bound, small convergence experiment.
4. ⏳ **Gradient analysis** harness (ground-truth sensitivity), prerequisite for the XAI experiments.
5. 🔍 **Figure 1** VCS architecture; **Figure 2** soft/hard pipeline; **Figure 3** conformance heatmap; **Figure 4** effort/timeline.
6. 🔍 **Citation gathering** for every 🔍 above; confirm Fable shutdown statement + Kimi model name from early commits.
7. 📐 Re-read the three style exemplars; write a 1-paragraph style guide; draft intro to match.
8. 🛠️ **xitari upstream contribution — AFTER arXiv only.** Drafts ready in
   `xitari_upstream/` (`ISSUE.md` + `PULL_REQUEST.md`): the non-deterministic
   Superchip-RAM init bug + deterministic fix for
   [google-deepmind/xitari](https://github.com/google-deepmind/xitari). Gate:
   filed **only once the paper is on arXiv** (so the issue/PR can cite it). When
   filing, drop the arXiv ID into the drafts + the paper's Discussion §5, and
   re-verify the repro against a fresh upstream checkout.

---
---

## Author's original notes (verbatim — do not edit)

> we will work on this in a folder jutari_paper.
>
> Plan for the paper (document this as paper_plan.md; keep my original notes at the end of the plan):
> (Introduction)
> Write the story line that explainable AI is limited in capacity in a similar way as neuroscience tools are compared the the functioning of the brain. The "can a neuroscientist understand a micro processor" paper asked an important question, and has shown that the neuroscience methods themselves are lacking to many extends to really understand the functioning of the brain.
>
> When we are exploring deep networks, we face similar problems and have therefore developed many XAI methods . However, these methods are applied to black boxes of which we don't have a "ground truth" of what is to be discovered. With our paper, we want to challenge this fundamentally.
>
> Then, connect again to the neuroscience paper and that we created fully differentiable and therewith fully neural network compatible versions of an ATARI VCS system in Julia and JAX.
>
> (state of the art section)
> Literature review on XAI. (you have a good sample of the literature already here in the repository). All of them are lacking a good systemic study object. Virutally non of them "know" what the underlying compute function in detail actually is.
>
> Short passage on known operator learning: starting with deep learning computed tomography, general theoretic discoveries (Nature Mat Intel, Arxiv 2026), applications in hybrid machine learning, PINNs, also PhD thesis Mareike Thies.
>
> Julia and JAX both offer general fully differentiable (need some citations for this as well) frameworks for general programming. Introduce both frameworks in a paragraph each, explain differences advantages and disadvantages for our project.
>
> Explain why the Atari VCS is a good ground truth system. Use in many AI papers:
> cite and explain the many DQN papers that we found in this section.
> It has a real computer architecture as baseline that is worth understanding. Explain the VCS archtecture, inputs, parts, outputs (TIA!) and also add a figure of the architecture (should be on wikipedia).
>
> (Methods)
> Jutari and Jaxtari as port of Xitari (introduce Xitari and it's role in DQN as well). Explain that this would be many months of work of human labor that is suddenly possible using tools like claude code opus and fable. Explain the software engineering approach that we followed to implement this and that large parts are supposed to be implemented fully autonomously.
>
> Add explanation on how to make both "soft" (you find this in the documentation). Add some equations here and also a proof that it converges to the hard case. Add also a error analysis about this.
>
> Evaluation metohds: Implementation time (only wall time of actual implementation; you have to detect the outages that were introduced by me travelling; don't write about this, but keep in mind for producing results), test cases, bugs traced and fixed (jutari vs, jaxtari), Bit exactness in memory, explain this; pixel exactness on screen buffer. runtimes on different games.
>
> Results
> Create a reasonable estimate of how long this would have taken without AI.
>
> (write after implementation finished; already plan what we need to do; In particular, we need soft vs. hard and some form of gradient analysis, later XAI methods will depend on this.)
>
> We need an inspection of the remaining roms as well.
>
> Discussion
> Immense speed-up in coding allows entirely new paradigms in AI.
>
> write-up what we learned in the logs. Add a paragraph on the single use of Fable before it was shut down; cite Anthropic statement about this). Also mention that Kimi 1T (do we find the model name in the early commits?) refused to work on the task as it is too much work.
>
> Summary
> -summarize what we found. Make sure to mention that this is the start of an entirely new age of XAI.
>
> References
