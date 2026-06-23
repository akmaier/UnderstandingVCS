# Sprint 13 — Revision R5: framing (intro + discussion + endmatter)

## Goal
The framing layer: the semantics/understanding reframe in the intro, the discussion's scope-narrowing
+ exemplar labels + Fig 6(map) caption, and the endmatter Code-Availability true-up to the resolved
GitHub→Zenodo→Code Ocean decision. Three disjoint owners.

## Planning (≤3 local workers; pairwise-disjoint file_scopes)
- **P2-R-S01-intro** — `sections/01_intro.tex`. number_audit §2: SAE Pong add "on Pong" (01:69-70;
  intro already labels ACDC "for Breakout"). Sharpen the reframe (faithfulness ≠ understanding; the
  "honesty contract" — what we DO and do NOT claim); tone down overclaims; keep the Jonas–Kording
  vivid register (drama from the problem + numbers, not ornament). No new experimental numbers.
- **P2-R-S08-discussion** — `sections/08_discussion.tex`. number_audit §2: M7 cell 84
  "decodable"→"present-but-unused" (08:78); ACDC "F=M=1.0/S=0.44" add "on Breakout" (08:40-43);
  "1−S=0.56 on an exactly recovered circuit" (08:99) — Breakout gives 0.5556, family mean 0.675, say
  which; SAE "on Pong" (08:41-43). Apply the **Fig 6 (VCS→NN map = fig5_representativeness_map)
  caption handoff** (r2_caption_handoffs.md: shorten, NN column = "documented analogue, not proven",
  §3.2 oracle, cite the 11 analogue works, point to the full SI version). Scope-narrowing: the VCS
  gap is a necessary-condition stress test, NOT a proven NN floor (improvement P0); remove residual
  "no business"/"remove the excuse" heat; point to SUPP S6 "does NOT claim" box.
- **P2-R-S09-endmatter** — `sections/09_endmatter.tex`. Rewrite the **Code/Data Availability**
  statement per revision_plan PO#3 + REPRODUCIBILITY §0: "available for review at
  github.com/akmaier/UnderstandingVCS; a versioned snapshot will be archived on Zenodo (DOI) and an
  optional Code Ocean capsule reproduces the leaderboard and figures" — replace "upon acceptance"
  with "available for review"; ROMs not redistributed (AutoROM + the SHA-256 hash table in the SI);
  emulator substrate public at arXiv:2606.22447. Keep the no-human-subjects + ethics wording.

## Contract (all R5 items)
number_audit §1 canonical numbers; ACDC=Breakout exemplar / SAE=Pong exemplar / 1.0-vs-0.0=position
regime / gap not robust off position. STYLE.md voice (no run-in \paragraph headings; vivid but
restrained; Siming Bayer is 2nd author). Never fabricate. Recompile-clean before done.

## Review
_(to be filled at the R5 barrier)_
