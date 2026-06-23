---
id: P2-R-REPRO
title: Reproducibility doc — review-time artifact, ROM SHA-256 + retrieval, action-stream hashes
epic: RV (Revision — reproducibility)
status: done
sprint: 9
owner:
where: local
depends_on: []
file_scope:
  - xai_paper/xai_2_interpretability/REPRODUCIBILITY.md
  - tools/xai_study/repro/rom_hash_table.csv
  - tools/xai_study/repro/action_stream_hashes.csv
  - tools/xai_study/repro/make_hash_tables.py
estimate: M
spec_ref: ../SPEC.md#e9-reproducibility--submission-prep-sprint-7-local-sm-led
---

## Goal
Make the XAI benchmark reviewable and reproducible (improvement_instructions P0#1; P1#20; P6
§53–56; final-review §Reproducibility, §Ethics). Produce:
1. **REPRODUCIBILITY.md** — review-time artifact availability (P0#1): the repo structure with
   one-command reproduction (`scripts/reproduce_all.sh`-style pointer), environment files
   (Project.toml/Manifest.toml + pyproject), commit hashes, seeds, and the explicit statement
   that the XAI scoring suite + records are **available for review now** (NOT "released upon
   acceptance"); cite the companion emulator code at https://github.com/akmaier/UnderstandingVCS
   (arXiv:2606.22447). Include the data/ROM policy (ROMs not redistributed) + the review-mode
   stub-test plan (P6#56).
2. **ROM hash table (P1#20):** `rom_hash_table.csv` with game name, ROM source/retrieval
   procedure, **SHA-256**, validation horizon. Generate it from the in-place ROM collection
   (`xitari/games/Atari-2600-VCS-ROM-Collection/ROMS/`; use in place, never commit ROMs).
3. **Action-stream hashes (P1#20):** `action_stream_hashes.csv` with game, action-stream hash,
   seed, start state, #frames, output ids — from the committed action streams used by the oracle.
4. **make_hash_tables.py** — the script that regenerates both CSVs (so the procedure is explicit
   and verifiable), printing a checksum manifest.

## Definition of Done
- [x] runs to completion: `python tools/xai_study/repro/make_hash_tables.py` → writes both CSVs + prints a manifest (deterministic; `--verify` re-hashes ROMs → PASS)
- [x] `rom_hash_table.csv` lists SHA-256 for every core game used in the paper (6/6, real digests verified against bytes); no ROM bytes are committed (no `.bin` in repro/)
- [x] REPRODUCIBILITY.md states review-time availability (§0, not acceptance-time) + the github/arXiv pointers (arXiv:2606.22447 / github.com/akmaier/UnderstandingVCS); adds §5 hash tables + §6.5 artifact-traceability
- [x] nothing outside `file_scope` changed; Paper-1 gates untouched/green (no emulator core touched — doc + offline analysis script only)
- [x] committed + pushed to main (rebase-before-push); primary pulled ff-only
- [x] `status: done`

## Notes / handoff
The S09-endmatter item owns the in-paper "code availability" prose ("released upon acceptance"
→ "available for review"); REPRODUCIBILITY.md is the standalone doc + the hash artifacts. The
two must agree on the wording — recorded in revision_plan.md. Does not touch any `paper/`
LaTeX. The anonymized-link mechanics for double-blind review are a PO decision (see
revision_plan §PO).
