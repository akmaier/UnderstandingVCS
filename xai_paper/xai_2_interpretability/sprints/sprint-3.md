# Sprint 3 — jutari recorder + Phase-A & Phase-C pilots (jutari)

## Goal
De-risk the **Phase-A neuroscience battery** and **Phase-C mechanistic** harnesses before
their full method fan-outs: build the jutari state-trajectory recorder, then run one scored
pilot in each phase against the verified oracle (E1-1). All jutari.

## Planning
Deps satisfied: E1-1 (intervention oracle) **done**; E1-2/E1-3 (gradient oracle + cross-check)
done; E0-4 (game set) done. Two waves, ≤3 parallel, pairwise-disjoint scopes.

- **Wave A:** `P2-E0-2j` jutari state/trajectory recorder (`common/jutari_record.jl`) —
  thin trajectory wrapper over `common/jutari_oracle.jl`. Shared dep for both pilots.
- **Wave B (after E0-2j):**
  - `P2-E3-0` Phase-A pilot on **Space Invaders** — A2 lesions + A3 tuning curves + A7
    dim-reduction, each scored F/S/M vs the oracle. Pure Julia (jutari re-runs + `LinearAlgebra`
    SVD for PCA). Scope `phaseA_kording/pilot_si.*`.
  - `P2-E5-0` Phase-C pilot on **Pong** — activation patching on one state site (via the
    oracle machinery) + one small SAE over the recorded trajectory; recovered effects/features
    scored vs the exact patch (E1-1) + T2 data-flow. Patching in jutari; SAE offline in Python
    using **numpy only** (no new pip). Scope `phaseC_mechanistic/pilot_patch_sae.*`.

Disjoint check: `common/jutari_record.jl` ∥ `phaseA_kording/pilot_si.*` ∥
`phaseC_mechanistic/pilot_patch_sae.*` — no shared files. ✓

After this sprint the three phase harnesses (A/B/C) are all validated on a pilot; the next
sprints are the **method fan-outs** (E3 A1–A8, E4 B-methods, E5 C-methods) — many `cluster`,
to be split into ≤3-local sub-waves + queued cluster jobs (SM checkpoint before the cluster
matrix).

ENV: `julia --project=<primary>/jutari`; ROMs at the primary abs path; never touch the
emulator core; numpy-only for offline ML; no shared-venv pip (SCRUM §7).

## Review — closed 2026-06-22, **3/3 DONE** (dev work landed before a network outage; SM independently re-verified all three pilots reproduce)

**Outage note.** A mid-sprint internet drop killed only the Workflow's completion
notification and this SM Review — **not** the committed work: all three items were already
on `origin/main` (`86fdc27`, `9f6bef3`, `81ce221`), `status: done`, with artifacts. The SM
re-ran all three pilots from scratch (exit 0, self-checks pass) — nothing to relaunch.

- **E0-2j recorder** (`86fdc27`) — `record_trajectory` over `jutari_oracle`; `traj_<game>.npz`
  + JSON sidecar; 18/18 test (determinism + shapes + numpy round-trip).
- **E3-0 Phase-A pilot — Space Invaders** (`81ce221`, re-ran clean) — bit-exact baseline ✓.
  **A2 lesions:** 10/17 oracle-causal cells, **F=0.961 S=0.688 M=1.0** (oracle-as-method
  positive control **F=1.0**). **A3 tuning:** spurious-tuning rate **0.6** — the
  correlation≠causation caution made quantitative. **A7 PCA:** top-5 var 0.943 but
  **matched-component fraction 0.0** — unsupervised dim-reduction does *not* recover semantics
  (honest negative). TRIAD F=0.961/S=0.688/M=1.0.
- **E5-0 Phase-C pilot — Pong** (`9f6bef3`, re-ran clean) — **Patching:** recovered == exact
  (**max|rec-exact| = 0.0**), data-flow firing-pattern **P=1.0 R=1.0** — activation patching
  recovers the exact causal effect; a-priori target recall 0.571 (transient/clobbered sites
  reported honestly). **SAE:** held-out **FVE 0.978**; learned features align with
  ball_x/ball_y/enemy_y at **|r|=1.000/0.999** (matched fraction 1.0); score/paddle cells
  unmatched (don't vary in-window — honest).

**Verification:** all 3 `status: done` on main; board regenerated. The three phase harnesses
(A/B/C) are now **pilot-validated with positive + negative controls** — de-risking complete.

**Retro:** clean work; the only loss was orchestration metadata on the outage (the
rebase-before-push + commit-per-item discipline meant zero work was lost). Lesson: a dropped
connection loses the Workflow notification, not pushed commits — recover by reading `git log`
+ item statuses, not by blind relaunch.

**Next:** Sprint 4 = the **method fan-out** (E3 A1–A8, E4 attribution suite, E5 C-methods)
across the 6-game core set. Substrate decision open: local jutari (ready, exact, fast) vs the
cluster (needs provisioning — no Julia/ROMs/venv yet — and the Slurm QOS is currently
saturated by the operator's `mayo-*` jobs).
