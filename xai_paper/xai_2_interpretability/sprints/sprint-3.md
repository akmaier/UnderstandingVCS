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

## Review
_(to be filled at the Sprint-3 barrier)_
