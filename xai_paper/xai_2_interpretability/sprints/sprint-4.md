# Sprint 4 — Method fan-out wave 1: Phase-A battery (local jutari) + cluster provisioning

## Goal
Two tracks in parallel (PO decision: **both — local now + provision cluster**):
1. **Local:** run the full Phase-A neuroscience battery (A1, A3–A8) + the N/A audit on
   **jutari**, each scored F/S/M vs the per-game oracle — the headline Phase-A results.
2. **Cluster:** provision LME (Julia + ROMs + `Pkg.instantiate` jutari + a Julia array
   sbatch + an oracle smoke-test) so the cluster-tagged Phase-B/C matrix can be submitted
   in Sprint 5.

## Planning
Deps satisfied: E3-0 (Phase-A pilot) done; oracle (E1-1) + recorder (E0-2j) + game set
(E0-4) done. Local items have pairwise-disjoint `phaseA_kording/A*.*` (+ `phaseB_attribution/`)
scopes. ≤3 concurrent local workers → 3 waves.

**LOCAL (jutari, this machine):**
- Wave 1: `E3-1` A1 data-flow graph · `E3-3` A3 tuning curves · `E3-4` A4 pairwise correlations
- Wave 2: `E3-5` A5 pooled-activity spectra · `E3-6` A6 Granger causality · `E3-7` A7 NMF/PCA
- Wave 3: `E3-8` A8 whole-state baseline · `E4-13` N/A audit (Grad-CAM/attention/VIPER write-up)

Each method runs on the **6 core games** (`game_set.json`), scored F/S/M vs the per-game
oracle (intervention map), with the oracle-as-method positive control; per-game §R records to
`phaseA_kording/out/`.

**CLUSTER (background, user `maier`):** provision = Julia (cluster module if present, else
user-space juliaup) → `git pull` repo to latest → copy the ROM set up (rsync) →
`Pkg.instantiate`/`precompile` jutari → write `tools/cluster/xai_array_jl.sbatch` (Julia
array runner; the existing `xai_array.sbatch` is Python) → smoke-test the oracle on the
cluster. Deliverable: **cluster READY for jutari jobs.** QOS is currently saturated by the
operator's `mayo-*` jobs, so provisioning submits ≤1 tiny validation job and never cancels
others' work.

**DEFERRED to Sprint 5 (after provisioning):** the cluster method matrix — `E3-2` (A2 lesion
sweep), `E4-1..12` (attribution suite), `E5-1..10` (mechanistic methods): implement each
runner, then submit as Slurm array/GPU jobs over the 6 core games.

## Review
_(to be filled at the Sprint-4 barrier)_
