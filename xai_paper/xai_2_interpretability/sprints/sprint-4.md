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

## Review — closed 2026-06-22, **8/8 local DONE + cluster READY** (both tracks complete)

**LOCAL — Phase-A neuroscience battery** (jutari, 6 core games, scored vs the per-game
oracle; **oracle-as-method positive control F=1.0 on every game/method**; bit-exact baselines
true). 7 runners + 47 §R records in `phaseA_kording/out/`:
- **E3-1 A1 connectomics/data-flow** (`efbe167`) — classical single-shot recovery mean **F1=0.41**
  vs control F1=1.0; under-recovers long-latency/sign-dependent flow (pong/ms_pacman recall 0,
  seaquest 0.09); breakout/qbert hit 1.0 only because their state has one edge.
- **E3-3 A3 tuning curves** (`b77968e`) — nonzero **spurious-tuning rate** (pong 1.0, SI 0.67,
  qbert 0.53) despite fully-known structure; the tuning-curve trap quantified.
- **E3-4 A4 pairwise correlations** (`68b6ba6`) — weak-pairwise/strong-global; correlation
  clusters **over-group** vs the true variable grouping; coupling-oracle F/S/M.
- **E3-5 A5 pooled-activity spectra** — descriptive (frame/game periodicities); reveals cadence,
  not semantics.
- **E3-6 A6 Granger** — **F=0.0, false-edge=1.0, missed-edge=1.0** vs control F1=1.0 — Granger's
  temporal precedence fails to recover true data-flow (SM-reverified).
- **E3-7 A7 NMF/PCA** — NMF matched-frac **0.8** > PCA **0.6** descriptively, but both **negative
  F** as attribution — descriptive match ≠ faithful (SM-reverified).
- **E3-8 A8 whole-state baseline** (`2b63d34`) — trivially faithful (F high), **M at the floor** —
  the non-minimal reference the other methods are measured against.
- **E4-13 N/A audit** — `phaseB_attribution/na_audit.md`: Grad-CAM/attention/VIPER/CAM do **not**
  apply to the VCS (no conv maps / no transformer / no learned policy) — scopes Phase B honestly.

**Headline (quantified Kording):** across the whole battery, classical neuroscience/unsupervised
methods score **low faithfulness** against the exact oracle while the oracle-as-method control
scores **perfectly** — a method can look structured/plausible yet not recover the true causal
program. Measured, not asserted.

**CLUSTER — provisioning DONE + verified** (`a4bc4e1`): Julia 1.12.6 on `/cluster/maier/.julia`,
jutari instantiated (64 deps), ROMs synced (collection + curated `xitari/roms/*.bin`), Julia array
sbatch committed, **oracle smoke 8/8 bit-exact ON the cluster**. **READY.** Constraint: the
operator's queue still holds ~17 pending `mayo-*` jobs, so Sprint-5 cluster jobs queue behind them
(throttle `%N`).

**Verification:** 8/8 `status: done` on main; 47 §R records; SM spot-re-ran A6+A7 (exit 0,
self-checks, bit-exact). **Retro:** clean; several items drifted `.py`→`.jl` (Julia mandated) —
consistent + noted; per-game jutari RomSettings added where missing (seaquest booted Generic, recorded honestly).

**Next:** Sprint 5 = the **method matrix** — implement the remaining method runners (local jutari,
≤3 workers) and run cheap ones locally / submit heavy ones to the cluster (throttled): Phase-B
attribution suite (E4-1..12), Phase-C mechanistic (E5-1..10), A2 lesion sweep (E3-2).
