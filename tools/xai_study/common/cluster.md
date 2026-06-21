# Cluster workflow — LME Slurm (Paper 2, `xai_study`)

How `where: cluster` items (E3 lesion/occlusion sweeps, E4 attribution sweeps,
E5 SAE training / batched gradients) submit, run, and collect on the LME Slurm
cluster. Templates live in `tools/cluster/`:

| Template | Use | Resource |
|---|---|---|
| `tools/cluster/xai_array.sbatch` | CPU **array** sweep — one task per shard (game / state / lesion index) | `--cpus-per-task=4`, `16G`, `4h`, no GPU |
| `tools/cluster/xai_gpu.sbatch`   | GPU **batch** — one job, batched gradients / IG-over-ROM / SAE training | `--gres=gpu:<type>:1`, `32G`, `8h` |

Both mirror the Paper-1 `tools/cluster/bench_gpu.sbatch` pattern (output-dir
trap, `set -euo pipefail`, `nvidia-smi` non-fatal, a `jax.devices()` GPU
assertion before walltime is spent). They are **runner-agnostic**: you pass the
`tools/xai_study/*.py` runner via `XAI_CMD` and extra flags via `XAI_ARGS`, so a
new experiment needs no new sbatch file — only a runner that accepts the shared
flags below.

## Environment facts (the cluster)

- **Login / account:** user `maier` (confirmed). One SSH session per wakeup;
  never `sudo`; never bypass a host-key change.
- **Repo on the cluster:** `/cluster/maier/UnderstandingVCS` — submit from the
  repo root so the `#SBATCH --output=results/slurm/...` relative paths resolve
  (`SLURM_SUBMIT_DIR`). Keep the cluster checkout on `main`, pulled before a run.
- **GPU venv (jaxtari):** `jaxtari/.venv/bin/python` with **`jax[cuda12]`** —
  the same venv Paper-1's GPU bench uses. CPU array tasks default to the same
  venv (`jax` CPU backend is enough); override with `VENPY=...` if a runner needs
  the jutari/Julia path instead.
- **Shared XLA cache:** `/cluster/maier/.jax_cache` (`CACHE_DIR`) — reuse across
  jobs so the first compile is paid once.
- **ROMs:** the full Atari-2600 collection is on the cluster under
  `/cluster/maier/UnderstandingVCS/xitari/games/Atari-2600-VCS-ROM-Collection/ROMS`
  (`ROMS_DIR` default). ROMs are **gitignored — use in place, never commit**.
- **Slurm tooling is not on the dev laptop** (M1 Max) — author + lint sbatch
  here, submit only on the cluster. `sbatch --test-only tools/cluster/xai_*.sbatch`
  validates a header against the live partitions once you are on a login node.

## Shared runner contract (what `XAI_CMD` must accept)

Both templates invoke the runner with a fixed flag set so any `xai_study` runner
plugs in unchanged. Results follow SPEC §R:
`results/xai/<phase>/<exp>_<game>[_<state>].json` (+ sibling `.npz` for arrays),
each row appended to `results_index.csv` at SM Review (never by the worker).

- `xai_array.sbatch` -> `--shard <i> --nshards <N> --shard-kind <game|state|lesion|shard> --roms-dir <dir> --out-dir results/xai --where cluster <XAI_ARGS>`
  - the shard is **also** exported as env (`SLURM_ARRAY_TASK_ID`, `SHARD_INDEX`,
    `NSHARDS`, `SHARD_KIND`) so a runner may read either.
- `xai_gpu.sbatch`   -> `--roms-dir <dir> --out-dir results/xai --where cluster --device gpu --cache-dir /cluster/maier/.jax_cache <XAI_ARGS>`

## Submit

```bash
# --- CPU array sweep: one task per game in a 6-game core set ---
sbatch --array=0-5 \
       --export=ALL,XAI_CMD=tools/xai_study/phaseA_kording/A2_lesions.py,SHARD_KIND=game \
       tools/cluster/xai_array.sbatch

# --- CPU array sweep: 64 shards of a flat work list, 8 concurrent (% throttle) ---
sbatch --array=0-63%8 \
       --export=ALL,XAI_CMD=tools/xai_study/ground_truth/oracle_intervene.py,NSHARDS=64 \
       tools/cluster/xai_array.sbatch

# --- GPU batch: integrated gradients on a Quadro 8000 ---
sbatch --gres=gpu:q8000:1 \
       --export=ALL,GPUTYPE=q8000,XAI_CMD=tools/xai_study/phaseB_attribution/integrated_gradients.py,XAI_ARGS="--game pong --batch 4096" \
       tools/cluster/xai_gpu.sbatch
```

Override any default at submit time via `--export` (`VENPY`, `ROMS_DIR`,
`CACHE_DIR`, `OMP_NUM_THREADS`, ...) or on the `#SBATCH` line (`--time`,
`--mem`, `--gres`). The array `%K` cap is the **one knob that throttles cluster
contention** — set it so a sweep does not monopolise the partition.

## Collect (at SM Review)

1. One SSH session: `squeue -u maier` to confirm the array/job finished; check
   `results/slurm/<jobname>-<jobid>[_<task>].{out,err}` for the `=== DONE ===`
   line and no `FATAL`.
2. Pull the per-shard records under `results/xai/` back to the primary checkout
   (e.g. `rsync` / `scp` the JSON+NPZ; large `.npz` only if the figure needs it).
3. The SM appends each record to `results_index.csv` (append-only, **SM-merged**)
   — workers never touch the aggregate (SCRUM §0).

## Gotchas (carry from Paper-1, SCRUM §7)

- **Always `git fetch && git rebase origin/main` before push; `git -C <primary>
  pull --ff-only` after** — a background jaxtari agent pushes concurrently.
- **Heavy gates run alone:** serialize GPU jobs that share a node/GPU (dependency
  chain `sbatch --dependency=afterok:<jobid> ...`, or distinct `--gres` types).
  jaxtari pytest can hang at 0% CPU under load — do not co-schedule a heavy gate
  with a sweep on the same node.
- **jutari before jaxtari** (jaxtari ~205× slower) — never let a jaxtari/GPU job
  block a local deliverable; tag jaxtari work cluster/background.
- **One SSH session per wakeup;** never `sudo`; never bypass host-key changes.
- **Never modify the emulator core** to make a sweep run — experiment hooks live
  in `tools/xai_study/`, the Paper-1 64/64 bit-exact gates stay green.
