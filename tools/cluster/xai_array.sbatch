#!/bin/bash
# Paper-2 (xai_study) CPU array sweep — one Slurm array task per shard of an
# embarrassingly-parallel job (e.g. one game, one frame/state, or one lesion
# index per task). Mirrors the Paper-1 tools/cluster/bench_gpu.sbatch pattern;
# CPU here (lesion/occlusion enumeration, oracle re-runs, T3 verification) — use
# tools/cluster/xai_gpu.sbatch for batched-gradient / SAE GPU work.
#
# Submit (from the repo root on /cluster/maier — see tools/xai_study/common/cluster.md):
#   # one task per game in a 6-game core set (indices 0..5):
#   sbatch --array=0-5 \
#          --export=ALL,XAI_CMD="tools/xai_study/phaseA_kording/A2_lesions.py",SHARD_KIND=game \
#          tools/cluster/xai_array.sbatch
#   # split a flat work list into 64 shards (indices 0..63), 8 running at once:
#   sbatch --array=0-63%8 \
#          --export=ALL,XAI_CMD="tools/xai_study/ground_truth/oracle_intervene.py",NSHARDS=64 \
#          tools/cluster/xai_array.sbatch
#
# The launched module receives the shard via env (SLURM_ARRAY_TASK_ID /
# SHARD_INDEX, NSHARDS, SHARD_KIND) AND as CLI flags (--shard / --nshards /
# --shard-kind) so runners can read either; extra args pass through XAI_ARGS.
#SBATCH --job-name=xai_array
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm/%x-%A_%a.out
#SBATCH --error=results/slurm/%x-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p results/slurm results/xai            # output-dir trap (Paper-1 guide §4.3)

# --- shard identity (array task -> shard; falls back to 0/1 for a plain run) ---
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS="${NSHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
SHARD_KIND="${SHARD_KIND:-shard}"             # game | state | lesion | shard

# --- what to run ---
XAI_CMD="${XAI_CMD:?FATAL: set XAI_CMD=<path to a tools/xai_study/*.py runner> via --export}"
XAI_ARGS="${XAI_ARGS:-}"                       # extra flags passed verbatim to the runner
VENPY="${VENPY:-.venv/bin/python}"            # jaxtari/jutari CPU venv (jax CPU is enough here)
ROMS_DIR="${ROMS_DIR:-/cluster/maier/UnderstandingVCS/xitari/games/Atari-2600-VCS-ROM-Collection/ROMS}"

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-?} array=${SLURM_ARRAY_JOB_ID:-?}_${SHARD_INDEX} kind=${SHARD_KIND} nshards=${NSHARDS} $(date -u +%FT%TZ) ==="
echo "    cmd=${XAI_CMD} args=${XAI_ARGS}"
[ -f "$XAI_CMD" ] || { echo "FATAL: runner $XAI_CMD not found (paths are repo-relative from the submit dir)"; exit 1; }
[ -d "$ROMS_DIR" ] || echo "WARN: ROMS_DIR $ROMS_DIR missing — runner must locate ROMs itself (gitignored; use in place)"

# Keep CPU math single-threaded per task; Slurm gives us the cores via --array
# parallelism, not BLAS threads (Paper-1 sweep pattern: one core per job).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export XLA_FLAGS="${XLA_FLAGS:---xla_force_host_platform_device_count=1}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export PYTHONPATH="${PYTHONPATH:-$PWD}"        # so tools/xai_study/common imports resolve

[ -x "$VENPY" ] || { echo "FATAL: venv python $VENPY not executable (activate jaxtari/jutari venv on /cluster/maier)"; exit 1; }
"$VENPY" -c "import sys; print('python', sys.version.split()[0])"

# Shard is provided BOTH as env (already exported above) and as CLI flags so the
# runner can consume whichever it prefers; --out lands in results/xai/<schema>.json.
# shellcheck disable=SC2086
"$VENPY" "$XAI_CMD" \
  --shard "$SHARD_INDEX" --nshards "$NSHARDS" --shard-kind "$SHARD_KIND" \
  --roms-dir "$ROMS_DIR" --out-dir "results/xai" --where cluster \
  $XAI_ARGS

echo "=== DONE array task ${SHARD_INDEX}/${NSHARDS} $(date -u +%FT%TZ) ==="
