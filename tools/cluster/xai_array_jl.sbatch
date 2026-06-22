#!/bin/bash
# Paper-2 (xai_study) JULIA CPU array sweep — the jutari (Julia) twin of
# tools/cluster/xai_array.sbatch. One Slurm array task per shard of an
# embarrassingly-parallel jutari job (e.g. one game, one frame/state, or one
# lesion index per task): oracle re-runs (intervention/gradient), the
# F∧S∧M T3 verifier, Phase-A lesion/occlusion enumeration, etc. CPU here —
# jutari is a pure-Julia CPU emulator and the oracle/T3 work is bit-exact
# replay, not GPU math; use tools/cluster/xai_gpu.sbatch for the batched
# jaxtari gradient / SAE GPU work.
#
# Submit (from the repo root on /cluster/maier — see tools/xai_study/common/cluster.md):
#   # one task per game in a 6-game core set (indices 0..5):
#   sbatch --array=0-5 \
#          --export=ALL,XAI_CMD="tools/xai_study/ground_truth/oracle_intervene.jl",SHARD_KIND=game \
#          tools/cluster/xai_array_jl.sbatch
#   # split a flat work list into 64 shards (indices 0..63); the user QOS caps
#   # concurrent jobs at 10, so throttle the array with %10 (SCRUM §7):
#   sbatch --array=0-63%10 \
#          --export=ALL,XAI_CMD="tools/xai_study/ground_truth/oracle_intervene.jl",NSHARDS=64 \
#          tools/cluster/xai_array_jl.sbatch
#   # pass runner-specific flags through XAI_ARGS (verbatim):
#   sbatch --array=0-5%10 \
#          --export=ALL,XAI_CMD="tools/xai_study/phaseA_kording/pilot_si.jl",XAI_ARGS="--game space_invaders --target-frame 120" \
#          tools/cluster/xai_array_jl.sbatch
#
# The launched runner receives the shard via env (SLURM_ARRAY_TASK_ID /
# SHARD_INDEX, NSHARDS, SHARD_KIND) AND as CLI flags (--shard / --nshards /
# --shard-kind) so it can read either; the jutari runners parse a permissive
# arg loop and ignore flags they don't use. Extra args pass through XAI_ARGS.
#SBATCH --job-name=xai_array_jl
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
XAI_CMD="${XAI_CMD:?FATAL: set XAI_CMD=<path to a tools/xai_study/*.jl runner> via --export}"
XAI_ARGS="${XAI_ARGS:-}"                       # extra flags passed verbatim to the runner
# Julia toolchain: juliaup-installed binary + DEPOT both live on /cluster/maier
# (NOT /home — home is congested). Override JULIA_BIN / JULIA_DEPOT_PATH via
# --export if the depot ever moves.
JULIA_BIN="${JULIA_BIN:-/cluster/maier/.juliaup/bin/julia}"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/cluster/maier/.julia}"
JULIA_PROJECT_DIR="${JULIA_PROJECT_DIR:-jutari}"   # --project points at the jutari env
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
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-1}"

[ -x "$JULIA_BIN" ] || { echo "FATAL: julia binary $JULIA_BIN not executable (install juliaup with depot on /cluster/maier)"; exit 1; }
"$JULIA_BIN" --project="$JULIA_PROJECT_DIR" -e 'println("julia ", VERSION, " depot=", DEPOT_PATH[1])'

# Shard is provided BOTH as env (already exported above) and as CLI flags so the
# runner can consume whichever it prefers; --out lands in results/xai/<schema>.json.
# shellcheck disable=SC2086
"$JULIA_BIN" --project="$JULIA_PROJECT_DIR" "$XAI_CMD" \
  --shard "$SHARD_INDEX" --nshards "$NSHARDS" --shard-kind "$SHARD_KIND" \
  --roms-dir "$ROMS_DIR" --out-dir "results/xai" --where cluster \
  $XAI_ARGS

echo "=== DONE array task ${SHARD_INDEX}/${NSHARDS} $(date -u +%FT%TZ) ==="
