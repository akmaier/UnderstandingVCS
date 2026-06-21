#!/bin/bash
# Paper-2 (xai_study) GPU batch — one Slurm job for a batched-gradient / SAE /
# Integrated-Gradients-over-ROM run on a single GPU. Mirrors the Paper-1
# tools/cluster/bench_gpu.sbatch pattern (GPU venv, jax.devices() assertion,
# shared JAX cache). Use tools/cluster/xai_array.sbatch for CPU array sweeps
# (lesion enumeration, oracle re-runs). The enabler is SOFT-STE: forward is
# bit-exact to HARD and GPU-batchable, so exact interventions + gradients run
# batched here (xai_study/README.md "Compute").
#
# Submit (from the repo root on /cluster/maier — see tools/xai_study/common/cluster.md):
#   sbatch --gres=gpu:q8000:1 --export=ALL,GPUTYPE=q8000,\
# XAI_CMD=tools/xai_study/phaseB_attribution/integrated_gradients.py \
#          tools/cluster/xai_gpu.sbatch
#   sbatch --gres=gpu:v100:1 --export=ALL,GPUTYPE=v100,\
# XAI_CMD=tools/xai_study/phaseC_mechanistic/sae_train.py,XAI_ARGS="--game pong --batch 4096" \
#          tools/cluster/xai_gpu.sbatch
#
# Serialize GPU jobs that contend for the same node/GPU (SCRUM §7: heavy gates
# run alone) — submit them as a dependency chain or distinct --gres types.
#SBATCH --job-name=xai_gpu
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p results/slurm results/xai            # output-dir trap (Paper-1 guide §4.3)

GPUTYPE="${GPUTYPE:-unknown}"
XAI_CMD="${XAI_CMD:?FATAL: set XAI_CMD=<path to a tools/xai_study/*.py runner> via --export}"
XAI_ARGS="${XAI_ARGS:-}"                       # extra flags passed verbatim to the runner
# jaxtari GPU venv (jax[cuda12]) — the same one Paper-1's bench_gpu.sbatch uses.
VENPY="${VENPY:-jaxtari/.venv/bin/python}"
ROMS_DIR="${ROMS_DIR:-/cluster/maier/UnderstandingVCS/xitari/games/Atari-2600-VCS-ROM-Collection/ROMS}"
CACHE_DIR="${CACHE_DIR:-/cluster/maier/.jax_cache}"   # shared XLA compile cache

echo "=== node=$(hostname) gpu=${GPUTYPE} job=${SLURM_JOB_ID:-?} $(date -u +%FT%TZ) ==="
echo "    cmd=${XAI_CMD} args=${XAI_ARGS}"
# Informational only — do NOT abort on a non-zero exit: nvidia-smi returns
# non-zero for a sibling GPU's "infoROM is corrupted" warning even when the
# allocated GPU is fine. The real check is the jax.devices() assertion below.
nvidia-smi || echo "WARN: nvidia-smi returned non-zero (sibling-GPU warning?) — continuing"
[ -f "$XAI_CMD" ] || { echo "FATAL: runner $XAI_CMD not found (paths are repo-relative from the submit dir)"; exit 1; }
[ -d "$ROMS_DIR" ] || echo "WARN: ROMS_DIR $ROMS_DIR missing — runner must locate ROMs itself (gitignored; use in place)"
[ -x "$VENPY" ] || { echo "FATAL: GPU venv python $VENPY not executable (jaxtari jax[cuda12] venv on /cluster/maier)"; exit 1; }

export PYTHONPATH="${PYTHONPATH:-$PWD}"        # so tools/xai_study/common imports resolve
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"  # require the CUDA backend

# Assert a GPU is actually visible to JAX before spending walltime (Paper-1 pattern).
"$VENPY" -c "import jax; ds = jax.devices(); assert any(d.platform == 'gpu' for d in ds), f'FATAL: no GPU device visible to JAX: {ds}'; print('jax', jax.__version__, 'backend', jax.default_backend(), 'devices', ds)"

# shellcheck disable=SC2086
"$VENPY" "$XAI_CMD" \
  --roms-dir "$ROMS_DIR" --out-dir "results/xai" --where cluster \
  --device gpu --cache-dir "$CACHE_DIR" \
  $XAI_ARGS

echo "=== DONE $(date -u +%FT%TZ) ==="
