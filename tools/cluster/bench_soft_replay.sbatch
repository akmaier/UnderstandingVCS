#!/bin/bash
# Full-system SOFT step on GPU: record / replay throughput (tools/bench_soft_system_replay.py).
# One Slurm job per GPU type. Submit from the repo root (a checkout or worktree on
# /cluster/<user>) e.g.:
#   mkdir -p results/slurm          # BEFORE sbatch: Slurm opens --output before this script
#                                   # runs, so a missing dir kills the job in ~1 s, no log
#   sbatch --qos=turbo --gres=gpu:v100:1     --export=ALL,GPUTYPE=v100     tools/cluster/bench_soft_replay.sbatch
#   sbatch --qos=turbo --gres=gpu:rtxa6000:1 --export=ALL,GPUTYPE=rtxa6000 tools/cluster/bench_soft_replay.sbatch
# QOS: the default `miti` caps a user at 4 running jobs (priority 0); `turbo` is the
# priority lane (priority 10000, 10 running jobs/user, 1-day wall). Both: partition main.
# The job imports jaxtari from THIS checkout (PYTHONPATH), not from wherever the shared
# venv's editable install points, and aborts if that is not the case.
#SBATCH --job-name=jaxtari_soft_replay
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p results/slurm results/gpu

GPUTYPE="${GPUTYPE:-unknown}"
VENV="${VENV:-/cluster/maier/UnderstandingVCS/.venv}"
ROM="${ROM:-tools/rom_sweep/roms/pong.bin}"
WINDOWS="${WINDOWS:-100 400 3000}"
BATCH_WINDOW="${BATCH_WINDOW:-400}"
BATCHES="${BATCHES:-1,16,64,256,1024,4096}"
CKPT="${CKPT:-100}"
REPEATS="${REPEATS:-5}"

echo "=== node=$(hostname) gpu=${GPUTYPE} commit=$(git rev-parse --short HEAD) $(date -u +%FT%TZ) ==="
nvidia-smi || echo "WARN: nvidia-smi returned non-zero (sibling-GPU warning?) — continuing"
[ -f "$ROM" ] || { echo "FATAL: ROM $ROM missing (ROMs are gitignored — copy or symlink them)"; exit 1; }

source "$VENV/bin/activate"
export PYTHONPATH="$PWD/jaxtari${PYTHONPATH:+:$PYTHONPATH}"
python - <<'PY'
import os, jax, jaxtari
here = os.path.realpath(os.path.join(os.getcwd(), "jaxtari"))
src = os.path.realpath(jaxtari.__file__)
assert src.startswith(here + os.sep), f"FATAL: imported jaxtari from {src}, expected under {here}"
ds = jax.devices()
assert any(d.platform == "gpu" for d in ds), f"FATAL: no GPU device visible to JAX: {ds}"
print("jaxtari", src, "| jax", jax.__version__, jax.default_backend(), ds)
PY

python tools/bench_soft_system_replay.py --rom "$ROM" \
  --windows $WINDOWS --checkpoint-every "$CKPT" \
  --batches "$BATCHES" --batch-window "$BATCH_WINDOW" --repeats "$REPEATS" \
  --cache-dir /cluster/maier/.jax_cache \
  --json "results/gpu/soft_replay_${GPUTYPE}.json"
echo "=== DONE $(date -u +%FT%TZ) ==="
