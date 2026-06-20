#!/bin/bash
# jaxtari GPU batched-throughput benchmark — one Slurm job per GPU type.
# Submit (from the repo root on /cluster/<user>) e.g.:
#   sbatch --gres=gpu:q8000:1 --export=ALL,GPUTYPE=q8000 tools/cluster/bench_gpu.sbatch
#   sbatch --gres=gpu:v100:1  --export=ALL,GPUTYPE=v100  tools/cluster/bench_gpu.sbatch
#   sbatch --gres=gpu:gtx1080ti:1 --export=ALL,GPUTYPE=gtx1080ti,BATCHES=1,16,64,256,1024 tools/cluster/bench_gpu.sbatch
#SBATCH --job-name=jaxtari_gpu_bench
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p results/slurm results/gpu            # output-dir trap (guide §4.3)

GPUTYPE="${GPUTYPE:-unknown}"
STEPS="${STEPS:-3000}"
REPEATS="${REPEATS:-10}"   # timed repeats per config; mean+-std reported (median too)
# Sweep well past where memory matters — soft state is tiny (~25-56 KB/env), so on
# these cards the limit is compute/occupancy, not memory. The bench records OOM and
# continues, so over-shooting the batch list is safe.
BATCHES="${BATCHES:-1,16,64,256,1024,4096,16384,65536}"
ROM="${ROM:-tools/rom_sweep/roms/pong.bin}"

echo "=== node=$(hostname) gpu=${GPUTYPE} $(date -u +%FT%TZ) ==="
# Informational only — do NOT abort on a non-zero exit: nvidia-smi returns non-zero
# for a sibling GPU's "infoROM is corrupted" warning even when the allocated GPU is
# fine. The real GPU check is the jax.devices() assertion below.
nvidia-smi || echo "WARN: nvidia-smi returned non-zero (sibling-GPU warning?) — continuing"
[ -f "$ROM" ] || { echo "FATAL: ROM $ROM missing (copy it — ROMs are gitignored)"; exit 1; }

source .venv/bin/activate
python -c "import jax, sys; ds = jax.devices(); assert any(d.platform == 'gpu' for d in ds), f'FATAL: no GPU device visible to JAX: {ds}'; print('jax', jax.__version__, 'backend', jax.default_backend(), 'devices', ds)"

python tools/bench_jaxtari_gpu.py --rom "$ROM" \
  --steps "$STEPS" --batch-sizes "$BATCHES" --modes fwd,fwdbwd --repeats "$REPEATS" \
  --device gpu --cache-dir /cluster/maier/.jax_cache \
  --out "results/gpu/${GPUTYPE}.json"
echo "=== DONE $(date -u +%FT%TZ) ==="
