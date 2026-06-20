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
BATCHES="${BATCHES:-1,16,64,256,1024,4096}"
ROM="${ROM:-tools/rom_sweep/roms/pong.bin}"

echo "=== node=$(hostname) gpu=${GPUTYPE} $(date -u +%FT%TZ) ==="
nvidia-smi || { echo "FATAL: nvidia-smi failed — no GPU in this allocation"; exit 1; }
[ -f "$ROM" ] || { echo "FATAL: ROM $ROM missing (copy it — ROMs are gitignored)"; exit 1; }

source .venv/bin/activate
python -c "import jax; print('jax', jax.__version__, 'backend', jax.default_backend(), 'devices', jax.devices())"

python tools/bench_jaxtari_gpu.py --rom "$ROM" \
  --steps "$STEPS" --batch-sizes "$BATCHES" --modes fwd,fwdbwd --repeats 3 \
  --device gpu --cache-dir /cluster/maier/.jax_cache \
  --out "results/gpu/${GPUTYPE}.json"
echo "=== DONE $(date -u +%FT%TZ) ==="
