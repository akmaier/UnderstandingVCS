#!/usr/bin/env python3
"""GPU batched-throughput benchmark for jaxtari's SOFT (differentiable) path.

Measures how JAX exploits a GPU for jaxtari by `vmap`-ing the compiled
`soft_run_scan` rollout over a batch of N environments, in two modes:

  * fwd     — forward rollout only (batched env-steps/s)
  * fwdbwd  — forward + reverse-mode gradient (the differentiable-RL / XAI case)

WHY this is the GPU story (and the HARD path is not): the HARD emulator is an
eager concrete-value Python interpreter (see tools/bench_jaxtari_rollout.py) —
not jittable, CPU-only. The SOFT path IS `jax.jit` + `lax.scan`, and its 256-way
`lax.switch` opcode dispatch evaluates ALL branches per lane under `vmap`. On a
CPU that serialises (aggregate asymptotes ~60k env-steps/s); on a GPU the
all-branch work maps onto the lanes, so large batches are the intended regime.
This script quantifies that: the scaling curve vs N, the CPU->GPU crossover, the
fwd/bwd ratio, and the per-GPU memory ceiling.

Usage (one GPU, sweep batch sizes, write JSON):
    python tools/bench_jaxtari_gpu.py --rom xitari/roms/pong.bin \\
        --steps 3000 --batch-sizes 1,16,64,256,1024,4096 \\
        --modes fwd,fwdbwd --repeats 3 --out results/gpu/q8000.json

Local CPU smoke test (small):
    python tools/bench_jaxtari_gpu.py --rom xitari/roms/pong.bin \\
        --steps 50 --batch-sizes 1,4 --repeats 1 --device cpu
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "jaxtari"))


def _setup_env(device: str, cache_dir: str | None) -> None:
    """Set JAX env BEFORE importing jax. `device` in {auto,gpu,cpu}."""
    if device == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    elif device == "gpu":
        os.environ.setdefault("JAX_PLATFORMS", "cuda")
    # else auto: let JAX pick (gpu if jaxlib-cuda present, else cpu).
    # Persistent compilation cache: unlike the eager HARD dump, soft_run_scan
    # DOES jit-compile, so the cache genuinely amortises compile across batch
    # sizes / repeated jobs.
    if cache_dir:
        os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.expanduser(cache_dir))
        os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0.5")


def _reset_pc(rom: np.ndarray) -> float:
    n = len(rom)
    return float((int(rom[n - 3]) << 8) | int(rom[n - 4]))   # 6502 reset vector @ $FFFC/$FFFD


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rom", type=Path, required=True)
    p.add_argument("--steps", type=int, default=3000, help="CPU instructions per env rollout")
    p.add_argument("--batch-sizes", type=str, default="1,16,64,256,1024",
                   help="comma-separated N values to sweep")
    p.add_argument("--modes", type=str, default="fwd,fwdbwd")
    p.add_argument("--repeats", type=int, default=3, help="timed repeats (median reported)")
    p.add_argument("--device", choices=("auto", "gpu", "cpu"), default="auto")
    p.add_argument("--cache-dir", type=str, default="~/.cache/jaxtari/jax_compilation_cache")
    p.add_argument("--out", type=Path, default=None, help="write results JSON here")
    args = p.parse_args(argv)

    _setup_env(args.device, args.cache_dir)

    import jax              # noqa: E402
    import jax.numpy as jnp # noqa: E402
    from jaxtari.diff.soft_state import initial_soft_cpu_state, initial_soft_bus  # noqa: E402
    from jaxtari.diff.soft_step import soft_run_scan                              # noqa: E402

    rom = np.frombuffer(args.rom.read_bytes(), dtype=np.uint8)
    pc0 = _reset_pc(rom)
    n_steps = args.steps
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    dev = jax.devices()[0]
    backend = jax.default_backend()
    env = {
        "rom": args.rom.name, "jax": jax.__version__, "backend": backend,
        "device": str(dev), "device_kind": getattr(dev, "device_kind", "?"),
        "steps": n_steps, "repeats": args.repeats,
    }
    print(f"# {json.dumps(env)}", flush=True)

    def _broadcast(tree, N):
        return jax.tree_util.tree_map(lambda x: jnp.broadcast_to(jnp.asarray(x)[None],
                                                                 (N,) + jnp.asarray(x).shape), tree)

    def _mem_bytes():
        try:
            st = dev.memory_stats() or {}
            return int(st.get("peak_bytes_in_use", st.get("bytes_in_use", -1)))
        except Exception:
            return -1

    # --- forward: vmap the compiled scan over N identical-init environments ---
    def build_fwd():
        run = lambda s, b: soft_run_scan(s, b, n_steps)
        return jax.jit(jax.vmap(run))

    # --- fwd+bwd: grad of a scalar loss wrt a per-env scalar perturbation eps
    # of the initial PC. eps flows through the whole scan, so the backward pass
    # exercises the full rollout. Robust to int leaves (grad is wrt eps only). ---
    def build_fwdbwd(b_single):
        def loss(eps):
            s = initial_soft_cpu_state(pc=pc0 + eps)
            out = soft_run_scan(s, b_single, n_steps)
            leaves = jax.tree_util.tree_leaves(out)
            return sum(jnp.sum(jnp.asarray(x, jnp.float32)) for x in leaves)
        return jax.jit(jax.vmap(jax.grad(loss)))

    def time_call(fn, *call_args):
        # warm-up (compile) excluded
        jax.block_until_ready(fn(*call_args))
        ts = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(*call_args))
            ts.append(time.perf_counter() - t0)
        return np.asarray(ts)

    results = []
    s0 = initial_soft_cpu_state(pc=pc0)
    b0 = initial_soft_bus(jnp.asarray(rom))

    for mode in modes:
        for N in batch_sizes:
            rec = {"mode": mode, "N": N}
            try:
                if mode == "fwd":
                    fn = build_fwd()
                    sB, bB = _broadcast(s0, N), _broadcast(b0, N)
                    ts = time_call(fn, sB, bB)
                elif mode == "fwdbwd":
                    fn = build_fwdbwd(b0)
                    epsB = jnp.zeros((N,), jnp.float32)
                    ts = time_call(fn, epsB)
                else:
                    raise ValueError(f"unknown mode {mode}")
                env_steps = N * n_steps
                med = float(np.median(ts))
                thr = env_steps / ts                       # per-repeat throughput
                rec.update({
                    "wall_s": round(med, 6),
                    "env_steps_per_s": round(env_steps / med, 1),         # median-based
                    "env_steps_per_s_mean": round(float(thr.mean()), 1),
                    "env_steps_per_s_std": round(float(thr.std()), 1),
                    "per_env_steps_per_s": round(n_steps / med, 1),
                    "mem_bytes": _mem_bytes(),
                    "ok": True,
                })
                print(f"  {mode:7s} N={N:<6d} {rec['env_steps_per_s']:>14,.0f} env-steps/s"
                      f"  (mean {rec['env_steps_per_s_mean']:,.0f} +/- {rec['env_steps_per_s_std']:,.0f})  mem={rec['mem_bytes']}", flush=True)
            except Exception as e:                       # OOM or trace error -> record + continue
                rec.update({"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"})
                print(f"  {mode:7s} N={N:<6d} FAILED: {rec['error']}", flush=True)
            results.append(rec)

    out_obj = {"env": env, "results": results}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out_obj, indent=2) + "\n")
        print(f"# wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
