#!/usr/bin/env python3
"""Full-system SOFT step throughput: eager loop vs record / replay.

The eager `soft_system_run` steps the hard console (eager Python) on every
instruction, so it cannot be jit-compiled or vmapped and its gradient is taken
through a Python loop. `record_soft_trace` + `soft_system_replay`
(jaxtari.diff.soft_replay) records the hard data once and replays the soft
handlers as one compiled `lax.scan`. This script reports, on whatever JAX
backend is active:

  record          hard console + recording (eager, CPU-bound)        instr/s
  eager fwd/grad  soft_system_run / jax.grad of it (--eager)         instr/s
  replay fwd/grad jit(soft_system_replay) / jit(grad), per window    instr/s
  batched         jit(vmap(...)) over B copies of one trace          env-instr/s

Compile time is reported separately and excluded from throughput. Batched runs
replicate one trace B times; the work per element is identical to B distinct
traces of the same length. Caveat (as in bench_soft_run_batched.py): the soft
opcode dispatch is a data-dependent 256-way `lax.switch`; under `vmap` XLA may
evaluate all branches, so batching need not accelerate on the CPU backend.
These are the measured numbers either way.

Usage:
  jaxtari/.venv/bin/python tools/bench_soft_system_replay.py \
      --rom tools/rom_sweep/roms/pong.bin --windows 100 400 3000 \
      --batches 1,16,64,256 --batch-window 400 --eager 100 --json out.json
  GPU sweep on the cluster: tools/cluster/bench_soft_replay.sbatch
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
import numpy as np

from jaxtari.diff import (
    record_soft_trace,
    soft_system_init,
    soft_system_replay,
    soft_system_run,
)
from jaxtari.env.stella_environment import StellaEnvironment
from jaxtari.games import GenericRomSettings, PongRomSettings


def _timed(fn, repeats):
    t = time.perf_counter()
    jax.block_until_ready(fn())
    compile_s = time.perf_counter() - t
    runs = []
    for _ in range(repeats):
        t = time.perf_counter()
        jax.block_until_ready(fn())
        runs.append(time.perf_counter() - t)
    return compile_s, statistics.median(runs)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rom", default="tools/rom_sweep/roms/pong.bin")
    ap.add_argument("--boot-frames", type=int, default=3)
    ap.add_argument("--windows", type=int, nargs="+", default=[100, 400, 3000])
    ap.add_argument("--checkpoint-every", type=int, default=100,
                    help="segment length for remat on windows longer than it")
    ap.add_argument("--batches", default="64",
                    help="comma-separated batch sizes for the vmap sweep; an OOM "
                         "is recorded and the sweep continues")
    ap.add_argument("--batch-window", type=int, default=400)
    ap.add_argument("--cache-dir", default=None,
                    help="persistent JAX compilation cache directory")
    ap.add_argument("--record-device", default="cpu", choices=["cpu", "default"],
                    help="device for booting + recording the eager hard console "
                         "(cpu: its tiny array ops are not dispatched to a GPU)")
    ap.add_argument("--eager", type=int, default=0,
                    help="also time the eager loop over this many steps (slow)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.cache_dir:
        jax.config.update("jax_compilation_cache_dir", a.cache_dir)

    rom_path = Path(a.rom)
    rom = np.frombuffer(rom_path.read_bytes(), dtype=np.uint8)
    settings = PongRomSettings() if rom_path.stem == "pong" else GenericRomSettings()
    batches = [int(b) for b in a.batches.split(",") if b]
    T = max(a.windows + [a.batch_window])
    rec_dev = jax.devices("cpu")[0] if a.record_device == "cpu" else jax.devices()[0]
    with jax.default_device(rec_dev):        # eager hard console + recording
        env = StellaEnvironment(rom, settings)
        env.reset(boot_noop_steps=a.boot_frames)
        con0 = env.console
        state, bus = soft_system_init(con0)
        t = time.perf_counter()
        trace_all, _ = record_soft_trace(copy.deepcopy(con0), T)
        rec_s = time.perf_counter() - t
    res = dict(rom=rom_path.name, backend=jax.default_backend(),
               devices=[str(d) for d in jax.devices()], record_device=str(rec_dev),
               windows={})
    print(f"backend={res['backend']} devices={res['devices']} rom={rom_path.name} "
          f"record_device={rec_dev}")
    res["record_instr_per_s"] = T / rec_s
    print(f"record   {T:6d} instr in {rec_s:8.2f}s  -> {T / rec_s:12.0f} instr/s")

    cut = lambda n: jax.tree.map(lambda v: v[:n], trace_all)

    if a.eager:
        n = a.eager
        t = time.perf_counter()
        s, b, _ = soft_system_run(state, bus, copy.deepcopy(con0), n)
        jax.block_until_ready(b.ram)
        fwd = time.perf_counter() - t
        con = copy.deepcopy(con0)

        def loss_e(r):
            s, b, _ = soft_system_run(state, bus._replace(rom=r), con, n)
            return jnp.sum(b.ram) + s.A

        t = time.perf_counter()
        jax.block_until_ready(jax.grad(loss_e)(bus.rom))
        grd = time.perf_counter() - t
        res["eager"] = dict(steps=n, fwd_instr_per_s=n / fwd, grad_instr_per_s=n / grd)
        print(f"eager    {n:6d} instr  fwd {n / fwd:10.1f} instr/s   grad {n / grd:10.2f} instr/s")

    for w in a.windows:
        tr = cut(w)
        ck = a.checkpoint_every if w > a.checkpoint_every else None
        fwd = jax.jit(lambda s, b, x: soft_system_replay(s, b, x, checkpoint_every=ck))

        def loss(r, x, ck=ck):
            s, b = soft_system_replay(state, bus._replace(rom=r), x, checkpoint_every=ck)
            return jnp.sum(b.ram) + s.A

        grd = jax.jit(jax.grad(loss))
        cf, tf = _timed(lambda: fwd(state, bus, tr), a.repeats)
        if "placement" not in res:
            out = fwd(state, bus, tr)[1].ram
            res["placement"] = dict(trace=str(list(tr.ram.devices())[0]),
                                    replay_output=str(list(out.devices())[0]))
            print(f"placement trace={res['placement']['trace']} "
                  f"replay_output={res['placement']['replay_output']}")
        cg, tg = _timed(lambda: grd(bus.rom, tr), a.repeats)
        res["windows"][w] = dict(checkpoint_every=ck, fwd_instr_per_s=w / tf,
                                 grad_instr_per_s=w / tg, fwd_compile_s=cf,
                                 grad_compile_s=cg)
        print(f"replay   {w:6d} instr  fwd {w / tf:10.0f} instr/s   grad {w / tg:10.0f} "
              f"instr/s   (compile {cf:.1f}s / {cg:.1f}s, ckpt={ck})")

    w = a.batch_window
    tr = cut(w)
    ck = a.checkpoint_every if w > a.checkpoint_every else None
    res["batched"] = []
    fwdB = jax.jit(jax.vmap(lambda s, b, x: soft_system_replay(s, b, x, checkpoint_every=ck)))
    for B in batches:
        rep = lambda x, B=B: jax.tree.map(
            lambda v: jnp.broadcast_to(v, (B,) + jnp.shape(v)), x)
        sB, bB, trB = rep(state), rep(bus), rep(tr)

        def lossB(roms, sB=sB, bB=bB, trB=trB):
            s, b = jax.vmap(lambda s, b, x: soft_system_replay(
                s, b, x, checkpoint_every=ck))(sB, bB._replace(rom=roms), trB)
            return jnp.sum(b.ram) + jnp.sum(s.A)

        grdB = jax.jit(jax.grad(lossB))
        entry = dict(batch=B, window=w, checkpoint_every=ck)
        line = f"batched  B={B:6d} x {w} instr"
        for mode, fn in (("fwd", lambda: fwdB(sB, bB, trB)), ("grad", lambda: grdB(bB.rom))):
            try:                                     # OOM etc.: record, continue
                c, t_ = _timed(fn, a.repeats)
                entry[f"{mode}_env_instr_per_s"] = B * w / t_
                entry[f"{mode}_compile_s"] = c
                line += f"  {mode} {B * w / t_:12.0f} env-instr/s (compile {c:.1f}s)"
            except Exception as e:
                entry[f"{mode}_error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
                line += f"  {mode} {entry[f'{mode}_error'][:60]}"
        res["batched"].append(entry)
        print(line)

    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2))
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
