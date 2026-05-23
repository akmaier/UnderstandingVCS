# UnderstandingVCS — results & findings (P9 writeup)

> A paper-shaped summary of what this project built and what it found.
> Snapshot at the end of the P9 milestone — see [`PORTING_PLAN.md`](PORTING_PLAN.md)
> and [`STATUS.md`](STATUS.md) for the per-phase commit ledger.

## Motivation

The XAI literature has rich attribution methods — gradients,
Integrated Gradients, attention rollout, mechanistic interpretability —
but its targets are mostly neural networks whose "ground truth"
mechanism is unknown. We can ask whether IG localises to a region
that *looks* right, but rarely whether it localises to the bytes that
*actually* produced the output.

**UnderstandingVCS** tests XAI methods on a system whose mechanism is
fully known: the Atari 2600 VCS, a 6507 CPU + TIA video chip + RIOT
I/O / timer running a copyrighted-but-disassemblable 2–4 KB cartridge
ROM. The cart code is the entire mechanism. If an attribution method
points at the *wrong* byte, we can prove it; if it points at the
*right* byte, we have a verifiable success.

To run XAI on the VCS we need the VCS to be **differentiable**. So
this project's first three quarters were a from-scratch port of the
xitari emulator to JAX (Python) and Julia, in *two* execution modes:

- **HARD mode** — bit-exact integer emulation. All conformance tests
  run here. The reference for what real Atari hardware does.
- **SOFT mode** — float-valued parallel emulator built on the same
  opcode dispatch and TIA rendering, but with one-hot ROM/RAM reads,
  soft branches, and a JIT-compiled `lax.scan` loop. The whole
  pipeline is differentiable from ROM bytes to framebuffer pixels.

The final quarter (P8, P8-c, P8-cx) puts XAI methods on top.

## What was built

### Both ports

- **CPU** — all 151 documented NMOS 6502 opcodes + the undocumented
  USBC alias (`$EB`), full N/Z/C/V flag semantics, BCD ADC/SBC, all
  addressing modes, JSR/RTS, BRK/RTI, branches.
- **Bus** — 6507 13-bit address mirror, RAM, cart, TIA / RIOT
  dispatch.
- **TIA** — playfield, two player sprites, two missiles, ball, all 8
  collision latches, VSYNC frame-edge, VBLANK output-blanking, end-of-
  scanline rendering.
- **RIOT** — 8-bit interval timer with 1/8/64/1024 prescaler, INTIM /
  INSTAT, SWCHA / SWCHB with DDRs.
- **Cart** — 2K / 4K plain plus F8 / F6 / F4 bank-switching via the
  hotspot read+write convention.
- **Env** — ALE-shape `reset` / `step(action) → reward` / `get_screen`
  / `get_ram` / `lives` / `game_over` with the camelCase aliases for
  drop-in compatibility, plus ALE-equivalent boot-burn (60 NOOP +
  4 RESET frames) on opt-in.

### SOFT mode (in addition to the above)

- **`SoftCPUState` / `SoftBus`** — float-valued parallel state types.
  All registers, all RAM cells, all ROM bytes are `Float32`.
- **`soft_step`** — differentiable parallel of `cpu.m6502.step`. Full
  151-opcode coverage (P7c-a … P7c-f), packed N/Z/C/V flag updates
  via straight-through-cast, BCD ADC/SBC, sentinel-less BRK (proper
  IRQ jump).
- **`soft_run_scan`** — `jax.lax.scan`-backed runner so a 50,000-
  instruction trace compiles once and runs in tens of seconds, not
  the tens of minutes the unrolled `soft_run` takes.
- **`soft_render_scanline` / `soft_render_frame`** — differentiable
  TIA render covering the full visible-object set (P7f-a/b/c) plus
  the 8 collision latches (P7f-d). A pixel is
  `pf_mask*COLUPF + (1−pf_mask)*COLUBK` composited with sprite layers;
  the colour gradients flow back to ROM, the structural masks are
  integer-extracted.
- **RIOT timer state in the SoftBus** (P8-cx) — INTIM ticks per
  instruction at the prescaler rate, INSTAT latch, TIM*T writes load
  the timer. This was the prereq for real-ROM execution (Pong busy-
  waits on INTIM in its startup).
- **`integrated_gradients`** (P8-a) — IG with midpoint Riemann rule,
  PyTree inputs, completeness axiom enforced.
- **`occlusion_attribution`** (P8-b) — per-element ablation, the
  right primitive for the discrete-opcode VCS where naive zero-
  baseline IG misleads on opcode bytes.

### Infrastructure

- **`tools/trace_dump.cpp`** — linked against xitari's `libxitari.a`,
  produces JSONL frame traces of RAM (+ optionally screen, reward,
  lives) for any ALE ROM + action sequence.
- **`tools/check_trace.{py,jl}`** — replays a JSONL trace against
  jaxtari / jutari and diffs per-frame RAM. Pytest test + Julia
  `@test_broken` testset record the current bit-exact gap as
  expected-failed.
- **`tools/bench_soft_run.{py,jl}`** — throughput benchmarks for
  `soft_run` on both ports (P9).

### Test coverage

- **jaxtari: 532 tests + 1 xfail.**
- **jutari: 1,122 tests + 1 broken.**
- **1,655 effective across both ports.**

Both xfail / broken markers track the PXC1-x conformance gap (10 of
128 RAM bytes diverge from xitari at frame 1 of `pong_noop_10` —
identically in both ports, so PXC2 is implicitly satisfied for this
fixture).

## Key findings

### 1. `∂pixel / ∂ROM` works end-to-end, on a real cart

The headline P8-c demo runs `pong.bin` for 50,000 SOFT-mode
instructions, which is enough for Pong's kernel to paint colour
registers, position sprites, and enter its game loop. `jax.grad` of a
rendered pixel flows back through the full chain — SOFT TIA →
SoftBus → soft_step → operand reads — to the specific ROM byte that
supplied the pixel's colour. The Pong-like synthetic-state demos
verify this for paddle, wall, and background pixels: each attributes
one-hot to its colour register.

### 2. Naive zero-baseline IG is misleading on discrete-opcode emulators

A documented finding (`test_naive_zero_baseline_ig_collapses` in
test_p8_b): IG with the standard zero baseline on a SOFT-mode ROM
returns ≈ 0 for clearly critical bytes. The path
`baseline=0 → target=rom` interpolates the opcode bytes through
arbitrary intermediate values that aren't actual opcodes — the
simulator hits the non-raising `_branch_default` for most of the
path and the gradient is zero almost everywhere.

The fix is either **a smart baseline** (the same program with the
attributed operand zeroed — keeps opcodes valid along the path) or
**occlusion attribution** (ablate one byte at a time; no
interpolation). The repository ships both. This is the kind of
methodological finding the project was set up to surface.

### 3. The conformance harness immediately found two real emulator bugs

The xitari-trace harness landed pre-fix at a 25-byte divergence on
`pong_noop_10`. PXC1-x round 1 closed two real bugs:

- **Frame counter double-count** — `tia_advance` was bumping `frame`
  on scanline-wrap AND the VSYNC 1→0 handler also did. Every real
  frame got counted twice. `run_until_frame` was completing every
  other "frame" in ~80 CPU cycles (one scanline) instead of the
  natural ~19,900. Fixed: frame is now VSYNC-only.
- **Missing ALE boot-burn** — xitari's `resetGame()` burns 60 NOOP +
  4 RESET-switch frames before the user's first `act()` so the
  cart's startup routine has time to settle. jaxtari / jutari ran
  zero. Fixed: opt-in via `env.reset(boot_noop_steps=60,
  boot_reset_steps=4)`.

After round 1, the divergence dropped to 10 RAM bytes (with the same
10 bytes diverging identically in both ports, which implicitly
satisfies PXC2 — JAX↔Julia bit-for-bit cross-check — for this
fixture). The remaining 10 are deep TIA scanline-cycle / RIOT timing
detail; closing them needs a per-cycle xitari debug-trace patch
(PORTING_PLAN.md §4.1) — that patch is the prereq for further PXC1-x
rounds.

### 4. SOFT mode requires deliberate model choices that diverge from HARD

Several SOFT-mode simplifications turned out to be load-bearing
design decisions, not bugs to fix:

- **Bus collapse** — TIA / RIOT register reads share the same
  128-byte `bus.ram` cells as zero-page writes via `addr & 0x7F`.
  This is wrong forward (a `STA $02` to VSYNC writes ram[2], where a
  read of SWCHB would also land) but it's also why the renderer can
  read its TIA register file straight out of `bus.ram[0:64]` — no
  separate TIA state needed. P8-cx adds defaults (0xFF for SWCHA/
  SWCHB reads, 0x80 for INPT* reads) to break the read-side of the
  conflict; the write-side stays.
- **BRK as halt-sentinel → proper interrupt** — P7b kept BRK halting
  in place for fixed-length XAI traces; P8-cx flipped this to the
  proper IRQ-vector jump because Pong uses BRK *intentionally* as a
  vertical-blank trigger.
- **Branch predicates are HARD** — the SOFT-mode branch handlers use
  `jnp.where` on the (integer-cast) flag bit. The forward PC is
  exact but gradient through the branch predicate is broken at the
  cast. Restoring it needs a float-valued flag representation in
  `SoftCPUState` (P7c-dx) — deferred as architecture work.

## Throughput

`tools/bench_soft_run.{py,jl}` runs N SOFT instructions of `pong.bin`
and reports steps/second. On the laptop used for this writeup
(Apple Silicon, JAX CPU backend, Julia 1.10):

| Port    | Steps     | Compile + first run | Cached median | Throughput        |
| ------- | --------- | -------------------- | -------------- | ----------------- |
| jaxtari | 50,000    | 30.9 s              | 30.0 s         | ~1,666 steps/s    |
| jutari  | 50,000    | 0.62 s              | 0.14 s         | ~363,800 steps/s  |

**jutari is ~218× faster than jaxtari** on per-instruction SOFT
throughput for this workload. The gap is dominated by per-op kernel-
launch overhead in JAX: each `soft_step` reads one opcode + dispatches
through a 256-way `lax.switch`, and at this granularity the launch
overhead dwarfs the work. Julia's mutable structs + JIT inlining of
small handlers through the `_HANDLERS` array close the gap entirely.

The benchmark is honest about what it measures — *cached-run* speed
for the inner loop after JIT. For an XAI workflow that takes a single
`jax.grad` forward+backward pass per attribution, the jaxtari ~30 s
per 50K-step trace is comfortably interactive (and JIT compilation
amortises over many attributions). For an RL-style training loop that
runs millions of frames, jutari's throughput would matter much more —
the JAX path would want `vmap`-batched parallel-environment rollouts
to amortise the per-launch overhead.

## Open work

In priority order:

1. **xitari per-cycle debug-trace patch** (prereq for PXC1-x round 2+):
   close the remaining 10-byte conformance gap with cycle-level
   diagnostics rather than frame-level RAM-only.
2. **P7c-dx**: float-valued flag representation in `SoftCPUState` so
   `soft_branch` carries gradient through the predicate. Unlocks
   attribution through control flow.
3. **P7e-x**: functional `soft_step` for jutari (or wire Enzyme.jl)
   so Zygote can take real gradients through full traces in Julia.
   jaxtari already has this via JAX.
4. **Per-subsystem deferrals** (each listed in STATUS.md with its
   phase ID):
   - TIA: NUSIZ multi-copy, VDELP/VDELBL, beam-accurate render, audio.
   - Cart: SC variants of F8/F6/F4, E0 / FE / 3F / 3E, MB / MC / AR / DPC.
   - RIOT: PA7 edge interrupt, paddle dump-pot timing.
   - Console: phosphor blending, per-game `RomSettings`.
5. **Real-game XAI study** (a paper, not just a writeup): pick a
   target ROM (probably Pong or Breakout), pose a specific
   attribution question — "which 50 ROM bytes most determine ball
   trajectory?" — and run IG / occlusion / smart-baseline IG on it.
   With the differentiable VCS in place this is now a study, not an
   infrastructure problem.

## Reproducibility

Everything in this repo is reproducible from a fresh clone:

```sh
# Build xitari (one-time, used by the conformance harness)
cd xitari && cmake . && make

# Build the trace-dump tool
cd ../tools && make

# jaxtari — Python tests
cd ../jaxtari && python -m venv .venv && source .venv/bin/activate
pip install -e . && pytest

# jutari — Julia tests
cd ../jutari && julia --project=. -e 'using Pkg; Pkg.test()'

# Run the conformance harness on the bundled fixture
python tools/check_trace.py --rom xitari/roms/pong.bin \\
    --trace tools/fixtures/traces/pong_noop_10.jsonl
julia --project=jutari tools/check_trace.jl --rom xitari/roms/pong.bin \\
    --trace tools/fixtures/traces/pong_noop_10.jsonl

# Run the throughput benchmark
python tools/bench_soft_run.py --rom xitari/roms/pong.bin \\
    --steps 50000 --repeats 5
julia --project=jutari tools/bench_soft_run.jl --rom xitari/roms/pong.bin \\
    --steps 50000 --repeats 5
```

xitari ROMs are gitignored (external dependency); the conformance and
benchmark tools skip cleanly when ROMs are absent.

## Provenance

This document was produced by Claude Opus 4.7 (1M context) over a
multi-session implementation arc that ported xitari to JAX and Julia
phase by phase (P0 → P9). The commit messages on `main` contain the
full prompt-to-commit chain.
