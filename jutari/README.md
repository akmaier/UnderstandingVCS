# JuTari

Differentiable Julia port of [xitari](https://github.com/google-deepmind/xitari) — the DeepMind fork of the Arcade Learning Environment built on Stella.

This package is one half of the **UnderstandingVCS** project (the other is the JAX port [jaxtari](../jaxtari/)). The bit-exact emulator runs in HARD mode today; the SOFT-mode differentiability primitives (`RomTensor`, `soft_select`, `soft_memory_read`, `soft_branch`, straight-through round/clamp) are in `JuTari.Diff` with forward-behaviour tests (gradient-stack rrules via Zygote / ChainRulesCore are deferred to P7b). See [`../STATUS.md`](../STATUS.md) for the per-phase ledger and [`../PORTING_PLAN.md`](../PORTING_PLAN.md) for the design.

## Quickstart

```julia
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'   # 776 tests
```

Build a console and run a frame:

```julia
using JuTari
using JuTari.Env: env_reset!, env_step!, get_screen
using JuTari.IO:  NOOP

rom = zeros(UInt8, 4096)
# ... load your ROM bytes into `rom` ...

env = StellaEnvironment(rom)
env_reset!(env)
while !game_over(env)
    reward = env_step!(env, Int(NOOP))
    frame  = get_screen(env)         # 192 × 160 Matrix{UInt8} indexed colour
end
```

## Layout

```
jutari/
├── Project.toml
├── README.md
├── src/
│   ├── JuTari.jl           # module root — `include`s all the sub-modules
│   ├── Types.jl            # CPUState mutable struct
│   ├── cpu/                # 6502 / 6507 — all 151 documented NMOS opcodes (P1)
│   │   ├── Tables.jl
│   │   ├── Addressing.jl
│   │   ├── ALU.jl
│   │   └── M6502.jl
│   ├── bus/Bus.jl          # 6507 13-bit address decode + region routing (P2)
│   ├── tia/TIA.jl          # TIA — register file, timing, sprites, collisions,
│   │                       # VSYNC/VBLANK, INPT* (P3a–f + P6)
│   ├── riot/RIOT.jl        # M6532 timer + I/O ports + DDRs (P4)
│   ├── cart/Cart.jl        # Bank-switched cartridges 2K/4K/F8/F6/F4 (P5)
│   ├── Console.jl          # Console (CPU + Bus) + reset! + step! + run_until_frame! (P6)
│   ├── io/IO.jl            # ALE Action enum + apply_action! + console_switches! (P6)
│   ├── games/RomSettings.jl  # abstract RomSettings + GenericRomSettings stub (P6)
│   ├── env/StellaEnvironment.jl  # ALE-style env_reset! / env_step! / etc. (P6)
│   └── diff/               # Differentiability primitives (P7)
│       ├── Modes.jl
│       ├── RomAsWeights.jl
│       ├── SoftSelect.jl
│       ├── SoftMem.jl
│       ├── SoftBranch.jl
│       └── StraightThrough.jl
└── test/runtests.jl        # 776 tests across 19 @testsets
```

## Function-name collisions worth knowing about

Julia's `Base` exports `peek` (iterator peek) and `step` (range stepper).
JuTari intentionally does NOT re-export `step`, `peek`, or `poke!` from
the top-level `using JuTari` — those collide with `Base` and would leave
the names undefined. Use the qualified imports instead:

```julia
using JuTari
using JuTari.CPU: step          # CPU instruction step
using JuTari.Bus: peek, poke!   # bus-level memory access
using JuTari.Diff: peek          # RomTensor differentiable peek
```

This decision is documented in `src/JuTari.jl` at the export block.

## What this port can do today

Identical to jaxtari's HARD-mode capabilities:

- Run any documented NMOS 6502 instruction sequence.
- Run a complete VCS through `StellaEnvironment` — frame, RAM, lives.
- Auto-detect cart format from ROM size; bank-switch on hotspot read/write.
- Translate ALE-style actions into RIOT bits + TIA trigger; drive console switches.
- Forward-behaviour-correct diff primitives (no gradient stack hooked up yet).

## What this port does NOT yet do

See [`../STATUS.md`](../STATUS.md) for the complete deferral list. Julia-specific notes:

- **No Zygote / ChainRulesCore wiring** — the diff primitives' forward behaviour matches jaxtari, but `Zygote.gradient` calls are not yet tested (and would require adding Zygote as a test dep). jaxtari is currently where the gradient verification lives.
- Everything else in jaxtari's "does NOT yet do" list applies equally here.
