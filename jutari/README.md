# JuTari

Differentiable Julia port of [xitari](https://github.com/google-deepmind/xitari) — the DeepMind fork of the Arcade Learning Environment built on Stella.

This package is one half of the **UnderstandingVCS** project (the other is the JAX port [jaxtari](../jaxtari/)). It is being built bit-exactly against xitari first, then layered with a differentiability mode (HARD = bit-exact, SOFT = relaxed for gradients) so XAI methods can be applied to the simulator itself. See [`../PORTING_PLAN.md`](../PORTING_PLAN.md) for the full plan and milestone phasing.

## Status

**Phase P0 — scaffolding.** No emulator works yet. The package layout, opcode tables, and a `Test` runner are in place. Phase P1 will fill in the 6502 instruction set, validated against per-cycle traces from xitari.

## Quickstart

```julia
julia --project=.
julia> using Pkg; Pkg.instantiate()
julia> Pkg.test()
```

## Layout

```
jutari/
├── Project.toml
├── README.md
├── src/
│   ├── JuTari.jl           # module root
│   ├── Types.jl            # shared state types (CPUState today; more later)
│   ├── cpu/                # M6502 / M6507 core
│   │   ├── Tables.jl       # opcode → addressing mode + cycle count tables
│   │   └── M6502.jl        # fetch–decode–execute (stub in P0)
│   └── diff/               # HARD vs SOFT differentiability layer
│       └── Modes.jl
└── test/
    └── runtests.jl
```

Additional submodules (`bus/`, `riot/`, `tia/`, `cart/`, `io/`, `env/`, `games/`, `xai/`) will be added as their phases land — see PORTING_PLAN.md §3.2 and §5.
