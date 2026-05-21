# jaxtari

Differentiable JAX port of [xitari](https://github.com/google-deepmind/xitari) — the DeepMind fork of the Arcade Learning Environment built on Stella.

This package is one half of the **UnderstandingVCS** project (the other is the Julia port [jutari](../jutari/)). It is being built bit-exactly against xitari first, then layered with a differentiability mode (HARD = bit-exact, SOFT = relaxed for gradients) so XAI methods can be applied to the simulator itself. See [`../PORTING_PLAN.md`](../PORTING_PLAN.md) for the full plan and milestone phasing.

## Status

**Phase P0 — scaffolding.** No emulator works yet. The package layout, opcode tables, and test runner are in place. Phase P1 will fill in the 6502 instruction set, validated against per-cycle traces from xitari.

## Quickstart

```bash
pip install -e ".[dev]"
pytest
```

## Layout

```
jaxtari/
├── pyproject.toml
├── README.md
├── jaxtari/
│   ├── __init__.py
│   ├── types.py            # shared state types (CPUState today; more later)
│   ├── cpu/                # M6502 / M6507 core
│   │   ├── tables.py       # opcode → addressing mode + cycle count tables
│   │   └── m6502.py        # fetch–decode–execute (stub in P0)
│   └── diff/               # HARD vs SOFT differentiability layer
│       └── modes.py
└── tests/
    └── test_smoke.py
```

Additional submodules (`bus/`, `riot/`, `tia/`, `cart/`, `io/`, `env/`, `games/`, `xai/`) will be added as their phases land — see PORTING_PLAN.md §3.1 and §5.
