# jaxtari

Differentiable JAX port of [xitari](https://github.com/google-deepmind/xitari) — the DeepMind fork of the Arcade Learning Environment built on Stella.

This package is one half of the **UnderstandingVCS** project (the other is the Julia port [jutari](../jutari/)). The bit-exact emulator runs in HARD mode (the only behavioural mode wired up today); the SOFT-mode differentiability primitives (`RomTensor`, `soft_select`, `soft_memory_read`, `soft_branch`, straight-through round/clamp) are in `jaxtari.diff` with gradient tests, but `cpu.m6502.step()` does not yet route through them. See [`../STATUS.md`](../STATUS.md) for the per-phase ledger and [`../PORTING_PLAN.md`](../PORTING_PLAN.md) for the design.

## Quickstart

```bash
cd jaxtari
python3.13 -m venv .venv     # JAX needs Python ≤ 3.13 as of writing
source .venv/bin/activate
pip install -e ".[dev]"
pytest                       # 321 tests
```

Build a console and run a frame:

```python
import jax.numpy as jnp
from jaxtari.env import StellaEnvironment
from jaxtari.io.action import Action

rom = jnp.zeros((4096,), dtype=jnp.uint8)
# ... load your ROM bytes into `rom` ...

env = StellaEnvironment(rom)
env.reset()
while not env.game_over():
    reward = env.step(Action.NOOP)
    frame  = env.get_screen()      # (192, 160) uint8 indexed colour
```

XAI demo (ROM-byte attribution via the `RomTensor` primitive):

```python
import jax
import jax.numpy as jnp
from jaxtari.diff import RomTensor

rom_bytes = jnp.full((256,), 5.0)
def simulator(rom_arr):
    rom = RomTensor(rom_arr)
    return rom.peek(0x42) ** 2

grad = jax.grad(simulator)(rom_bytes)
# grad is one-hot at position 0x42 with value 2 * rom[0x42] (= 10.0); zero elsewhere.
```

## Layout

```
jaxtari/
├── pyproject.toml
├── README.md
├── jaxtari/
│   ├── __init__.py
│   ├── types.py            # CPUState NamedTuple
│   ├── cpu/                # 6502 / 6507 — all 151 documented NMOS opcodes (P1)
│   │   ├── tables.py       # opcode → addressing-mode + cycle-count tables
│   │   ├── addressing.py   # 12 effective-address resolvers
│   │   ├── alu.py          # set_zn / compare_flags / bit_flags / adc / sbc /
│   │   │                   # asl_op / lsr_op / rol_op / ror_op
│   │   └── m6502.py        # fetch–decode–execute dispatch + post-step TIA/RIOT hooks
│   ├── bus/                # 6507 13-bit address decode + region routing (P2)
│   │   └── system.py       # Bus NamedTuple + peek/poke + Bus<->flat-memory dispatch
│   ├── tia/                # TIA video chip (P3a–f)
│   │   └── system.py       # register file, scanline / frame timing, WSYNC, playfield,
│   │                       # players, missiles, ball, collisions, VSYNC/VBLANK, INPT*
│   ├── riot/               # M6532 timer + I/O ports (P4)
│   │   └── system.py       # INTIM / INSTAT / TIM*T + SWCHA/SWCHB + DDRs
│   ├── cart/               # Bank-switched cartridges (P5)
│   │   └── system.py       # Cart (2K/4K/F8/F6/F4) — mutable for hotspot reads
│   ├── console.py          # Console (CPU + Bus) + reset + step + run_until_frame (P6)
│   ├── io/
│   │   └── action.py       # ALE Action enum + apply_action + console_switches (P6)
│   ├── games/
│   │   └── rom_settings.py # RomSettings Protocol + GenericRomSettings stub (P6)
│   ├── env/
│   │   └── stella_environment.py  # ALE-style reset / step / get_screen / get_ram (P6)
│   └── diff/                       # Differentiability primitives (P7)
│       ├── modes.py                # Mode enum + global toggle
│       ├── rom_as_weights.py       # RomTensor with one-hot peek
│       ├── soft_select.py          # softmax-weighted mixture
│       ├── soft_mem.py             # NTM-style positional read
│       ├── soft_branch.py          # sigmoid-relaxed PC gate
│       └── straight_through.py     # STE round / clamp via jax.custom_vjp
└── tests/
    ├── test_smoke.py
    ├── test_cpu_p1a.py … test_cpu_p1f.py
    ├── test_bus.py
    ├── test_tia.py / test_tia_playfield.py / test_tia_players.py /
    │   test_tia_missiles_ball.py / test_tia_collisions.py / test_tia_vsync_vblank.py
    ├── test_riot.py
    ├── test_cart.py
    ├── test_p6.py          # Console + IO + StellaEnvironment
    └── test_diff.py        # P7 primitives + ROM-byte attribution demo
```

## What this port can do today

- Run any documented NMOS 6502 instruction sequence.
- Run a complete VCS through `StellaEnvironment.step(action)`, producing a `(192, 160)` framebuffer + 128 B RAM each frame.
- Auto-detect cart format from ROM size; bank-switch on hotspot read OR write (F8/F6/F4).
- Translate ALE-style actions into RIOT joystick bits + TIA INPT4 trigger; expose console switches via SWCHB.
- Compute `jax.grad` of any output that depends on a `RomTensor.peek` — one-hot at the accessed address, ready for XAI attribution.

## What this port does NOT yet do

See [`../STATUS.md`](../STATUS.md) for the complete deferral list. The biggest items:

- **`step()` doesn't route through the P7 diff primitives** (P7b is the next phase) — SOFT mode is a flag with no behavioural effect.
- **No xitari-trace conformance** — coverage is via 321 unit tests, not against real ROM runs. The `tools/trace_dump.cpp` harness sketched in PORTING_PLAN.md §4 isn't built.
- TIA: NUSIZ multi-copy, VDELP*, sub-pixel beam-accurate rendering, audio (AUDC/AUDF/AUDV) — all deferred.
- RIOT: paddle dump-pot timing, PA7 interrupt — deferred.
- Cart: SC variants, E0/FE/3F/3E/MB/MC/AR/DPC — deferred.
- Env: per-game `RomSettings`, phosphor blend, random-noop-reset — deferred.
