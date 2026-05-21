# UnderstandingVCS — implementation status

Snapshot of where the differentiable VCS ports stand. For the why and the
phase-design rationale, see [PORTING_PLAN.md](PORTING_PLAN.md). For the
overall project goal, see [README.md](README.md).

## Phase ledger

| Phase | Scope (one line) | Commit | jaxtari Δtests | jutari Δtests | Status |
|---|---|---|---|---|---|
| **P0**  | Scaffolding + `tools/trace_dump.cpp` sketch | [`74e864f`](https://github.com/akmaier/UnderstandingVCS/commit/74e864f) | +6 | +11 | ✅ |
| **P1a** | LDA/LDX/LDY/STA/STX/STY/TAX/TAY/TXA/TYA/TSX/TXS (37 opcodes) | [`4c4826d`](https://github.com/akmaier/UnderstandingVCS/commit/4c4826d) | +22 | +47 | ✅ |
| **P1b1**| AND/ORA/EOR/CMP/CPX/CPY/BIT (40 opcodes) | [`e82c3ab`](https://github.com/akmaier/UnderstandingVCS/commit/e82c3ab) | +13 | +39 | ✅ |
| _fix_   | jaxtari dtype helper + jutari import depth + `step` vs `Base.step` | [`b1fd25b`](https://github.com/akmaier/UnderstandingVCS/commit/b1fd25b) | (turns tests green) | (turns tests green) | ✅ |
| **P1b2**| ADC + SBC binary + BCD/decimal mode (17 opcodes) | [`40fe53c`](https://github.com/akmaier/UnderstandingVCS/commit/40fe53c) | +20 | +46 | ✅ |
| **P1c** | ASL/LSR/ROL/ROR accumulator + memory (20 opcodes) | [`fa6f2f2`](https://github.com/akmaier/UnderstandingVCS/commit/fa6f2f2) | +13 | +42 | ✅ |
| **P1d** | Branches + JMP (abs+ind) + JSR/RTS + stack helpers (12 opcodes) | [`3dd577c`](https://github.com/akmaier/UnderstandingVCS/commit/3dd577c) | +16 | +33 | ✅ |
| **P1e** | PHA/PHP/PLA/PLP + SEC/CLC/SEI/CLI/SED/CLD/CLV + NOP (12 opcodes) | [`68687b2`](https://github.com/akmaier/UnderstandingVCS/commit/68687b2) | +15 | +37 | ✅ |
| **P1f** | INC/DEC + INX/INY/DEX/DEY + BRK/RTI (14 opcodes) — completes P1 | [`f580a58`](https://github.com/akmaier/UnderstandingVCS/commit/f580a58) | +13 | +41 | ✅ |
| **P2**  | 6507 bus + 13-bit mirror + RAM/ROM/TIA-stub/RIOT-stub regions | [`3473b2d`](https://github.com/akmaier/UnderstandingVCS/commit/3473b2d) | +11 | +30 | ✅ |
| **P3a** | TIA register file + scanline/frame timing + WSYNC | [`91868bc`](https://github.com/akmaier/UnderstandingVCS/commit/91868bc) | +18 | +48 | ✅ |
| **P3b** | Playfield (PF0/PF1/PF2 + CTRLPF mirror + COLU*) | [`c9f3e37`](https://github.com/akmaier/UnderstandingVCS/commit/c9f3e37) | +16 | +45 | ✅ |
| **P3c** | Player sprites P0/P1 + GRP + REFP + RESP + HMP + HMOVE/HMCLR | [`74fd4c7`](https://github.com/akmaier/UnderstandingVCS/commit/74fd4c7) | +19 | +75 | ✅ |
| **P3d** | Missiles M0/M1 + ball BL (sizes, EN*, RES*, HM*) | [`4995227`](https://github.com/akmaier/UnderstandingVCS/commit/4995227) | +15 | +54 | ✅ |
| **P3e** | 8 collision latches CXM0P..CXPPMM + CXCLR | [`76c6d96`](https://github.com/akmaier/UnderstandingVCS/commit/76c6d96) | +16 | +46 | ✅ |
| **P3f** | VSYNC frame-edge + VBLANK output blanking — completes P3 | [`f27ae81`](https://github.com/akmaier/UnderstandingVCS/commit/f27ae81) | +10 | +30 | ✅ |
| **P4**  | RIOT M6532 timer (TIM*T/INTIM/INSTAT) + 2 I/O ports (SWCHA/SWCHB + DDRs) | [`1635048`](https://github.com/akmaier/UnderstandingVCS/commit/1635048) | +26 | +43 | ✅ |
| **P5**  | Cart 2K / 4K / F8 / F6 / F4 with hotspot bank-switching | [`53e393f`](https://github.com/akmaier/UnderstandingVCS/commit/53e393f) | +20 | +47 | ✅ |
| **P6**  | Console + IO actions + StellaEnvironment + TIA INPT* triggers | [`7a8f303`](https://github.com/akmaier/UnderstandingVCS/commit/7a8f303) | +29 | +39 | ✅ |
| **P7**  | Diff primitives — RomTensor, soft_select, soft_memory_read, soft_branch, STEs | [`4b4d989`](https://github.com/akmaier/UnderstandingVCS/commit/4b4d989) | +23 | +22 | ✅ |
| **P7b** | Parallel SOFT-mode `soft_step` (8 opcodes) + end-to-end `jax.grad` back to ROM | _next commit_ | +34 | +18 | ✅ |
| **P8**  | XAI hooks + first attribution experiment | — | — | — | ☐ |
| **P9**  | JAX-vs-Julia benchmark + paper-shaped XAI study | — | — | — | ☐ |

**Totals after P7b: jaxtari 339 tests, jutari 832 tests, 1171 green across both ports.**

## What each port can do today

End-to-end, both ports can:

- Run any of the 151 documented NMOS 6502 opcodes plus the undocumented USBC alias (`$EB`).
- Run a complete VCS — CPU + Bus (13-bit address mirror) + RAM + TIA + RIOT + bank-switched cart.
- Render the framebuffer: playfield (with mirror) + 2 player sprites (with REFP / HMOVE positioning) + 2 missiles + ball + collision latches.
- Handle a full Atari frame structure: software VSYNC pulse drives the frame counter; VBLANK blanks output during vsync/overscan; INSTAT-readable RIOT timer ticks down with selectable 1/8/64/1024 prescaler.
- Auto-detect cart format by ROM size (2K / 4K / 8K-F8 / 16K-F6 / 32K-F4) and bank-switch on hotspot read or write.
- Accept ALE-style discrete actions (NOOP, FIRE, the 8 joystick directions, the 8 direction+fire combinations); apply them through SWCHA + INPT4; expose them to the running ROM the next frame.
- Drive the SELECT / RESET / colour / difficulty switches on SWCHB.
- Expose an ALE-shaped front-end: `reset` / `step(action) → reward` / `get_screen` / `get_ram` / `game_over` / `lives`, plus the camelCase aliases (`act`, `getScreen`, etc.) for drop-in compatibility with code written against the original C++ ALE.
- (jaxtari only, P7) compute gradients: `RomTensor.peek(addr)` → gradient is one-hot at `addr`; `soft_select` / `soft_memory_read` / `soft_branch` are saturation-equivalent to their hard counterparts and provide useful backward signal under relaxation; STE round/clamp give identity backward through discrete forward.
- (jaxtari only, P7b) **run a real 6502 program in SOFT mode end-to-end**: `soft_step(state, bus)` dispatches over a 256-way `jax.lax.switch` opcode table (8 opcodes handled — NOP, LDA imm/zp, LDX imm, STA zp, STX zp, JMP abs, BRK; rest fall through to a non-raising default), all register state is `float32`, ROM and RAM access is one-hot-dot-product differentiable, and `jax.grad(ram[0])(rom)` of the headline `LDA #$42 / STA $00` program returns a one-hot gradient at the immediate-operand byte. That's "this ROM byte explains this RAM cell" in working code.

End-to-end XAI demo (jaxtari, P7b): the test `test_grad_lda_imm_then_sta_zp_one_hot_at_immediate` runs the two-instruction program `LDA #$42 / STA $00` in SOFT mode and asserts that `jax.grad(RAM[0])(rom)` is exactly 1.0 at `rom[1]` (the immediate operand) and 0 elsewhere. Same primitive demo as P7's `test_xai_rom_byte_attribution_demo` but with the full opcode-dispatch path involved end-to-end. This is the project's whole point in one test.

## Module layout (after P7b)

```
jaxtari/jaxtari/                 jutari/src/
├── types.py                      ├── Types.jl
├── cpu/                          ├── cpu/
│   ├── tables.py                 │   ├── Tables.jl
│   ├── addressing.py             │   ├── Addressing.jl
│   ├── alu.py                    │   ├── ALU.jl
│   └── m6502.py                  │   └── M6502.jl
├── bus/                          ├── bus/
│   └── system.py                 │   └── Bus.jl
├── tia/                          ├── tia/
│   └── system.py                 │   └── TIA.jl
├── riot/                         ├── riot/
│   └── system.py                 │   └── RIOT.jl
├── cart/                         ├── cart/
│   └── system.py                 │   └── Cart.jl
├── io/                           ├── io/
│   └── action.py                 │   └── IO.jl
├── env/                          ├── env/
│   └── stella_environment.py     │   └── StellaEnvironment.jl
├── games/                        ├── games/
│   └── rom_settings.py           │   └── RomSettings.jl
├── diff/                         ├── diff/
│   ├── modes.py                  │   ├── Modes.jl
│   ├── rom_as_weights.py         │   ├── RomAsWeights.jl
│   ├── soft_select.py            │   ├── SoftSelect.jl
│   ├── soft_mem.py               │   ├── SoftMem.jl
│   ├── soft_branch.py            │   ├── SoftBranch.jl
│   ├── straight_through.py       │   ├── StraightThrough.jl
│   ├── soft_state.py             │   ├── SoftState.jl       (P7b)
│   └── soft_step.py              │   └── SoftStep.jl        (P7b)
├── console.py                    ├── Console.jl
└── __init__.py                   └── JuTari.jl
```

## Per-phase deferrals (each documented inline in its module header)

The phase-by-phase deferrals build up. Listing them once here as a single
reference of what remains:

### CPU (P1)
- PA7 edge-triggered IRQ (BRK-only software interrupts are implemented; the external IRQ / NMI / RESET pins are not — those need bus-level integration in P3+).
- Cycle-counting fine print beyond page-cross and branch-taken: per-opcode quirks (e.g. RMW double-write, undocumented opcode set beyond USBC).

### Bus (P2)
- Settled: 6507 13-bit mirror, RAM with stack-page mirror.

### TIA (P3)
- NUSIZ multi-copy / 2×/4×-wide player scaling (single 1×-wide copy per player today).
- VDELP* / VDELBL vertical-delay sprite updates.
- Sub-pixel beam-accurate rendering (mid-scanline register changes affecting the line being drawn). P3 renders at end-of-scanline.
- Audio (AUDC*/AUDF*/AUDV* registers are stored but inert; TIASnd not modelled).
- HMOVE +8 nibble timing quirk on real hardware.
- CTRLPF.D2 priority swap (default priority only).

### RIOT (P4)
- PA7 edge-triggered interrupt (INSTAT.D6 not modelled).
- Reading INSTAT does NOT clear the expired flag (only TIM*T writes do; the chip's exact behaviour here varies between implementations).
- Real paddle / driving-controller wiring (paddle dump-pot timing — INPT0-3 stay at `$80` "centred").

### Cart (P5)
- SC variants of F8 / F6 / F4 (128 B on-cart RAM at `$1000–$10FF`).
- E0, FE, 3F, 3E, MB, MC, AR, DPC formats.
- Signature-based detection for ROMs whose size is ambiguous between formats (e.g. an 8 KB ROM could be F8, E0, or 3F — currently size → format is 1:1).

### Console / IO / Env (P6)
- Paddle pot dump-pot timing (INPT0-3 default `$80`).
- Phosphor blending (Stella post-processes the framebuffer for flicker).
- Per-game `RomSettings` (Pong / Breakout / Pitfall / Atari Zoo etc. all need reverse-engineered RAM-address probes).
- Random no-op reset (Mnih-style "skip 0..30 NOOPs at episode start" — this is a wrapper concern).
- Two-player joystick (P1 directions stay defaulted-released).

### Diff (P7 / P7b)
- **P7c** is the next item: extend the SOFT-mode opcode handler table from the 8 P7b opcodes (NOP, LDA imm/zp, LDX imm, STA zp, STX zp, JMP abs, BRK) to the rest of the 151 NMOS set. Unhandled opcodes today fall through to a `_branch_default` that advances PC by 1 — gradient-safe, but forward-wrong if a real ROM hits one.
- Status-flag updates (N/Z/C/V) in the SOFT handlers — register movement is there, flags are not.
- TIA / RIOT writes via SOFT mode (STA $0xxx currently drops silently into the RAM region).
- Cart bank-switching from SOFT mode (P5's hotspot mechanism isn't wired into `soft_step` yet).
- RomTensor replacing the existing Cart class in the Bus's cart slot (the SoftBus carries a raw `jnp.ndarray`, not the `RomTensor` wrapper, because the wrapper isn't a PyTree).
- Julia gradient verification — jutari has the same forward behaviour as jaxtari but no Zygote / ChainRulesCore `rrule` wired in yet (would need adding Zygote as a test dep).

### Cross-cutting
- **xitari-trace conformance harness** (PORTING_PLAN.md §4) — `tools/trace_dump.cpp` is sketched but never built; no golden traces exist yet. Both ports are validated against hand-built unit tests, not against real ROM runs. This is the most important infrastructure debt; it would catch dozens of subtle bugs at once.
- JAX-vs-Julia bit-for-bit cross-check (PORTING_PLAN.md §4.4).
- CI hook (no automated test runs yet).

## How to run the test suites

```bash
# jaxtari (requires Python 3.13 + a venv with jax + pytest)
cd jaxtari
source .venv/bin/activate   # if not already created: python3.13 -m venv .venv && pip install -e ".[dev]"
pytest

# jutari (requires Julia 1.10+)
cd jutari
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

Both should print all green. There is currently no CI runner.

## The XAI use case in one paragraph

The whole point of UnderstandingVCS is to ask whether modern XAI methods can produce a hierarchical, mechanistic understanding of a system whose ground truth we fully possess — the Atari 2600 VCS. By making the simulator end-to-end differentiable (P7+), we can compute gradients like `∂(pixel value)/∂(ROM byte)` and ask whether attribution methods correctly localise the bytes that explain a given pixel. The differentiable port also opens up concept-probe attacks (CAVs / network-dissection analogues) on TIA register state, mechanistic-interpretability circuit tracing across the CPU's micro-architecture, and ablation studies that can swap any byte or register and measure the downstream effect. The P7 RomTensor primitive — `peek(addr) = one_hot(addr) · rom` — is the single most important hammer the whole project is built around; its proof-of-concept is `test_xai_rom_byte_attribution_demo` in `jaxtari/tests/test_diff.py`.
