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
| **P7b** | Parallel SOFT-mode `soft_step` (8 opcodes) + end-to-end `jax.grad` back to ROM | [`11c8388`](https://github.com/akmaier/UnderstandingVCS/commit/11c8388) | +34 | +18 | ✅ |
| **P7c-a** | Full load/store/transfer opcode coverage (37 opcodes) + N/Z flag updates + cart-vs-RAM bus-read dispatch | [`d6e2c18`](https://github.com/akmaier/UnderstandingVCS/commit/d6e2c18) | +33 | +28 | ✅ |
| **P7c-b** | Arithmetic + logic + compare + BIT (49 opcodes — ADC/SBC binary mode incl. USBC; AND/ORA/EOR/CMP/CPX/CPY/BIT) + N/Z/C/V flag updates | [`742fcbe`](https://github.com/akmaier/UnderstandingVCS/commit/742fcbe) | +36 | +22 | ✅ |
| **P7c-c** | Shifts and rotates (20 opcodes — ASL/LSR/ROL/ROR accumulator + 4 memory modes each); RMW memory ops; N/Z/C flag updates | [`1530c6e`](https://github.com/akmaier/UnderstandingVCS/commit/1530c6e) | +33 | +17 | ✅ |
| **P7c-d** | Branches (8 conditionals), JMP indirect (NMOS page-wrap bug), JSR/RTS with SOFT stack | [`a320ee3`](https://github.com/akmaier/UnderstandingVCS/commit/a320ee3) | +22 | +15 | ✅ |
| **P7c-e** | Stack push/pull (PHA/PHP/PLA/PLP), status-flag opcodes (CLC/SEC/CLI/SEI/CLV/CLD/SED), INC/DEC memory, INX/INY/DEX/DEY — 23 opcodes | [`7eafd63`](https://github.com/akmaier/UnderstandingVCS/commit/7eafd63) | +30 | +21 | ✅ |
| **P7c-f** | RTI — **completes the full 151-opcode documented NMOS set (+ USBC) in SOFT mode** | _next commit_ | +36 | +7 | ✅ |
| **P7d** | RomTensor as a custom JAX PyTree — usable directly as the `SoftBus.rom` slot, `jax.grad` threads the cotangent back as a RomTensor (jaxtari only — PyTree is a JAX concept) | _next commit_ | — | +9 | ✅ |
| **P7e** | Julia gradient stack — Zygote / ChainRulesCore `rrule`s for the SOFT primitives so jutari can take real gradients | — | — | — | ☐ |
| **P7f** | Differentiable bus + TIA — route SOFT writes through real TIA/RIOT register dispatch + cart hotspots, and a differentiable TIA so `jax.grad` flows from a framebuffer pixel back to ROM | — | — | — | ☐ |
| **P8**  | XAI hooks + first attribution experiment | — | — | — | ☐ |
| **P9**  | JAX-vs-Julia benchmark + paper-shaped XAI study | — | — | — | ☐ |

**Totals after P7c (complete): jaxtari 449 tests, jutari 1022 tests, 1471 green across both ports.**

**P7c milestone: the full 151-opcode documented NMOS 6502 set (+ the undocumented USBC `$EB` alias) now executes in SOFT mode on both ports** — `soft_step` is a complete differentiable parallel to the HARD `step()` at the instruction level. What remains for a fully-differentiable VCS is **P7f** (real TIA/RIOT/cart bus dispatch + a differentiable TIA).

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
- **run a real 6502 program in SOFT mode end-to-end** (jaxtari with `jax.grad`; jutari forward-only): `soft_step(state, bus)` dispatches over a 256-way `jax.lax.switch` opcode table covering **all 151 documented NMOS opcodes + the USBC `$EB` alias** (P7c-a…f). All register state is `float32`, ROM/RAM access is one-hot-dot-product differentiable, and `jax.grad(ram[0])(rom)` of the headline `LDA #$42 / STA $00` program returns a one-hot gradient at the immediate-operand byte. That's "this ROM byte explains this RAM cell" in working code. The opcode set is complete; what is still simplified is the *bus* — TIA/RIOT register writes land in the RAM array rather than affecting chip state (real dispatch + a differentiable TIA is P7f).

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

Every deferral now has a phase identifier (see PORTING_PLAN.md "Deferral identifiers" section) so it can be picked up in any order.

### CPU (P1)
- **P1g**: PA7 edge-triggered external IRQ + NMI / RESET pins (bus-level integration; BRK-only software interrupts work today).
- **P1h**: Undocumented opcode set beyond USBC (`$EB`).

### Bus (P2)
- Settled: 6507 13-bit mirror, RAM with stack-page mirror. **No deferrals.**

### TIA (P3)
- **P3g**: NUSIZ multi-copy / 2×/4×-wide player scaling (single 1×-wide copy per player today).
- **P3h**: VDELP* / VDELBL vertical-delay sprite updates.
- **P3i**: Sub-pixel beam-accurate rendering (mid-scanline register changes affecting the line being drawn). P3 renders at end-of-scanline.
- **P3j**: Audio (AUDC*/AUDF*/AUDV* registers are stored but inert; TIASnd not modelled).
- **P3k**: HMOVE +8 nibble timing quirk on real hardware.
- **P3l**: CTRLPF.D2 priority swap (default priority only).

### RIOT (P4)
- **P4b**: PA7 edge-triggered interrupt (INSTAT.D6 not modelled).
- **P4c**: Real paddle / driving-controller wiring (paddle dump-pot timing — INPT0-3 stay at `$80` "centred").
- **P4d**: Reading INSTAT does NOT clear the expired flag (only TIM*T writes do; the chip's exact behaviour here varies between implementations).

### Cart (P5)
- **P5b**: SC variants of F8 / F6 / F4 (128 B on-cart RAM at `$1000–$10FF`).
- **P5c**: E0, FE, 3F, 3E formats.
- **P5d**: MB, MC, AR, DPC formats.
- **P5e**: Signature-based detection for ROMs whose size is ambiguous between formats (e.g. an 8 KB ROM could be F8, E0, or 3F — currently size → format is 1:1).

### Console / IO / Env (P6)
- **P6b**: Phosphor blending (Stella post-processes the framebuffer for flicker).
- **P6c**: Per-game `RomSettings` (Pong / Breakout / Pitfall / Atari Zoo etc. all need reverse-engineered RAM-address probes).
- **P6d**: Random no-op reset (Mnih-style "skip 0..30 NOOPs at episode start" — this is a wrapper concern).
- **P6e**: Two-player joystick (P1 directions stay defaulted-released).

### Diff (P7 / P7b / P7c)
- **P7c-a … P7c-f are ✅ complete** — the full 151-opcode documented NMOS set executes in SOFT mode. Three deliberate SOFT-mode simplifications remain, each with its own sub-identifier:
  - **P7c-bx**: BCD (decimal-mode) ADC/SBC — the binary path always runs regardless of the D flag.
  - **P7c-dx**: gradient through branch predicates — branches are HARD (`jnp.where` on the flag bit); restoring predicate gradient needs a float-valued flag representation in `SoftCPUState`.
  - BRK stays the end-of-trace sentinel (intentional for fixed-length XAI traces).
- **P7d**: RomTensor replacing the raw `jnp.ndarray` in the SoftBus's `rom` slot (the SoftBus carries a raw array today because `RomTensor` is a Python class, not a PyTree). Requires registering `RomTensor` as a custom JAX PyTree.
- **P7e**: Julia gradient verification — jutari has the same forward behaviour as jaxtari but no Zygote / ChainRulesCore `rrule` wired in yet (would need adding Zygote as a test dep).
- **P7f**: Differentiable bus + TIA — `soft_step`'s `_bus_write` collapses all non-cart writes into the 128-byte RAM array, so SOFT-mode TIA/RIOT register writes have no chip-level effect. Real dispatch + a differentiable TIA is what lets `jax.grad` flow from a framebuffer pixel back to ROM. This is the largest remaining piece for an end-to-end differentiable VCS.

### Cross-cutting
- **PXC1**: xitari-trace conformance harness (PORTING_PLAN.md §4) — `tools/trace_dump.cpp` is sketched but never built; no golden traces exist yet. Both ports are validated against hand-built unit tests, not against real ROM runs. **The most important single piece of infrastructure debt** — it would catch dozens of subtle bugs at once.
- **PXC2**: JAX-vs-Julia bit-for-bit cross-check (PORTING_PLAN.md §4.4).
- **PXC3**: CI hook (no automated test runs yet).
- **PXC4**: Klaus Dormann `cpu_klaus_dormann.jsonl.gz` regression run (referenced as the P1 acceptance criterion but never wired up).

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
