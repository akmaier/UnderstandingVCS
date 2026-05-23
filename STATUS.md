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
| **P7e** | Julia gradient stack — Zygote reverse-mode AD verified through every pure SOFT primitive; one-hot constructions rewritten as broadcast comparisons so Zygote can trace them | _next commit_ | +18 | — | ✅ |
| **P7f-a** | Differentiable TIA — `soft_render_scanline` / `soft_render_frame` (background + playfield); reads the TIA register file out of `bus.ram[0:64]`. **`jax.grad` of a framebuffer pixel back to a ROM byte now works end-to-end.** | [`2253d32`](https://github.com/akmaier/UnderstandingVCS/commit/2253d32) | +14 | +10 | ✅ |
| **P7f-b** | Differentiable TIA — player sprites P0/P1 (GRP, REFP, single 1× copy) compositing over the playfield; SOFT convention: RESP0/RESP1 cells hold the player X | [`43371ce`](https://github.com/akmaier/UnderstandingVCS/commit/43371ce) | +25 | +11 | ✅ |
| **P7f-c** | Differentiable TIA — missiles M0/M1 + ball BL (solid 1/2/4/8-px blocks); full HARD compositing order bg ← pf ← ball ← M1 ← P1 ← M0 ← P0 | [`99e2ec7`](https://github.com/akmaier/UnderstandingVCS/commit/99e2ec7) | +18 | +12 | ✅ |
| **P7f-d** | Differentiable TIA — collision detection: `soft_collision_registers` returns the 8 CX latches (15 pairwise object overlaps) | _next commit_ | +11 | +12 | ✅ |
| **P8-a** | XAI primitive — Integrated Gradients on top of `jax.grad` (midpoint Riemann; exact for linear/quadratic `f`; completeness axiom enforced) | _next commit_ | — | +7 | ✅ |
| **P8-b** | First attribution experiment — three attribution semantics (plain gradient / occlusion / smart-baseline IG) on a SOFT-mode kernel, plus the recorded finding that naive zero-baseline IG collapses on opcode bytes | _next commit_ | — | +4 | ✅ |
| **P8-c** | Pong-like state attribution: hand-set TIA registers representing a mid-Pong-frame; IG / `jax.grad` localises a paddle pixel to COLUP0, a wall pixel to COLUPF, a background pixel to COLUBK. **Plus** `soft_run_scan` (jax.lax.scan-backed) for thousands-of-instruction SOFT traces, smoke-tested on real `pong.bin`. | _next commit_ | — | +6 (+1 xfail) | ✅ |
| **P8-cx** | SOFT-mode RIOT timer + proper BRK interrupt + ROM-mirror wrap + RIOT/SWCHB read defaults + TIM*T region-guard fix. **Pong now runs end-to-end in SOFT mode** — `soft_run_scan(pong.bin, 50_000)` paints COLUP0/COLUP1/COLUBK/RESP0/RESP1 etc.; the previously-xfailed real-Pong test now passes. | _next commit_ | — | — | ✅ |
| **P9**  | JAX-vs-Julia benchmark (`tools/bench_soft_run.{py,jl}`) + paper-shaped writeup (`RESULTS.md`). **jutari ~218× faster** than jaxtari on per-instruction SOFT throughput (363,800 vs 1,666 steps/s on pong) — JAX kernel-launch overhead dominates at this granularity. | _next commit_ | — | — | ✅ |

**Totals after P8-cx: jaxtari 532 tests (+ 1 xfail), jutari 1122 tests (+ 1 broken) — 1655 effective.** One xfail remains, recording the PXC1-x round 2+ bit-exact gap (10 RAM bytes).

**Real Pong now executes in SOFT mode end-to-end.** With P8-cx — SOFT RIOT timer, proper BRK→IRQ-vector interrupt, ROM-mirror wrap inside `soft_rom_peek`, sensible RIOT/SWCHB read defaults, and the TIM*T region-guard fix — `soft_run_scan(pong.bin, 50_000)` paints COLUP0/COLUP1/COLUBK/RESP0/RESP1 etc. (Pong's kernel really runs). The previously-xfailed real-Pong test now passes. Execution-based attribution (full `jax.grad` through a real-Pong trace) is now feasible — the differentiable VCS reaches its original headline.

**P8 milestone — first XAI signal from the differentiable VCS.** A SOFT program executes, the differentiable TIA renders a pixel, and three attribution methods correctly identify the ROM byte that explains it: plain `jax.grad` (source), occlusion (necessity), smart-baseline IG (quantified contribution). The naive zero-baseline IG fails on opcode bytes — a recorded finding about XAI on discrete-input emulators.

**The project's headline claim is now live in code, end-to-end.** A SOFT program executes 6502 instructions, writes colours into TIA registers, and `soft_render_scanline` turns the register file into pixels — then `jax.grad(pixel)(rom)` is one-hot at the ROM byte that painted that pixel. `∂pixel / ∂ROM` — "this ROM byte explains this pixel" — runs from instruction fetch through CPU execution through TIA compositing to a framebuffer pixel (test `test_grad_background_pixel_one_hot_at_colour_rom_byte`).

**P7c milestone:** the full 151-opcode documented NMOS 6502 set (+ the undocumented USBC `$EB` alias) executes in SOFT mode on both ports — `soft_step` is a complete differentiable parallel to the HARD `step()` at the instruction level.

**P7f milestone:** the SOFT-mode differentiable TIA renderer covers background, playfield, both players, both missiles, the ball, and all 8 collision latches — full P3a-e visible-object parity. Genuinely deferred: VBLANK output-blanking, wiring collisions into bus *reads* (P7f-dx), and cart-hotspot bank-switching (P7f-e).

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
│   ├── soft_step.py              │   ├── SoftStep.jl        (P7b/P7c)
│   └── soft_tia.py               │   └── SoftTIA.jl         (P7f-a..d)
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
- **P3l is ✅ complete** — CTRLPF bit 2 (PFP) priority swap is honoured by both the HARD `render_scanline` and the SOFT `soft_render_scanline` (jaxtari + jutari). Default (PFP=0) keeps the original `bg ← pf ← bl ← M1 ← P1 ← M0 ← P0` order; PFP=1 swaps to `bg ← M1 ← P1 ← M0 ← P0 ← pf ← bl` so playfield + ball composite on top of sprites. The SOFT renderer blends the two composites by the integer-extracted PFP bit; gradients w.r.t. every colour register remain exact in either mode. Tests: `jaxtari/tests/test_p3l_priority.py` (8) + the `JuTari P3l — CTRLPF.D2 priority swap` testset (10) covering both renderers and a Zygote gradient through COLUPF under PFP=1.

### RIOT (P4)
- **P4b**: PA7 edge-triggered interrupt (INSTAT.D6 not modelled).
- **P4c**: Real paddle / driving-controller wiring (paddle dump-pot timing — INPT0-3 stay at `$80` "centred").
- **P4d**: Reading INTIM should clear the timer-expired flag (real MOS 6532), and reading INSTAT should clear the PA7 interrupt latch. We currently clear `timer_expired` only on TIM*T writes. **Why not yet fixed:** `riot_peek` / `peek` return a plain `int`, so a read can't mutate bus state. Threading a new bus through every memory read would touch the CPU step + every test ROM (the 6502 reads memory ~10× per opcode), which makes this a real architecture change, not a quick win. Picked up by a future *peek-with-side-effects* refactor (also unblocks PXC1-x round 3+ if a TIM*T-related read divergence shows up).

### Cart (P5)
- **P5b**: SC variants of F8 / F6 / F4 (128 B on-cart RAM at `$1000–$10FF`).
- **P5c**: E0, FE, 3F, 3E formats.
- **P5d**: MB, MC, AR, DPC formats.
- **P5e**: Signature-based detection for ROMs whose size is ambiguous between formats (e.g. an 8 KB ROM could be F8, E0, or 3F — currently size → format is 1:1).

### Console / IO / Env (P6)
- **P6b**: Phosphor blending (Stella post-processes the framebuffer for flicker).
- **P6c**: Per-game `RomSettings` (Pong / Breakout / Pitfall / Atari Zoo etc. all need reverse-engineered RAM-address probes).
- **P6d**: Random no-op reset (Mnih-style "skip 0..30 NOOPs at episode start" — this is a wrapper concern).
- **P6e is ✅ complete** — `apply_action` (Python) / `apply_action!` (Julia) now take a `player=0|1` keyword. P0 routes the action's directions onto the high nibble of SWCHA + INPT4 trigger; P1 routes to the low nibble + INPT5. The non-driven player's nibble is preserved across calls, so the two-player idiom is `apply_action(c, p0_action, player=0); apply_action(c, p1_action, player=1); run_until_frame(c)`. Tests: 10 new P6e cases in `jaxtari/tests/test_p6.py` and a `P6e` block in `jutari/test/runtests.jl` covering each direction, FIRE→INPT5, P0+P1 composition, and nibble independence.

### Diff (P7 / P7b / P7c / P7d / P7c-bx)
- **P7c-a … P7c-f are ✅ complete** — the full 151-opcode documented NMOS set executes in SOFT mode.
- **P7c-bx is ✅ complete** — BCD (decimal-mode) ADC/SBC now dispatch on the D flag; binary path stays gradient-clean, BCD path is integer.
- **P7d is ✅ complete** — `RomTensor` is a JAX PyTree, usable directly in the `SoftBus.rom` slot.
- **P7c-dx is ✅ complete (jaxtari)** — `SoftCPUState` now carries `P_N` / `P_Z` / `P_C` / `P_V` float mirrors alongside the packed `P` byte. Every flag-touching opcode routes through a single `_with_p` helper that keeps both representations in lock-step. `_do_branch` uses a straight-through trick: the *forward* PC is read from the packed `P` (so the existing P7c-d / PXC1 forward semantics stay bit-exact), but the *gradient* runs through `soft_branch` keyed by the float mirror — so an XAI caller that injects a soft Z / N / C / V into `SoftCPUState` gets a non-zero gradient through both `pc_taken` and `pc_not_taken`. Tests: `jaxtari/tests/test_p7c_dx.py` (12) covering mirror sync, forward-bit-exactness with only `P` set, and the sigmoid-gated gradient through `P_Z`. Jutari mirror waits on P7e-x (a mutating soft_step!/soft_run! can't carry the gradient anyway).
- BRK stays the end-of-trace sentinel (intentional for fixed-length XAI traces).
- **P7e is ✅ complete** — Zygote (reverse-mode AD) is a jutari test dependency and gradients are verified through every pure SOFT primitive (`soft_rom_peek`, `soft_ram_peek`, `RomTensor.peek`/`peek_many`, `soft_select`, `soft_memory_read`, `soft_branch`). The one-hot constructions were rewritten as broadcast comparisons so Zygote can trace them.
  - **P7e-x is ✅ partial — functional core landed** — alongside the mutating `soft_step!` / `soft_run!`, jutari now ships pure-functional `soft_step` / `soft_run` (no bang). They build a new `SoftCPUState` / `SoftBus` per step via `update_state` / `update_bus` constructors and a Zygote-friendly `_set_ram` broadcast (no `setfield!` / `setindex!` on the gradient path). Initial opcode coverage is the **P7b core** + common P7c-a opcodes: NOP, BRK (halt-in-place sentinel), JMP abs, LDA / LDX / LDY imm/zp (+ LDA abs), STA / STX / STY zp (+ STA abs), TAX/TAY/TXA/TYA/TSX/TXS, CLC/SEC — ~22 opcodes. Uncovered opcodes fall through to `_func_default` (PC+=1, cycles+=2) — the trace stays gradient-clean, just forward-wrong past an unhandled opcode. **Zygote can now take real gradients through full Julia traces** built from the covered set; `Zygote.gradient(rom -> soft_run(state, SoftBus(ram, rom), N)[2].ram[k], rom0)` correctly produces a one-hot at the relevant operand byte. Tests: 24 in the new `JuTari P7e-x` block — forward equivalence vs `soft_run!`, Zygote ∂A/∂ROM is one-hot at LDA's immediate byte, end-to-end gradient through a 4-instruction load/store chain reaches the ROM, transfer-chain gradient survives TAX, unhandled opcodes fall through cleanly. **Extending the handler table to the full 151-opcode set is mechanical follow-up work** — each remaining handler is the same shape as the ones already done (mutating ones live in the same file as reference). The mutating path retains full coverage for non-AD use.
- **P7f-a … P7f-d are ✅ complete** — `soft_render_scanline` / `soft_render_frame` render the full visible object set differentiably: background, playfield, both player sprites, both missiles, and the ball, composited in the HARD TIA's priority order; `soft_collision_registers` detects the 8 CX collision latches. `jax.grad` of a pixel reaches the ROM. The renderer reads the TIA register file out of `bus.ram[0:64]` (the SOFT bus collapses TIA addresses into the low RAM cells, so no SoftBus change was needed).
- **Still deferred after P7f** (each a deliberate, documented gap, not a bug):
  - **P7f-dx — collision *reads***: `soft_collision_registers` is a standalone function; a program cannot yet *read* $00-$07 as collision data because `_bus_read` returns the register-file value. Proper TIA read-register dispatch would wire it in.
  - **P7f-e — cart-hotspot bank-switching from SOFT mode**: needs the SoftBus to carry bank state — an architectural addition.
  - **VBLANK output-blanking**: the render is the active-display path only.
  - **P7f-b…d positioning**: sprite/missile/ball X comes from the SOFT-mode convention that the RES* cells hold position; faithful strobe-timing positioning needs TIA timing state in the SoftBus (see the P7f architectural-prerequisite note in PORTING_PLAN.md).
- **P7f-b…d positioning caveat**: a SOFT-mode convention treats the `RESP0`/`RESP1` cells as holding the player X position (real hardware sets it by strobe timing). Faithful strobe-timing positioning needs the SoftBus to carry TIA timing state — see the P7f architectural-prerequisite note in PORTING_PLAN.md.

### Cross-cutting
- **PXC1 + PXC1-x round 1 are ✅ shipped** — `tools/trace_dump.cpp` builds against xitari and produces frame-level JSONL traces (`tools/fixtures/traces/pong_noop_10.jsonl` is committed). `tools/check_trace.py` / `tools/check_trace.jl` replay them against jaxtari / jutari. **PXC1-x round 1** closed two real emulation gaps — (a) ALE-equivalent boot-burn (60 NOOP frames + 4 RESET-switch frames) is now an opt-in via `env.reset(boot_noop_steps=60, boot_reset_steps=4)`; and (b) the TIA frame counter was being double-incremented per real frame (the scanline-wrap "safety fallback" fired in addition to the VSYNC 1→0 handler), so `run_until_frame` was completing every other "frame" in ~80 CPU cycles instead of ~19,900 — frame is now VSYNC-only. The divergence on `pong_noop_10` dropped from **25 → 10 RAM bytes**, and **the same 10 bytes still diverge identically in both ports** (PXC2 stays implicitly satisfied).
- **PXC1-x round 2+** — diagnostic step landed (next commit): `tools/trace_dump --cpu` now dumps the full M6502 register file (A/X/Y/SP/P/PC) per frame, via the `friend class CpuDebug` declaration xitari already had — **no xitari modifications needed**. `tools/check_trace.py` verifies the CPU state. **The CPU matches exactly** on the `pong_noop_10` fixture (jaxtari's A/X/Y/SP/P/PC after frame 1 are bit-equal to xitari's), so the remaining 10-byte RAM divergence is *not* an execution-path bug — it's purely in data-path reads (TIA collisions, INPT*, INSTAT, etc.). That radically narrows where to look in round 3+ — closing the remaining 10 bytes is now a targeted "which TIA/RIOT *read* returns the wrong value at one specific cycle" investigation rather than a generic "find any bug in our CPU".
- **PXC2** (implicit for the pong_noop_10 fixture — both ports diverge identically): formal JAX-vs-Julia bit-for-bit cross-check across more fixtures + a fresh test that asserts jaxtari ≡ jutari directly.
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
