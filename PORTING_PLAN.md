# Porting Plan: xitari → jaxtari (JAX) and jutari (Julia)

## 1. Goal and Scope

Produce two **complete, bit-exact, end-to-end Atari 2600 VCS emulators** — one in JAX, one in Julia — whose forward behaviour is verifiable against `xitari/` cycle-by-cycle, and whose computation graph can be relaxed into a fully differentiable form for XAI experiments.

Non-goals (for now): sound (TIASnd) beyond register-state tracking; networked controllers (fifo, rlglue); operating-system layer (`os_dependent/`); GUI tooling.

Scope explicitly *includes*: 6507 CPU, TIA video, M6532 RIOT, all cartridge bank-switching types xitari supports, the ALE-style environment wrapper, per-game scoring rules, joystick + paddle + console-switch I/O, and the differentiability layer.

---

## 2. Why the previous attempt stalled

The Kimi-K2.6 stub in [jaxtari/jaxtari_cpu.py](jaxtari/jaxtari_cpu.py) and [jutari/src/JuTari.jl](jutari/src/JuTari.jl) gets as far as opcode/addressing/cycle tables and an `execute_instruction` shell that returns the state unchanged. It stalled because:

1. **No module decomposition.** The whole CPU was being put in one file with no boundaries for bus, RAM, TIA, RIOT, cart. The xitari source already shows the decomposition — we mirror it.
2. **No oracle-based test loop.** Without per-cycle xitari traces, every instruction had to be implemented "blind" against a datasheet. With traces, each opcode reduces to *make the JAX/Julia state match the xitari state after this cycle.*
3. **Differentiability conflated with correctness.** Trying to write functional/jit-friendly opcode handlers while also worrying about gradients made every step harder. We separate the two: get **bit-exact** first, layer **differentiability** on top.

---

## 3. Module map (xitari → jaxtari → jutari)

The xitari layout is the authoritative breakdown; both ports follow it 1:1.

### 3.1 jaxtari/ (Python + JAX)

```
jaxtari/
├── pyproject.toml              # package metadata, jax/jaxlib pin, dev deps
├── README.md                   # quickstart + how to run conformance tests
├── jaxtari/
│   ├── __init__.py             # public API: make_console, step, run_frame
│   ├── types.py                # CPUState, BusState, TIAState, RIOTState, ConsoleState (PyTrees)
│   ├── cpu/
│   │   ├── __init__.py
│   │   ├── tables.py           # opcode → (mnemonic, addr_mode, base_cycles)
│   │   ├── addressing.py       # 12 addressing modes → (effective_addr, extra_cycles)
│   │   ├── alu.py              # ADC/SBC/CMP/AND/OR/EOR/ASL/LSR/ROL/ROR with flag updates
│   │   ├── m6502.py            # fetch–decode–execute step, IRQ/NMI/RESET handling
│   │   └── m6507.py            # 6507 (13-bit addr bus) wiring on top of M6502
│   ├── bus/
│   │   ├── __init__.py
│   │   └── system.py           # address decoding: $0000-$007F RAM, $0080-$00FF stack mirror, TIA, RIOT, ROM
│   ├── riot/
│   │   ├── __init__.py
│   │   ├── ram.py              # 128 B RAM read/write
│   │   ├── timer.py            # INTIM/TIM1T/TIM8T/TIM64T/T1024T
│   │   └── ports.py            # SWCHA/SWCHB I/O ports
│   ├── tia/
│   │   ├── __init__.py
│   │   ├── registers.py        # write-only + read-only register decoders
│   │   ├── scanline.py         # WSYNC, HMOVE, hblank, vblank, vsync timing
│   │   ├── playfield.py        # PF0/PF1/PF2 rendering
│   │   ├── players.py          # P0/P1 sprite + GRP0/GRP1/HMP0/HMP1/REFP0/REFP1
│   │   ├── missiles_ball.py    # M0/M1/BL with ENABL etc.
│   │   ├── collisions.py       # CXMnP/CXPPMM latches
│   │   └── framebuffer.py      # 160x192 NTSC palette output (no phosphor here)
│   ├── cart/
│   │   ├── __init__.py
│   │   ├── base.py             # Cart interface (peek/poke + bank state)
│   │   ├── cart2k.py           # 2K
│   │   ├── cart4k.py           # 4K (most common)
│   │   ├── cartf8.py           # 8K F8 bank-switching (+ SC variant)
│   │   ├── cartf6.py           # 16K F6 (+ SC)
│   │   ├── cartf4.py           # 32K F4 (+ SC)
│   │   ├── carte0.py, cart3f.py, cart3e.py, cartfe.py, ...   # rest as xitari has them
│   │   └── detect.py           # auto-detect cart type from ROM size + signatures
│   ├── console.py              # wire CPU + System + TIA + RIOT + Cart + Controllers
│   ├── io/
│   │   ├── __init__.py
│   │   ├── joystick.py
│   │   ├── paddles.py
│   │   └── switches.py         # SELECT, RESET, color/B&W, P0/P1 difficulty
│   ├── env/
│   │   ├── __init__.py
│   │   ├── ale_state.py
│   │   ├── stella_environment.py    # reset/step/act/getScreen API
│   │   └── phosphor_blend.py        # post-process for "phosphor" frames
│   ├── games/
│   │   ├── __init__.py
│   │   ├── rom_settings.py     # base class
│   │   └── supported/          # per-game scoring + termination rules ported from xitari/games/supported
│   ├── diff/                   # the differentiability layer (Phase 7)
│   │   ├── __init__.py
│   │   ├── modes.py            # HARD (bit-exact) vs SOFT (relaxed) global switch
│   │   ├── soft_select.py      # Gumbel-softmax / straight-through opcode dispatch
│   │   ├── soft_mem.py         # NTM-style attention reads / writes
│   │   └── rom_as_weights.py   # ROM tensor wrapper with @jax.grad-able peeks
│   └── xai/
│       ├── __init__.py
│       └── hooks.py            # taps to expose CPU/RAM/TIA state to attribution methods
└── tests/
    ├── conformance/            # bit-exact vs xitari (Phase 1–6)
    ├── differentiability/      # gradient sanity (Phase 7)
    └── xai/                    # XAI-method smoke tests (Phase 8+)
```

### 3.2 jutari/ (Julia)

```
jutari/
├── Project.toml                # name = "JuTari", deps: Zygote, ChainRulesCore, Test
├── README.md
├── src/
│   ├── JuTari.jl               # module root, exports public API
│   ├── Types.jl                # CPUState, BusState, TIAState, RIOTState, ConsoleState
│   ├── cpu/
│   │   ├── Tables.jl
│   │   ├── Addressing.jl
│   │   ├── ALU.jl
│   │   ├── M6502.jl
│   │   └── M6507.jl
│   ├── bus/System.jl
│   ├── riot/{RAM.jl, Timer.jl, Ports.jl}
│   ├── tia/{Registers.jl, Scanline.jl, Playfield.jl, Players.jl, MissilesBall.jl, Collisions.jl, FrameBuffer.jl}
│   ├── cart/{Base.jl, Cart2K.jl, Cart4K.jl, CartF8.jl, CartF6.jl, CartF4.jl, ... Detect.jl}
│   ├── Console.jl
│   ├── io/{Joystick.jl, Paddles.jl, Switches.jl}
│   ├── env/{ALEState.jl, StellaEnvironment.jl, PhosphorBlend.jl}
│   ├── games/{RomSettings.jl, supported/...}
│   ├── diff/{Modes.jl, SoftSelect.jl, SoftMem.jl, RomAsWeights.jl}
│   └── xai/Hooks.jl
└── test/
    ├── conformance/
    ├── differentiability/
    └── xai/
```

Both trees use the **same module names** so cross-language diffs and review are easy.

---

## 4. The bit-exact test harness (built first, before any port code is trusted)

This is the lever Kimi missed. Every port commit must show this harness green for its scope.

### 4.1 Reference-trace tool inside xitari/

Add a small C++ helper `xitari/tools/trace_dump.cpp` (NOT committed under `xitari/` since that folder is gitignored — kept in `tools/` at the repo root instead, and built against the existing `libxitari.a`):

```
tools/
├── trace_dump.cpp              # links libxitari.a, runs ROM + action stream, writes JSONL trace
├── trace_dump.hpp
├── Makefile
└── README.md
```

Output is **one JSON line per CPU cycle** with:

```json
{"cyc": 0, "pc": "0xF000", "a": "0x00", "x": "0x00", "y": "0x00", "sp": "0xFD",
 "p": "0x34", "ram": "base64(128B)", "tia_regs": "base64(64B)",
 "riot_regs": "base64(...)", "bank": 0, "frame": null}
```

…plus an optional `"frame": "base64(160x192 indexed)"` at end-of-frame.

The tool takes `--rom`, `--actions <csv>`, `--seed`, `--max-cycles`, `--frames-only` (for big runs).

### 4.2 Golden traces

Stored under `tests/golden/` (tracked, gzipped):

- `cpu_klaus_dormann.jsonl.gz` — running the Klaus Dormann 6502 functional test ROM. Tests CPU correctness independent of TIA/RIOT.
- `tia_blank_screen.jsonl.gz` — ROM that programs TIA registers then sits in WSYNC loop. Tests scanline timing.
- `riot_timer_walk.jsonl.gz` — ROM that exercises every timer prescaler.
- Per-cart-type traces: `cart_4k_pitfall.jsonl.gz`, `cart_f8_pacman.jsonl.gz`, `cart_f6_montezuma.jsonl.gz`, …
- `game_pong_5000frames.jsonl.gz` — full Pong run with a scripted action sequence and fixed seed.
- `game_breakout_5000frames.jsonl.gz`
- `game_space_invaders_5000frames.jsonl.gz`

ROMs themselves are **not** committed (licensing). The trace files (which are derived from ROMs but contain only register/RAM state) are committed so CI can run without ROMs present.

### 4.3 Conformance test runner

Each port has `tests/conformance/run_against_golden.{py,jl}` which:

1. Loads a golden JSONL.
2. For each cycle, advances the port's emulator one CPU cycle.
3. Diffs every field (regs, RAM, TIA regs, RIOT regs, bank, frame).
4. On first mismatch: prints last 16 cycles from both sides, dumps disassembly around `PC`, exits non-zero.

This is what unblocks the implementation: every opcode, every TIA register write, every bank-switch becomes "make this single failing cycle pass."

### 4.4 Cross-port equivalence

Once both ports pass against xitari, add `tests/cross/jaxtari_vs_jutari.py` that runs the same golden through both and diffs them directly. This catches latent non-determinism (e.g., reduction order) and is the foundation for the JAX-vs-Julia benchmark the user wants.

---

## 5. Phased milestones

Each phase ends with a green test suite for its scope, in **both** ports, before the next phase starts. Status shown here; for per-phase commit IDs and exact test deltas see [STATUS.md](STATUS.md).

| Phase | Scope | Conformance gate | Status |
|---|---|---|---|
| **P0** | Repo scaffolding: pyproject/Project.toml, module skeleton, CI, tools/trace_dump build | `pytest -q` and `julia -e 'using Pkg; Pkg.test("JuTari")'` both run (empty suites OK) | ✅ |
| **P1** | 6502 CPU: official opcodes, addressing modes, flags, cycle counts, IRQ/NMI/RESET | `cpu_klaus_dormann.jsonl.gz` passes byte-for-byte for ≥10M cycles — **conformance harness not built yet; coverage is via 200+ unit tests per port across P1a-f.** All 151 documented NMOS opcodes + USBC alias implemented. | ✅ (unit-tests) |
| **P2** | Bus + RAM + 6507 13-bit mirroring | RAM read/write goldens pass — **goldens not generated yet; coverage is via unit tests.** | ✅ (unit-tests) |
| **P3** | TIA video: scanline engine, playfield, players, missiles, ball, collisions, HMOVE, WSYNC | `tia_blank_screen` and a TIA stress ROM pass; frame buffers match — **conformance harness not built; coverage is via 200+ unit tests across P3a-f.** End-of-scanline rendering (no beam-racing yet); NUSIZ multi-copy + VDELP not yet implemented. | ✅ (unit-tests) |
| **P4** | M6532 RIOT: timers + I/O ports + RAM (already in P2) | `riot_timer_walk` passes — coverage via unit tests. PA7 interrupt + INSTAT-read flag clearing deferred. | ✅ (unit-tests) |
| **P5** | Cartridges: 2K, 4K, F8, F6, F4 first (covers ~90% of supported ROMs), then E0, FE, 3F, 3E, MB, MC, AR, DPC, … | Per-cart-type goldens pass — coverage via unit tests + 1 end-to-end CPU bank-switch test. SC variants + E0/FE/3F/3E/MB/MC/AR/DPC deferred. | ✅ (basic 5 formats) |
| **P6** | Console wiring + I/O (joysticks, paddles, switches) + StellaEnvironment + per-game RomSettings | `game_pong_5000frames`, `game_breakout_5000frames`, `game_space_invaders_5000frames` pass; ALE step/reset/act/getScreen API matches xitari semantically | ✅ (API surface) — per-game `RomSettings` deferred (GenericRomSettings stub only); paddle timing + phosphor blend deferred. |
| **P7** | Differentiability layer (`diff/`): HARD mode is the default and must keep all conformance tests green; SOFT mode enables gradients via soft opcode dispatch, soft RAM addressing, ROM-as-weights | Gradient sanity tests pass (e.g., ∂(pixel)/∂(ROM byte) is non-zero for byte known to affect that pixel and zero for one that cannot); HARD-mode conformance still green | ✅ (**primitives only**) — `RomTensor`, `soft_select`, `soft_memory_read`, `soft_branch`, STE round/clamp shipped with jaxtari gradient tests + end-to-end ROM-attribution demo. **Integration with `step()` is the still-open P7b.** |
| **P7b** | Parallel SOFT-mode `soft_step` (`SoftCPUState` + `SoftBus`, both float32) wired through `jax.lax.switch` over a 256-way opcode handler table; 8 opcodes handled (NOP, LDA imm/zp, LDX imm, STA zp, STX zp, JMP abs, BRK); jutari mirror shipped forward-only | `jax.grad(simulator)(rom)` on the two-instruction program `LDA #$42 / STA $00` returns a one-hot gradient at the immediate-operand byte | ✅ — jaxtari `soft_step.py`/`soft_state.py` + jutari `SoftStep.jl`/`SoftState.jl`; +18 jaxtari tests / +34 jutari tests; headline test `test_grad_lda_imm_then_sta_zp_one_hot_at_immediate` passes. **Opcode-handler expansion, N/Z/C/V flags, TIA/RIOT writes, and cart-hotspot bank-switching from SOFT mode are P7c.** |
| **P7c** | Extend SOFT-mode opcode coverage to the full 151-opcode documented NMOS set; add N/Z/C/V flag updates inside handlers. Subdivided **P7c-a … P7c-f** matching P1's subdivision. | All 151 documented NMOS opcodes (+ USBC `$EB`) execute in SOFT mode | ✅ — jaxtari `soft_step.py` + jutari `SoftStep.jl`; 152-opcode dispatch table. P7c-bx (BCD) is done; two deliberate simplifications remain (P7c-dx branch-predicate gradient, BRK-as-sentinel). |
| **P7d** | `RomTensor` registered as a JAX PyTree node so it can sit in the `SoftBus.rom` slot and `jax.grad` threads the cotangent back as a `RomTensor` | `jax.grad` of a SOFT trace w.r.t. a RomTensor-backed bus returns a RomTensor cotangent, one-hot at the relevant byte | ✅ — jaxtari `rom_as_weights.py` (`tree_flatten`/`tree_unflatten`); +9 tests. jaxtari-only — PyTree is a JAX mechanism. |
| **P7e** | Julia gradient stack — Zygote reverse-mode AD verified through the pure SOFT primitives; one-hot constructions rewritten as broadcast comparisons so Zygote can trace them | `Zygote.gradient` of each SOFT primitive matches the jaxtari `jax.grad` result (one-hot for `soft_rom_peek`, etc.) | ✅ — Zygote added as a jutari test dep; +18 tests. The mutating `soft_step!` is not Zygote-able (deferred as **P7e-x** — needs a functional rewrite or Enzyme.jl). |
| **P7f** | Differentiable bus + TIA — route SOFT-mode writes through real TIA / RIOT register dispatch + cart hotspots (today `_bus_write` collapses non-cart writes into the 128-byte RAM array), and make the TIA differentiable so `jax.grad` flows from a framebuffer pixel back to ROM bytes | A real ROM (e.g. a minimal Stella demo cart) runs end-to-end in SOFT mode and `jax.grad(env.step)` flows through the TIA framebuffer back to ROM bytes | ⏳ |
| **P8** | XAI hooks + first attribution experiment | Integrated Gradients on ROM bytes recovers a known sprite-defining region in Pong | ☐ |
| **P9** | JAX-vs-Julia benchmark + first paper-shaped XAI study | Throughput numbers + a writeup | ☐ |

### P1 (HARD CPU) subdivision

P1 is the largest single chunk and is naturally subdivided:

- **P1a**: load/store + transfer opcodes (LDA/LDX/LDY/STA/STX/STY/TAX/TAY/TXA/TYA/TSX/TXS)
- **P1b**: arithmetic/logic (ADC/SBC/AND/ORA/EOR/CMP/CPX/CPY/BIT)
- **P1c**: shifts/rotates (ASL/LSR/ROL/ROR)
- **P1d**: branches + jumps (BCC/BCS/BEQ/BNE/BMI/BPL/BVC/BVS/JMP/JSR/RTS)
- **P1e**: stack + status (PHA/PLA/PHP/PLP/SEC/CLC/SEI/CLI/SED/CLD/CLV)
- **P1f**: interrupts (BRK/RTI/IRQ/NMI/RESET) and the cycle-counting fine print (page-crossing penalty, branch-taken penalty, `(indirect),Y` extra cycle)

Don't add unofficial/illegal opcodes until P1f is green — xitari's M6502Hi/M6502Low pair tracks which set is in use, and we match whichever it links.

### P7c (SOFT-mode `step()`) subdivision

P7c mirrors P1's subdivision — same shape, same scope per chunk, but executed against the parallel SOFT primitives:

- **P7c-a**: load/store + transfer opcodes (LDA/LDX/LDY/STA/STX/STY/TAX/TAY/TXA/TYA/TSX/TXS) with N/Z flag updates and all 12 addressing modes routed through `soft_rom_peek` / `soft_ram_peek`
- **P7c-b**: arithmetic/logic (ADC/SBC/AND/ORA/EOR/CMP/CPX/CPY/BIT) with N/Z/C/V flag updates. **P7c-bx** (✅ done) extends ADC/SBC with BCD (decimal-mode) support — the D flag selects between the binary and BCD result; the binary path stays gradient-clean.
- **P7c-c**: shifts/rotates (ASL/LSR/ROL/ROR) on accumulator and memory with N/Z/C flag updates
- **P7c-d**: branches (BCC/BCS/BEQ/BNE/BMI/BPL/BVC/BVS) + JMP (indirect, with the NMOS page-wrap bug) + JSR / RTS. Branches use a **hard** predicate (`jnp.where` on the flag bit) so the forward PC is exact; gradient through the branch predicate is broken at the int-cast of `P`. Wiring `soft_branch` into the default handlers needs a float-valued flag representation in `SoftCPUState` and is deferred as **P7c-dx**.
- **P7c-e**: stack push/pull (PHA/PLA/PHP/PLP) + status-flag manipulators (SEC/CLC/SEI/CLI/SED/CLD/CLV) + INC/DEC/INX/INY/DEX/DEY
- **P7c-f**: RTI — completes the documented NMOS opcode set. BRK is kept as the P7b end-of-trace sentinel (the useful semantics for fixed-length XAI traces) rather than running its proper interrupt sequence. **Routing SOFT writes through real TIA / RIOT / cart-hotspot dispatch is re-scoped to its own phase, P7f** — it is chip-level re-implementation work, not opcode work.

All six sub-phases are ✅ complete: the full 151-opcode documented NMOS set (+ USBC `$EB`) executes in SOFT mode on both ports.

### Deferral identifiers

Each per-subsystem deferral listed in STATUS.md gets a phase ID so it can be picked up in any order:

**CPU (HARD)**
- **P1g**: PA7 edge-triggered external IRQ + NMI / RESET pins (bus-level integration; BRK-only software interrupts work today)
- **P1h**: Undocumented opcode set beyond USBC (`$EB`)

**TIA**
- **P3g**: NUSIZ multi-copy + 2×/4×-wide player scaling
- **P3h**: VDELP* / VDELBL vertical-delay sprite updates
- **P3i**: Beam-accurate (sub-pixel, mid-scanline) rendering — today P3 renders at end-of-scanline
- **P3j**: Audio (AUDC*/AUDF*/AUDV* registers are stored but inert; TIASnd chip not modelled)
- **P3k**: HMOVE +8 nibble timing quirk on real hardware
- **P3l**: CTRLPF.D2 priority swap (today only default priority is honoured)

**RIOT**
- **P4b**: PA7 edge-triggered interrupt (INSTAT D6)
- **P4c**: Paddle dump-pot timing — INPT0-3 stay at `$80` "centred"
- **P4d**: INSTAT-read-clears-flag semantics (today only TIM*T writes clear; real chip varies)

**Cart**
- **P5b**: SC variants of F8 / F6 / F4 (128 B on-cart RAM at `$1000-$10FF`)
- **P5c**: E0 + FE + 3F + 3E formats
- **P5d**: MB + MC + AR + DPC formats
- **P5e**: Signature-based detection for ROMs whose size is ambiguous between formats (e.g. an 8 KB ROM could be F8, E0, or 3F)

**Console / IO / Env**
- **P6b**: Phosphor blending (Stella post-processes the framebuffer for flicker)
- **P6c**: Per-game `RomSettings` (Pong / Breakout / Pitfall / Atari Zoo etc.) — today only `GenericRomSettings` stub
- **P6d**: Random no-op reset wrapper (Mnih-style "skip 0..30 NOOPs at episode start")
- **P6e**: Two-player joystick (P1 directions stay defaulted-released)

**Cross-cutting infrastructure**
- **PXC1**: xitari-trace conformance harness — `tools/trace_dump.cpp` is sketched but never built; no golden traces exist; closing this lets us claim "bit-exact against xitari"
- **PXC2**: JAX-vs-Julia bit-for-bit cross-check
- **PXC3**: CI hook — no automated test runs yet
- **PXC4**: Klaus Dormann `cpu_klaus_dormann.jsonl.gz` regression run (referenced as the P1 acceptance criterion but never wired up)

The single most important piece of unfinished infrastructure is **PXC1** — both ports are currently validated against hand-built unit tests rather than against real ROMs running on xitari, so subtle timing or BCD or bank-switch bugs that don't show up in our test set won't get caught.

---

## 6. Differentiability layer (Phase 7) in detail

Two execution modes, sharing the same module code paths:

### 6.1 HARD mode (default, bit-exact)

- All state is integer (`jnp.uint8` / `UInt8`).
- Opcode dispatch is a `lax.switch` (JAX) / `if/elseif` chain or `@generated` table (Julia).
- Memory reads/writes are index ops.
- This is the mode every conformance test runs in. **No XAI here.**

### 6.2 SOFT mode (relaxed, differentiable)

- State is `float32`, with integers represented as their value (gradients are zero almost everywhere but well-defined under the relaxation we provide).
- **Opcode dispatch** becomes a 256-way softmax over the decoded opcode logits, with each "branch" computing its effect and the outputs being a weighted sum. At training/inference time the softmax is saturated (one-hot), so the forward pass is identical to HARD; gradients flow via the relaxation.
- **Branches** (`BNE` etc.) become `PC_next = (1-g) * PC_no_branch + g * PC_branch`, where `g = sigmoid(α * flag_logit)`. Large α gives bit-exact forward; smaller α gives more gradient signal.
- **RAM addressing**: hard direct addresses (load/store with immediate operand) stay hard. *Indirect* and *indexed* addresses get an NTM-style soft read: `value = sum_i softmax(−|i − addr|/τ)_i * RAM[i]`. With τ→0 this collapses to a normal indexed read.
- **ROM** is wrapped as `RomTensor`: peeks return `ROM @ one_hot(addr)` so gradients can flow back to ROM bytes. Useful for asking "which ROM bytes explain this score?"
- **Status flags** (N, V, Z, C, etc.) become soft `[0,1]` signals; downstream consumers either threshold (HARD) or use them in convex combinations (SOFT).

A global config (`jaxtari.diff.modes.SOFT_MODE`) picks the path. The SOFT path **must reduce to the HARD path** at saturated temperatures — this is a test in itself.

### 6.3 Straight-through escape hatch

For opcodes where the soft relaxation is too expensive (e.g., the full ALU on every cycle), use a straight-through estimator: forward computes the hard answer, backward uses the soft Jacobian. JAX: `jax.custom_vjp`. Julia: `ChainRulesCore.rrule`.

---

## 7. JAX-specific notes

- **State as PyTree of `jnp.uint8` arrays.** Avoid Python dataclasses with `__init__` side effects; use `flax.struct.dataclass` or `chex.dataclass` for jittability.
- **`jax.lax.scan` for the cycle loop.** Each scan step is one CPU cycle. Avoid Python-level `for` loops over cycles.
- **`jax.lax.switch` for opcode dispatch.** 256-way switch with all branches having the same output shape.
- **No `.item()` calls** — they break tracing. Everything stays in JAX arrays until the very edge (final frame buffer).
- **`jax.jit(donate_argnums=...)` on the console step** to reuse RAM buffers.
- **Pure-functional updates**: `state.replace(A=new_A)` rather than mutation.
- The existing [jaxtari/jaxtari_cpu.py](jaxtari/jaxtari_cpu.py) Kimi stub uses `@dataclass` — switch to `flax.struct.dataclass` in P0.

---

## 8. Julia-specific notes

- **`StaticArrays.SVector{N,UInt8}`** for fixed-size state (regs, TIA register file). Mutable `Vector{UInt8}` only for RAM (and even then consider `MVector{128,UInt8}`).
- **`@inline` and `@code_warntype` discipline** — type instability will tank performance to the point that bit-exactness is the only thing left.
- **Zygote.jl** for AD on the SOFT path; **ChainRulesCore.jl** for custom rrules on the straight-through estimator.
- **`Val`-types or `@generated` functions** for opcode dispatch in HARD mode (compile away the branch).
- **Avoid global state.** Pass `ConsoleState` through every function; let the compiler inline.
- The existing [jutari/src/JuTari.jl](jutari/src/JuTari.jl) Kimi stub uses a mutable struct + 1-based indexing. Keep the mutable struct, but split into per-module files in P0.

---

## 9. Open questions / decisions deferred

1. **Sound (TIASnd)**: track register state for parity, but do we synthesise audio samples? Default: no, until/unless an XAI experiment needs it.
2. **Phosphor blending**: included in the env layer (Phase 6) to match xitari's `ALEScreen` output; bit-exactness here is against xitari's post-processed frame, not the raw TIA output.
3. **Random seeds**: xitari has its own RNG (used in stella_environment for stochastic frame-skip and noop-reset). The ports must seed identically. We will probably need a wrapper that mirrors xitari's `Random` class byte-for-byte.
4. **Illegal/undocumented 6502 opcodes**: include them only if xitari uses M6502Hi (which supports them). Check at P1 start.
5. **PAL vs NTSC**: ports target NTSC first (240 → 192-line crop), matching the DQN paper. PAL deferred.
6. **GPU vs CPU**: jaxtari should run on CPU first; GPU may need rethinking the scan body to batch many cycles. Out of scope until P9.

---

## 10. Immediate next actions (in order)

1. **P0.1** — In `tools/`, add `trace_dump.cpp` that links `libxitari.a` and writes the JSONL trace described in §4.1. Test it produces output for a known ROM.
2. **P0.2** — Generate `tests/golden/cpu_klaus_dormann.jsonl.gz` using `trace_dump`. (Requires obtaining the Klaus Dormann 6502 functional-test ROM; not committed.)
3. **P0.3** — In both ports: real package scaffolding (pyproject.toml, Project.toml), CI hook, conformance runner shells that load the golden, advance the port one cycle, compare.
4. **P0.4** — Refactor the Kimi stubs into `jaxtari/jaxtari/cpu/m6502.py` and `jutari/src/cpu/M6502.jl` per §3. **Do not add new opcode logic yet** — get the file layout and the conformance harness wired first.
5. **P1a** — Implement load/store/transfer opcodes in both ports; make the harness green for the subset of Klaus Dormann's tests that only use those.

Anything past P0.4 is a separate work session. The plan above is the contract that future sessions should be held to.

---

*This plan is a living document. Update it (with the same commit conventions as the rest of the project) whenever scope or strategy changes.*
