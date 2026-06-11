# Understanding VCS: XAI for the Atari Simulator via Differentiable Emulation

## Project Overview

This project asks: **can modern XAI methods produce a hierarchical, mechanistic understanding of the Atari 2600 VCS — a system whose ground truth we fully possess?** It is directly inspired by Jonas & Kording's "Could a Neuroscientist Understand a Microprocessor?" (2017), but inverts the usual XAI experiment.

The conventional approach uses XAI to probe a black-box neural network (e.g., a DQN agent). This project flips the target: **the simulator itself becomes the XAI subject**. The DQN is, at most, a behaviour policy that drives the simulator into interesting states — we are not trying to explain the DQN.

To make XAI methods applicable to a classical hand-written C++ emulator, we are building **two differentiable, end-to-end ports of xitari** — one in **JAX (Python)** and one in **Julia**. Once the simulator is differentiable, gradient-based attribution, concept probing, mechanistic interpretability, and ablation analyses can be turned on it directly. Because we already know the true hierarchy (6507 CPU → TIA → RIOT → cartridge bank-switching → console), every XAI claim is testable against ground truth.

### Conceptual framing of the differentiable simulator

- **ROM as a hardwired neural network.** A cartridge ROM is a fixed bit pattern that the CPU "executes." Read as a tensor, it is the weight matrix of a network whose forward pass is one machine cycle. Backpropagating into ROM bytes asks "which bits explain this pixel?"
- **RAM as a Neural-Turing-Machine-like tape.** The 128 B of VCS RAM is small enough to carry as a differentiable state vector with soft (attention-style) read/write addressing. Unlike a vanilla NTM, the addressing is sometimes hard (direct, indexed) — we relax those into convex combinations only where gradients must flow.
- **Branches and case-selects as soft switches.** `if flag then PC+=offset` and opcode-indexed dispatch become gated mixtures (softmax / Gumbel-softmax / straight-through estimators). Forward behaviour stays bit-exact when the gates are saturated; gradients flow when they are relaxed.

This is the lever that lets us aim XAI tools at a system we already understand bit-for-bit.

---

## Current status

Both ports run a full VCS — CPU + Bus + RAM + TIA + RIOT + cart (2K, 4K, F8, F6, F4, F8SC, F6SC, F4SC, E0) — wrapped in an ALE-style `StellaEnvironment` (`reset` / `step(action)` / `get_screen` / `get_ram`). All **151 documented NMOS 6502 opcodes** + USBC + **37 common undocumented opcodes** (NOP/LAX/SAX) are implemented in both ports and in both execution modes (**HARD** = bit-exact uint8 dispatch, **SOFT** = differentiable float32 dispatch). The TIA renders playfield, both player sprites with NUSIZ multi-copy + 2×/4× scaling + VDELP shadow, both missiles, the ball with VDELBL, and all 8 collision latches; drives the framebuffer through VSYNC / VBLANK / VBLANK-output-blanking; supports HMOVE positioning + the floating-bus quirk on TIA reads. RIOT has the timer (4 prescalers, with INTIM-read-clears-flag P4d semantics) + I/O ports (SWCHA/SWCHB + DDRs, 2-player joystick wiring). Differentiability primitives (`RomTensor`, `soft_select`, `soft_memory_read`, `soft_branch`, straight-through round/clamp) are integrated end-to-end via `soft_step` / `soft_run` / `soft_run_scan`, with full Zygote support in jutari via the functional `_FUNC_HANDLERS` table.

**~1950 tests green** across the two ports (jaxtari ~780 + jutari ~1170). **PXC2** (jaxtari ≡ jutari cross-check) covers 6 ROMs — pong, breakout, space invaders, pitfall, seaquest, enduro — with byte-identical RAM between the two ports at every frame. **PXC1** (jaxtari↔xitari RAM, `*_noop_10` last frame): **pong, breakout, space_invaders, pitfall, enduro bit-exact (0 bytes)** (pitfall via task #81's `getStartingActions` fix; enduro NEWLY bit-exact via task #84's GRP* defer fix — was 8 b/f), seaquest 4. **PXC-S** (a new test the P3i-g pt4 work added — per-frame *screen* diff vs xitari): breakout **0 px (BIT-EXACT)**, **pong 24** (was 32 — task #84 GRP* defer eliminated the paddle 1-row-early bug; remaining 24 = 8 row-0 HMOVE comb [#83] + 16 rows 35-37 phantom score sprites [#85]), space_invaders **12** (near bit-exact), pitfall 322, seaquest 1104, enduro 1197. See [bug_fix_log.md](bug_fix_log.md) for the scoreboards and the running history of fixes.

For the per-phase commit ledger, what each port can do today, and the complete list of deferrals, see **[STATUS.md](STATUS.md)**. For the design rationale and the still-pending phase plan, see **[PORTING_PLAN.md](PORTING_PLAN.md)**. For the **active 5-phase deep-dive** that closes the remaining open bugs (pong residuals, breakout ball-doesn't-die, jaxtari pong freeze) via precise CPU↔TIA cycle threading, see **[P3I_G_THREADING_PLAN.md](P3I_G_THREADING_PLAN.md)** — that doc names the failure modes of the two prior reverted attempts and the discipline that keeps the third one from breaking again.

> **Agents working on emulation / conformance:** read **[bug_fix_log.md](bug_fix_log.md)** first — it's the running history of bugs hunted, patches landed, dead-ends ruled out (e.g. the `bisect.py` stdlib-shadow gotcha), and ideas still open (e.g. enduro collision-timing convergence). **Append to it** whenever you fix a bug or rule out a hypothesis, so the next agent inherits the context. It also carries the live jaxtari↔xitari conformance scoreboard. For the active CPU↔TIA cycle-threading work-arc, **[P3I_G_THREADING_PLAN.md](P3I_G_THREADING_PLAN.md)** is the technical plan; bug_fix_log is its running journal.

---

## Hand-off — pick up here

**Big session 2026-06-07**: closed task #75 (jutari TIM*T-load timing
drift / "jumping scanlines"), task #73 (jaxtari pong $04/$3c residual /
SwapPaddles=YES paddle resistance routing), task #77 (pong
$3f/$40 swap / SwapPaddles=YES FIRE button routing — both ports),
task #76 (jutari breakout auto-reset / RomSettings terminal latch),
and task #78 (pong score addresses **wrong in both ports** —
PongRomSettings was reading RAM[$14]/[$15] but xitari uses RAM[$0D]/[$0E];
sprite-pattern bytes at $15 hit 0x82=130 within 60 frames of FIRE,
falsely triggering env.terminal and freezing the paddle).

Concrete wins:
  - **jutari↔xitari NOOP-300 RAM**: pong / breakout / space_invaders
    bit-exact across 300 frames (was just 10).
  - **PXC1 jutari pong + breakout** noop-10 now bit-exact (was
    @test_broken; harness wasn't passing the right per-ROM RomSettings).
  - **PXC1 jaxtari pong + breakout** noop-10 now bit-exact too (same
    TIM*T-load + SwapPaddles=YES fixes mirrored to jaxtari).
  - **Breakout video alignment** (jutari vs xitari, full 3600 frames):
    11.6% → 16.3% pixel-exact; gating to actual gameplay (frames 0-596
    before lives → 0), 99.7% pixel-exact post-#76.
  - **Pong env.terminal** (post-#78): no longer false-triggers on
    RAM[$15]=130 sprite-pattern reads; jutari pong paddle responds to
    LEFT/RIGHT across the full 200-frame smoke test (was frozen after
    ~60 frames). jutari pong screen diff vs xitari = ~32 px/frame
    (mean), 0.1% of pixels — paddle moves, residual is row-0 HMOVE-comb
    + score-digit color (filed as background tasks).

**Earlier-known user-visible bugs were action-driven** (mostly closed
by the SwapPaddles=YES FIRE routing + the TIM*T-load timing fix):

1. ~~Pong freezes within ~100 frames~~ — jaxtari's paddle resistance
   was wired to INPT0 instead of INPT1, freezing the user paddle.
   **Closed by task #73 (commit `de00af8` + `c3d6d42`)**.
2. ~~Breakout ball doesn't die~~ — closed by the breakout VDELBL fix
   (commit `20b5de0` + `a418e4c`). Subsequent TIM*T-load timing fix
   (commit `4ddb0b7`) brought breakout pixel-accuracy from 11.6%
   to 16.3% (98.3% on live gameplay).

### Big new finding (2026-06-02): jutari is much closer to xitari than jaxtari is

A 3-way per-frame RAM diff on pong + the seed-42 random action stream
(`tools/pong_3way_ram_diff.py`):

| Comparison       | Bytes diff / frame (typical) | After Phase 1c (jutari) |
|------------------|---|---|
| **jutari ↔ xitari**  | **4**           | **<1** (only frame-20 $3f/$40 swap remains) |
| jaxtari ↔ xitari | 15–18           | unchanged (jaxtari side not yet refined) |
| jaxtari ↔ jutari | 12–16           | unchanged (same reason) |

**jutari (Julia port) tracks xitari almost exactly. jaxtari (JAX port) does not.** PXC2 misses this because PXC2 only tests NOOP fixtures (where both ports happen to agree with xitari and each other). Under random actions, the two ports diverge — and the bug is in **jaxtari**, not jutari.

**Update 2026-06-02 (later)**: Phase 1c of [P3I_G_THREADING_PLAN.md](P3I_G_THREADING_PLAN.md) refined the collision catch-up endpoint to include `pending_tia_cycles * 3` (matching xitari's `mySystem->cycles()` POST-increment semantic from `M6502High::peek`). Measured via `tools/jutari_xitari_ram_diff.py`: the jutari↔xitari pong RAM gap dropped from **~4 bytes/frame typical** to **<1 byte/frame** (mean ~0, max 3 across 600 frames), with the only persistent residual being the documented frame-20 `$3f`/`$40` swap. **The pong RAM deep-dive target is effectively closed.** However, the pong SCREEN residual is unchanged at ~30 px/frame — the visual bugs are renderer-only (HMOVE-blank misfire on scanline 34, sprite Y off-by-1 on rows 35-37/149), orthogonal to CPU-cycle threading.

**Confirmed via a parallel pong-progresses test** (`/tmp/pong_jutari_progress.jl` runtime + `/tmp/pong_screen_random.py` runtime): jutari pong's screen progresses cleanly across frames (72–344 px diff between sample frames), while jaxtari's screen **freezes** within ~100 frames.

#### Concrete frame-by-frame data

  - **Frames 0–58**: jaxtari ≡ jutari, byte-identical. **Both** ports diverge from xitari at frame 20 (FIRE) by 2 bytes (`$3f`, `$40` — values 0xc0/0x00 are swapped between the ports and xitari). This is a shared bug to fix later.
  - **Frame 59** (action=4=LEFT): **jaxtari first diverges from jutari** at 6 bytes (`$04`, `$08`, `$0c`, `$31`, `$36`, `$3b`). At this single frame:
    - byte `$04`: jx=0x6e, ju=0x72, xi=0x6e (jaxtari happens to match xitari).
    - other 5 bytes: jaxtari differs from BOTH ju and xi (mostly off-by-1).
  - From frame 59 onwards jaxtari drifts further while jutari stays within 4 bytes of xitari.

#### The single highest-leverage next move

Find the **specific emulator-core difference** between `jaxtari/jaxtari/*` and `jutari/src/*` that activates at frame 59. Both ports share the same high-level `apply_action` / `_apply_paddle_action` / SWCHA paddle wiring (just verified by reading both files). So the divergence is at a lower level — CPU instruction cycle accounting, TIA peek timing, RIOT timer ticking, or bus state propagation. Concrete recipe:

  1. Re-run the probe: `cd jaxtari && .venv/bin/python ../tools/pong_3way_ram_diff.py`. Refresh the jutari RAM trace with the command in the probe's docstring if needed.
  2. Instrument both ports to dump the CPU+TIA+RIOT state at frame 58 (last identical frame). Diff the snapshots — RAM matches, but `total_cycles`, `dump_disabled_cycle`, `paddle_resistance`, etc. may not. Whichever non-RAM field differs is the entry point.
  3. Trace forward through the frame-59 step's bus-poke sequence; the first poke that writes a different RAM byte in jaxtari vs jutari pinpoints the responsible instruction stream.
  4. Fix jaxtari to match jutari's behavior at that point. Mirror to jutari only if jutari turns out to be the wrong one (unlikely given the 4-byte vs 15-18-byte gap).
  5. Then continue on the jutari path: find the remaining 4-byte/frame jutari↔xitari residual (the frame-20 FIRE swap of `$3f`/`$40` is the lead).

#### Deeper look at the frame-20 FIRE divergence (both ports vs xitari)

The shared bug at frame 20 — both jaxtari and jutari diverge from xitari at exactly 2 bytes (`$3f`/`$40`, values swapped) — was investigated this session via `tools/jutari_pong_fire_probe.jl` + `tools/pong_fire_full_compare.py` (just persisted). Findings:

  - **Pong stores to BOTH `$3f` AND `$40` every frame** (steady-state value 0xc0; pokes are ~40 cycles apart per frame). At the FIRE frame, exactly ONE of them flips 0xc0→0x00.
  - **xitari clears `$40`** (keeps `$3f`=0xc0). **My ports clear `$3f`** (keep `$40`=0xc0). Adjacent addresses, off by 1.
  - The instruction is a `STA <addr>` with value 0x00 — data_bus_state at the moment of poke = 0x00, so A = 0x00 was loaded then stored.
  - Pong's CPU writes the same VALUE (0x00) to a different ADDRESS — strongly suggests **indexed addressing** (`STA $XX,X` or `STA $YY,Y`) where the index register holds a different value in my ports vs xitari at the STA moment.
  - Both ports' SWCHA at frame 20 = 0x7F (bit 7 cleared = P0 FIRE pressed). This matches xitari's expected value with `SwapPaddles=YES` wiring per `xitari/emucore/Paddles.cxx`. So SWCHA is NOT the divergence source.
  - INPT0/INPT1 dump-pot timing: both ports compute the same threshold (`needed = 7779` cycles for paddle_resistance = 408823). Both reach D7-flip at the same cycle. So INPT polling result is the same.
  - INSTAT: my port returns 0x80 if `timer_expired` else 0x00. xitari has a more nuanced formula involving `myTimerReadAfterInterrupt`, but my port's INTIM-read-clears-timer_expired gives an equivalent end-state. Not the smoking gun (yet).
  - Likely culprits still on the suspect list: **CXxx (collision register) read at the FIRE moment** (collision detection cycle precision could differ between ports), or some other floating-bus / data-bus state difference at the specific instruction that loads the index register.

  **Concrete next probe** (next agent or session):
  1. Add PC tracking to the bus poke trace in `tools/jutari_pong_fire_probe.jl` (modify the temporary debug line in `jutari/src/bus/Bus.jl::poke!` to capture `bus.tia.total_cycles` + an externally-passed CPU PC). Find the exact CPU PC of the STA that writes to `$3f` at frame 20.
  2. Disassemble pong.bin around that PC (e.g., `tools/trace_dump --decode` if available, or use a 6502 disassembler on the ROM). Identify the addressing mode and what register the index comes from.
  3. Trace backwards from that LDA/LDX/LDY to find which TIA/RIOT register read produces the divergent index value.
  4. Fix that read in jutari (it's likely a peek formula off-by-one similar to pt8 INTIM). Mirror to jaxtari.

### Test gates (parallel by default after commit `dd317b6`)

```bash
# jaxtari (~22 min in parallel; -n auto picks all cores)
cd jaxtari && .venv/bin/python -m pytest -q

# jutari (~20 s)
cd jutari && julia --project=. -e 'using Pkg; Pkg.test()'

# Quick smoke: PXC1+PXC2 only (~22 min parallel)
cd jaxtari && .venv/bin/python -m pytest tests/test_pxc1_conformance.py tests/test_pxc2_jaxtari_vs_jutari.py -q

# Override default parallelism with -n 1 for serial debugging.
```

### Bug-bisection methodology — read this before chasing divergences

[BUG_BISECTION_METHODOLOGY.md](BUG_BISECTION_METHODOLOGY.md) — the
per-bus-op trace technique that closed breakout (commit `20b5de0`).
Decision tree, tooling pointers, fix patterns, anti-patterns. Use
it for any new RAM-divergence investigation.

### jutari↔xitari RAM conformance scoreboard (2026-06-07, post-INTIM-fix)

300 frames of NOOP (`jutari_xitari_ram_diff.py`):

| ROM            | Mean b/f | Max b/f | Status   |
|----------------|----------|---------|----------|
| pong           | 0.0      | 0       | ✅ bit-exact (300 frames) |
| breakout       | 0.0      | 0       | ✅ bit-exact (300 frames, was 9.9 b/f before 20b5de0) |
| space_invaders | 0.0      | 0       | ✅ bit-exact (300 frames) |
| asteroids      | 0.0      | 0       | ✅ bit-exact |
| seaquest       | 5.5      | 18      | next target (was 2.6 over 100f) |
| pitfall        | 18.5     | 21      | next target (was 19.8 over 100f) |
| enduro         | 52.7     | 67      | next target (was 45.0 over 100f) |

Plus, post-INTIM-fix (commit `4ddb0b7`) + auto-reset fix
(commit `037526c`), **breakout video alignment** under random actions:

| measurement              | pre-fix     | post-fix         |
|--------------------------|-------------|------------------|
| 3600-frame pixel-exact   | 11.6%       | **99.7%** (3590/3600) |
| first-life (0..596)      | variable    | **98.3%** (587/597) |

The remaining 10 mismatched frames are 1-2-pixel ball-paddle bounce
sub-cycle artifacts (pairs every 120 frames = once per ball-death
cycle).

### Recent commits worth knowing about

  - **🏆 Breakout ball-doesn't-die FIXED (2026-06-03, commit
    `20b5de0` jutari + `a418e4c` jaxtari)**:
    - **jutari↔xitari breakout RAM: 9.9 b/f → 0.0 b/f
      (BIT-EXACT, 300 frames)**.
    - **Ball lives counter (`RAM[$39]`) now decrements every
      ~120 frames** matching xitari (5→4→3→2→1→0 at frames
      117/237/357/477/597). Was decrementing only ONCE before.
    - Root cause: ENABL writes are deferred (P3i-g pt6), but the
      VDELBL shadow latch fires synchronously at GRP1 writes —
      so the live ENABL register was stale at shadow-capture
      time, retaining "ball enabled" across what should have been
      a clear. Fix drains pending ENABL writes whose
      activation_clock ≤ current beam_cc at GRP1 poke time, so
      the shadow captures the correct effective value.
    - Discovery via per-bus-op xitari trace (commit `d66b290`)
      + temporary `_BLPF_LOG`/`_ENABL_OLD_DEBUG` instrumentation
      — bisected to scn 51-52 of frame 92 then to the GRP1
      shadow-capture path. Full story in bug_fix_log.md.
  - **Phase 2b of [P3I_G_THREADING_PLAN.md](P3I_G_THREADING_PLAN.md)
    landed (2026-06-02, both ports — major progress)**:
    - `2aec8c5` jutari + `a8f2bd7` jaxtari: per-opcode cycle-counter
      validation. `bus.pending_tia_cycles` now sums to
      `CYCLE_TABLE[opcode]` for ALL 189 opcodes on both ports.
      Validated by new `scratch_cycle_audit.{jl,py}` test runners.
    - **Measured pong RAM bit-exactness**:
      - jutari↔xitari: 4 → **0.0 b/f** (frames 0-300 bit-exact except
        well-known frame-20 FIRE shared bug).
      - jaxtari↔xitari: 13 → **4 b/f** (frames 0-23 bit-exact, frame
        24+ has residual 4 b/f at $04/$3c — Phase 5 candidate).
    - **CYCLE_TABLE bug fix**: 0xA7 LAX zp was 4 on both ports, fixed
      to 3 (matches xitari + NMOS reference).
  - **Phase 1 of [P3I_G_THREADING_PLAN.md](P3I_G_THREADING_PLAN.md)
    landed (2026-06-02, this session)**:
    - `fb72495` — jutari Bus trace tap + `tools/cpu_tia_cycle_trace.jl`
      (per-bus-op CSV diagnostic, zero-cost when disabled).
    - `6cdc99a` — `tools/cycle_trace_inspect.py` query language
      (scanline / poke-trace / hmove / diff / summary subcommands).
    - `408c516`+`adcacc2` — collision catch-up endpoint now uses
      sub-instruction `effective_cc` (xitari `M6502High::peek`
      semantic). Closes jutari↔xitari pong RAM gap **4 → <1
      bytes/frame** (max 3, mean ~0 across 600 frames). jaxtari
      partial (13/frame, refinement is in the wrong code path
      for its residual).
    - `0769a5e` — `tools/jutari_xitari_ram_diff.py` general per-frame
      RAM diff tool. Used to measure the Phase 1c impact + identify
      breakout's first-divergence frame (frame 92, 6 bytes at $37
      $5f $61 $65 $67 $6c).
  - **Task #66 (`8531bb8`)** — jutari pong paddles MOVE (SwapPaddles
    routes user paddle to INPT1).
  - `1c96314` — refined handoff in `bug_fix_log.md` with the freeze diagnosis (pong stops within ~100 frames, not just static paddles).
  - `dd317b6` — pytest-xdist `-n auto` default: 1 h 41 m → 22 m on the PXC sweep.
  - `c016087` — Task #65 paddle games skip SWCHA in `apply_action`.
  - `15cff40` — pt8 RIOT INTIM `-1` fix (74 % screen-residual drop across 5 ROMs).
  - Earlier pt5–pt7: SCOREMODE, ENAM/NUSIZ/COLU/CTRLPF defer.

### Open work — pick up here on the next session

  1. **Pong renderer-only bugs** (32 px/frame screen residual on
     jutari, RAM now bit-exact post-Phase-2b). The 3 reproducible
     bug sites — HMOVE-blank misfire on scanline 34, sprite Y
     off-by-1 on rows 35-37/149 — live in `render_pixel` /
     `render_scanline` / the HMOVE-blank state machine. Orthogonal
     to P3I_G_THREADING_PLAN.md cycle threading.
  2. **Breakout ball-doesn't-die — FIXED** (2026-06-03,
     commit `20b5de0`). jutari↔xitari RAM bit-exact, ball lives
     counter decrements every ~120 frames. Closed. ✅
  3. **jaxtari pong 4 b/f residual at $04/$3c (frame 24+)** — same
     bytes flagged in bug_fix_log's "frame-1 with LEFT" finding,
     now suppressed to frame 24 thanks to the cycle counter fix.
     A specific INPT/floating-bus issue jaxtari has and jutari does
     not. **Phase 5** of P3I_G_THREADING_PLAN.md or further
     investigation of `bus.system.peek`'s noise-OR semantics in
     jaxtari would close this.

### Uncommitted experimental edits left in the working tree

`jaxtari/jaxtari/io/action.py` + `jutari/src/io/IO.jl` have a pending diff (paddle FIRE routed to SWCHA bits 6/7 instead of INPT4 — xitari Paddles controller wiring per `xitari/emucore/Paddles.cxx`). 67/67 PXC + RIOT + bus tests still pass with the diff applied; pong screen still freezes (so the diff doesn't fix the visible bug). Next agent to decide whether to commit, revert, or leave as a stash. See `bug_fix_log.md` "Where we left off" section for the full context. **NB**: those edits are on a separate machine — current main is clean.

---

## Repository Structure

```
UnderstandingVCS/
├── README.md                  # This file
├── PORTING_PLAN.md            # Phase plan + design rationale
├── STATUS.md                  # Per-phase commit/test/deferral ledger
├── bug_fix_log.md             # Running bug/patch history + open debugging ideas (agents: read & update)
├── P3I_G_THREADING_PLAN.md    # Active 5-phase plan for CPU↔TIA cycle threading (closes pong/breakout residuals)
├── .gitignore             # Excludes papers/, dqn/, xitari/ (external deps) and .DS_Store
├── literature/            # AI-readable markdown versions of papers with BibTeX (13 papers)
├── jaxtari/               # JAX port — see jaxtari/README.md
├── jutari/                # Julia port — see jutari/README.md
├── tools/                 # trace_dump.cpp sketch (xitari conformance helper, not built yet)
├── papers/                # PDF downloads (excluded from git, reproducible via DOIs)
├── dqn/                   # DeepMind DQN repository clone (excluded — used as black-box agent)
└── xitari/                # DeepMind Xitari (ALE fork) — the bit-exact reference (excluded)
```

`xitari/`, `dqn/`, and `papers/` are external dependencies, cloned/downloaded locally and not version-controlled here. `jaxtari/` and `jutari/` are the primary deliverables of this project and **are** version-controlled.

---

## Rules / Working Setup

### Developer Log Book via Commits

**Every command, change, or action performed in this project is committed and pushed to GitHub immediately after each turn.** The commit history serves as a complete developer log.

### Commit Message Format

Each commit message MUST include:
- **The full user prompt** that triggered the changes
- **The AI model used** (e.g., `claude-opus-4-7[1m]`)
- **A concise summary** of what was changed and why

This ensures reproducibility and full traceability.

### Literature Management

1. **papers/**: Downloaded PDFs. Excluded from git (binary, reproducible via URLs).
2. **literature/**: Markdown conversions with full text and a closing BibTeX block. Tracked in git.

### External Dependencies

The following live locally but are **excluded from git**:
- `dqn/` — DeepMind's DQN implementation (Lua/Torch). Used here as an *optional* black-box action source.
- `xitari/` — DeepMind's ALE fork. The **reference oracle** that the differentiable ports must match cycle-by-cycle.

---

## Key Papers and Their Role

### 1. Could a Neuroscientist Understand a Microprocessor? (Jonas & Kording, 2017)

**DOI**: [10.1371/journal.pcbi.1005268](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005268)

**Core insight**: Standard neuroscience methods applied to a fully simulated MOS 6502 — with perfect data and ground truth — failed to recover the processor's hierarchical organisation. The bottleneck was the methods, not the data.

**Relevance**: This is our foundational paper. Where Jonas & Kording asked whether neuroscience methods could understand a microprocessor running an Atari game, we ask whether **XAI methods can understand the whole VCS** (CPU + TIA + RIOT + cart + ROM) once we expose it through gradients. The 6507 / VCS is our "known artifact." Any failure of XAI here is informative regardless of outcome.

### 2. Human-level Control through Deep Reinforcement Learning (Mnih et al., 2015)

**DOI**: [10.1038/nature14236](https://www.nature.com/articles/nature14236) · **arXiv**: [1312.5602](https://arxiv.org/abs/1312.5602)

**Relevance to the *new* project goal**: The DQN is **not** the XAI target. We use it (or any other agent) as a *policy* that drives the simulator into game-relevant trajectories, so that the XAI we run on the simulator is conditioned on realistic state distributions rather than random play. The original "explain the DQN" framing is preserved in the bibliography for context, but is no longer the research question.

---

## DQN and Xitari Breakdown

### DQN (Deep Q-Network)

**Source**: `https://github.com/google-deepmind/dqn`. Lua/Torch implementation of the Nature 2015 DQN. Treated here as a black-box action source. We do not modify or instrument it.

### Xitari (Arcade Learning Environment fork)

**Source**: `https://github.com/google-deepmind/xitari`. C++ Atari 2600 emulator based on Stella, with an ALE-style RL interface.

Top-level layout that the ports mirror:

| xitari path | Role | Port target |
|---|---|---|
| `emucore/m6502/src/` | 6502/6507 CPU (M6502, M6502Hi/Low, System) | `jaxtari/cpu/`, `jutari/src/cpu/` |
| `emucore/TIA.{cxx,hxx}` | Television Interface Adapter (video + audio) | `jaxtari/tia/`, `jutari/src/tia/` |
| `emucore/M6532.{cxx,hxx}` | RIOT: 128 B RAM + I/O + timer | `jaxtari/riot/`, `jutari/src/riot/` |
| `emucore/Cart*.{cxx,hxx}` | Cartridge / bank-switching types | `jaxtari/cart/`, `jutari/src/cart/` |
| `emucore/Console.{cxx,hxx}` | Top-level VCS console wiring | `jaxtari/console.py`, `jutari/src/Console.jl` |
| `emucore/Control.cxx`, `Joystick`, `Paddles`, `Switches` | Controllers + console switches | `jaxtari/io/`, `jutari/src/io/` |
| `environment/` | ALE-style RL wrapper, phosphor blend | `jaxtari/env/`, `jutari/src/env/` |
| `games/` | Per-game ROM scoring/termination rules | `jaxtari/games/`, `jutari/src/games/` |
| `controllers/` | Agent IPC (fifo, internal, rlglue) | Out of scope for ports (use direct API) |
| `agents/` | Sample agents | Out of scope for ports |

The full module-by-module plan, including the differentiability layer, the bit-exact test harness against xitari, and milestone phasing, is in **[PORTING_PLAN.md](PORTING_PLAN.md)**.

### Why Xitari is the right reference

Xitari is deterministic given a ROM, an action stream, and an RNG seed. That makes it a perfect oracle: every CPU register, every RAM byte, every TIA register, and every frame buffer the JAX/Julia ports produce can be compared byte-for-byte against xitari's trace for the same inputs. Disagreement is unambiguously a port bug.

---

## Project Plan and Hypotheses

### Core Hypothesis

> **A complete, differentiable port of the Atari VCS — where ROMs act as weights, RAM acts as a soft tape, and control flow acts as gated switches — lets us turn modern XAI methods on a system whose true mechanism is fully known. The mismatch between XAI explanations and the verified ground-truth hierarchy will quantify the methods' limits more sharply than any biological or neural-network target can.**

### Specific Questions

1. Can gradient-based attribution (Integrated Gradients, Grad-CAM analogues) localise the ROM bytes / RAM cells / TIA registers that "explain" a given pixel or score change?
2. Do concept probes (CAVs, network dissection analogues) recover known hardware concepts — bank-switch bits, sprite registers, collision latches, timer reloads?
3. Does mechanistic / circuit-style interpretability reconstruct the true hierarchy (instruction fetch → decode → ALU → bus → TIA/RIOT → frame) when the gates are relaxed?
4. How do the JAX and Julia ports compare on (a) throughput vs. xitari and (b) gradient-pass cost? Are the two languages' AD systems equivalent for this workload?
5. Where do soft-switch relaxations leak — i.e., for which instructions/branches does the differentiable version diverge from the bit-exact reference, and by how much?

### Methodology

1. **Port xitari to JAX and Julia, bit-exactly**, validated against per-cycle xitari traces. (See PORTING_PLAN.md.)
2. **Layer differentiability on top**, with a flag that toggles "hard / bit-exact" mode vs. "soft / differentiable" mode.
3. **Drive trajectories** with either random play, scripted actions, or a pre-trained DQN.
4. **Apply XAI methods** to ROM bytes, RAM cells, TIA registers, and intermediate CPU state.
5. **Compare to ground truth**: 6502 datasheet, TIA documentation, disassembly of the target ROM, known cartridge bank-switching schemes.

### Expected Outcomes

- **Best case**: XAI methods cleanly recover the documented VCS hierarchy on at least one game, validating the methods on a transparent target.
- **Likely case**: Methods recover *parts* of the hierarchy (e.g., correctly attribute a sprite pixel to a TIA player register) but miss higher-level structure (e.g., the game's score logic in ROM).
- **Worst / most informative case**: XAI produces confident, plausible, but **wrong** explanations on a system we can verify — a stronger version of Jonas & Kording's negative result.

---

## Bibliography

```bibtex
@article{Jonas2017Could,
  author    = {Jonas, Eric and Kording, Konrad Paul},
  title     = {Could a Neuroscientist Understand a Microprocessor?},
  journal   = {PLOS Computational Biology},
  year      = {2017},
  volume    = {13},
  number    = {1},
  pages     = {e1005268},
  doi       = {10.1371/journal.pcbi.1005268},
  issn      = {1553-7358},
  publisher = {Public Library of Science},
  url       = {https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005268}
}

@article{Mnih2015Human,
  author    = {Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and Rusu, Andrei A. and Veness, Joel and Bellemare, Marc G. and Graves, Alex and Riedmiller, Martin and Fidjeland, Andreas K. and Ostrovski, Georg and Petersen, Stig and Beattie, Charles and Sadik, Amir and Antonoglou, Ioannis and King, Helen and Kumaran, Dharshan and Wierstra, Daan and Legg, Shane and Hassabis, Demis},
  title     = {Human-level control through deep reinforcement learning},
  journal   = {Nature},
  year      = {2015},
  volume    = {518},
  number    = {7540},
  pages     = {529--533},
  doi       = {10.1038/nature14236},
  publisher = {Nature Publishing Group},
  url       = {https://www.nature.com/articles/nature14236}
}

@article{Bellemare2013Arcade,
  author    = {Bellemare, Marc G. and Naddaf, Yavar and Veness, Joel and Bowling, Michael},
  title     = {The {Arcade} {Learning} {Environment}: An Evaluation Platform for General Agents},
  journal   = {Journal of Artificial Intelligence Research},
  year      = {2013},
  volume    = {47},
  pages     = {253--279}
}

@article{Graves2014NTM,
  author    = {Graves, Alex and Wayne, Greg and Danihelka, Ivo},
  title     = {Neural {Turing} {Machines}},
  journal   = {arXiv preprint arXiv:1410.5401},
  year      = {2014},
  url       = {https://arxiv.org/abs/1410.5401}
}
```

---

*This project is developed with AI assistance. Every change is committed and pushed to GitHub with full prompt and model documentation.*
