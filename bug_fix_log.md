# Bug-fix & patch log

> **For agents:** This file is the running history of bugs hunted, patches
> landed, dead-ends ruled out, and ideas still on the table for the two
> differentiable VCS ports (`jaxtari`, `jutari`). **Read it before starting
> emulation/conformance work**, and **append to it** whenever you fix a bug,
> rule out a hypothesis, or discover a gotcha — newest entry on top of the
> "Patches landed" section. It complements (does not duplicate) the
> phase/test ledger in [STATUS.md](STATUS.md): STATUS.md says *what each
> phase delivered*; this file says *what broke, why, and what we tried*.

Keep entries concrete: name the files, the symptom, the root cause, the
measured before/after, and any conformance (PXC) numbers that moved.

---

## Where we left off — pick up here (2026-06-01)

This section is what the **next agent** should read first to know what
to chase. Skip down to "Conformance scoreboard" and "Patches landed"
for the full history. Test infrastructure note: `pyproject.toml`'s
pytest `addopts` is now `-q -n auto --dist worksteal`, so `pytest`
parallelises across all cores by default. Override with `pytest -n 1`
to debug a single test deterministically.

### Active user-reported bugs

**A. Pong FREEZES under random actions** (user, 2026-05-31, refined 2026-06-02).
Reported as "paddles don't move" in `tools/breakout_video/output/pong_xitari_vs_jaxtari.mp4`,
but it's worse than that — the **entire game state freezes** within
~100 frames. Diagnosis:
  - `/tmp/pong_screen_random.py` against the seed-42 random-action
    stream from the video: frames 100, 200, 300, 500, 700, 1000 are
    **byte-identical** in jaxtari. The CPU has fallen into a
    steady-state loop that produces the same screen every frame.
  - xitari on the same action stream: every pair of those frames
    differs by 88-472 px. Game is alive and progressing.
  - PXC1 noop-10 RAM is still bit-exact, so the freeze is **action-
    triggered**. The first FIRE in the random stream is at frame 20.
  - Pong's TIA reads are **INPT1 (paddle 1) and INPT3 (paddle 3)**,
    91 polls each per frame — NOT INPT0/INPT2 (per `/tmp/pong_all_peeks.py`).
    Pong (md5 60e0ea3c…) has `Controller.SwapPaddles = "YES"` in xitari's
    `stella.pro`, so paddle wiring inside the Paddles controller is
    swapped. INPT1 = `m_right_paddle` (= default `408823`, NOT updated
    by single-player ALE.act since `player_b_action = PLAYER_B_NOOP`),
    INPT3 = paddle 3 (= unset event = minimumResistance → returns 0x80
    immediately after VBLANK D7 clear).
  - jaxtari's paddle_resistance plumbing IS wired up correctly:
    `_apply_paddle_action` writes `paddle_resistance[0] = _left_paddle`
    (action-driven) and `paddle_resistance[1] = _right_paddle`
    (default). Same as xitari.
  - With identical INPT1/INPT3 timing inputs and the same ROM, xitari
    progresses and jaxtari freezes. **The divergence is therefore in
    CPU branching driven by some other read** — RIOT timer
    (INTIM/INSTAT), SWCHA (joystick after FIRE), CXxx (collision
    latches), or floating-bus noise on INPT* reads (xitari ORs
    `mySystem->getDataBusState() & 0x3F` into INPT reads; my port
    returns clean `0x80`/`0x00`).
  - Frame-1 RAM with LEFT-only diverges in 2 bytes (`$04`, `$3c`)
    after Task #65's SWCHA-skip fix. After 100 LEFT actions, jaxtari
    changes byte `$3b` but xitari changes `$33` AND `$3c` (the bytes
    the renderer reads). The CPU takes a different STA path.
  - **Concrete next step**: floating-bus noise. xitari's TIA peek for
    every INPT/CX register returns `data | noise` where
    `noise = mySystem->getDataBusState() & 0x3F` (the lower 6 bits of
    whatever the bus was last set to). My port returns the clean bit
    only. If pong reads INPT or a CX register and uses any of the
    lower 6 bits (e.g., stored to RAM, used as table index, branched
    on), my port and xitari differ. Add a `data_bus_state` field to
    `BusState` that tracks the last value written/read on the bus,
    then OR `(data_bus_state & 0x3F)` into every TIA peek that returns
    a sub-8-bit semantic value (CX*, INPT*). High-leverage like pt8 —
    likely also unsticks (B).

**B. Breakout ball doesn't die under random actions** (user, earlier).
Reported in `breakout_xitari_vs_jaxtari.mp4`: ball reaches the bottom
of the screen and continues to bounce, instead of disappearing and
triggering a new round. Diagnosis:
  - RAM byte 57 = breakout lives counter (per `xitari/games/supported/Breakout.cpp`).
  - xitari (random-actions, 600 frames): lives 5→4 at frame 116,
    then 4→3, 3→2, 2→1, 1→0 every ~120 frames; game ends at frame 597.
  - jaxtari: lives 5→4 at frame **241** (very late), then NEVER changes
    again; runs all 600 frames without ending.
  - First RAM divergence: frame 20 — the first `FIRE` in the action
    stream — at offsets [95, 99, 101, 103, 105].
  - Same class as (A): action-driven CPU divergence. Likely the same
    INPT0/INPT4 polling timing residual. Fixing (A) should also fix (B).

**C. Uncommitted experimental edits in working tree** (Task #3):
`jaxtari/jaxtari/io/action.py` + `jutari/src/io/IO.jl` have a pending
change that routes paddle FIRE to SWCHA bits 6/7 (xitari Paddles
controller wiring per `xitari/emucore/Paddles.cxx`) instead of INPT4.
67/67 PXC tests pass with the change in place, but it does NOT fix
(A) — the visible paddle still doesn't move. Decide whether to commit,
revert, or leave as a stash.

### Per-cycle-accuracy class (lower priority)

The remaining PXC-S residuals on `*_noop_10` are concentrated in
score-digit areas and are all the per-cycle bus accuracy class:
  - pong **32** (8 px row-0 cols-0-7 HMOVE/VBLANK quirk + 24 px PF1
    bit-0 timing at rows 35-37/95/111).
  - space_invaders **12** (near bit-exact).
  - pitfall **322**, seaquest **1104**, enduro **1197** (all score-area
    digit-shape differences).
These are unlikely to yield to a single-fix lever like pt5/pt6/pt7/pt8
did; they need broader per-cycle bus accuracy work (store-mode
`abs,X`/`abs,Y`/`(zp),Y` always-dummy-on-store, RES* defer, etc.).

### Quickstart for the next agent

```bash
# Make sure tests run parallel (already configured in pyproject.toml):
cd jaxtari && .venv/bin/python -m pytest -q                # all tests, ~20 min
.venv/bin/python -m pytest tests/test_pxc1_conformance.py tests/test_pxc2_jaxtari_vs_jutari.py -q  # 22 min
.venv/bin/python -m pytest tests/test_screen_conformance.py -q   # ~5 min

# Measure all 6 ROMs' PXC-S worst frames at once (no pin):
.venv/bin/python /tmp/all_pxc_s.py   # if /tmp probes are gone, regenerate from the pt7/pt8 commit messages

# Reproduce the pong-paddle-doesn't-move bug:
.venv/bin/python /tmp/pong_screen_compare.py     # NOOP vs LEFT vs RIGHT at frame 100; should differ but shows 0 px

# Side-by-side comparison videos:
.venv/bin/python tools/breakout_video/render_breakout_compare.py --rom xitari/roms/pong.bin --n-frames 1800
# (breakout default; pass --rom for any ROM in tools/breakout_video/dump_jaxtari_frames.py::_SETTINGS_BY_BASENAME)
```

---

## Conformance scoreboard (jaxtari↔xitari, `*_noop_10` fixtures, last frame)

Ground truth is **xitari** (`tools/trace_dump`). `jaxtari ≡ jutari` is the
PXC2 invariant (the two ports must diverge from xitari *identically*).
Numbers = RAM bytes differing from xitari on the last frame.

| ROM | divergence | notes |
|---|---|---|
| pong | **0** | bit-exact (was 6 before P3i-g) |
| breakout | **0** | bit-exact (was 2 before P3i-g) |
| space_invaders | 0 | bit-exact |
| seaquest | 4 | improved from 6 by P3i-g |
| pitfall | 19 | joystick; unchanged |
| enduro | 43 | = pre-P3i-g parent baseline (part 1 briefly hit 46; part 2 beam_cc+always-defer fixed it → NO net regression) |

**Screen scoreboard** (PXC-S, per-frame max screen ndiff vs xitari, `*_noop_10` fixtures, after P3i-g pt3):

| ROM | screen ndiff | notes |
|---|---|---|
| breakout | **0** | BIT-EXACT (8→0 by pt6 defer-ENAM/ENABL) |
| pong | **32** | 29760→920 (pt4 COLU mask)→568 (pt5 SCOREMODE)→32 (pt8 INTIM `-1`) |
| space_invaders | **12** | 2145→2079 (pt7)→12 (pt8 INTIM `-1` — near bit-exact) |
| pitfall | **322** | 1786 unchanged through pt7; 1786→322 by pt8 INTIM `-1` |
| seaquest | **1104** | 3941 (pt7)→1104 by pt8 INTIM `-1` |
| enduro | **1197** | 1972→1954 (pt7)→1197 by pt8 INTIM `-1` |

The pinned counts live in `jaxtari/tests/test_pxc2_jaxtari_vs_jutari.py`
(`_PXC2_CASES`). **If you change emulation behaviour and a pin moves, update
the pin AND regenerate the jutari fixtures in the same commit** so both ports
move in lock-step (recipe in that file's header comment).

---

## Patches landed (newest first)

### 🏆 Breakout FIXED — ENABL pending-write into GRP1 shadow (2026-06-03)

**Commit `20b5de0` (jutari) + `a418e4c` (jaxtari mirror) — the
breakout-frame-92 ball-doesn't-die bug is CLOSED.**

Measured: jutari↔xitari breakout RAM **9.9 b/f → 0.0 b/f**
(BIT-EXACT across all 300 frames of seed-42 random actions).
Lives counter `RAM[$39]` now decrements **every ~120 frames**
matching xitari (5→4→3→2→1→0 at frames 117, 237, 357, 477, 597).

**Root cause** (final): xitari's M6502Low applies ENABL writes
IMMEDIATELY; when GRP1 fires the shadow latch (`myDENABL =
myENABL`, TIA.cxx:2492), the latest ENABL value is captured.
Jutari/jaxtari defer ENABL writes (P3i-g pt6, needed for the
mid-scanline-stomp brick-rendering fix) — so when GRP1 captures
`tia.registers[W_ENABL+1]` into `enabl_old`, the live register
file holds a STALE value (the deferred ENABL write hasn't
activated yet). The shadow then retains a bit-1-SET value across
what should have been a clear, and `_vdel_enabl()` keeps
returning "ball enabled" when xitari has cleared it. The
collision evaluator then sets BL-PF where it shouldn't, which
the game uses to take the wrong CPU branch.

**Concrete sequence** (frame 91 scanline 229 of breakout):

  ```
  sc 35: ENABL = $00     ← clear ball
  sc 38: GRP1  = $00     ← triggers shadow latch
  
  xitari (immediate writes):
    sc 35: myENABL = ($02 & $00) = 0
    sc 38: myDENABL = myENABL = 0     ← shadow captures cleared
  
  jutari/jaxtari (before fix):
    sc 35: queue ENABL=$00 in pending_writes (activates sc 35+delay)
    sc 38: enabl_old = tia.registers[W_ENABL+1] = $37 ← STALE
    later: deferred ENABL=$00 activates — too late, shadow already wrong
  ```

**Fix**: at GRP1 poke, scan pending_writes for the LATEST ENABL
write whose `activation_clock ≤ current beam_cc`. Use that as
the effective ENABL value for the shadow capture, rather than
the stale live register.

**Discovery path** (so future agents can apply the technique):

  1. Built per-bus-op xitari trace (`d66b290` — extends
     `tools/trace_dump.cpp` + `xitari/.../System.{hxx,cxx}`).
  2. Diff'd jutari's existing per-bus-op trace vs xitari's at
     frame 93 — found CXBLPF reads $b6 in jutari vs $36 in
     xitari (= spurious BL-PF collision bit 7).
  3. Tried VBLANK collision-skip fix — semantically correct
     vs xitari but didn't close the gap.
  4. Added temporary `_BLPF_LOG` instrumentation to
     `_apply_pixel_collisions!` — found BL-PF gets set at scn
     51-52 of frame 92 with `bl_x=7`, `enabl_old=$37`,
     `VDELBL=$b9`.
  5. Recognized the VDELBL shadow path; traced ENABL state at
     each GRP1 write (the shadow-latch trigger); found last
     GRP1 of frame 91 captured `enabl_old=$37` in jutari but
     the corresponding ENABL state in xitari was $00 (cleared).
  6. The difference: jutari's deferred ENABL hadn't activated
     yet at the GRP1 moment, xitari's had (immediate writes).
  7. Fix: GRP1 shadow capture now drains pending ENABL writes.

The temporary `_BLPF_LOG` + `_ENABL_OLD_DEBUG` instrumentation
hooks have all been REMOVED in the final commit.

### Breakout — root cause identified at scanline 51-52, VDELBL shadow path (2026-06-03)

Used a temporary `_BLPF_LOG` debug hook in `_apply_pixel_collisions!`
to capture EVERY (scanline, color_clock) where BL-PF latch gets
set in jutari's frame 92 (= breakout step 92 in the diff). Output:

```
scn=51 x=7 bl_x=7 ENABL=0x00 enabl_old=0x37 VDELBL=0xb9
scn=51 x=8 bl_x=7 ENABL=0x00 enabl_old=0x37 VDELBL=0xb9
scn=52 x=7 bl_x=7 ENABL=0x34 enabl_old=0x37 VDELBL=0xb9
scn=52 x=8 bl_x=7 ENABL=0x34 enabl_old=0x37 VDELBL=0xb9
```

Decoded:
  - `ENABL = $00` / `$34` — both have bit 1 = 0 → ball "disabled" by
    current ENABL.
  - `enabl_old = $37` — shadow value, bit 1 = 1 → ball "enabled"
    by shadow.
  - `VDELBL = $b9` — bit 0 = 1 → use shadow (`enabl_old`).
  - Result: jutari's `_vdel_enabl` returns $37 → bit 1 set → ball
    is in the BL pixel set. Overlaps with PF at x=7, x=8 → BL-PF
    latch set.

**The bug is in the VDELBL shadow path interaction.** Either:

  1. xitari's `myDENABL` is FALSE at scn 51-52 (so it doesn't
     enable BL there), even though jutari's `enabl_old` is $37.
     xitari's `myDENABL = myENABL` at GRP1-write time, where
     `myENABL = value & 0x02` (just bit 1). So myDENABL is
     boolean. Jutari's `enabl_old` is the FULL byte, masked at
     read time. These should yield the same result, BUT...
  2. ...or the LAST GRP1 write happened at a different ENABL
     state in jutari vs xitari, so the shadow latched a
     different value.
  3. ...or xitari simply doesn't evaluate collisions at this
     position because `myCurrentBLMask[hpos]` is 0 (its ball
     position is different from jutari's bl_x=7).

**Concrete next steps for next session**:

  - Extend `xitari/emucore/TIA.cxx` to log when `myDENABL` /
    `myENABL` / `myVDELBL` change AND when `myEnabledObjects &
    myBLBit` is set. Compare against jutari's `enabl_old` /
    ENABL / VDELBL trace at scn 51-52 of frame 92.
  - Most-likely culprit: jutari's `enabl_old` retains an ENABL
    value with bit 1 set somewhere xitari's `myDENABL` got
    overwritten to false. Or xitari's `myDENABL` doesn't track
    when the ball goes "off" via ENABL=0 + GRP1 write.

The `_BLPF_LOG` debug hook is REMOVED in the final commit — only
used for this single diagnostic pass.

### Breakout deeper — BL-PF set during VISIBLE scanlines (2026-06-03)

Continuation of the per-bus-op trace investigation below. Applied
two safe correctness fixes:

  - `jutari` (commit `a8cdcc9`): skip collision evaluation during
    VBLANK in both the render branch (`tia_advance!`) and the bus
    peek catch-up (`_tia_catch_up_collisions!`). Matches xitari's
    `TIA::updateFrameScanline` (TIA.cxx:1121) early-exit.
  - `jaxtari` (commit `fa2a371`): symmetric port.

**Neither fix moved the breakout RAM gap** (still 9.9 b/f from
frame 92). So the spurious BL-PF latch in jutari is set during
VISIBLE scanlines (28-261 of frame 92), not during VBLANK.

ENABL pokes diff at frame 93:
  - jutari scn 97 sc 0: ENABL = $f4
  - xitari scn 97 sc 0: ENABL = $34  (xitari catches up at scn 99)
  - jutari scn 226 sc 9: ENABL = $37 (ball ENABLED bit set)
  - xitari scn 226 sc 9: ENABL = $b4 (ball disabled)

The ENABL divergence is DOWNSTREAM of the BPL branch divergence
at scn 1 sc 26 (caused by the spurious CXBLPF=$b6 latch read).
The root cause is upstream: WHY does jutari set the BL-PF latch
at some visible scanline in frame 92 where xitari does not?

Next-step recipe: instrument `_apply_pixel_collisions!` in
`Bus.jl` / `TIA.jl` to log every (scanline, color_clock) where
BL-PF is set in frames 89-92. Compare against an equivalent
trace from xitari (requires extending xitari's TIA.cxx with
similar logging, OR walking the per-bus-op trace to identify
when myCollision bit 0x8000 = BL-PF gets set in xitari's
`updateFrameScanline` switch).

### Phase 1+ — per-bus-op xitari trace extension + breakout bug ID (2026-06-03)

**Phase 1 follow-up**: extended `tools/trace_dump.cpp` (xitari) +
`xitari/emucore/m6502/src/System.{hxx,cxx}` to emit per-bus-op
CSV via a global `BusTraceCallback` hook. CSV format matches
jutari's `tools/cpu_tia_cycle_trace.jl` output one-for-one:
`global_idx,frame,kind,scanline,scanline_cycle,color_clock,addr,value`.

Usage:

    tools/trace_dump --rom ... --actions ... --max-frames N \
        --bus-trace OUT.csv [--bus-trace-frames LO,HI]

**Immediately used this to bisect breakout-frame-92**:

  - Diffed jutari trace vs xitari trace at trace frame 93.
  - All 4 emulator bus ops in the first ~10 events match modulo
    NMOS RMW dummy writes (xitari is M6502Low — no NMOS quirks).
  - **First semantic divergence** at scanline 1 sc 15:

    ```
    xitari  peek CXBLPF ($0036) = $36   (bit 7 = 0, no BL-PF collision)
    jutari  peek CXBLPF ($0036) = $b6   (bit 7 = 1, BL-PF COLLISION SET)
    ```

  - **The bug is in jutari's BL-PF collision detection.**
    `CXBLPF` is latched whenever the BALL overlaps with the
    PLAYFIELD on any pixel since the last CXCLR. The latch
    persists across scanlines until cleared by a CXCLR write.

  - **When does the spurious latch get set?** CXCLR poke happens
    at scn 27 sc ~73 of each frame, so the collision latch state
    at scn 1 sc 15 of frame 93 reflects events between scn 27 of
    frame 92 and end of frame 92 (scanlines 28..261). Jutari
    decides BL overlaps PF at SOME pixel in that range; xitari
    decides not.

  - **Confirmed reproducible**: frames 90, 91, 92 return
    CXBLPF = $36 in BOTH ports. Frame 93 returns $36 in xitari
    but $b6 in jutari — the FIRST frame jutari over-reports.

  - **Concrete next step**: extend jutari's per-pixel collision
    evaluation (`_apply_pixel_collisions!` in
    `jutari/src/bus/Bus.jl`) with logging of WHICH (scanline,
    color_clock) sets BL-PF in frame 92, and compare against
    xitari's render. Possible causes: (a) jutari renders BL at a
    slightly different X due to HMBL/HMOVE drift, (b) jutari's
    PF mask differs from xitari's at the BL location, (c) jutari
    evaluates the collision on a pixel where the BALL is
    *disabled* (ENABL = 0). The fix once identified will
    likely close the breakout RAM gap entirely.

### Phase 5 jutari + jaxtari — RIOT-read threading (2026-06-03)

  - **Commit `a505300`** (jutari first attempt) + **`7bd08a3`** (jutari
    `cycles - 1` refinement) + **`dbd3e06`** (jaxtari mirror).

  - **What it does**: `riot_peek` now takes a `pending_extra_cycles`
    argument; the bus wires `pending_tia_cycles` (post-increment, i.e.
    cycles consumed in current instruction including this bus op).
    INTIM read computes an effective sub-instruction extra using
    `max(0, pending_extra_cycles - 1)` (the `- 1` matches xitari's
    `cycles = mySystem->cycles() - 1` at M6532.cxx:161 — the current
    bus op's own incrementCycles(1) does NOT count toward the timer
    delta). Read-only — does NOT mutate riot.intim (real advance
    happens in `_tia_post_step` / per-step).

  - **Why M6532-specific**: xitari's `M6532::peek` uses `cycles() - 1`
    but `TIA::peek` uses `cycles() * 3` directly (no `- 1`). So
    Phase 5 for RIOT needs the `- 1` adjustment that Phase 5 for TIA
    does not.

  - **Measured impact**:
    - jutari↔xitari pong RAM: 0.0 b/f mean (unchanged — was already
      bit-exact, no regression).
    - jutari↔xitari breakout RAM: 9.9 b/f from frame 92 (unchanged —
      so breakout's bug is NOT in INTIM read timing).
    - jaxtari pong & jutari full test suite remain green.

  - **Conclusion**: the breakout-frame-92 bug is NOT INTIM-driven.
    Phase 5 is semantically correct vs xitari and preserved; the
    real breakout bug requires a per-bus-op xitari trace to
    pinpoint (extend `tools/trace_dump.cpp` to emit per-bus-op CSV
    matching jutari's existing trace format, then diff event-by-
    event — the first differing tuple is the entry point).

### Breakout frame-92 — narrowed to a collision-register BPL branch (2026-06-03)

Continuation of the deep dive below. Walked the jutari cycle trace
to localize the EXACT divergent instruction sequence:

  - **Trace coords**: trace frame 93 (= RAM-diff step 92), scanline 1.
  - **The smoking-gun instruction**: `BPL +$13` at cart byte $14af.
    - The LDA $e5 at $14ab reads `$b8` in jutari (RAM is bit-exact at
      end of trace frame 92, so xitari also reads $b8 at the same PC).
    - CMP #$28 on A=$b8 → result $90 → N=1 → BPL NOT taken in jutari.
    - Falls through to LDA #$80; STA $e1 — this is the FIRST
      divergent write to RAM[$61] (mirror of $e1 with bit 7 clear).
  - **What ran before the BPL**: 3 collision-register reads:
    ```
    scn 1 sc 1: peek CXP0FB ($0032) = $32   (bits 6+7 = 00, no col)
    scn 1 sc 9: peek CXP1FB ($0033) = $33   (bits 6+7 = 00)
    scn 1 sc 15: peek CXBLPF ($0036) = $b6  (bit 7 = 1, BL-PF collision!)
    ```
  - **Hypothesis**: one of these collision values DIFFERS in xitari.
    If xitari reports a different collision bit (e.g. CXBLPF returns
    no collision because the BALL hasn't actually hit the playfield
    yet in xitari's tracking), the CPU takes a different branch
    EARLIER in trace frame 93 — and by the time we reach the LDA $e5
    at $14ab, the PC is at a different address entirely (the LDA $e5
    we see in jutari may not even execute in xitari).

  - **Confirmation needed (next-step recipe)**: extend
    `tools/trace_dump.cpp` to emit per-bus-op CSV. The FIRST tuple
    `(scanline, scanline_cycle, addr, value)` that differs between
    jutari and xitari traces inside trace frame 93 IS the bug entry
    point. Most likely candidate: one of the 3 collision reads at
    scanline 1 sc 1/9/15.

  - **Why Phase 1c catch-up wasn't enough**: Phase 1c's
    `_tia_catch_up_collisions!` extends collision bits to the current
    beam color clock. For a peek at scanline 1 sc 1 (color_clock 3),
    the catch-up only covers a tiny prefix. If the BALL collided with
    PF earlier in scanline 0 (before VBLANK), that bit should ALREADY
    be latched. So either: (a) the collision latch from scanline 0
    differs between jutari/xitari, or (b) jutari is mis-evaluating
    when a collision occurs in scanline 0.

  - **Cycle count matches**: confirmed jutari's per-frame CPU cycle
    count for breakout is exactly **19912** — matches xitari's
    19912 to the cycle. So no cycle drift; the divergence is purely
    in the BYTES returned by some TIA read.

### Breakout frame-92 RAM divergence — deep dive (2026-06-03)

  - **Confirmed unchanged after Phase 2b**: same 6 bytes diverge at
    frame 92 ($37, $5f, $61, $65, $67, $6c). Mean 5.5 b/f. Phase 2b
    cycle counter fix moved the needle elsewhere but NOT here.

  - **Per-frame delta diagnostic** (`/tmp/breakout_frame_diff.py`):
    Reading frame 91 → 92 (action=NOOP, not action-triggered):
    ```
    xitari changes: $5a $5f $63 $65 $69    (5 bytes)
    jutari changes: $5a $5f $61 $63 $65 $67 $69  (+ $37 $6c at exit
                                                   from frame 91)
    ```
    Where jutari and xitari changes overlap:
      $5f: xi +5 (0x40 → 0x45)    ju +71  (0x40 → 0x87) — different!
      $65: xi +1 (0xb8 → 0xb9)    ju -1   (0xb8 → 0xb7) — OPPOSITE!
      $69: xi -1 (0x01 → 0xff)    ju -1   (0x01 → 0xff) — same
      $5a, $63: same +1 in both

    So jutari takes a SUBSTANTIALLY different code path at frame 92.
    The $65 going -1 (jutari) vs +1 (xitari) is the smoking gun — a
    counter is decrementing where it should increment, suggesting a
    branch driven by a TIA flag returned a different value.

  - **TIA register read traffic at frame 92** (jutari trace):
    - `$0284` INTIM (RIOT timer): 382 reads (most-read register)
    - Cartridge data lookup tables in `$133a-$133f`, `$1611-$1616`,
      `$10b1-$10ba`
    - RIOT RAM `$00bb $00e5 $00b6 $011e $011f $0038`
    - **NO collision register reads (`$00-$07`) in frame 92** —
      collision peeks are only ~78 across 93 frames (~0.84/frame).
    - **NO INPT reads at frame 92**.
    - So the divergent read is NOT a collision OR a paddle peek.

  - **Most-likely culprit**: INTIM. 382 reads/frame means the game
    polls the RIOT timer constantly. Previous fix `15cff40` (pt8)
    addressed an INTIM `-1` quirk that closed several big screen
    residuals — but maybe not all the edge cases. Specifically, with
    382 reads/frame, even a 1-tick mismatch at the wrong moment can
    trip a branch.

  - **Next-step diagnostic**: extend `tools/trace_dump.cpp` (xitari)
    to also dump per-bus-op events (same CSV format as Phase 1's
    jutari trace) so they can be diffed event-by-event. The first
    `(scanline, scanline_cycle, color_clock, addr, value)` tuple
    where jutari and xitari disagree IS the bug entry point. ~30
    min of C++ work; described in P3I_G_THREADING_PLAN.md §Phase 1.

### Phase 2b jaxtari mirror (P3I_G_THREADING_PLAN.md) — cycle-counter validation (2026-06-02)

  - **Commit `a8f2bd7`**: mirror of jutari Phase 2b (commit `2aec8c5`)
    applied to jaxtari. All 189 opcodes now satisfy
    `bus.pending_tia_cycles == CYCLE_TABLE[opcode]`.

  - **Validation via `jaxtari/tests/scratch_cycle_audit.py`**:
    Before: 64 of 189 opcodes mismatched (more than jutari, because
    jaxtari's NOP handler was even simpler — only ABS,X did anything).
    After: ALL 189 pass.

  - **Same fix shape as jutari**: 21 opcode-handler additions
    (implied 2-cycle internal ticks via `pending_tick`; PHA/PHP/PLA/
    PLP/RTS/RTI cycle-2 internal; branch NOT taken peek of operand;
    NOP full bus-op pattern across all 6 modes) + CYCLE_TABLE fix
    at 0xA7 (LAX zp 4 → 3).

  - **Measured impact (pong jaxtari↔xitari, 200 frames seed-42 actions
    via `tools/pong_3way_ram_diff.py`)**:
    - Before Phase 2b mirror: 13 b/f mean, 277/300 frames diverge
    - After  Phase 2b mirror: **first divergence at frame 24** (post-
      shared-FIRE-bug-at-20), **4 b/f sustained**. Frames 0-23
      bit-exact across all 3 emulators (modulo frame-20 FIRE).
    - **3-way snapshot**: jutari is now bit-exact with xitari for the
      first 200 pong frames except for the frame-20 FIRE shared bug.
      jaxtari has a small remaining 4 b/f at offsets $04 + $3c — same
      drift pattern as the old frame-59 divergence the previous agent
      identified, just compressed forward (now visible at frame 24).
    - **Phase 4 acceptance met**: criterion was "single digits"; we
      hit 4.

  - **What's next**: the residual jaxtari 4 b/f at $04 / $3c is a
    SINGLE specific bug — likely a jaxtari-only issue not shared
    with jutari. The bug_fix_log "where we left off" recipe applied
    to frame 23→24 transition would localize it. Phase 5 optional
    TIA-READ threading might also help.

  - **Pre-existing test failures discovered (NOT caused by Phase 2b)**:
    Reproducing on commit `8b5fc34` then again on a checked-out copy
    of `main~10` (i.e. before Phase 2b landed) shows 6 jaxtari tests
    were ALREADY failing — `test_p3g_nusiz.py`'s 4 `test_hard_*` cases
    and `test_p3h_vdel.py`'s 2 `test_vdelbl_*` cases. The failures
    are in `render_scanline` output (asserted pixel-by-pixel against
    expected NUSIZ patterns). Likely orphaned by a subsequent TIA
    renderer refactor that wasn't paired with a test update. NOT a
    regression — separate cleanup item.

  - **What's next**: Phase 3 verification (re-run PXC-S screen diff,
    regen pong/breakout videos). Phase 5 optional TIA-READ threading
    not started.

### Phase 2b jutari (P3I_G_THREADING_PLAN.md) — per-opcode cycle-counter validation (2026-06-02)

  - **Commit `2aec8c5`**: every documented + undocumented opcode
    now has `bus.pending_tia_cycles` summing to `CYCLE_TABLE[opcode]`
    when `_step_inner!` returns. Validated by
    `jutari/test/scratch_cycle_audit.jl` over all 189 opcodes.

  - **What was wrong**: `pending_tia_cycles` counts bus ops + explicit
    `pending_tick!` ticks. For 21 opcodes, internal cycles (no bus op)
    weren't ticked, undercounting the bus-op cycle count by 1-3 cycles.
    Each missed cycle means `_bus_poke!` for TIA writes scheduled the
    write 3 color clocks too early (since `pending_tia_cycles * 3`
    drives the effective `beam_cc`).

  - **What was fixed**: added `pending_tick!` for cycle-2 internal of
    all implied 2-cycle ops (TAX/TAY/TXA/TYA/TSX/TXS, INX/INY/DEX/DEY,
    CLC/SEC/CLI/SEI/CLV/CLD/SED, ASL A / LSR A / ROL A / ROR A),
    PHA/PHP/PLA/PLP cycle-2 internal, RTS/RTI cycle-2 internal. Added
    the NMOS-mandated operand peek for branch-NOT-taken. Added all 6
    NOP-variant bus-op patterns (IMM operand discard / ZP operand+val /
    ZP,X operand+tick+val / ABS lo+hi+val / ABS,X lo+hi+val+optional
    cross). Fixed a CYCLE_TABLE bug: 0xA7 LAX zp was 4 (jutari) should
    be 3 (matches jaxtari + NMOS reference).

  - **Measured impact (pong jutari↔xitari, 300 frames seed-42 actions)**:
    - Before Phase 2b: first div frame 20, mean 0.5 b/f, max 3 b/f
    - After  Phase 2b: first div frame 20, **mean 0.0 b/f**, max 3 b/f
    - Only 3 of 300 frames have any RAM diff (frame 20 FIRE shared bug;
      frames 140 + 260 each show 3 bytes transiently then return to
      bit-exact next frame). Effectively bit-exact through 300 frames.

  - **Measured impact (breakout jutari↔xitari)**: first div still at
    frame 92 with the same 6 bytes ($37 $5f $61 $65 $67 $6c). Breakout's
    bug is downstream of cycle accounting — likely a RIOT-timer or
    paddle/joystick wiring quirk specific to that ROM. Mean 9.9 b/f
    post-divergence (was ~10 before too).

  - **Test suite**: full jutari test pass remains green. The Phase 1
    diagnostic catch-up changes + Phase 2b cycle bumps together close
    PXC1 conformance.

  - **What's next for the threading plan**:
    - Phase 3 (verification): re-run PXC-S screen diff for all 6 ROMs;
      regen jutari pong/breakout videos with the cycle-accurate model.
    - Phase 4 (jaxtari mirror): apply the same 22 fixes to jaxtari.
      Expected to close jaxtari pong 13 b/f → near-0 (parity with jutari).

### Phase 1 (P3I_G_THREADING_PLAN.md) — diagnostic harness + collision catch-up refinement (2026-06-02)

  - **Phase 1a** (commit `fb72495`): jutari Bus gains a zero-cost
    trace tap. `trace_enable!()` / `trace_disable!()` /
    `trace_take!()` toggle a module-level buffer that records
    `(kind, scanline, scanline_cycle, color_clock, addr, value)`
    for every peek + poke + internal-cycle tick. Fast path when
    disabled is a single Ref nil-check per bus op. Tap lives on
    a module-level Ref (not on `BusState`) so the BusState layout
    is unperturbed — no test fixture churn.

  - **Phase 1a** also (commit `fb72495`):
    `tools/cpu_tia_cycle_trace.jl`: runs jutari for N frames with
    tracing on across boot-burn too, dumps a CSV with one row per
    event + synthetic `frame_boundary` markers. For pong 25
    frames → 1.37 M events, 44 MB CSV.

  - **Phase 1b** (commit `6cdc99a`):
    `tools/cycle_trace_inspect.py`: query language for the trace
    CSVs (scanline / poke-trace / hmove / diff / summary
    subcommands). Surfaces TIA register names. Used to refute the
    earlier "HMOVE blank misfires on scanline 34" hypothesis —
    pong frame 1 only writes HMOVE at scanline=27,sc=0 (in the
    blank-trigger window). The "row 0 = scanline 34" pixel
    divergence in the comparison video has a different cause.

  - **Phase 1c** (commits `408c516` jutari + `adcacc2` jaxtari):
    `_tia_catch_up_collisions` now accepts an `effective_cc`
    argument and `_bus_peek` passes
    `tia.color_clock + pending_tia_cycles * 3`. xitari's
    `M6502High::peek` increments cycles BEFORE the bus op, so
    `mySystem->cycles() * 3` (= the value `TIA::peek`'s
    `updateFrame` sees) is the POST-increment cycle. Our peek
    now matches that.

  - **Effect**: jutari + jaxtari focused tests stay green. The
    breakout ball-doesn't-die bug is UNCHANGED by this refinement
    (probe: still 1 death at frame 242 across 600 frames). So
    collision register read timing was NOT the proximate cause
    of that bug — investigation continues in a different
    direction (likely RAM-tracked ball Y vs paddle Y, or a
    different TIA read whose timing differs).

  - **Discovery from cross-reading xitari M6502High.cxx**:
    `M6502High::peek` does `mySystem->incrementCycles(1)` BEFORE
    `mySystem->peek(address)`, so the `cycles()` value during the
    bus op is the POST-increment value (= cycle this bus op
    completes at). Our `pending_tia_cycles += 1` at the top of
    `peek/poke!` produces the same value, so the existing
    `beam_cc = color_clock + pending_tia_cycles * 3` is **off by
    zero**, not off by one as the earlier session speculation
    suggested. The collision catch-up endpoint was the only
    place using `tia.color_clock` directly (= the conservative
    instruction-start value); that's now fixed.

  - **Measured impact of Phase 1c collision catch-up refinement (pong)**:
    `tools/jutari_xitari_ram_diff.py --rom xitari/roms/pong.bin
    --rom-settings pong --max-frames 600` shows **max 3 bytes/frame**
    and **mean ~0 bytes/frame** post-divergence — down from the
    pre-refinement "4 bytes/frame typical" figure in the earlier
    handoff. The only persistent divergence is the documented
    frame-20 FIRE shared bug (`$3f`/`$40` swap, 2 bytes) which
    matches the bug_fix_log "where we left off" entry. Effectively
    closes the jutari↔xitari pong RAM gap that was the original
    deep-dive target.

  - **BUT**: pong screen residual is UNCHANGED (~30 px/frame avg
    across 1800 frames, matching the user's "32 px at frame 200"
    visual report). RAM matches → renderer is producing different
    pixels from the same TIA register state. So the pong visual
    bugs (HMOVE-blank misfire on scanline 34, sprite Y off-by-1
    on rows 35-37/149) are **renderer-only**, not CPU-cycle. The
    cycle-threading work is in the wrong layer for them. Their
    fix requires investigating our `render_pixel` /
    `render_scanline` / HMOVE-blank state machine — orthogonal to
    Phases 1-5 of P3I_G_THREADING_PLAN.md.

  - **jaxtari pong measurement (Phase 1c mirror was already in
    `adcacc2`)**: vs xitari over 300 frames of random actions:
    first divergence at frame 20 (FIRE), mean 13 bytes/frame, max
    22, 277/300 frames diverge. **The refinement didn't close
    jaxtari nearly as much as jutari** — jutari went 4 →  <1, but
    jaxtari stayed 15-18 → 13. That's because the catch-up only
    helps the specific code paths that read collision registers
    (regs $00-$07) — jutari's residual happened to be in those
    paths, jaxtari's isn't. To close jaxtari's gap, need Phase 2
    (explicit per-cycle counter on `_step_inner` with `CYCLE_TABLE`
    validation) or a different deeper investigation. The bug_fix_log
    "where we left off" recipe (instrument frame 58→59 transition,
    find first divergent poke) remains the right path for jaxtari.

  - **Investigation finding for the breakout ball-doesn't-die bug**:
    using `tools/cycle_trace_inspect.py` on a 280-frame breakout
    trace, the collision register reads CONTINUE every frame past
    the only death at frame 242 (CXP0FB / CXP1FB / CXM0FB /
    CXM1FB / CXBLPF each read once per frame for ALL 280 frames).
    So the game IS still polling collisions post-death — it's
    just that what we return doesn't trigger the death condition
    again. Two remaining hypotheses:
      a. CXBLPF returns 0 every frame post-death; xitari would
         return non-zero on the cycle ball Y crosses the playfield
         floor. Need an xitari trace to confirm.
      b. The death trigger isn't CXBLPF at all but a RAM-tracked
         ball Y > paddle Y check; our ball physics computation
         differs from xitari's so ball Y never crosses the death
         threshold post-respawn.
    Hypothesis (b) is consistent with the "RAM diverges at frame
    20 (first FIRE)" note in the earlier handoff section — the
    physics state is wrong from frame 20 onward. The first death
    at frame 242 happens because that's when the wrong ball Y
    HAPPENS to be at-or-past the death threshold; post-death,
    the wrong-Y'd ball never crosses again.
    **Diagnostic next step**: extend `cpu_tia_cycle_trace.jl`
    to also dump RAM[$XX] for a configurable set of addresses
    per frame. Identify which 4 bytes diverge first at frame 20
    in jutari, then walk back to the causing read/write.

### Task #66 — jutari pong paddles move (SwapPaddles routes user paddle to INPT1) (2026-06-02)

  - **Symptom**: jutari pong paddles stayed completely frozen at the
    centred default — pressing LEFT/RIGHT had no visible effect on
    the on-screen paddle, despite `_apply_paddle_action!` running
    each frame and the paddle_resistance array updating.
  - **Root cause**: Pong's `stella.pro` entry has `Controller.SwapPaddles
    "YES"`. xitari's `Paddles::Paddles(jack=Left, event, swap=true)`
    swaps which pin reads which paddle resistance:
      * Pin Five (Left) ← `PaddleZeroResistance`  (user-driven)
      * Pin Nine (Left) ← `PaddleOneResistance`  (default, never updated)
    The TIA wires `INPT0 ← Pin Nine`, `INPT1 ← Pin Five`, so with
    swap the user paddle lands at **INPT1**, not INPT0. Our
    `_apply_paddle_action!` was unconditionally writing the user
    paddle to `paddle_resistance[0]` — meaning Pong was reading
    INPT1 (the default 408823) instead of the user's actual paddle
    position.
  - **Fix**: added `romsettings_swap_paddles(::RomSettings)`
    interface method (default `false`); `PongRomSettings` overrides
    to `true`. `_apply_paddle_action!` now branches on the flag and
    writes `paddle_resistance[1] = left_paddle` (user),
    `paddle_resistance[0] = right_paddle` (default) for swap games.
    Files: `jutari/src/games/RomSettings.jl`,
    `jutari/src/games/PaddleGames.jl`,
    `jutari/src/env/StellaEnvironment.jl`.
  - **Verification**: 8 LEFT presses now flip `paddle_resistance[1]`
    `0x63cf7 → 0x90bb7` (was flipping `[0]` before the fix);
    screen-motion probe: 30 NOOP + 50 LEFT + 50 RIGHT → 156 px
    changed after LEFT, 180 px after RIGHT (was 0/0 — completely
    frozen).
  - **Remaining cosmetic gaps in pong jutari** (visible in
    `tools/breakout_video/output/pong_xitari_vs_jutari.mp4`, frame
    ~200): 32 px / frame still differ. Three reproducible bug
    sites:
      1. **Row 0 (= scanline 34) leftmost 8 px**: xitari blanks
         them (HMOVE blank quirk firing on this scanline);
         jutari renders the full gray wall. Probably the HMOVE
         write timing on the visible/VBLANK boundary scanline.
      2. **Rows 35–37**: spurious COLUP1 green pixels at x=16-19
         AND spurious COLUP0 orange pixels at x=140-143. Looks
         like sprite Y-edge bleeding 1 scanline beyond xitari's
         range (P0/P1 GRP write timing off by 1 scanline on the
         top edge).
      3. **Row 149**: xitari draws orange P0 at x=140-143, jutari
         doesn't (sprite Y-edge bleeding 1 scanline beyond on the
         bottom edge — mirror of bug #2).
    These three are all CPU↔TIA cycle-accuracy bugs of the same
    family as the deferred jaxtari work — a 1-scanline timing
    drift in either the WSYNC release or the GRP* write delay.
    Likely closed together once jutari's CPU cycle accounting
    matches xitari's at the scanline boundary. Same root cause
    family as the jutari breakout ball-doesn't-die bug below.

### jutari breakout ball doesn't die after first death (open) (2026-06-02)

  - **Symptom**: under the seed-42 random action stream,
    `RAM[$39]` (lives) decrements **once at frame 242** then
    never again — even though the ball appears to keep going
    off the bottom of the screen. xitari decrements every
    ~120 frames. The on-screen ball doesn't disappear when it
    reaches the bottom.
  - **Same as the documented jaxtari bug** (search "Breakout
    ball doesn't die" higher in this log). Same shape: lives
    decrements once, then `RAM[$39]` stays stable while the
    ball visibly continues to move. RAM divergence from xitari
    likely starts at frame 20 (first FIRE) — confirmed jaxtari
    pattern, plausible same for jutari.
  - **Deferred cause family**: this is the same CPU↔TIA cycle
    accuracy issue that breaks the jutari pong sprite Y bugs
    above. The ball-death trigger relies on the
    `CXBLPF`-style collision register or a RAM Y-comparison
    that misses by 1-2 cycles per frame. Fix requires the same
    deep CPU-cycle threading work that has been attempted and
    deferred multiple times (the P3i-g part 2 revert, the
    write-side-only bus-op counting attempt). **Not addressable
    in a quick patch**; needs its own session.
  - **Diagnostic next step**: dump CPU PC + `RAM[$3F]`/`$40` at
    every frame from 18 to 25. Find the first frame where
    jutari's PC/RAM diverges from xitari's. xitari has already
    been demonstrated to read INPT0/INPT1 with very specific
    cycle timing at frame 20's FIRE; the bug is downstream of
    that read.

### Task #65 — paddle games skip SWCHA write (pong paddles move in jaxtari/jutari) (2026-05-31)
**Symptom:** User report — in pong jaxtari/jutari video, both paddles
do not move while xitari moves them just fine.
**Diagnosis:** Compared frame-1 pong RAM jaxtari vs xitari for the LEFT
action: 3 bytes diverged at offsets $04, $3c, $40 — pong's first frame
already takes a different code path with LEFT input. Walked through
xitari's `applyActionPaddles` (in `ale_state.cpp`): for paddle games it
ONLY updates paddle resistance + sets the fire event, and crucially
*does not* drive joystick direction events. My port's
`StellaEnvironment.step()` was calling `apply_action()` AFTER
`_apply_paddle_action()`; `apply_action()` always wrote LEFT/RIGHT
to SWCHA, so pong saw a phantom joystick LEFT bit on the bus, branched
differently from xitari, and never wrote the paddle-position byte
that the rendering code reads — the paddle stayed stuck.
**Fix (both ports):** added a `paddle_mode=False` keyword to
`apply_action` / `apply_action!`. When True, only the fire-button
trigger is updated and SWCHA is left untouched. `StellaEnvironment`
passes `paddle_mode=uses_paddles()` so paddle ROMs (pong, breakout,
warlords, …) skip the SWCHA write.
**Result:** pong frame-1 LEFT RAM divergence: 3 bytes → 2 bytes
(byte $40 fixed: was 0xc0 → 0x00 → now 0xc0). The remaining 2 bytes
($04, $3c) differ by exactly 4 and 2 — the paddle-position state
tracked from INPT0 polling — pointing to a sub-cycle INPT0 timing
gap. PXC-S noop unchanged for all ROMs. PXC1+PXC2 RAM all 21 green.
jaxtari unit tests 67 passed. jutari unit tests 820 passed.

### P3i-g part 8 — RIOT INTIM `-1` offset (pong 568→32, SI 2079→12, pitfall 1786→322, seaquest 3941→1104, enduro 1954→1197) (2026-05-31)
**Symptom:** Pong's PXC-S residual 568 px was *structurally* three full
boundary rows (24/34/194) where xitari renders the strip-on PF colour but
jaxtari renders the previous-row colour — a 1-scanline lag on the
`PF=$ff` strip-activation writes. Same pattern in SI / pitfall /
seaquest / enduro at varying scales.
**Diagnosis path:**
  1. Confirmed via row-fingerprint probe that rows 24/34/194 in pong
     match xitari with a +1-row shift.
  2. Instrumented `_bus_poke` — pong's `PF=$ff` writes land at sl 59 cc=39
     in jaxtari vs sl 58 xpos=39 in xitari (same intra-scanline beam,
     one scanline later).
  3. Counted WSYNCs: xitari has 133 in pong frame 1, jaxtari only 131.
     The 5th + 6th WSYNCs (xitari sl 14 + sl 22) are missing in jaxtari;
     re-aligned WSYNCs then drift +1 scanline.
  4. Frame-1 RAM is **identical** between ports — so it's not a RAM-state
     branch divergence.
  5. Instrumented all TIA/RIOT peeks: pong polls **INTIM** intensely
     starting at sl 14 (returns 0x10, 0x0f, 0x0f, ... every 21 cc).
  6. Read `xitari/emucore/M6532.cxx` case 0x04: xitari's INTIM formula is
     `myTimer - (delta>>shift) - 1` — an **extra `- 1`** that makes
     xitari's INTIM appear 1 less than the raw register value at all
     times. My port returned the raw register value → my INTIM is always
     1 higher than xitari's at the same CPU instruction → polling loops
     exit 1+ iterations LATER → 76 CPU cycles drift per polling loop
     (compounded across multiple loops to give the observed full-line
     drift on PF writes).
**Fix (both ports):** in `riot_peek` for reg=4 (INTIM), return
`(intim - 1) & 0xFF` instead of `intim`. When `intim == 0` this returns
`0xFF`, matching xitari's just-expired post-formula value. Pre-/post-
expired state transitions are unchanged in the underlying state — only
the read output is shifted.
**Result (PXC-S, worst frame across 10 frames):**
  - **pong: 568 → 32** (−536, **94%** reduction)
  - **space_invaders: 2079 → 12** (−2067, **99.4%**)
  - **pitfall: 1786 → 322** (−1464, **82%**)
  - **seaquest: 3941 → 1104** (−2837, **72%**)
  - **enduro: 1954 → 1197** (−757, **39%**)
  - breakout: unchanged (still 0, BIT-EXACT)
  - Total worst-frame screen diff: ~10,328 → 2,667 (a 74% drop)
PXC1 RAM stays bit-exact for pong / breakout / space_invaders, and the
other pre-existing PXC1 residuals (seaquest 4, pitfall 19, enduro 43)
are unchanged. PXC2 (jaxtari ≡ jutari) byte-identical across all 12
ROMs. **One unit-test update** (`tests/test_riot.py::test_intim_readable_via_peek`
+ jutari mirror): the test wrote 42 and expected to read back 42;
now correctly expects 41 per xitari semantics.

### P3i-g part 7 — extend defer to NUSIZ/COLU/CTRLPF/REFP (space_invaders 2145→2079, enduro 1972→1954) (2026-05-31)
**Symptom:** After pt6 closed breakout PXC-S to 0 and pt5 closed pong to
568, the other ROMs still had large screen residuals (space_invaders
2145, pitfall 1786, seaquest 3940, enduro 1972).
**Diagnosis:** Re-read xitari's `TIA::poke` more carefully. The first
thing it does for **every** poke is `updateFrame(clock + delay)` —
advance the renderer up to the activation cc, THEN apply the
register change. So even a delay=0 write (COLU*/CTRLPF) doesn't
affect pixels rendered before the write's CPU cycle on the same
scanline. My port was applying these immediately on poke, then the
whole-scanline batched render saw only the post-write state — lumping
pre-write pixels into the post-write colour. This is why
space_invaders and enduro had so much residual: per-scanline COLU /
CTRLPF / NUSIZ writes that should only affect mid-scanline+ pixels
were affecting the *whole* scanline.
**Fix (both ports):** extend the pt6 defer set to also cover
**NUSIZ0/NUSIZ1/COLUP0/COLUP1/COLUPF/COLUBK/CTRLPF/REFP0/REFP1** —
the "no-side-effect render registers" (the renderer just reads them).
Same `pending_writes` machinery, queued at `beam_cc + delay` and
drained at the right cc inside the per-color-clock render loop.
**Result:**
  - space_invaders: **2145 → 2079** (-66)
  - enduro: **1972 → 1954** (-18)
  - breakout, pong, pitfall: unchanged
  - seaquest worst: **3940 → 3941** (+1) but most frames *improved*
    (3768 / 3650 / etc., down from a flat 3940). A single
    worst-case frame regressed by 1 px — accept as a tradeoff;
    document the +1 in the pin.
**NOT deferred** (have side effects beyond a register store, would
need to defer the side effect too):
  - GRP0/GRP1 (latch VDELP shadows on write)
  - WSYNC, VSYNC, RES*, HMOVE, CXCLR, VBLANK (immediate-apply
    semantics matter for stall/strobe behavior)
**Test maintenance:** regenerated all 6 jutari screen fixtures in
the same commit. Pin updates above. PXC1+PXC2 RAM still bit-exact
(this is render-only). Pong's 568 unchanged — that's the cross-scanline
1-line-drift class still tracked under open ideas.

### P3i-g part 6 — defer ENAM0/ENAM1/ENABL to activation cc (breakout screen 8 → 0) (2026-05-30)
**Symptom:** The pinned breakout PXC-S residual of 8 px/frame, *every* frame,
at row 195 cols 0-7 — a single 8-px-wide block of palette index 182 that
xitari paints but jaxtari/jutari leaves as background (0). All other rows
matched bit-exactly.
**Diagnosis:** Probed `xitari/tools/trace_dump --screen` poke log around
TIA scanline 229 (= screen row 195 + Y_START 34). Found ENAM1 was set
to `$ff` (enabled, bit 1 set) all the way back at sl 51 and then *not
disabled until sl 229 xpos=105* — i.e. mid-scanline, in the visible
region. M1 is at position 0 with NUSIZ1=$ff (width 8) so it covers cols
0..7. **Xitari's per-color-clock renderer sees ENAM1 enabled at cc
68..104 (= cols 0..37) and disabled at cc 105+**, so M1 paints cols
0..7 before the disable. My port applied the ENAM1=0 write
*immediately* on poke, before the scanline render loop ran — so M1 was
already disabled when the renderer reached cc=68, and cols 0..7 stayed
background.
**Fix (both ports):** extend the existing PF0/PF1/PF2 deferred-write
mechanism (`pending_writes` queue, drained per-color-clock inside
`tia_advance`) to also cover **ENAM0/ENAM1/ENABL**. A write in the
visible region (`beam_cc >= HBLANK_COLOR_CLOCKS`) is queued with
`activation_clock = beam_cc + delay` (delay 0 per xitari's
`ourPokeDelayTable`); the render loop applies it at the right cc, so
sprites enabled at the start of visible can still paint their leading
pixels before a mid-scanline disable. HBLANK writes (cc < 68) keep
the immediate-apply path (xitari semantics give the same result for
scanline-setup writes).
**Result:** **breakout screen 8 → 0 px/frame** (full bit-exact screen
in PXC-S). Other ROMs unchanged in the noop fixtures (most don't
disable sprites mid-scanline at the leading edge). PXC2 still
byte-identical (this is a render-only change; RAM is unaffected).
**Test-pin maintenance:** the same regen-jutari-fixture + tighten-pin
recipe — `tools/fixtures/screens/breakout_noop_10_jutari.screen.gz`
regenerated, `_CASES["breakout_noop_10"].max_screen_diff` 8 → 0.
**Stale jutari unit tests surfaced:** with the bigger PXC-S unit
suite reaching tests it had been failing-out-of, four tests asserting
on odd COLU* values (pt4 `& 0xFE` mask) and three NUSIZ-wide-mode
position tests (pt3 `+1 px` offset) needed updating. Both classes were
pre-existing test debt from pt3/pt4 commits — not new regressions.

### P3i-g part 5 — CTRLPF.D1 SCOREMODE (pong screen 920 → 568) (2026-05-30)
**Symptom:** PXC-S pong's 920-px residual concentrated in the score-area
strip near the top of the screen. Pixel-byte probing showed xitari drew the
LEFT digit (player 1's score) in palette index **250** (= COLUP0 in pong)
and the RIGHT digit (player 2's score) in **138** (= COLUP1). My ports drew
both halves in **210** (= COLUPF). The pixel *positions* matched — only the
*colour* differed across the half-screen seam.
**Diagnosis:** Looked up CTRLPF bit definitions. Bit 0 = REFLECT, bit 2 =
PFP, bits 4-5 = ball size — all implemented. Bit 1 is **SCOREMODE**: in
score mode the playfield-ON pixels in the LEFT half are coloured with
**COLUP0** and in the RIGHT half with **COLUP1**, instead of COLUPF (ball
stays on COLUPF). Pong sets `CTRLPF = 0x02` during the score band. My
emulator ignored the bit entirely.
**Fix (both ports):** `render_pixel` / `render_playfield_scanline` /
`_overlay_playfield` now read `(ctrlpf & 0x02) != 0`; when set, the
LEFT-half PF colour is `COLUP0` and the RIGHT-half PF colour is `COLUP1`.
Ball, players, missiles, background are untouched. Mirrored to jutari's
`render_pixel`, `render_playfield_scanline`, and `_overlay_playfield!`.
**Result:** **pong screen 920 → 568 px/frame** (~38% reduction). Both ports
still byte-identical to each other (jaxtari ≡ jutari at 568). Other ROMs
unchanged (none of breakout / space_invaders / pitfall / seaquest / enduro
use score mode in these noop fixtures). RAM unchanged everywhere.
Regenerated `tools/fixtures/screens/pong_noop_10_jutari.screen.gz` and
tightened the pin in `test_screen_conformance.py` from 920 → 568.

### P3i-g part 4 — mask `COLU* & 0xFE` (pong screen 29760 → 920) (2026-05-30)
**Symptom:** PXC-S pong screen ndiff was 29760/frame — almost the entire
160×210 framebuffer differing from xitari, despite RAM being bit-exact and
the rendered frames *looking* identical to the eye in the side-by-side
video.
**Diagnosis:** Compared the per-pixel byte values: xitari produced even
palette indices (228, 138, 250); jaxtari/jutari the odd siblings
(229, 139, 251). The Stella NTSC palette has the same RGB at index N and
N+1, so the rendered colour is identical — only the encoded byte differs.
Checked xitari `TIA::poke` cases 0x06–0x09 (COLUP0/COLUP1/COLUPF/COLUBK):
they each mask `value & 0xFE` ("bit 0 of the color-luminance registers is
unused on real NMOS hardware"). My code stored the raw value, so any ROM
that wrote an odd luminance produced odd palette indices.
**Fix:** in both ports' `tia_poke`/`tia_poke!`, mask `value & 0xFE` when
`reg` is COLUP0/COLUP1/COLUPF/COLUBK (`0x06..0x09`).
**Result:** **pong screen 29760 → 920** (97% reduction; the remaining
920 px is real sprite/timing divergence — paddles/ball/score positions).
Other ROMs unchanged (they happened to only write even luminance values
already). RAM unchanged everywhere (this is a TIA-register store, not
game logic).

### P3i-g part 3 — NUSIZ wide-mode +1 pixel offset (Breakout paddle 1 px off) (2026-05-30)
**Symptom:** After P3i-g pt2 fixed Breakout's walls + bricks + red columns,
the paddle was still 1 px off from xitari (jaxtari at x=98-113, xitari at
99-114; same paddle width, just shifted left by one). The numerics agreed
end-to-end — RESP0 → POSP0=96, HMOVE delta +2 → POSP0=98 — so the difference
had to be on the render side, not the position side. (A 1-cycle RESP0 shift
would be 3 px, so the residual couldn't be cycle-accuracy.)
**Diagnosis:** Probed jaxtari's full RESP0/HMP0/HMOVE/HMCLR chain for the
boot+1 frame, confirmed it matches xitari's `UV_TIA_POKES` log exactly.
Then inspected xitari's `computePlayerMaskTable`: the mask init for player
mode 7 (quad-size, NUSIZ=7, 4× scale) and mode 5 (double-size, NUSIZ=5,
2× scale) both have the comment "for some reason in double/quad size mode
the player's output is delayed by one pixel thus we use > instead of >=" —
a documented NMOS-TIA quirk that shifts wide-mode players one pixel right.
**Fix (both ports):** in `_overlay_player` / `_player_set`, add a +1 pixel
offset to the rendered/collision base position when `scale > 1` (modes 5
and 7). Mirrors xitari's mask offset exactly.
**Result:** Breakout paddle now at x=99-114, byte-identical to xitari on
rows 189-194; screen-vs-xitari ndiff dropped **16 → 8 per frame** (the
only residual is an 8-px corner block xitari draws on row 195 that we
miss — looks like a P1 VDELP-shadow scanline). The same fix improved
**seaquest 3946→3940** and **enduro 1988→1972** (those games also use
the wide player modes). pong/space_invaders/pitfall unchanged. RAM stays
bit-exact for breakout (collision-set patch is correctness-faithful, not
behaviour-changing).

### P3i-g part 2 — beam_cc threading + always-defer PF writes (Breakout "red columns" / flicker) (2026-05-29)
**Symptom (user report):** Breakout's *first* frame already showed big red
vertical columns in the lower (below-bricks) screen, and heavy frame-to-frame
flicker. RAM was bit-exact (PXC green) — a pure **rendering** bug, invisible
to the RAM-only tests.
**Diagnosis tools:** `UV_TIA_POKES=1 ./tools/trace_dump …` (xitari's per-poke
xpos/delay/activation log) + a monkeypatch logger of jaxtari's `tia_poke`
PF2 writes. Found: xitari clears the playfield for the lower screen with
`PF2=$00` at **scanline 127, xpos=0** (HBLANK), after which no more PF writes
— so the lower screen stays black. My code issued the same `PF2=$00` but at
**scanline 126, beam_cc=225** (1 CPU cycle / 3 color clocks earlier, a cumulative
sub-cycle drift), where `activation = 225+3 = 228 ≥ 228` → it fell through to
**immediate-apply**. Then the scanline-126 render **drained the still-pending
`PF2=$ff@68` and `PF2=$3f@156`**, which clobbered the register back to `$3f`
(bits 0-5 = pixels 12-17 = x48-71). That `$3f` then persisted into the lower
screen — the red columns. Frame-to-frame the brick scroll changed which value
won the clobber race → flicker.
**Two-part fix (both ports):**
  1. **beam_cc threading** (replaces the P3i-g-part-1 inline `tia_advance`
     flush): `Bus._bus_poke` passes the *effective* sub-instruction beam
     position (`beam_cc = color_clock + pending*3`, `beam_sc = scanline_cycle
     + pending`) to `tia_poke`, which uses it for the PF defer + RES*/HMOVE.
     The TIA is **not** advanced/rendered mid-instruction — it advances exactly
     ONCE per instruction in `_tia_post_step`. (The part-1 inline flush rendered
     scanlines mid-instruction, which is what let a pre-clear PF pattern leak.)
  2. **Always-defer PF writes**: removed the "fall through to immediate-apply
     when `activation ≥ 228`" path in `tia_poke`/`tia_poke!`. A PF write whose
     effect crosses into the next scanline now stays in `pending_writes`;
     `tia_advance`'s drain-remaining step applies it LAST (after the in-loop
     drains), into `final_registers` — so it is NOT clobbered by an earlier
     same-register pending write, and it correctly carries to the next
     scanline. This is what fixes the `PF2=$00` clear.
**Result:** Breakout screen-vs-xitari diff dropped **~8000 → 16 px/frame**
(the only residual is a 1 px paddle-X offset, 7 rows). Lower screen black,
no flicker. TIA/playfield/p3i/p3l unit tests green; jutari byte-identical to
jaxtari. The paddle X also tightened from 13 px off → 1 px off.
**Test gap closed (user's idea):** added `jaxtari/tests/test_screen_conformance.py`
(PXC-S) — diffs the *rendered framebuffer* xitari↔jaxtari and xitari↔jutari
per frame, with per-ROM `max_screen_diff` pins. This would have caught the
red columns immediately (RAM tests never could). Screen fixtures live in
`tools/fixtures/screens/*.screen.gz` (xitari + jutari, gzip — they compress to
<7 KB). Generator: `tools/jutari_screen_dump.jl`. **Finding it surfaced:**
most ROMs have large screen-vs-xitari divergence (pong ~4760, seaquest ~3946,
…) despite bit-exact RAM — pre-existing rendering-accuracy gaps now visible
and pinned.

### P3i-g — mid-instruction TIA-write cycle threading (2026-05-29)
**Commit:** `2e44bab` (code) + follow-up (PXC pins + jutari fixtures).
**Symptom (user report):** Breakout side wall on the right was at the wrong
position (pulled inward to x≈140 with a grey wall segment embedded in the
brick rows), a center seam through the bricks, and the paddle offset.
**Root cause:** the TIA was advanced once per CPU instruction in
`_tia_post_step`, so a TIA register *write* issued mid-instruction recorded
the beam at the **instruction-START** color clock — off by the cycles
already consumed (0..6 CPU cycles = 0..18 color clocks). Breakout draws its
walls + bricks by writing PF0/PF1/PF2 *mid-scanline* (racing the beam), so
every such write landed ~12 px too early. Verified against xitari's own
`UV_TIA_POKES` debug env var (set it when running `tools/trace_dump` to get
exact `xpos`/`delay`/`activation` per poke): on a brick scanline xitari
writes PF at xpos 66/111/132/153; jaxtari recorded 54/99/120/141 (−12 each).
**Fix:**
- `Bus` gains `pending_tia_cycles` (every peek/poke = one 6502 cycle) +
  `tia_advanced_this_instruction`. A TIA-region **write** flushes
  `tia_advance(tia, pending)` BEFORE the poke, so PF*/RESP*/HMOVE/COLU* land
  at the precise sub-instruction color clock — matching xitari's
  `M6502High::poke` (incrementCycles-before-access) → `TIA::poke` →
  `updateFrame(cycles*3)`. `_tia_post_step` drains the remainder.
- Indexed zero-page resolvers (`zp,X`/`zp,Y`) count their **cycle-3 dummy
  read** via `Bus.pending_tick` so `STA zp,X` to a PF/COLU register flushes 4
  cycles not 3 (closes the last 3 px of the wall: 149→152). A cycle-only
  tick is observably equivalent to the real dummy read because its bus value
  is always overwritten by the instruction's final access — and it avoids
  routing low-zero-page operands through the TIA collision/INPT catch-up path
  (which is ruinously slow in the Python renderer).
- jutari mirrors all of the above on its mutable `BusState`; jutari's
  `tia_poke!` `beam_cc`/`beam_sc` params (a parallel attempt at the same fix)
  default to the post-flush `color_clock`, so the two mechanisms compose.
**Result:** walls now flush at cols 0-7 / 152-159 in both ports (exact match
to xitari); brick rows continuous. **Bonus:** the accurate write timing also
closed the data-path gap — **pong 6→0, breakout 2→0 (BIT-EXACT)**, seaquest
6→4. **Cost:** enduro +3 (see "stale pin" note). PXC2 `jaxtari ≡ jutari`
preserved (ports byte-identical). jaxtari CPU/bus/TIA/P3 suites green.

### jutari SWCHB test fix (2026-05-29)
**Commit:** `065171f`. Task #64 changed the SWCHB default 0xFF→0x3F but only
updated the jaxtari test; the jutari "SELECT + RESET press" test still
asserted `0xFF & ~0x03`. Corrected to `0x3F & ~0x03` (= 0x3C). Pre-existing,
unrelated to P3i-g.

### jutari SOFT-PFP Zygote test (2026-05-29)
**Commit:** `efda8a3` (spawned task). Pre-existing SOFT-mode gradient test at
`jutari/test/runtests.jl:4911` that the SWCHB abort had been masking; the
COLUP0 assert now uses `atol` for denormal AD noise.

---

## Dead ends & gotchas (so we don't repeat them)

- **DO NOT name a throwaway script `bisect.py`** (or any stdlib module name)
  and run it from its own directory. Python puts the script dir on
  `sys.path[0]`, so stdlib `random.py`'s `import bisect` picks up *your*
  file. If your file imports jax, you get a baffling
  `ImportError: cannot import name 'typeof' from partially initialized
  module 'jax._src.core' (circular import)`. Cost us ~an hour chasing a
  phantom "broken jax env". Name scratch scripts uniquely.
- **jax 0.10.1 import-order fragility:** importing `jax.numpy`/`jax.lax`
  before a full `import jax` can trigger the same `typeof` circular import on
  a cold bytecode cache. jaxtari's normal entry points are fine; ad-hoc
  scripts should `import jax` first.
- **Frame-alignment off-by-one:** when hand-checking a port against xitari,
  make sure you compare the *same* frame index. An ad-hoc check comparing
  jaxtari frame 12 to a `trace_dump --max-frames 13 | tail -1` (= frame 13)
  produced a phantom "5-byte divergence" on breakout that does not exist
  (breakout is bit-exact). Use the canonical `*_noop_10.jsonl` fixtures.
- **Stale PXC pins:** the enduro pin claimed 29, but the suite had been
  aborting at an unrelated SWCHB test (task #64 era) *before* reaching the
  divergence check, so the pin was never refreshed. Measured live, the
  pre-P3i-g parent (`6aa18b5`) enduro is actually **43**. Always re-measure
  the baseline before attributing a "regression" to your change.
- **Read-side TIA flushing is too slow in the Python renderer:** flushing the
  TIA before every INPT/collision *read* (the fully xitari-faithful model)
  forces a scanline re-render on each INPT poll — Breakout's paddle loop
  reads INPT0 dozens of times per frame → ~6× slowdown — and it did *not*
  move the paddle. So P3i-g threads **writes only**; reads stay on the
  per-instruction model (`_tia_catch_up_collisions` partial catch-up).
- **`total_cycles` decoupling was a no-op:** we hypothesised the inline
  write-flush perturbed the dump-pot cycle counter (`total_cycles`) and added
  a `count_total_cycles=False` decoupling; frame-12 RAM was byte-identical
  with and without it, and PXC pong/breakout passed either way, so it was
  reverted. The dump-pot is not on the write path.

---

## Open ideas / planned (not yet done)

- **Breakout ball-death detection broken under random actions (NEW — 2026-05-31).**
  Reproduction: `tools/breakout_video/output/breakout_random_actions.txt` (seed
  42, 1800 frames). xitari decrements RAM byte 57 (lives) on a clean cadence:
  5→4 at frame 116, 4→3 at 236, 3→2 at 356, 2→1 at 476, 1→0 at 596 (game over
  at frame 597). jaxtari only loses ONE life (5→4 at frame 241) — then never
  loses another and runs all 1800 frames without ending. So in the
  random-action video the ball never dies cleanly: it bounces forever instead
  of disappearing and triggering a new round. RAM divergence first appears at
  **frame 20** (the first FIRE/serve in the action sequence), 5 bytes at
  offsets [95, 99, 101, 103, 105], growing from there. PXC1 noop-10 RAM is
  STILL bit-exact (the bug only manifests when FIRE is exercised), so the
  divergence is downstream of FIRE-action handling. Suspect classes:
  (a) `INPT4` (joystick fire button) read returns wrong value at the moment
  the game polls it; (b) the StellaEnvironment translation of `A_FIRE` to a
  button-press event differs subtly from xitari's `ALEState::act`; (c)
  ball/paddle collision detection drops the "ball below paddle" event on the
  first ball, leaving the ball-state machine stuck. Concrete next probe:
  instrument `INPT4` peeks during frames 18-25 (the FIRE window) and compare
  to xitari. The video is at `tools/breakout_video/output/breakout_xitari_vs_jaxtari.mp4`.

- **Pong screen 568 → lower (one-row-late cross-scanline transitions).** After pt4 (COLU mask 29760→920) and pt5 (SCOREMODE 920→568), the residual is now structurally three "full row diff" rows (24, 34, 194 — each 160 px = 480 px) plus ~88 px scattered (score digit fragments + rows 36-38 + col 0-7 on row 0). Confirmed (2026-05-31) by row-fingerprint probe: rows 24/34/194 in xitari render the strip-on colour, jaxtari renders the previous BG colour — exact 1-scanline lag. Instrumented `_bus_poke` showed pong's PF=$ff "strip-on" writes land at sl 59 cc=39 in jaxtari vs sl 58 xpos=39 in xitari — same intra-scanline beam position, one scanline later. Deeper trace: xitari emits 133 WSYNCs (STA $02) in pong frame 1; jaxtari emits 131. Sl-by-sl alignment: WSYNCs 1-4 (sl 3/6/9/12) match in both ports; jaxtari is missing 2 of xitari's early-VBLANK WSYNCs (around xitari sl 14, 22). PXC1 RAM stays bit-exact because STA $02 doesn't write RAM. The likely root: pong polls INPT (paddle) and/or CXxx (collision) and branches on the value; jaxtari's TIA read returns slightly different values at those scanlines, causing pong to skip the WSYNC instructions xitari executes, which compresses jaxtari's CPU stream by ~76 cycles and shifts every subsequent PF/COLU write 1 scanline later. Same per-cycle bus-accuracy class as the bug log open ideas; not a clean small fix.
- **Enduro convergence (43 → lower).** Enduro's large base divergence (43/128)
  is pre-existing and unrelated to P3i-g; P3i-g additionally broke 5
  collision-timing-sensitive cells (`$36 $47 $67 $68 $76`) while fixing 2
  (`$2e $46`). Hypothesis: the write-threading shifts *collision-detection*
  timing — enduro repositions sprites (cars) per-scanline via RESP/HMOVE, and
  those writes are applied **immediately** (not deferred like PF*), so a
  mid-scanline RESP changes the sprite position for the *whole* scanline
  render rather than only after the write. Next step: instrument collision
  reads (`tools`-style monkeypatch of `tia_peek` for regs $00-$07) on enduro
  parent-vs-mine, find the first CXxx read whose value diverges, then trace
  which object overlap at which color clock. A draft logger lives in the
  session notes (`collog.py`). Likely fix: **defer RESP/HMOVE sprite-position
  changes** to their activation color clock the same way PF* writes are
  deferred (per-pixel sprite X), so collision detection sees the right
  position at the right beam position. Same class of problem that caused the
  *original* P3i-g part-2 revert (`ed1e498`).
- **Breakout center seam (4 px at x≈76-79 on brick rows).** Pure-rendering
  residual at the playfield half-boundary — does NOT affect RAM (breakout is
  bit-exact). Needs full per-cycle bus accuracy / sub-pixel PF activation.
- **Paddle random-run drift.** On the *random-action* Breakout run (ball
  launched), the game state still diverges from xitari by deep frames (≈200+)
  even though noop is bit-exact — the post-launch ball/paddle collision
  dynamics accumulate small timing differences. Tied to the same
  collision-timing accuracy as enduro.
- **P3i-d2 (read-side threading), task #3.** The faithful "flush TIA before
  every read" model — deferred for the performance reason above. Would need a
  cheaper renderer (JIT/vectorised per-scanline) to be viable.
- **P4c dump-pot timing model, task #4.** Formula is already faithful
  (`tia_peek` INPT0-3 matches xitari's `(1.6·r·0.01e-6)·1.19e6` threshold +
  the VBLANK-D7 `dump_disabled_cycle` capture). The residual paddle-game gap
  was the write-timing chain, now closed (pong/breakout bit-exact). Remaining
  read-path accuracy is the per-cycle bus work above.
- **P7f-dx, task #5.** Wire collision *reads* into the SOFT bus + migrate the
  remaining SOFT-mode collision tests. Independent of the HARD-path work here.
- **Per-cycle bus accuracy for store-mode `abs,X`/`abs,Y`/`(zp),Y`.** These
  always dummy-read on a store (5/6-cycle) even without a page cross; jaxtari
  currently only dummy-reads on a page cross. Adding the unconditional
  store-dummy (cycle count + data_bus_state) is the documented PXC1 "missing
  internal-cycle bus exposures" remainder.

---

## How to reproduce the key checks

```bash
# xitari ground truth (+ exact poke timing):
UV_TIA_POKES=1 ./tools/trace_dump --rom xitari/roms/breakout.bin \
    --actions tools/fixtures/actions/breakout_noop_10.txt --screen --max-frames 11

# PXC conformance (jaxtari↔xitari pins + jaxtari≡jutari):
cd jaxtari && .venv/bin/python3 -m pytest tests/test_pxc1_conformance.py \
    tests/test_pxc2_jaxtari_vs_jutari.py -q

# Regenerate a jutari fixture after an emulation change (keep ports lock-step):
julia --project=jutari tools/jutari_trace_dump.jl --rom xitari/roms/<rom>.bin \
    --trace tools/fixtures/traces/<name>.jsonl \
    --out   tools/fixtures/traces/<name>_jutari.jsonl

# Side-by-side comparison videos:
jaxtari/.venv/bin/python3 tools/breakout_video/render_breakout_compare.py \
    --out-dir tools/breakout_video/output --n-frames 600 --seed 42
```
