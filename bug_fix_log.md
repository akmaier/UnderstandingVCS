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

## Where we left off — pick up here (2026-06-09/10)

### Session 2026-06-09 / 2026-06-10 rundown (jutari-focused)

PXC2 confirms 18/18 jaxtari ≡ jutari cross-port tests still pass with
the new pinned divergences after today's per-ROM RomSettings +
starting-actions work.

| Task           | Status   | Commits                       | Result                              |
|----------------|----------|-------------------------------|-------------------------------------|
| #78 pong score addrs $0D/$0E   | ✅ closed | `7fee15f` + `b10321a`         | pong paddle no longer freezes      |
| #79 re-render pong videos      | ✅ closed | `c9ed0c5`                    | all 4 new game videos on disk      |
| #80 seaquest boot residual     | 🔬 partial | `533f706`                    | 6 b/f frame 0 → 1 byte at boot end |
| #81 pitfall starting actions   | ✅ closed | `08b98d4`                    | 19.8 → **0** b/f BIT-EXACT       |
| #82 enduro starting actions    | ✅ partial | `08b98d4`                    | 45 → **17** b/f (3 b/f @ frame 0) |
| #83 jutari row-0 HMOVE comb   | 🔬 deferred | (this entry)                 | needs xitari-faithful "clear only after render past HBLANK+8" refactor |
| #84 jutari sprite Y 1-row off  | 🔬 deferred | (this entry)                 | needs per-cycle scanline/PF write timing alignment |

### Earlier session 2026-06-07 rundown

| Task           | Status   | Commits                       | Result                              |
|----------------|----------|-------------------------------|-------------------------------------|
| #73 jaxtari pong $04/$3c       | ✅ closed | `de00af8` + `c3d6d42`         | bit-exact RAM at frame 24+ |
| #75 jutari breakout jumping    | ✅ closed | `b7cd741` + `4ddb0b7`         | 99.7% breakout pixel-exact         |
| #76 jutari auto-reset          | ✅ closed | `037526c`                    | env.terminal flips correctly       |
| #77 pong $3f/$40 swap          | ✅ closed | `c3d6d42`                    | both ports bit-exact at frame 20   |

### 🔬 Phase C remaining pong 24 px residual (after #84 closed)

Per-row pong jutari↔xitari diff (frame 0, post-#84):

| Row(s) | Cols | Px | Source | Task |
|--------|------|-----|--------|------|
| 0 | 0-7 | 8 | HMOVE comb missing | #83 |
| 35 | 140-143 | 4 | Phantom 4-px sprite color 138 (COLUP1) | #85 |
| 36 | 16-19, 140-143 | 8 | Phantom 4-px sprites color 250+138 | #85 |
| 37 | 16-19 | 4 | Phantom 4-px sprite color 250 (COLUP0) | #85 |

Investigation summary for the rows 35-37 phantoms (task #85, new):
  - Inspected jutari TIA state at scanlines 65-72 (= rows 31-38 of
    get_screen). ENAM0/ENAM1 toggle correctly per xitari trace
    (bit-1 on/off alternating across consecutive scanlines).
  - Confirmed pong cart accesses ENAM0 via $011D / ENAM1 via $011E
    (the address mirrors that decode to TIA via A7=0 path) — both
    engines decode these identically.
  - The 4-px width + COLUP0/COLUP1 colors strongly suggest M0/M1
    missile rendering at unexpected positions, OR NUSIZ multi-copy
    of P0/P1 score sprites rendering an extra copy in jutari.
  - Deferred — needs targeted trace of m0_x/m1_x positions + NUSIZ
    register state at scanlines 69-71 vs xitari's render math.

### 🏆 Task #84 CLOSED (2026-06-10) — GRP* defer fix

**Fix:** Added W_GRP0/W_GRP1 to the deferred-writes block in
`tia_poke!`. VDELP* / VDELBL latch SIDE EFFECTS still run immediately
at the cart's write moment (matching xitari), but the RENDERED
register value is deferred to its activation_clock so pre-write pixels
use the OLD GRP value.

**Result:**
  - pong PXC-S screen residual: **32 → 24 px** (paddle 1-row-early
    fully GONE; rows 95/111 cleared from diff list)
  - All 5 PXC1 ROMs stay bit-exact (pong, breakout, space_invaders,
    pitfall, **enduro now bit-exact too!**) — seaquest unchanged at
    6 b/f boot
  - All ~1170 jutari runtests pass

**Initial false-alarm:** First measurement of "breakout RAM regressed
to 4.3 b/f" was caused by my Phase B boot_end emit in `trace_dump`
(commit `533f706`) shifting the diff-tool comparison by one frame.
xitari emits an extra `boot_end:true` record before user-action
frames; jutari does not. The diff tool was index-comparing
xitari[0]=boot_end vs jutari[0]=first_user_frame.

**Fix applied to diff tool** (`tools/jutari_xitari_ram_diff.py`):
skip records with `"boot_end":true` when parsing xitari output. With
this, all comparisons are aligned and the GRP defer fix is clean.

### 🔬 EARLIER (REVERTED then SUPERSEDED): Task #84 dead-end notes

**The actual bug:** GRP0/GRP1 writes are NOT in jutari's deferred-writes
list. They apply immediately to `tia.registers[]` and the whole
scanline renders with the new value. xitari renders incrementally so
pre-write pixels use OLD GRP value and post-write pixels use NEW value.

Pong's right paddle: cart writes GRP1=240 at scanline 129 cycle 71
(cc 213). Paddle position cols 140-143 = cc 208-211 (BEFORE the write).
xitari: pre-write cycles use OLD GRP1=0 → no paddle on scanline 129;
paddle starts at scanline 130. Jutari: whole scanline uses NEW
GRP1=240 → paddle visible at scanline 129 (= row 95 vs xitari's 96).

Confirmed via per-bus-op trace diff (jutari vs xitari pong frame 1):
both engines write GRP1=240 at identical (scanline, scanline_cycle)
positions. The cart code execution is byte-identical. The divergence
is purely render-side.

**The fix (REVERTED):** Add GRP0/GRP1 to the deferred-writes block,
with VDELP* / VDELBL latch SIDE EFFECTS still running immediately at
the cart's write moment.

  - ✅ pong screen residual: 32 → 24 px (paddle 1-row-early bug fully
    GONE — rows 95/111 cleared, remaining 24 px = row-0 HMOVE comb +
    rows 35-37 PF score-mode phantoms)
  - ✅ jutari runtests all pass (EXIT=0)
  - ❌ breakout PXC1 RAM (jutari↔xitari noop_10) regressed from
    BIT-EXACT to 4.3 b/f mean
  - ❌ enduro/pitfall PXC1 RAM also worse

**Why breakout regresses:** breakout's cart code relies on the
SPURIOUS-IMMEDIATE-GRP behavior in jutari to maintain its RAM
sequence. The cart probably reads collision latches that pick up
phantom sprites caused by the whole-scanline-GRP application; removing
those phantoms changes the collision pipeline output, which changes
the cart's branch decisions, which changes RAM.

**Lesson:** Multiple jutari TIA "bugs" silently compensate for each
other. Fixing one in isolation exposes another. To ship the GRP defer
without breakout regression, the related compensating bug(s) must be
identified and fixed first. Likely candidates: collision detection
during VBLANK boundary, mid-instruction TIA-read catch-up, or floating-bus
value when reading TIA write-only registers.

Task #84 root cause is now KNOWN. The fix is a 30-line patch that
shipping requires a parallel investigation into what breakout depends
on. Deferred until that investigation is done.

### Phase C WSYNC scanline-boundary "76 stall" investigation (2026-06-10) — REVERTED

Attempted fix for the 1-row-early sprite/PF bug (task #84). Hypothesis:
jutari's WSYNC stall semantics at `scanline_cycle = 0` returns 0 cycles
while xitari's `(228 - cycleOfScanline) / 3` returns 76 cycles (one
full scanline advance).

Change: removed the `mod()` in `tia_apply_wsync!` so stall = 76 when
scanline_cycle = 0 (xitari-faithful). Updated the existing unit test
to expect 76 stall + advance to scanline 1.

Result:
  - boot-end RAM: pong/breakout/space_invaders/pitfall STAY bit-exact,
    enduro IMPROVES from 3 → 0 bytes ✓
  - noop_300 frame-by-frame RAM: enduro REGRESSED from ~17 b/f →
    ~30 b/f mean. **pitfall REGRESSED from BIT-EXACT (0 b/f) to ~6 b/f.**
  - pong screen residual: UNCHANGED at 32 px (this WSYNC edge case
    isn't what's driving the paddle 1-row offset).

Net: regression. Reverted both TIA.jl + test changes.

Lesson: jutari's existing "0 stall at boundary" behavior is silently
compensating for some other timing bug elsewhere. Naive xitari-faithful
WSYNC breaks that compensation. The 1-row-early paddle bug (task #84)
needs a different angle of attack.

### Phase C HMOVE comb DEEP investigation (2026-06-10) — REVERTED again

Tried (3rd attempt) to make jutari render the row-0 HMOVE comb that
xitari produces. Used the diagnostic infrastructure to trace HMOVE
writes in both engines via the bus trace tool.

Findings:
  - Both engines have ONLY ONE HMOVE strobe per frame in pong: at
    scanline 27 cycle 0. Same instruction in the cart.
  - Both engines write VBLANK off at scanline 27 cycle 3 (= color
    clock 9, still in HBLANK).
  - xitari produces an 8-px HMOVE comb at row 0 of get_screen (=
    scanline 34 in TIA timing). 7 scanlines AFTER the HMOVE strobe.
  - Jutari produces the comb at scanline 27 (= framebuffer row 27,
    BELOW Y_START=34, not in get_screen output) → no comb visible.

Root cause analysis: xitari's rendering is INCREMENTAL per CPU-cycle
chunk (`mySystem->m6502().execute(25000)` runs until stop, intermediate
state writes call `updateFrame` which renders the chunk-of-clocks
that just elapsed). The `myHMOVEBlankEnabled` flag is checked +
cleared per chunk based on whether the chunk covered past HBLANK+8.

Jutari's rendering is PER-SCANLINE (whole row rendered in one shot
at the end of any instruction that advances past a scanline boundary).
Cannot easily decide whether comb was "actually painted" without
breaking the per-scanline render model.

Attempts that didn't work:
  - "Only clear hmove_blank_pending if !vblank_active" — pong cart
    flips VBLANK off mid-scanline-27, so scanline 27 ends as visible
    and the flag clears anyway.
  - "Only set, never clear in HMOVE handler" — pong only strobes
    HMOVE once per frame, no clobber to prevent.
  - Combination of both — same outcome, comb still missing.

The fundamental issue: jutari's per-scanline render model cannot
replicate xitari's per-chunk decision about whether the comb was
painted. Needs a refactor to render incrementally per CPU-cycle chunk
(matching xitari's TIA.cxx:1776 condition exactly) — a much larger
piece of work than naive flag-clear-condition tweaks.

Task #83 stays open. Deferred until per-cycle rendering refactor.

### Phase C pong screen residual: localized to 2 distinct TIA bugs (2026-06-09)

Detailed per-row jutari↔xitari pong screen comparison (frame 0):

| Issue | Rows | Cols | Pixels | Description |
|-------|------|------|--------|-------------|
| Missing HMOVE comb | 0 | 0-7 | 8 | xitari renders 8 black px at row 0; jutari renders background |
| Phantom sprites | 35 | 140-143 | 4 | jutari has color-138 sprite; xitari has background |
| Phantom sprites | 36 | 16-19, 140-143 | 8 | jutari has color-250 + 138 sprites |
| Phantom sprites | 37 | 16-19 | 4 | jutari has color-250 sprite |
| Paddle 1-row early | 95 | 78-81 | 4 | jutari paddle starts at row 95; xitari at row 96 |
| Paddle 1-row early | 111 | similar | 4 | symmetric for right paddle |

Both bugs are 1-row-related vertical-timing issues:
  - The HMOVE comb requires `myHMOVEBlankEnabled` to be true at the
    START of scanline 34's render (= row 0 of get_screen). Pong cart
    strobes HMOVE during VBLANK; jutari's per-scanline clear (line
    715 of TIA.jl) clobbers it before reaching scanline 34. A naive
    "only clear if !vblank_active" fix didn't work (reverted) because
    cart turns VBLANK off mid-scanline, defeating the gate. xitari's
    semantic (TIA.cxx:1776-1786) is: clear ONLY when a render call
    completed past HBLANK+8. Needs a deeper refactor.
  - The paddle/sprite 1-row-early bug suggests jutari's VDELP or
    framebuffer y-tracking is off by 1 for specific sprite-rendering
    paths. xitari rows 96..N+13 = jutari rows 95..N+12. Same width.
    Tied to GRP1 latch timing or sprite-Y bounds checking.

Total: pong 32 px screen residual = 8 (HMOVE comb) + 24 (1-row-early
sprite renderings). Both deferred.

### Phase C row-0 HMOVE-comb investigation (2026-06-09) — REVERTED

Attempted to fix the row-0 HMOVE comb missing on pong (8 px of the
32 px pong screen residual). Two candidate fixes both reverted after
empirical traces showed they didn't address the actual semantic gap.

Diagnostic findings:
  - Pong cart strobes HMOVE at scanline 27 cycle ~0-20 (cart's own
    "scanline 27" in TIA counter — corresponds roughly to xitari's
    HBLANK of one of the early VBLANK scanlines).
  - jutari's `_hmove_blank_enabled_at` returns true for this cycle,
    so `hmove_blank_pending = true` correctly.
  - At end of scanline 27's render path, `vblank_active` has been
    flipped to false by the cart writing VBLANK off mid-scanline.
  - With my candidate fix "only clear flag if !vblank_active", the
    flag still got cleared because vblank was already off by end of
    render.
  - xitari's actual semantic (TIA.cxx:1776-1786): clear `myHMOVEBlankEnabled`
    only when an updateFrameScanline call rendered past HBLANK+8.
    This is a per-render-chunk decision, not per-scanline-end.
    Replicating it requires a more invasive refactor: track whether
    THIS scanline's render entered the "blank-applied" path AND
    completed past HBLANK+8, not just whether vblank was active.

For now: the row-0 comb is left as-is. The 8 px row-0 contribution
remains part of pong's 32 px screen residual. Documented for future
Phase C work — needs a refactor where the flag clear is gated on
"the blank was actually consumed by THIS scanline's render", not
the timing approximation tried here.

### Phase B / Task #80 partial (2026-06-09) — seaquest boot-end localized to 1 byte

Localized seaquest's 6-byte frame-0 RAM divergence to a SINGLE byte at
boot-end (before any user action): RAM[\$01].

  | Source | RAM[\$01] | Other bytes |
  |--------|----------|-------------|
  | xitari (60+4 boot)   | \$3e    | identical |
  | jutari (60+4 boot)   | \$3f    | identical |
  | jutari (60+3 boot)   | \$3e    | matches xitari |
  | jutari (59+4 boot)   | \$3e    | matches xitari |

The other 5 bytes that diverge at frame 0 in the noop-300 diff are
DOWNSTREAM propagation of this 1-byte boot mismatch — once the cart's
loop iteration drifts by 1, accumulated state cascades.

Diagnosis: seaquest's cart code increments RAM[\$01] every frame (a
plain frame counter). jutari ends at $3f (64 increments), xitari at
$3e (63 increments). Despite both engines running the same 60-NOOP +
4-RESET boot burn, ONE FRAME's worth of cart code executes differently.

The 8 other shipped ROMs (pong, breakout, space_invaders, pitfall,
enduro, asteroids, qbert) are bit-exact at boot-end. Beamrider +
mspacman trace_dump appears to fail (separate issue).

Suspected root cause: seaquest specifically reads INTIM (RIOT timer)
or INPT4 (joystick trigger) inside its frame-counter increment path,
and the first-frame-of-burn timing for those reads differs subtly
between engines. Same class of issue as the per-cycle bus accuracy
documented as the cause of the PXC-S screen residuals.

Decision: leave the 1-byte boot residual to be swept up by Phase C
(per-cycle bus accuracy work) rather than chase it independently.
Task #80 stays open with this localized diagnosis recorded.

Tooling added:
  - `tools/boot_state_diff.jl` — dumps jutari's post-boot RAM hex for
    any ROM (matched by `tools/trace_dump`'s new `"boot_end":true`
    frame-0 emission).
  - `tools/trace_dump.cpp` — now emits a synthetic frame-0 record with
    boot-end RAM BEFORE any user action, so the two ports can be
    compared at the boot/burn boundary directly.

### 🏆 Tasks #81 + #82 (2026-06-08) — getStartingActions for pitfall + enduro

xitari's `StellaEnvironment::reset()` doesn't just do the 60-NOOP +
4-RESET boot burn — it ALSO emulates `m_settings->getStartingActions()`
afterwards (gated on the default-true `use_starting_actions` setting).
Pitfall returns `[PLAYER_A_UP]` and Enduro returns `[PLAYER_A_FIRE]` —
both put the agent into a known starting pose that real human players
would take after the cart's title screen.

jutari + jaxtari were doing the boot burn + `m_settings->reset()`
but skipping the starting-actions step. That left both ports 1 frame
behind xitari at frame 0 for Pitfall + Enduro, which compounded into
the 19.8 b/f (pitfall) / 45 b/f (enduro) RAM divergence over 300 NOOPs.

Fix:
  - Both ports: add `starting_actions()` to RomSettings (default `[]`).
  - Both ports: `env.reset()` / `env_reset!` emulates each starting
    action after the boot burn + settings reset.
  - Jutari: new `JoystickGames` module with `PitfallRomSettings`
    (returns `[2]`) and `EnduroRomSettings` (returns `[1]`); ports'
    `_SETTINGS_BY_BASENAME` registries updated.
  - Jaxtari: add `starting_actions()` to `PitfallRomSettings` and
    `EnduroRomSettings`. Defaults stay empty for the other 4 ROMs.

Measured jutari↔xitari RAM diff (300 NOOPs):
  | ROM      | Before | After | Status |
  |----------|--------|-------|--------|
  | pitfall  | 19.8 b/f | **0.0** b/f | ✅ BIT-EXACT |
  | enduro   | 45.0 b/f | **~17** b/f | residual 3 b/f @ frame 0 |
  | seaquest | 2.6 b/f  | 2.6 b/f    | no starting actions — separate fix needed |

Enduro's residual 3 b/f at frame 0 (RAM[\$00] off-by-1, RAM[\$23] big
diff, RAM[\$2e] off-by-1) is a smaller boot-state mismatch left for
future work (task #82 stays open as a partial). Seaquest's 5 b/f
boot mismatch is task #80 (also distinct from starting actions).

### 🏆 Task #78 closed (2026-06-07) — pong score addresses wrong in both ports

`PongRomSettings._scores()` (jaxtari) and `_pong_scores()` (jutari)
were reading RAM[\$14] and RAM[\$15] as P0/P1 scores. Those addresses
are a leftover from an early-stage guess that was never cross-checked
against xitari. They actually hold sprite-pattern bytes that briefly
hit `0x82 = 130` within ~60 frames of FIRE+LEFT. Then
`max(0, 130) >= 21` returned True, `env_step!` returned early on
`env.terminal && return 0`, and the user paddle FROZE without any
visible game-over indication.

xitari/games/supported/Pong.cpp:55-56:
```cpp
int x = readRam(&system, 13); // cpu score      → $0D
int y = readRam(&system, 14); // player score   → $0E
```

Fix: change both ports' constants and the live-RAM read path to
\$0D/\$0E, update docstrings to point at the xitari ground truth,
and add regression tests pinning the addresses + a "no false terminal
across 100 frames of FIRE+LEFT" assertion in both ports.

Empirical: jutari smoke (200 frames of FIRE+LEFT after boot burn):
  pre-fix:  env.terminal flips True at frame ~60, paddle frozen
  post-fix: **env.terminal stays False all 200 frames**, RAM[\$0D]=RAM[\$0E]=0

jutari runtests.jl now has `pong score-address fix + no false terminal
(task #78)` testset (6/6 asserts pass). jaxtari now has
`tests/test_pong_score_addresses.py` with the same shape (constants +
no-false-terminal check).


Plus per-ROM RomSettings now mirror xitari semantics (jutari Pong +
Breakout, jaxtari Breakout) — both ports use the started+terminal
sticky-latch idiom to avoid boot-time false terminal. Regression
tests added on both ports for the auto-reset terminal latch (jutari
`@testset "breakout auto-reset terminal latch (task #76)"`; jaxtari
`tests/test_breakout_auto_reset_terminal.py`). PXC2 cross-port
conformance still passes 18/18 after all changes.

### 🏆 Task #76 closed (2026-06-07, commit `037526c`) — jutari auto-reset

jutari's `BreakoutRomSettings` was using `RomSettingsModule`'s no-op
defaults for `is_terminal` / `get_reward` / `lives` — so `env.terminal`
never flipped True even after lives hit 0. `dump_jutari_frames.jl`'s
auto-reset loop never fired; post-game-over frames showed a "stuck"
breakout while xitari did a full ale.resetGame() and started a new
episode. Result: 3000+ frames of the comparison video diverged after
frame 597 (game-over).

Fix: implement xitari-faithful `is_terminal` / `lives` / `get_reward`
in `BreakoutRomSettings` (started+terminal sticky flag, score from
RAM[\$4C]/[\$4D] BCD nibbles). Also fixed the `_ram(addr)` helper to
index the bus's RAM array directly (don't go through `bus.peek` —
addresses \$39/\$4C/\$4D are TIA read-side registers, not RAM mirrors).

Also added equivalent PongRomSettings (score from \$14/\$15, terminal
at 21, ΔP0−ΔP1 reward) to mirror jaxtari/jaxtari/games/pong.py.

Empirical: breakout video (jutari vs xitari, 3600 frames):
  pre-fix:  587/3600  = 16.3% pixel-exact (only frames 0-596 lived)
  post-fix: **3590/3600 = 99.7% pixel-exact** (all 3600 frames live)

Only 10 sub-pixel ball-paddle-bounce divergences remain.

### 🏆 Task #75 closed (2026-06-07, commit `4ddb0b7`) — TIM*T-load timing bug

The 76-cycle / 1-scanline drift was the **timer-load instruction's own
cycles being counted toward the new timer**. xitari's M6502Low records
`myCyclesWhenTimerSet = mySystem->cycles()` at the poke moment, but
`incrementCycles` ran at the START of the load instruction so cycles
== END of the load instruction. Effectively **none of the load
instruction's cycles count toward the new timer**.

jutari was setting `cycles_since_tick = 0` at riot_poke, then
`riot_advance!(instr_cycles)` at end of instruction added all 3-5 of
the load instruction's cycles to the timer — overcounting by exactly
the load instruction's cycle count.

Fix: pass `bus.pending_tia_cycles` to `riot_poke!`, set
`cycles_since_tick = -pending_extra_cycles` on TIM*T loads so the
trailing `riot_advance!` cancels them out.

Empirical result on breakout (jutari vs xitari):

| run                      | pre-fix    | post-fix     |
|--------------------------|-----------:|-------------:|
| first 100 frames         |  72/100    | **98/100**   |
| full 3600 (game ends 597)| 416/3600   |  587/3600    |
| **frames 0-596 (live)**  | varies     | **587/597 = 98.3%** |

All jutari tests still pass; the breakout `ball-death` regression is
still bit-exact RAM. The "jumping scanlines" bug is **closed for
actual gameplay** — the 1-row vertical shift no longer occurs.

The remaining 10 misaligned frames (91, 93, 211, 213, 331, 333, 451,
453, 571, 573) each differ by only 1-2 pixels at row 195 — likely
sprite sub-pixel ball-paddle bounce position. Pairs appear every 120
frames = once per ball-death cycle. Not a fundamental drift — a
residual sub-cycle sprite rendering artifact.

The much-larger 3000-frame post-frame-597 divergence is a SEPARATE
bug: jutari's auto-reset (env_reset! with 60 NOOPs + 4 RESET) does
not match xitari's `ale.resetGame()` semantic. Tracked as a separate
follow-up.

### 🏆 Task #73 closed (2026-06-07, commit `de00af8`) — pong SwapPaddles

jaxtari was always sending paddle resistance to INPT0 regardless of
stella.pro's `Controller.SwapPaddles` setting. For Pong (Video
Olympics, which has SwapPaddles=YES) the user's paddle is wired to
INPT1 — without the swap, jaxtari's paddle resistance went to a
register pong never read as the user paddle, freezing it at the
centred default.

Fix: add `swap_paddles()` method to RomSettings base, override in
PongRomSettings → True, conditional swap in StellaEnvironment's
paddle action handler. Mirror of jutari task #66.

Verified: jaxtari pong RAM at frame 24+ now matches jutari/xitari
($04 = 0x68 etc.) under the standard random action sequence.

### 🏆 More wins (2026-06-07)

  - **PXC1 jutari pong + breakout now bit-exact** (commits `4ddb0b7`
    + `75e5312`): the TIM*T-load timing fix + repairing the test
    harness to auto-detect per-ROM `RomSettings` (pong needs
    `PongRomSettings` for paddles + SwapPaddles=YES) closed the
    long-standing pong_noop_10 4-byte divergence. Jutari PXC1 testset
    promoted from `@test_broken` to `@test`.

  - **Scoreboard refresh** (commit `c0fb21a`): 300-frame NOOP RAM diff
    jutari↔xitari → pong/breakout/space_invaders are **bit-exact across
    300 frames** (was just 10). enduro/pitfall/seaquest residuals
    captured for the next round.

  - **Jaxtari INTIM mirror** (commit `5736f57`): all 25 jaxtari tests
    pass with the new TIM*T-load semantics + the pong SwapPaddles
    routing.

### 🏆 Task #77 closed (2026-06-07, commit `c3d6d42`) — Pong $3f/$40 swap

The previous SwapPaddles=YES fix only swapped the paddle RESISTANCE
routing (INPT0/INPT1). xitari ALSO swaps the FIRE button wiring under
the `swap` branch — Pin Three → PaddleZeroFire (= USER's fire) lands
on a different TIA pin than without swap.

In jutari/jaxtari, paddle-mode FIRE was always landing on SWCHA bit 7
(= paddle 0's wiring). For Pong (SwapPaddles=YES) the USER's fire
needs SWCHA bit 6 (= paddle 1's wiring) instead.

Fix: `apply_action` gains a `swap_paddles` kwarg that flips
`paddle_fire_bit` (0x80 ↔ 0x40) for paddle-mode. `StellaEnvironment`
threads `settings.swap_paddles()` through alongside `paddle_mode`.

Verified: jutari + jaxtari pong frame 20 (action=FIRE) now both read
$3f=0xc0 $40=0x00 — matching xitari. All paddle tests (jutari + 22
jaxtari) still pass.

### Open work

  - **Enduro / pitfall / seaquest** RAM residuals (52 / 18 / 5 b/f over
    300 NOOPs vs xitari): cycle-accuracy work remaining.
  - **Jutari auto-reset divergence**: post-game-over breakout frames
    don't match xitari (tracked as task #76).

### Update — vsync_reset_pending fix landed (2026-06-04 evening, commit `b7cd741`)

Implemented the "next-step recipe" below (`tia.vsync_reset_pending::Bool`
flag, drained at end of `tia_advance!`). All tests pass; the P3f frame-
ending testset was updated; breakout `ball-death` RAM regression still
bit-exact. **However**, the fix turns out to be a no-op for breakout's
visible jumping: the per-bus-op trace and the per-frame screen diff
both show identical content pre- vs. post-fix (11.6% exact-match
unchanged). Reason: for breakout the VSYNC=0 instruction consistently
fires at `scanline_cycle=0` (frame ends right on a scanline boundary),
so the OLD mid-instruction reset and the NEW deferred reset converge
to the same end-of-instruction state.

The fix is still worth keeping — it now matches xitari's startFrame()
semantics for games where VSYNC fires mid-scanline (the more general
case). Just doesn't move breakout.

**Empirical confirmation that scanline_cycle MUST be preserved (not
reset) at the deferred drain**: also resetting `scanline_cycle=0`
(`color_clock=0`) at the drain point dropped breakout match rate from
11.6% → 1.7% — preserving is the right call.

### Deeper finding — cycle accumulation drift (2026-06-04 evening)

Tracked the jutari-vs-xitari elapsed-CPU-cycle gap across the first VBLANK
clear poke in breakout frame 22 (the first frame after FIRE):

| anchor event                  | xi cyc | ju cyc | gap |
|-------------------------------|-------:|-------:|----:|
| first poke `$0296` (TIM1T)    |      6 |      2 |   4 |
| first peek `$0282` (SWCHB)    |     18 |     14 |   4 |
| first VBLANK clear `$01=0`    |   2207 |   2128 |  79 |

The gap STARTS at 4 cycles (= xitari records POST-instr cycles for poke
events vs jutari's PRE-instr recording — the 4 comes from STA absolute
= 4 cycles) and GROWS to 79 cycles at the VBLANK clear. **Excess gap
= 79 - 3 (STA zp post-vs-pre offset) = 76 cycles = exactly 1 scanline.**

That 76-cycle drift IS the 1-row vertical shift. jutari's beam is 1
scanline BEHIND xitari's at the point game code clears VBLANK, so
jutari's "scanline 34" rendering picks up game state that xitari has
at "scanline 35". `framebuffer[35,:]` in jutari = `framebuffer[36,:]`
in xitari → entire frame shifted up by 1 row.

**Where the 76 cycles disappear**: still narrowing down. Each INC/DEC
zp/abs RMW adds 1 extra BUS OP (jutari's NMOS-correct dummy write
that xitari's M6502Low skips), but CYCLE counts match. Each taken-BNE
adds 1 extra BUS OP (jutari's phantom fetch xitari omits) — but again
cycles match. Suspect classes for the cycle drift:

  - STA abs,X / STA (zp),Y unconditional-dummy-read: bug_fix_log
    already lists this as the open remainder; jaxtari "only
    dummy-reads on a page cross". jutari likely has the same shape.
    Each STA abs,X is supposed to take 5 cycles unconditionally;
    if jutari counts 4, that explains accumulating cycle loss.
  - Page-cross detection differs between jutari and M6502Low.
  - Some other addressing mode's cycle count is off by 1 in jutari.

**Next probe** (highest signal):
  1. Add a per-instruction cycle-count differ tool: capture both
     traces, for each "matching" instruction (same low-12 PC), record
     cycle delta (=cycles_after - cycles_before in each engine), diff.
  2. The first instructions where ju delta != xi delta are the
     culprits. Cluster by opcode to find the broken addressing mode.
  3. Fix the addressing-mode cycle count in jutari's `CYCLE_TABLE` or
     in `_step_inner` page-cross logic.

This is a separate, deep bug from the deferred-reset fix. Closing it
would likely close the breakout jumping bug AND improve pong's
remaining 2-byte residual (`$3f`/`$40`) and other cycle-sensitive
games. Tracked as task #75.

**Confirmed (2026-06-04)** by counting total CPU cycles per frame:

```
frame  xi cycles  ju cycles   delta
  19     19912      19912        0
  20     19912      19912        0
  21     19912      19912        0   ← last aligned frame (FIRE)
  22     19912      19836      -76   ← first misaligned frame
```

Frame 22 is jutari's frame after FIRE. xitari uses 262 scanlines × 76
= 19912 CPU cycles. jutari uses 261 scanlines × 76 = 19836 — exactly
**76 cycles less = 1 scanline less of CPU work**. Same game code,
same boot sequence (frames 1-21 agree byte-for-byte). One or more
instructions in jutari's frame-22 execution undercounts cycles by a
total of 76. Suspect opcodes (by occurrence count in breakout ROM):
LDY abs,X (16), LDA abs,X (11), LDA abs,Y (9), DEC abs,X (8), LDX
abs,Y (3), LDA (zp),Y (6) — each a page-cross-sensitive +1-cycle
addressing mode. If even one of these has wrong page-cross handling
and is hit ~76 times per frame in the FIRE path, the drift matches.

**Localized (2026-06-04)** — first observable divergence in frame 22:

Compared TIA/RIOT-only peek sequences (both engines, same code path).
First difference: the 32nd INTIM (`$0284`) read returns 10 in jutari
vs 11 in xitari. Both read the same address on the same instruction;
the timer counts down by 1 ONE READ EARLIER in jutari. xitari catches
up on the very next iteration (read 33 = 10 in both). The 1-cycle
drift in INTIM is the entry point: whichever Pong loop branches on
this value picks a different path in jutari → game executes different
total cycles → frame ends 76 cycles short.

The INTIM formula difference candidate:

  - xitari `M6532::peek` case 0x04: `uInt32 cycles = mySystem->cycles() - 1;`
    Then `delta = cycles - myCyclesWhenTimerSet`. For M6502Low the
    `mySystem->cycles()` value INCLUDES the current instruction's
    full incrementCycles bump.
  - jutari `riot_peek!` INTIM: `eff = max(0, pending_extra_cycles - 1)`.
    Then `extra = (cycles_since_tick + eff) >> shift`.

The `-1` in jutari subtracts from `pending_extra_cycles` (= bus ops
processed THIS instruction including current). xitari subtracts from
`mySystem->cycles()` (= cumulative system cycles MINUS 1). Whether
these are exactly equivalent depends on a per-bus-op vs per-instruction
cycle delta that doesn't always match — likely a 1-cycle off-by-one.

Concrete next probe: capture the EXACT cycles value at the divergent
INTIM read in both engines via an extra trace field, compute what each
engine's formula returns, and align them. Likely fix: tweak jutari's
`eff` formula (perhaps drop the `- 1` or change the cycle reference
point) to match xitari's M6502Low INTIM semantics.

### NEW investigation — jumping is FIRE-action triggered (2026-06-04)

Per-frame analysis of the post-fix 3600-frame breakout video:

  - Frames 1-20: 100% pixel-aligned with xitari (no shift).
  - Frame 21: **first FIRE action** (action_id=1) in the action stream
    — instantaneous 1-row vertical shift up.
  - Frames 22-25: aligned again.
  - Frames 26-35: shifted (10 consecutive frames).
  - Frames 36-55: aligned (20 consecutive).
  - Frames 56-onwards: alternating windows of shifted/aligned.

Total: 416/3600 frames (11.6%) pixel-exact; 3137/3600 frames (87.1%)
have top-most-row index off by exactly -1 (jutari shifted UP by 1
scanline).

**Concrete observation** for frame 21:

  - jutari's first non-zero pixel row = row 4 (= internal scanline 38)
  - xitari's first non-zero pixel row = row 5 (= internal scanline 39)
  - jutari row 1 == xitari row 2 (exact match — pure 1-row shift)

So jutari's "scanline 38" contains what xitari renders at "scanline 39"
— jutari is **one scanline ahead in beam-position-vs-game-state**
during the visible region.

**Two-state pattern** suggests: at certain game-logic moments (after
FIRE press, brick-break events?) jutari's TIA renders the leading edge
of the visible region one scanline earlier than xitari. Likely path:
a TIA state read (CXxx / INPT4) returns slightly different value at
the VBLANK-clear instruction, causing jutari's game code to branch
through a path 1-3 cycles shorter than xitari's, so VBLANK=0 fires
1 scanline earlier in jutari for that frame.

**Concrete next probe**:
  1. Capture jutari & xitari per-bus-op traces frames 19-22, find the
     first event where jutari's "VBLANK clear" poke (`STA $01` with
     bit 1 = 0) happens at a different scanline than xitari.
  2. Walk backward to the most recent TIA-read peek diverging in
     value (likely CXBLPF / CXP0FB / INPT4 in the fire-handling
     region).
  3. Fix that read's value to match xitari at the divergent cycle.

The deferred VSYNC reset is in place — the jumping is a SEPARATE bug
with the same per-bus-op-trace methodology pointing the way.

### Investigation note — jumping scanlines deep dive (2026-06-04)

Used per-bus-op trace bisection (per `BUG_BISECTION_METHODOLOGY.md`)
to narrow down the scanline drift. Findings:

**Quantified per-bus-op cycle drift** (frame 21, breakout):

```
idx  op                              xi_sc  ju_sc  diff
0    poke $0296=34 (TIM1T)             6      5    +1
1    peek $00da=19                    11      9    +2
2    poke $00da=20 (INC RAM[$5a])     11      9    +2
3    peek $0282=63 (SWCHB)            18     17    +1
4    poke $00dc=255                   32     32     0   ← converges
13   peek $01ff=242 (cart)            87     84    +3
16   peek $00de=209                   90     90     0
```

**Key observation**: jutari is consistently 1 CPU cycle BEHIND
xitari at every frame's FIRST bus op (`poke $0296=34`). Across
frames 18-22, xitari first event always at sc=6, jutari at sc=5.

**Root cause hypothesis**: xitari's `Console::run()` calls
`TIA::frameReset()` AFTER the VSYNC=0 instruction completes — and
the reset preserves the partial-scanline beam progress via
`clocks = (cycles*3 - frame_start_clocks) % 228`. M6502High
increments cycles per bus op, but M6502Low (the default in ALE)
increments at instruction end. xitari's first new-frame instruction
starts with cycles=0 (post-resetCycles), but the FIRST BUS OP
records the post-increment cycles=1 → sc=0. Subsequent bus ops:
sc=2, sc=4, etc. (each ++ by some amount).

jutari's VSYNC handler resets `tia.scanline_cycle = 0`
immediately on the VSYNC=0 poke (mid-instruction). The next
instruction starts with `scanline_cycle = 3` (residual from the
3-cycle STA $0=0). xitari starts with scanline_cycle = ~4 due
to the resetCycles+stop()+frameReset timing.

The 1-cycle difference makes ~30% of frames render with a +1
scanline drift visible in the breakout side-by-side video.

**Why fixes tried so far didn't work**:

  1. Preserve scanline_cycle at VSYNC=0 (don't reset): no-op
     since scanline_cycle was already 0 at the VSYNC=0 poke
     (frame ends at scanline boundary).
  2. Clear framebuffer at VSYNC=0: made things worse (now NO
     frames aligned because previous-frame data is needed for
     scanlines jutari doesn't write).

**Next-step recipe for next session**:

  - Add a `tia.vsync_reset_pending::Bool` field. VSYNC handler
    sets it to true. In `_tia_post_step!`, after `tia_advance`
    has run with the full instruction cycles, check the flag
    and do the actual reset there (+ optional "phantom fetch"
    cycle to match xitari's stop+resetCycles+next-fetch timing).
  - Validate by re-running the per-bus-op trace diff — frame
    21 first event should land at sc=6 in jutari, matching
    xitari.

Severity: video polish only — gameplay/RAM/scoring is bit-exact.

### NEW open bug — jutari "jumping scanlines" in breakout video

**Visible symptom**: in the 60s breakout comparison video
(`tools/breakout_video/output/breakout_xitari_vs_jutari.mp4`,
commit `5ccab08`), the jutari panel occasionally shows its render
shifted UP by 1 scanline compared to xitari, then snaps back.
Makes the bricks and paddle "jump" intermittently.

**Quantified**: per-frame screen diff over 500 frames of breakout
random actions:

  - 347/500 frames: jutari and xitari pixel-bit-exact
  - 145/500 frames: jutari is EXACTLY +1 scanline ahead
    (`jutari[r] == xitari[r+1]` for every diverging row, d=0)
  - 8/500 frames: neither (1-frame transitions)

The S-state (shift +1) lasts 1-15 frames at a time, then snaps
back. Frames where it triggers seem game-event-driven (start of
brick-break? collision?).

**Key constraint**: jutari↔xitari RAM is BIT-EXACT for all 300
frames (commit `20b5de0`). Same CPU cycles per frame (19912). Same
`tia.scanline` and `tia.scanline_cycle` at every frame end
(verified by instrumentation: scanline=0, scanline_cycle=3, cc=9
on every frame from 1..60). So the divergence is in TIA's
*intra-frame* beam tracking — the renderer writes to different
framebuffer rows in some frames despite identical CPU state.

**Next-step recipe**:

  1. Find the first frame where drift appears (frame 21 in the
     dump). Instrument `tia_advance!` to log
     `(line_advance, completed_line, tia.scanline_before,
      cpu_cycles, scanline_cycle_before)` per call.
  2. Find one TIA write inside frame 21 where jutari's
     `tia.scanline` differs from what xitari would compute. The
     answer is likely either in `line_advance = total ÷ 76`
     integer-truncation behavior, or in the WSYNC stall
     calculation (`(76 - scanline_cycle) mod 76`), or in
     `tia_advance!`'s `new_line = tia.scanline + line_advance`.
  3. Build a 2nd xitari hook (similar to commit `d66b290`'s bus
     trace) that emits TIA's `myFramePointer` row index per
     scanline render, so you can diff against jutari's
     `completed_line + 1`. First differing tuple is the entry
     point.

The bug is NOT user-blocking now (game plays correctly, RAM is
bit-exact, scoring matches xitari). It's a polish item for the
side-by-side video.

### Earlier (2026-06-03 evening) — Breakout BIT-EXACT

**Big wins this 2026-06-02/03 sprint** (skim "Patches landed" for
details + commits):

  - jutari pong RAM 4 b/f → **bit-exact** (Phase 2b cycle counter +
    Phase 5 RIOT-read threading).
  - jaxtari pong RAM 13 b/f → **4 b/f** at $04/$3c (Phase 2b +
    Phase 5 mirror — residual is a single jaxtari-only INPT/
    floating-bus issue).
  - **jutari breakout RAM 9.9 b/f → bit-exact** (this morning's
    VDELBL/deferred-ENABL shadow latch fix, commit `20b5de0`).
    Lives counter `RAM[$39]` now decrements 5→4→3→2→1→0 every
    ~120 frames matching xitari. Regression test landed:
    `jutari/test/runtests.jl` `breakout ball-death` testset.
  - Per-bus-op xitari trace + diff infrastructure (commit
    `d66b290`) — the unblocker that let us bisect the BL-PF bug
    to a single instruction.

**jutari↔xitari RAM scoreboard, 100-frame NOOP** (post-fix):

  | ROM            | Mean b/f | Status |
  |----------------|----------|--------|
  | pong           | 0.0      | ✅ |
  | breakout       | 0.0      | ✅ (was 9.9 before today) |
  | space_invaders | 0.0      | ✅ |
  | asteroids      | 0.0      | ✅ |
  | seaquest       | 2.6      | 6 bytes diverge at FRAME 0 (boot) |
  | pitfall        | 19.8     | next target |
  | enduro         | 45.0     | next target |

### Active user-reported bugs (DOWN from 3 to 1)

  - ~~Breakout ball-doesn't-die~~ — **CLOSED** by commit
    `20b5de0`.
  - ~~jutari pong paddles frozen~~ — closed by Task #66 SwapPaddles fix.
  - jaxtari pong remaining 4 b/f at $04/$3c — the original
    "where we left off" recipe (instrument frame 23→24
    transition, find first divergent INPT poll) still applies.

### Recipe for next agent — pick a target

  - **jaxtari pong $04/$3c** (4 b/f at frame 24+): per-bus-op
    trace technique that worked for breakout. Diff jaxtari trace
    vs xitari at frame 24, find first byte mismatch on a TIA
    read (likely INPT4/5 or floating-bus noise).
  - **seaquest frame-0 boot divergence** (6 bytes): something in
    the boot sequence (random NOOP reset? RAM seeding?). Compare
    initial RAM state at boot end.
  - **pitfall / enduro**: bigger divergences from frame 0.
    Likely similar root cause but separate investigation.

### Test infrastructure note

`pyproject.toml`'s pytest `addopts` is `-q -n auto --dist
worksteal`, so `pytest` parallelises across all cores by default.
Override with `pytest -n 1` to debug a single test deterministically.

---

## Earlier "where we left off" — kept for archive purposes (2026-06-01)

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
