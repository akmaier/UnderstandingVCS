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

### 🎯 Task #80 LOCALIZED (2026-06-13) — seaquest boot off-by-1 is at FRAME 2: xitari's first post-reset frame is PARTIAL

Instrumented xitari's per-boot-frame RAM[$01] (the seaquest frame counter) with a
temporary, env-var-guarded probe in `StellaEnvironment::emulate` (printed
`system().peek(0x81)` after each `mediaSource().update()` when `XITARI_BOOTPROBE`
is set; reverted + rebuilt clean afterwards — the xitari reference is pristine).
Ran it on seaquest and compared to jutari's `tools/seaquest_boot_probe.jl`:

```
frame:     1   2   3   4   5  ... 60(noop) 61  62  63  64(reset)  end
xitari:   00  00  01  02  03  ...   3a     3b  3c  3d   3e        $3e (62)
jutari:   00  01  02  03  04  ...   3b     3c  3d  3e   3f        $3f (63)
```

**The divergence is exactly at FRAME 2.** Both ports start at 00 (frame 1), but
xitari stays at 00 for frame 2 while jutari increments to 01 — and the +1 offset
then rides along for the entire run. Root: **xitari's first post-reset frame is
PARTIAL.** `system().reset()` leaves the beam mid-frame; the first
`mediaSource().update()` completes only the short remainder of that frame, so the
seaquest cart reaches its RAM[$01]-increment instruction one frame LATER (first
increment at frame 3). jutari's `run_until_frame!` runs a FULL first frame after
`console_reset!`, so its cart hits the increment a frame earlier (frame 2).

**Ruled out:** harness frame-count mismatch (#95-style). xitari `reset()` =
frame-less `system().reset()` + 60 NOOP + `system_reset_steps`(=4, Defaults.cpp:47)
RESET + (no startingActions for seaquest); jutari `env_reset!` = frame-less
`console_reset!` + 60 NOOP + 4 RESET. Counts + structure match (60+4=64). The
bug is purely the partial-vs-full FIRST frame.

**Why only seaquest:** the 4 bit-exact ROMs (PXC1) already match xitari using
jutari's CURRENT full-first-frame boot ⇒ they are INSENSITIVE to this 1-frame
boot-init offset (their state stabilizes regardless). seaquest's free-running
RAM[$01] counter (drives the HUD luminance toggle, see RECON below) is the only
thing that exposes it — hence the 6-byte frame-0 RAM diff (`$01 $02 $60 $66 $7e
$7f`, the counter + its derivatives) and the 904 px HUD band.

**Fix direction (deliberate, NOT yet applied — high-risk boot timing per #93/#95):**
make jutari's FIRST post-reset frame partial like xitari (so the cart's first
increment slips from frame 2 → frame 3). Because the bit-exact ROMs are
insensitive to this offset, the change *should* fix seaquest without breaking
them — but a fresh-reset partial-frame change touches every ROM's boot, so it
MUST be gated on the full PXC1 + PXC2 + all-6 PXC-S suites before landing. The
mechanism likely lives near jutari's reset→first-`run_until_frame!` handoff
(cf. task #72 `vsync_reset_pending`): xitari's `system().reset()` leaves a
partial frame that the first `update()` flushes; jutari's `console_reset!` zeroes
the TIA beam so the first `run_until_frame!` is a full frame. Reproduce xitari's
"partial first frame after reset" to close the gap.

---

### 🔬 Task #80/#96 RECON (2026-06-13) — seaquest 904px is the boot off-by-1 (#80), NOT a render bug

Pixel-level diff of the seaquest noop-10 worst frame (jutari↔xitari fixtures):
  - Per-frame diff: `[10, 894, 5, 0, 5, 4, 5, 0, 5, 904]` — only frames 1 and 9
    diverge; frames 3 and 7 are BIT-IDENTICAL. The big diff is confined to
    rows 45-54 (the score/oxygen HUD band), nearly full width (cols 8-159).
  - The band is a flat color that toggles luminance `$90`↔`$92` on an 8-frame
    period. At frame 9: xitari row45 = `144` ($90), jutari = `146` ($92).
    Decisive: `xitari[f9] == jutari[f8]` (True) and `xitari[f9] == xitari[f8]`
    (True) but `jutari[f9] != jutari[f8]`. So **jutari is exactly 1 frame AHEAD
    of xitari** on the HUD toggle.
  - That 1-frame phase lead IS the **#80 boot off-by-1** (jutari does 63
    cart-increments during the 64-frame boot, xitari 62) made visible: the HUD
    color is driven by the frame counter, so a 1-frame state offset puts the
    toggle out of phase exactly on the period-8 toggle frames (1, 9).

**Implication:** seaquest's 904 px is DOWNSTREAM of state (#80), not a
rendering-logic bug. #96 ("genuine render divergence, not settings") was right
that it's not settings, but it isn't a renderer bug either — fixing the #80
boot cart-increment off-by-1 should collapse BOTH the RAM divergence AND the
904 px HUD band together. Confirms the plan's "seaquest STATE-first" ordering.
Next: close #80 (per-bus-op boot trace; lead = partial-frame `m6502.stop()` /
RIOT tick skipping one cart increment).

---

### 🏆 Task #97 LANDED — jutari (jaxtari mirror pending env-confirm) (2026-06-13) — enduro HMOVE-blank comb 249 → 33 px

enduro strobes HMOVE on free-running (non-WSYNC) kernel lines, where jutari's
beam lands ~1 CPU cycle early — a line-N+1 `cyc 0` strobe gets recorded as
line-N `cyc 75`. xitari blanks the leftmost-8px comb on every such line; jutari
was missing 27/210 rows (249 px). Two fixes:

1. **Wrap fix** (commit 2ac7344): `_hmove_blank_enabled_at` wraps `beam_sc>=76`
   to the next line's cycle instead of returning false. enduro 249 → 137 (both
   ports, MEASURED).
2. **`_next` deferral** (this commit): an HMOVE write with `beam_sc>=76` crossed
   into line N+1 mid-instruction, so its comb is parked in a new
   `hmove_blank_pending_next` flag (not line N's `hmove_blank_pending`) and
   promoted once the beam advances. enduro 137 → 33.
   - jutari `TIA.jl`: field + W_HMOVE routing + promotion after the drain in
     `tia_advance!`.
   - jaxtari `system.py`: functional mirror (field + tia_poke routing +
     promotion in `tia_advance`'s return-tuple).

**Verification status (honest — see CLAUDE.md "verify, don't claim"):**
  - jutari enduro = **33 MEASURED** (fixture regenerated + committed); jutari
    bit-exact ROMs (pong/breakout/pitfall/SI) stay **0**; all jutari runtests
    pass.
  - jaxtari `_next`: full jaxtari **TIA unit suite 127 passed** (incl. the
    HMOVE-blank tests), and it is a line-by-line mirror of the verified jutari
    logic. (Also fixed 4 stale jaxtari TIA unit tests broken since #83's
    Y_START gate — separate commit 484bc07.)
  - jaxtari ENV screen confirmation is IN-FLIGHT but pathologically slow (the
    jaxtari env re-traces per frame → ~25 min/ROM just for the 64-frame boot,
    ~2 h for the full PXC-S set). Per the jutari-first rule (never let a jaxtari
    stage block a verified jutari deliverable), the shared PXC-S enduro pin is
    **HELD at 137** — the last value measured on BOTH ports — so jutari arm
    passes at 33≤137 and jaxtari arm at 137≤137. A follow-up tightens the pin
    137 → 33 once jaxtari env enduro=33 is measured.

**Residual 33 px = road-border 1-cc positioning** (NOT the comb). Worst frame
(f5): 9 rows (53,68,76,78,102,104,144,146,154), each a symmetric 1-px edge swap
about x≈87.5 — the road borders (color $C0) converging to the horizon (narrow
rows 53-78, widening to 144-154); jutari pulls both borders 1 px toward center
vs xitari. Same ~1-cycle CPU↔TIA beam offset, now in RESP/HMOVE object
positioning; lives in the P3i-g cycle core — HIGH RISK to the 4 bit-exact ROMs,
deferred. (Original diagnosis detail below.)

---

### 🔬 Task #97 DIAGNOSED (2026-06-13) — enduro 249px = HMOVE-blank comb mis-placed by jutari's 1-cycle beam offset at scanline boundaries

enduro RAM is bit-exact (40 frames) → pure render. Localized the noop-10
worst-frame (249px) diff:
  - The diff is the **leftmost 8 px (cols 0-7)**: xitari = `$00` (black),
    jutari = `$c0`. Enduro strobes HMOVE on EVERY scanline at cycle 0, so
    the HMOVE-blank "comb" blacks cols 0-7 of every line in xitari (210/210
    rows blanked). jutari blanks only 183/210 — **missing 27 rows**
    (display 53,54,66-69,76-79,87-90,102-105,121-124,144-147,154). The
    remaining ~9 px of the 249 are the car sprites (cols 36/37/88/138/139).
  - **Root cause** (bus-trace, RAM bit-exact so writes are "the same"
    instructions): xitari records one HMOVE per line at `cycle 0`
    (internal sl 84-93 all cyc 0). jutari records the SAME number but
    occasionally **1 cycle early** — e.g. sl88's cyc-0 strobe lands as
    sl87 **cyc 75** (end of the previous line). So that line (88) gets no
    HMOVE → no comb → its left 8 px aren't blanked. The strobe COUNT is
    right; the boundary PLACEMENT is off by 1 CPU cycle.
  - This is the **same ~1-cycle CPU↔TIA beam-position offset** measured in
    pitfall's INTIM reads (jutari bus ops land ~3 cc / 1 cycle before
    xitari). It only bites enduro because enduro strobes HMOVE exactly at
    cycle 0 (right on the boundary), where a 1-cycle slip flips the
    scanline attribution. pong/breakout/SI don't strobe on the boundary,
    so they stay bit-exact.

**Refinement — it's the FREE-RUNNING (non-WSYNC) lines.** xitari's enduro
kernel is a MIX: some lines end with `STA WSYNC` (sl84,85 WSYNC@cyc73),
but others free-run with no WSYNC (sl86,87,88 — HMOVE@cyc0 each, reached
by cycle-exact code = exactly 76 CPU cycles between HMOVEs). On the
WSYNC'd lines jutari lands HMOVE@cyc0 fine (the stall snaps to the
boundary). On the FREE-RUNNING lines jutari's beam drifts: two HMOVEs 76
CPU cycles apart should be line N cyc0 → line N+1 cyc0, but jutari places
the second at line N **cyc75** — i.e. over a 76-CPU-cycle gap jutari's
beam advanced only 75 color-clock-lines-worth at the poke point. So
`beam_sc = tia.scanline_cycle + pending_tia_cycles` (Bus.poke! :352-353)
undercounts by 1 at the poke on free-running lines, OR the per-
instruction TIA advance in `_tia_post_step!` leaves scanline_cycle 1
short. That 1-cycle slip flips the HMOVE onto the previous line → that
line misses its comb (27 lines, 240 of the 249 px).

**Fix is in the P3i-g cycle-accounting core** — HIGH RISK; all 4
bit-exact ROMs depend on it. Next step: trace PC + scanline_cycle across
a free-running line pair, find where jutari's beam is 1 short of xitari's
`(clock - frameStart)` at the HMOVE write, and correct the beam_sc /
per-instruction advance WITHOUT moving the WSYNC'd-line behaviour. Gate
on all 6 PXC-S pins + PXC1/PXC2. Do NOT rush (discipline from #93/#95):
a wrong nudge here desyncs every ROM. Likely shares a root with pitfall's
INTIM ~1-cycle read offset — fixing one may fix both.

---

### 🗺️ ACTIVE PLAN (2026-06-13) — remaining PXC-S screen differences

Post-#95 state (all rendered with correct RomSettings; full PXC-S suite
**12 passed**). 600-frame random-action video accuracy, jutari↔xitari:

  | ROM            | RAM (same action stream) | screen worst | nature |
  |----------------|--------------------------|--------------|--------|
  | breakout       | bit-exact                | **0 px**     | done   |
  | space_invaders | bit-exact                | **0 px**     | done   |
  | pitfall        | bit-exact                | **0 px**     | done (#95) |
  | pong           | bit-exact                | 2 px @ f459+ | #98 ball sub-cycle |
  | enduro         | **bit-exact (40 f)**     | 257 px       | #97 PURE RENDER |
  | seaquest       | **diverges f0** (6 bytes)| 1449 px      | #80 STATE-first |

**Decisive recon** (`jutari_xitari_ram_diff.py` on the video streams):
  - enduro RAM bit-exact ⇒ its 257 px is a pure RENDER bug (state
    identical). Broad: 92 rows differ, hottest 53-54, 66-71, 154.
  - seaquest RAM diverges at frame 0 (`$01 $02 $60 $66 $7e $7f` — the
    task #80 boot off-by-1) ⇒ its 1449 px (hot band rows 45-53 = score/
    oxygen HUD) is largely DOWNSTREAM of state. Fix #80 first.
  - pong RAM bit-exact ⇒ 2 px at ball rows 160/161/176/177 is a pure
    ball-bounce sub-cycle render artifact.

**Order: enduro (#97) → seaquest state (#80) → pong (#98).**
  1. **#97 enduro** — pure render, can't break RAM, fully PXC-S-gated.
     (a) per-pixel dump of the first diverging frame at the hot rows →
     name the objects (53-71 horizon/road, 154 car/dashboard) + driving
     registers; (b) bus-trace align (expect identical writes → render-
     timing: road playfield / HMOVE comb / per-scanline COLU sky); (c)
     fix one mechanism, re-measure all 6 PXC-S pins + PXC1/PXC2, regen
     fixture + video.
  2. **#80 seaquest** — close the 6-byte frame-0 RAM divergence FIRST
     (per-bus-op diff of the last boot frames; lead = RIOT-timer /
     partial-frame `m6502.stop()` per the #80 notes), THEN re-measure the
     screen; much of the 1449 px should evaporate.
  3. **#98 pong** — bus-trace the bounce frame, compare `bl_x` + RESBL/
     HMBL at the exact cycle. 2 px polish, last.

Guardrails: see CLAUDE.md "Hard-won methodology" (RAM-first, harness
parity, never gate on tia.frame, bus-trace alignment, ~3cc offset).

---

### ✅ Task #95 CLOSED (2026-06-13) — it was a TOOLING bug, not the emulator: screen tools missing pitfall/enduro RomSettings

**Resolution of the whole saga.** After three rounds of (wrong) emulator
hypotheses, the round-3 proof — *all 1252 TIA writes identical, RAM
identical, INTIM identical* — was the tell: the EMULATOR was correct.
The bug was that the screen-comparison TOOLING rendered the jutari/jaxtari
side with the WRONG game settings.

**The bug.** Three files had a settings map listing only breakout + pong:
  - `tools/breakout_video/dump_jutari_frames.jl`  (video panel)
  - `tools/jutari_screen_dump.jl`                  (PXC-S jutari fixtures)
  - `jaxtari/tests/test_screen_conformance.py`     (PXC-S jaxtari live arm)
pitfall + enduro fell through to `GenericRomSettings`, which omits each
game's `getStartingActions` (Pitfall = 1× UP, Enduro = 1× FIRE). xitari
(`trace_dump`) always applies the real settings, so the jutari/jaxtari
panel was literally running a different game (no UP-start → Harry's
trajectory desyncs → "doesn't jump").

**How it was found.** A clean mimic of `dump_jutari` rendered Harry's
jump at the correct scanline (display 99 = xitari), but the committed
`.raw` (and a fresh `dump_jutari` run) showed display 104. Same ROM,
actions, boot — so the difference had to be the env construction.
Diffing the two: the mimic used `PitfallRomSettings()`; `dump_jutari`'s
`_settings_for_rom` returned `GenericRomSettings()` because pitfall
wasn't in its map. The RAM tool (`jutari_trace_dump.jl`) *did* map
pitfall → that's why RAM was bit-exact all along while the screen wasn't.

**Fix + verification (zero emulator change → zero PXC1/PXC2 risk):**
add pitfall + enduro to all three maps.
  - pitfall VIDEO: Harry jumps identically — **BIT-EXACT vs xitari for
    60 frames of random-action gameplay including the jump** (0 px;
    frame 20 both rise to row 103, apex row 95, land row 104).
  - enduro PXC-S: **516 → 249 px** on BOTH ports (settings-mismatch
    artifact removed). pitfall noop-10 stays 0 (UP-start doesn't change
    the first 10 noop frames). PXC2 jaxtari==jutari preserved (both 0 /
    249). jutari PXC-S arm 6/6 pass.

**Method lessons (the dead ends, for the next agent):**
  1. Two earlier #95 diagnoses were wrong (FIRE-input timing; "1-scanline
     render-phase lag"). Both came from instrumenting with a frame gate
     (`tia.frame == 24`) that was the WRONG frame — `tia.frame` is the
     VSYNC counter (~+65 from boot), NOT the env-step / video-frame
     index. Every render-side measurement gated on it captured a boot
     frame. **Gate debug on a flag set around the exact env_step, or on
     the env-step index — never assume tia.frame == video-frame.**
  2. "All writes identical + screen differs" does NOT always mean a
     render bug — it can mean the two SIDES were fed different inputs
     (here: different RomSettings via different tools). Check the harness
     parity (settings, boot, actions) before blaming the renderer.
  3. seaquest is still generic on both ports (jutari has no
     SeaquestRomSettings yet; jaxtari does). Its 904-px PXC-S residual is
     therefore partly a settings mismatch too — adding a jutari
     SeaquestRomSettings (emulator code, not just tooling) is the
     follow-up that would make seaquest apples-to-apples vs xitari.

---

### 🔬 Task #95 ROUND 3 (2026-06-13) — PROVEN render-only: all 1252 TIA writes identical, player renders 5 scanlines low (VDELP0 / deferred-GRP0 render phase) [SUPERSEDED — see CLOSED entry above; "render-only" was right that the emulator was fine, but the cause was tooling settings, not the render phase]

Used xitari's `trace_dump --bus-trace` (per-bus-op CSV) to compare the
jump frame (random-action frame 24) event-for-event against jutari's
`Bus.trace_enable!` trace. Definitive results:

  - **All 1252 TIA writes are IDENTICAL** between xitari and jutari —
    same register, same value, AND same beam-scanline (only the final
    frame-wrap write at sl262/sl0 differs). `reg/value` sequence: 0
    mismatches. So the CPU issues Harry's GRP0 bytes identically.
  - **GRP0 writes identical**: 244 writes, first non-zero at the same
    `(sl43,cc57,$60)`; Harry's bitmap `$18,$18,$10,$18,$1a,$3e,$7c…$33`
    on the same beam-scanlines 132-147 in BOTH.
  - **INTIM reads identical** read-for-read (433 reads, 0 value diffs);
    Pitfall busy-polls INTIM (TIM64T) but the values match — only the
    bus-op *color-clock* is 3cc (1 CPU cycle) earlier in jutari, which
    shifts 6 reads across a scanline boundary but never changes a value.
  - **RAM identical** (confirmed again, 330 frames).

  → The CPU, kernel, RIOT timer, and every TIA write are exonerated.
    **The bug is purely in jutari's RENDERER.** A fix here CANNOT break
    PXC1/PXC2 RAM conformance (render-only) — only PXC-S screen.

**What the renderer does wrong** (instrumented the render of frame 24):
Harry uses VDELP0=1, so he's drawn from the `grp0_old` shadow. Pixel
comparison shows jutari draws the *identical* sprite (colours
`12/4a/c8/d2`) exactly 5 scanlines BELOW xitari (jutari row 104 =
xitari row 99), while the ground band (`$14`, row 133+) matches. The
render-side `grp0_old` (and even the live `GRP0`) only reflects the
sl132 write at render-row ~137-138 — a ~5-scanline lag between the
(identical) GRP0 write's beam-scanline and the row where the renderer
actually paints it. The GRP0 writes are LATE (cc156-168) and DEFERRED;
combined with the VDELP0 shadow latch (now at activation-clock per
task #94) the player's first-non-zero row lands ~5 scanlines late.

**Status: NOT fixed.** The fault is in the deferred-GRP0-write drain +
VDELP0-shadow render phase — the exact machinery task #94 tuned to make
the pitfall HUD bit-exact. Changing it risks the HUD and the 4 bit-exact
ROMs. After two prior wrong #95 diagnoses, the discipline is: do NOT
ship a render-core guess. The precise next experiment: instrument, per
render-row in frame 24, the color-clock at which the player's
`grp0_old`/`GRP0` is sampled vs the color-clock of the deferred GRP0
write's activation, and compare to xitari's beam draw of the player at
cc≈84 (p0_x=16). The 5-row lag must come from the deferred write
activating after the player's cc each row, cascading through the
shadow. Fix candidate: when a deferred GRP0/GRP1 write's activation
clock is AFTER the player's draw clock, its shadow effect should still
apply to the NEXT row's player exactly as xitari's beam does — verify
jutari's per-color-clock player draw samples grp0_old at the right cc.
Gate ANY change on all 6 PXC-S pins + the pitfall jump (frames 20-40).

**Tooling note for next session:** `tools/trace_dump --bus-trace PATH
--bus-trace-frames LO,HI` emits xitari's per-bus-op CSV
(global_idx,frame,kind,scanline,scanline_cycle,color_clock,addr,value);
jutari's `Bus.trace_enable!()/trace_take!()` gives the matching stream.
Aligning the two write-sequences by index is the fastest way to prove
register-vs-render divergence (that's how round 3 nailed render-only).

---

### 🎯 Task #95 RE-DIAGNOSED (2026-06-13, round 2) — jutari doesn't apply Harry's VERTICAL JUMP position (sprite Y pinned to ground)

**User caught what noop-only testing missed:** "at the beginning of the
pitfall video the player jumps on xitari while jutari stays on the
ground; later jutari also jumps. Is something mixed up with jump vs.
jump+fire in jutari?" This is a REAL, significant gameplay bug — and
both my earlier #95 diagnoses below (round 1 "render phase 1-scanline
late", and the original "FIRE input timing") were WRONG.

**Evidence chain (all verified this session):**

1. **RAM is byte-identical, jutari↔xitari, for 330 frames** of the
   random-action stream — including the FIRE frame (20) and the entire
   jump. `tools/jutari_xitari_ram_diff.py`, position-aligned.
2. **jutari's INPT4 reads FIRE correctly**: NOOP→$80 (released),
   FIRE→$00 (pressed). The button reaches the chip. (Input-mixup
   hypothesis killed.)
3. **The jump IS in RAM and jutari HAS it**: byte `$69` is Harry's
   vertical jump position. Over frames 18-40 xitari's `$69` runs
   `20 20 20 1f 1e 1d 1c 1b 1a 19 18 18 17 …15… 16` (rises then
   falls — the parabola). jutari's `$69` is **identical** at every
   position.
4. **Yet jutari renders Harry at a FIXED ground row** while xitari moves
   him per `$69`:

   | frame | xitari Harry top | jutari Harry top | `$69` |
   |-------|------------------|------------------|-------|
   | 19    | 104 (ground)     | 104              | 0x20  |
   | 20    | 103 (rising)     | 104              | 0x1f  |
   | 24    | 99               | 104              | 0x1b  |
   | 30    | 95 (apex)        | 104              | 0x16  |
   | 40    | 94               | 104              | 0x15  |
   | 58    | 104 (landed)     | 104              | 0x20  |

   The xitari→jutari gap equals the jump height (1 px at f20, 5 px at
   f24), so it is NOT a constant render offset — jutari is rendering
   Harry as if `$69` were always its ground value 0x20.

**Conclusion:** the game logic (CPU + RAM + INPT4 fire) is in perfect
lockstep — the fire correctly triggers the jump and `$69` animates
identically. The bug is purely in **how jutari converts `$69` →
on-screen sprite Y**. In a 2600, a sprite's vertical position is set by
a cycle/scanline-timed "coarse position" loop in the display kernel
(skip N WSYNC'd scanlines, where N derives from `$69`, then draw GRP0).
jutari's render produces a fixed top row regardless of N, so the
scanline-skip count is not translating into screen position the way
xitari's beam does. This is adjacent to the WSYNC↔scanline-phase area
flagged in round 1, but the EFFECT is "vertical position doesn't track
RAM," a behavioral render bug — far more than the cosmetic 1-row colour
shift I claimed in round 1.

**Why noop missed it:** under NOOP, `$69` stays at its ground value
0x20 every frame, so Harry never moves vertically and jutari's fixed-Y
render coincides with xitari. PXC-S (noop fixtures) is therefore blind
to it — pitfall noop is genuinely bit-exact; this only appears once
gameplay drives `$69`.

**REFINEMENT (same session, full 600-frame trajectory):** it is a
DESYNC, not a permanent pin. jutari's Harry DOES reach the full vertical
range (top rows 93-104 occur) — it just jumps at the WRONG frames:
  - jump window 1: xitari rises f20-57; jutari stays grounded the whole
    window (the jump the user saw missing).
  - window ~f145-160: jutari jumps ~13 frames EARLIER than xitari
    (jutari leaves row 133 at f145; xitari not until f158).
So jutari is neither always-grounded nor a fixed phase offset — the
vertical render is inconsistently early/late/missing vs xitari. Since
`$69` (and all RAM) is byte-identical at every frame, the vertical-
positioning kernel must consume a NON-RAM input that differs between
ports — most likely the RIOT timer (`INTIM`; we have prior INTIM
off-by-one history, task #15 pt8) used as a coarse-position countdown,
or a collision/INPT read mid-kernel. That is the thing to trace next,
NOT the GRP0/COLUP0 register paths. (Heuristic caveat: the row-133
frames are likely Pitfall's underground level region, so the per-frame
"Harry top" numbers conflate level + jump; the DESYNC conclusion holds
regardless.)

**Next step (concrete):** trace the kernel's reads of `$69` and the
scanline at which GRP0 first goes non-zero for Harry, in BOTH engines,
for frames 20-30. Find why jutari's "first GRP0 scanline" is fixed
while xitari's tracks `$69`. Likely the coarse-vertical-position loop
(WSYNC + DEC + BNE) interacts with jutari's WSYNC stall / scanline
counter so each iteration doesn't advance the beam the way xitari's
does. This is the same render-phase core as #94's neighbourhood — high
risk to the 4 bit-exact ROMs — so reproduce with a minimal trace and
gate every change on all 6 PXC-S pins.

---
*Round-1 (also incorrect) diagnosis kept for the record below.*

### 🔬 Task #95 round 1 (2026-06-13) — "renders 1 scanline late (render phase)" — SUPERSEDED

Investigated the post-#94 pitfall random-action divergence. **The initial
"FIRE input-timing, pong #77 class" hypothesis (below) was WRONG** —
disproved by the very first measurement, exactly the kind of mis-framing
the method lesson warns about. Verify-first paid off again.

**Step 1 — RAM diff (`tools/jutari_xitari_ram_diff.py`, pitfall, the
random action stream): jutari↔xitari RAM is BIT-EXACT for all 120 frames
checked**, including the FIRE frame (20) and well past it. So the game
state is in perfect lockstep — this is a PURE TIA-RENDER divergence, not
input/CPU. (Input-timing diagnosis killed.)

**Step 2 — screen diff at frame 20**: 31 px in a narrow vertical band,
cols 18-22, rows 103-124 — Harry's multi-colour player sprite. Rows
32-102 and the HUD (0-31) are bit-exact, so it is NOT global scanline
drift; it is a localized, self-correcting 1-line shift confined to the
player kernel's band.

**Step 3 — the player band renders ONE SCANLINE LATE.** Harry's kernel
is a per-scanline colour kernel: every line (WSYNC-aligned, HMOVE at
cc0) it writes `COLUP0` at cc≈18 (early HBLANK) for that line's colour
and `GRP0` late (cc≈120-168) for the NEXT line's graphics; VDELP0 is
toggled on so the player also rides the GRP0 shadow (task #94 path).
Both the colour AND the graphics land one display row low in jutari
(row 103: xitari shows the bar in colour $12, jutari still shows
background $d6).

**Step 4 — instrumented the poke + render directly (frame 20):**
  - render of internal scanline 148 uses `COLUP0=$c8`; xitari colours
    it `$d2`. jutari's COLUP0 only becomes `$d2` at internal sl 149.
  - the `COLUP0=$d2` poke fires at `tia.scanline = 149`, `scanline_cycle
    = 3` (beam_cc 18) — i.e. when the kernel writes the setup for the
    line it intends to colour, **jutari's scanline counter is already
    +1** relative to the display row that write must affect. The render
    of line 148 has already happened by then, so line 148 keeps the
    previous colour and the new value bleeds onto 149.

**Root cause (narrowed):** the HBLANK "scanline-setup" writes
(`COLUP0`/`GRP0`) in Pitfall's Harry kernel are applied immediately
(beam_cc < 68 → not deferred; see the deferred-block comment "applying
them immediately for the whole scanline gives the same render"). That
assumption holds when the line they set up is rendered AFTER the write —
true for pong/breakout/SI — but Pitfall's WSYNC phase puts jutari's
`tia.scanline` one ahead at the moment of the write, so the line they
mean to colour was already committed.

**Why not fixed this session:** the fix lives in the WSYNC ↔ scanline-
render PHASE (which beam line a post-WSYNC HBLANK write belongs to), the
highest-risk part of the core. pong / breakout / space_invaders / pitfall
are all BIT-EXACT under noop and must stay so; PXC-S gates on noop
fixtures, and this residual is gameplay-only (random-action video). A
rushed phase change here is exactly how #93 regressed 5 ROMs. Handing off
with the precise evidence instead.

**Next attempt (concrete):** compare jutari's `tia.scanline` at a known
post-WSYNC HBLANK write against xitari's `(clock-myClockWhenFrameStarted)
/228` scanline at the same write, on this kernel, to confirm the +1 and
locate where the phase diverges (likely `tia_apply_wsync!` advancing the
counter, or the render committing line N at entry rather than exit). Then
make post-WSYNC HBLANK writes land on the line that follows, WITHOUT
moving the boundary for the bit-exact ROMs — validate all 6 PXC-S ROMs
stay at their pinned numbers before/after.

---
*Original (incorrect) hypothesis, kept for the record:* "Same class as
pong's frame-20 FIRE bug (task #77): input-read timing at the FIRE edge."
Disproved by the RAM diff (bit-exact through the FIRE frame).

### 🏆🏆 Task #94 CLOSED (2026-06-12) — activation-time VDELP shadow latch (PITFALL BIT-EXACT, 4 of 6 ROMs now 0 px)

**The final root cause of the pitfall "time renders in wrong format" bug**
— and the #92 seaquest/enduro regression, in one mechanism.

**Verification-first this time**: before writing code, traced pitfall's
HUD scanlines via `JuTari.Bus.trace_enable!` (script pattern in
`/tmp/_pitfall_hud_trace.jl`). Found the classic Activision 6-digit
kernel: per HUD scanline (sl 43-50), GRP0/GRP1 are written SEVEN times —
3 in HBLANK (immediate) + 4 mid-visible-region (deferred, activation
x = 23 / 32 / 41 / 50):

```
sl=44 cc=  9 x_act=-58: GRP0=$00   (HBLANK — immediate)
sl=44 cc= 33 x_act=-34: GRP1=$00   (HBLANK — latches DGRP0=$00)
sl=44 cc= 57 x_act=-10: GRP0=$46   (HBLANK — first digit "2"!)
sl=44 cc= 90 x_act= 23: GRP1=$66   (deferred — latches DGRP0=$46)
sl=44 cc= 99 x_act= 32: GRP0=$66   (deferred — latches DGRP1=$66)
sl=44 cc=108 x_act= 41: GRP1=$66   (deferred — latches DGRP0=$66)
sl=44 cc=117 x_act= 50: GRP0=$66   (deferred — latches DGRP1=$66)
```

With VDELP0=VDELP1=1, the players render from the SHADOWS, and the
shadow value evolves mid-scanline: DGRP0 = $00 until x=23, $46 ("2")
for x in 23..40, $66 ("0") after. Each digit value lives in the shadow
for ~2 copy slots. jutari/jaxtari captured the shadow ONCE per write at
POKE time into a single field — the render then used the FINAL value
($66) for the whole row: wrong digit values (the user's literal
"renders numbers in wrong format") plus phantom copies left of x=23
where xitari's shadows were still $00.

Task #91's "latest pending value" lookback was a half-fix: it changed
WHICH single snapshot the whole row used (final instead of stale) —
right for 2 of pitfall's digit slots, wrong for others, and net-WORSE
for enduro/seaquest (the #92 regression).

**Fix** (both ports):

  - `jutari/src/tia/TIA.jl`: new `_apply_pending_write!(tia, reg, val)`
    — stores the register AND runs the GRP0/GRP1 VDELP/VDELBL shadow
    latch (`grp1_old = GRP1` on GRP0 writes; `grp0_old = GRP0`,
    `enabl_old = ENABL` on GRP1 writes). Used at all three drain sites
    (visible per-color-clock loop, defensive post-loop drain, VBLANK
    drain). The deferred branch of `tia_poke!` now ONLY queues — no
    poke-time shadow capture, no lookbacks. This is exactly xitari's
    ordering: `updateFrame(clock + delay)` then mutate (TIA.cxx case
    0x1B/0x1C) — store + latch effective at the activation clock,
    against the register file as of that moment.
  - The 2026-06-03 ENABL lookback (breakout frame-92 fix) is subsumed:
    a same-scanline ENABL write with an earlier activation clock is
    already stored when GRP1's latch reads it.
  - `jaxtari/jaxtari/tia/system.py`: functional mirror
    `_apply_pending_write(tia, reg, val) -> TIAState` at the same three
    drain sites + threading `grp0_old/grp1_old/enabl_old` from
    `tia_for_render` into `tia_advance`'s final `_replace` (previously
    the shadows were dropped on the floor there — only `registers` was
    threaded).

**Numbers (worst-frame screen diff vs fresh xitari fixtures, jutari):**

  | ROM            | post-#91 | post-#94            |
  |----------------|----------|---------------------|
  | pong           | 0        | **0**               |
  | breakout       | 0        | **0**               |
  | space_invaders | 0        | **0**               |
  | pitfall        | 166      | **0 BIT-EXACT** 🎉  |
  | seaquest       | 1087     | **904** (best ever; pre-#91 was 1043) |
  | enduro         | 1133     | **516** (best ever; pre-#91 was 660)  |

All 1170+ jutari runtests pass. Task #92 (the regression) is RESOLVED —
both ROMs now beat their pre-#91 numbers. Task #93 (skip-first-copy) was
the WRONG TREE for pitfall: the measured RESP case there is when=-1
(no skip; oldx==newx). Skip-first remains a real xitari semantic that
may matter inside the remaining seaquest/enduro residuals, but nothing
currently attributes to it — downgraded to low priority.

**Method lesson** (for the next agent): two blind implementation rounds
of #93 cost a session; ONE 40-line trace probe pinpointed the real
mechanism in minutes. For render divergences, dump the per-scanline
write timeline FIRST and check whether the divergence region's pixels
depend on state that EVOLVES mid-row.

---

### 🏆 Task #91 CLOSED (2026-06-12) — fresh xitari fixtures + GRP shadow lookback (space_invaders BIT-EXACT, pitfall HUD digits render)

**Two compounded bugs:**

1. **STALE xitari fixtures** — `tools/fixtures/screens/pitfall_noop_10.screen.gz`
   was off from a fresh `tools/trace_dump --screen` run by **5570 px** for
   pitfall and **1140 px** for enduro (almost certainly committed before
   tasks #81/#82 landed their pitfall/enduro starting-action fixes). The
   entire previous task #86 root-cause analysis ("xitari RESP* skip-first-
   copy semantic") was diagnosing against outdated ground truth and was
   WRONG. Reverted those exploratory changes; regenerated the two fixtures
   from fresh trace_dump output. pong/breakout/space_invaders/seaquest
   fixtures verified still bit-exact.

2. **GRP0/GRP1 shadow-latch missed pending writes** — in
   `jutari/src/tia/TIA.jl::tia_poke!`'s deferred path (line ~479), the
   GRP0 / GRP1 VDELP shadow capture read `tia.registers[W_GRP*+1]` for
   the OTHER player's grp*_old — but if that GRP* was itself pending
   (in `pending_writes`, not yet drained), the live register is STALE.
   xitari has no deferral, so `myDGRP0 = myGRP0` / `myDGRP1 = myGRP1`
   sees the just-written byte. Without lookback, jutari's
   `STA GRP0; STA GRP1` two-digit kernel captures the PRE-GRP0-write
   byte into grp0_old → with VDELP0=1 the second digit row renders
   the wrong sprite shape, and the first digit row renders solid
   (no sprite at all).

**Fix** (`jutari/src/tia/TIA.jl::tia_poke!` deferred GRP* branch):

  - When W_GRP0 is poked, scan pending_writes for the LATEST pending
    GRP1 byte (regardless of activation_clock — the latch fires at
    cart-write time per xitari) and use that as `grp1_old`.
  - When W_GRP1 is poked, scan pending_writes for the LATEST pending
    GRP0 byte → `grp0_old`. Same lookback already existed for ENABL.

**Numbers (worst-frame screen diff, after fixture regen + fix):**

  | ROM            | Pre-#91 (stale)  | Post-regen          | Post-fix       |
  |----------------|------------------|---------------------|----------------|
  | pong           | 0                | 0                   | **0**          |
  | breakout       | 0                | 0                   | **0**          |
  | space_invaders | 42               | 42                  | **0 BIT-EXACT**|
  | pitfall        | 553              | 184                 | **166** (HUD digits render correctly) |
  | seaquest       | 1043             | 1043                | 1087 (+44, exposed downstream) |
  | enduro         | 774              | 660                 | 1133 (+473, exposed downstream) |

The seaquest/enduro regression is from a different latent bug whose
wrong-but-consistent output was canceling the wrong-but-consistent
shadow-latch staleness. Now that the shadow latch matches xitari
exactly, this other bug is exposed. Filed as task #92; not addressed
in this commit.

**Pitfall HUD visualization** (rows 9-29, cols 60-67, color $D2 lit):

  Before fix (stale fixture diagnosed → wrong target):
  ```
  r 9: ████████████████████████████████  (TIME digit MISSING — solid)
  ...
  r22: ····████████████████████████████  (SCORE digit present but WRONG shape)
  r23: ███··███████████████████████████  (3-wide bar instead of 2)
  ```

  After fix:
  ```
  r 9: ····████████████████████████████  (TIME digit row TOP — matches xitari)
  r10: ·██··███████████████████████████  (TIME digit "0" left bar — matches xitari)
  r22: ····████████████████████████████  (SCORE digit row TOP — matches xitari)
  ```

All 1170+ jutari runtests pass.

---

### 🔬 Task #93 ATTEMPTED & REVERTED (2026-06-12) — xitari skip-first-copy semantic

After task #91 closed pitfall's HUD top/bottom (TIME row top + SCORE row
top now match xitari), the remaining 166 px residual is concentrated at
rows 9-29 cols 19-48 — jutari draws EXTRA digit copies to the LEFT of
where xitari's "0:00:00" time digits start. xitari skips the first NUSIZ
multi-copy via `myCurrentPMask = ourPlayerMaskTable[align][1][mode][...]`
when RESP fires OUTSIDE the "delay section" of a copy (case=0 or +1).

**Attempted implementation** in `jutari/src/tia/TIA.jl`:
  - Added `skip_first_p0/p1::Bool` fields to TIAState (defaults false).
  - Added `_resp_when_case(mode, oldx, newx)` helper returning -1/0/+1
    per `ourPlayerPositionResetWhenTable` (TIA.cxx:928-1042).
  - In W_RESP0/W_RESP1 handlers: compute case using OLD `tia.pX_x` as
    `oldx` and the new `_resp_player_position(beam_cc)` as `newx`, look
    at pending NUSIZ* writes via lookback for the latest CPU-written
    NUSIZ. Set `skip_first_pX = (when != -1)`.
  - In `_overlay_player!` and `_player_set`: when skip_first is set,
    iterate `copy_offsets[2:end]` to drop the leftmost copy.

**Result**: REGRESSED most ROMs:

  | ROM            | Pre #93 | Post #93 | Δ      |
  |----------------|---------|----------|--------|
  | pong           | 0       | 0        | =      |
  | breakout       | 0       | 64       | +64    |
  | space_invaders | 0       | 634      | +634   |
  | pitfall        | 166     | 485      | +319   |
  | seaquest       | 1087    | 1179     | +92    |
  | enduro         | 1133    | 1101     | -32    |

Reverted (`git checkout jutari/src/tia/TIA.jl`). Only pong stayed
bit-exact because its paddle RESP fires at `newx == oldx` every frame,
keeping case=-1 (delay section).

**Diagnosis hypotheses for the next attempt:**

1. **Case-table coverage** — xitari iterates `newx` in [0, 160+72+5) when
   building the table; the modulo-160 fold causes MULTIPLE assignments
   per `(mode, oldx, newx_mod)` cell. My impl only checked `nx` and
   `nx+160`. May need the full 237-step walk per cell.
2. **`updateFrame(clock + 11)` semantic** for case=1 — xitari renders
   the partial first-copy display BEFORE moving the player. Without
   this, the first copy's bit-pattern bleeds into the wrong row.
3. **Mode 5 / mode 7 offset-by-1 quirk** — the double/quad-size mask
   uses `>` not `>=` at x==0, delaying the leftmost pixel. My case-skip
   doesn't honor this — may incorrectly suppress visible pixels.
4. **Shadow / skip-first interaction** — task #91's GRP shadow lookback
   changed which GRP byte is rendered between RESP and the next render.
   If skip-first applies to the OLD shadow but the NEW shadow renders
   ALL copies (because skip-first only updates at the NEXT RESP), this
   is a temporal mismatch.
5. **Direct trace diff** — instrument BOTH xitari and jutari to dump
   `(scanline, cycle, RESP target, NUSIZ, oldx, newx, computed when,
   resulting enable axis)` at every RESP write and diff. That nails the
   exact divergence cell.

**ROUND 2 ATTEMPT (2026-06-12 same session)** — added the missing NUSIZ-
write and HMOVE-write resets (xitari case 0x04/0x05 hardcode
`ourPlayerMaskTable[align][0][...]`, i.e. `enable=0` — they RESET skip-
first). Also added end-of-scanline reset per TIA.cxx:1796-1802 TODO
comment ("reset at end of scanline since the other way would be too
slow"). RESULT: NO REGRESSIONS but also NO IMPROVEMENTS — all 6 ROMs
identical to baseline (pong/breakout/space_invaders BIT-EXACT, pitfall
166, seaquest 1087, enduro 1133). Skip_first never visibly fires.

**ROOT CAUSE OF ROUND 2 NULL RESULT** — jutari renders ONE FULL SCANLINE
per `tia_advance` call (per-pixel within the call, but the skip_first
state is captured at the START and the scanline render is monolithic
from the renderer's perspective). xitari, by contrast, mutates
myCurrentPMask MID-SCANLINE at RESP time, and the subsequent pixels
within the same scanline use the new mask. For pitfall's TIME row, the
RESP fires AT the leftmost-copy position; xitari's pixels AFTER that
position render with enable=1 (the leftmost copy is suppressed because
its display region was just passed). jutari's scanline render fires
AFTER the CPU instruction containing RESP, so by then the whole scanline
gets rendered with the post-RESP skip_first — either ALL pixels see
skip_first=true (over-suppressing leftmost-copy across whole row) or
NONE do (if the next NUSIZ/HMOVE clears it before render fires).

**ROUND 3 / 4** — tried removing the HMOVE reset (kept only NUSIZ +
scanline-end reset) per the actual xitari source layout. Same nulled
result.

**FIX REQUIRES** a per-color-clock skip_first tracker (analogous to the
pending_writes mechanism for register stores). The render loop already
iterates color clocks 68..227 with mid-scanline writes; it would need to
ALSO honor a "skip_first changes at color clock X" event from the RESP.
Round 2's TIAState fields were architecturally sound but the render
loop wasn't extended.

Reverted round 2 changes (`git checkout jutari/src/tia/TIA.jl`). Task
#93 stays open; the right next attempt is the per-color-clock skip_first
tracker, not a one-bit-per-scanline approximation.

---

### 🔬 Task #80 SWEEP RESULTS (2026-06-11)

Ran `tools/seaquest_boot_probe.jl` + a reset-timing sweep on jutari's
seaquest boot to rule out hypotheses:

  - **Reset switch state is irrelevant**: pressing reset for all 4
    boot reset frames vs leaving it unpressed yields the same
    `RAM[$01]=$3f`. Seaquest's cart code does NOT branch on SWCHB
    bit 0 — its frame-counter increment runs every frame regardless.
  - **Reset switch press timing is irrelevant**: pressing reset at
    any frame offset (60..63) leaves `RAM[$01]=$3f` unchanged.
  - **Hold count IS the lever**: reset_hold = 3 gives `$3e` (matches
    xitari); reset_hold = 4 gives `$3f` (the off-by-1 jutari shows).
    So jutari's 64-frame boot does 63 cart-increments while xitari's
    does 62.

Conclusion: the divergence is NOT a SWCHB/event-bus latency bug at
the reset transition (the earlier round 2 hypothesis). xitari must be
silently skipping ONE additional frame's cart code somewhere — most
likely related to either an interrupt-cycle / RIOT timer tick or a
partial-frame `m6502.stop()` interaction that doesn't reach the cart's
increment instruction.

Closing #80 properly needs a per-bus-op diff of the last few boot
frames between jutari and xitari (the cycle-trace tooling at
`tools/cpu_tia_cycle_trace.jl` + `tools/trace_dump --bus-trace` can
do this, but requires patching trace_dump to emit boot-burn bus
ops too — currently it only traces user-action frames). PXC1 RAM
divergence stays at 4 bytes for seaquest (1 byte at boot end +
3 downstream propagation).

### 🏆 Task #83 CLOSED (2026-06-11) — Y_START framebuffer-write gate (pong BIT-EXACT)

**Root cause** (the actual, narrow one — not the "needs per-cycle render refactor"
I'd written off it as in the round 2 notes):

xitari's `myClockStartDisplay = myClockWhenFrameStarted + 228*myYStart`
makes the framebuffer pointer skip the first `myYStart` (= 34 for pong)
scanlines of each frame — pixels "rendered" in that pre-display region
go nowhere, AND xitari's `myHMOVEBlankEnabled` flag only gets cleared
from inside the framebuffer-writing branch (TIA.cxx:1776-1786). Result:
xitari preserves the HMOVE-blank flag across all pre-Y_START scanlines
and consumes it at the FIRST visible scanline (= display row 0).

jutari's framebuffer was indexed by absolute scanline (0..243); the
visible-render branch wrote to `framebuffer[27, :]` for pong's HMOVE-
strobe scanline and CONSUMED `hmove_blank_pending` there. By the time
display row 0 (= absolute scanline 34) rendered, the flag was already
false — no comb, 8 px residual.

**Fix** (commit pending — `jutari/src/tia/TIA.jl::tia_advance!`):

  - Per-pixel rendering + collision detection still runs at every
    scanline (so unit tests that check collisions at scanline 0 keep
    working).
  - Framebuffer write is now gated on `tia.scanline >= Y_START`.
  - `tia.hmove_blank_pending = false` only happens inside the framebuffer-
    writing branch (matching xitari `TIA::updateFrame` exactly).

Mirror edit in `jaxtari/jaxtari/tia/system.py::tia_advance` with the
same gate + a `wrote_framebuffer_this_advance` flag threading through
the post-render `new_hmove_blank` decision (jaxtari is functional, no
in-place mutation, so the gate has to flow through the return-tuple).

3 jutari unit tests had to nudge `tia.scanline = Y_START` to keep
exercising the framebuffer-write path (`tia_advance! writes scanline
on boundary`, `tia_advance! writes multiple scanlines`, `program
writes playfield then WSYNC renders scanline`, `VBLANK clear resumes
framebuffer writes`, `P3i-b framebuffer matches pre-P3i render`).
Collision tests didn't need any changes — collisions still run at
every scanline.

**Numbers (worst-frame screen diff, after fixture regen):**

  | ROM            | Post-#85 | Post-#83 |
  |----------------|----------|----------|
  | **pong**       | 8        | **0 BIT-EXACT** |
  | breakout       | 0        | 0        |
  | space_invaders | 42       | 42       |
  | pitfall        | 553      | 553      |
  | seaquest       | 1043     | 1043     |
  | enduro         | 774      | 774      |

All 1170+ jutari runtests pass.

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

### 🔬 Task #86 NEW (2026-06-11) — pitfall HUD render diverges from xitari

**Symptom** (user-visible): pitfall's top score/timer overlay renders
with a BLACK background in jutari where xitari renders a brown ($0c)
background. Cross-ROM PXC-S shows 553 px residual (worst frame),
concentrated at:

  - Rows 9-29 cols 28-66 (top score + timer)
  - Rows 190-196 cols 29-60 (bottom HUD strip)

In both regions: **xitari pixel = $0c (brown), jutari pixel = $00
(black)**.

**Investigation done:**

  1. The cart writes COLUBK only 3 times per pitfall frame:
     `sl 13 → $d6`, `sl 151 → $00`, `sl 167 → $00` (real TIA addresses
     with A7=0 — the trace's $0089 / $00c9 events are RAM mirrors,
     NOT TIA, and decode to RAM[$09] / RAM[$49] respectively in both
     jutari and xitari).
  2. Between sl 13 and sl 151, jutari's COLUBK register stays at
     `$d6` — confirmed via per-scanline debug print in `tia_advance!`.
  3. The HUD scanlines (~ sl 43-63 = display rows 9-29) thus render
     PF-clear pixels as `$d6` (brown-purple).
  4. xitari's fixture shows `$0c` (brown) at those PF-clear pixels.
  5. PXC1 RAM is bit-exact for pitfall at end-of-frame, so the carts'
     RAM is identical. But mid-frame TIA register state must differ
     — somewhere xitari's COLUBK becomes `$0c` while jutari's stays
     `$d6`.

**Additional finding (user follow-up):** The DIGIT SHAPES themselves
are wrong, not just the BG color. xitari row 22 cols 28-66 has a
COMPLEX pattern with `$d2 / $0c` alternation in non-uniform segment
widths (the actual digit topology). jutari row 22 cols 28-66 is a
CLEAN 4-on / 4-off `$d2 / $00` alternation — a uniform block
pattern.

**Pixel-level analysis (sl 43 = display row 9):**

Bus trace shows at sl 41/42 the cart sets up the HUD render:
  `NUSIZ0 = NUSIZ1 = $13` (three copies close, 16-cc spacing)
  `COLUP0 = COLUP1 = $00` (reset from earlier $0c)
  `COLUPF = $d2`  `COLUBK = $d6` (from sl 13)  `CTRLPF = $01`
  `PF0 = $00, PF1 = $10, PF2 = $14`
  `RESP0 → p0_x = 21`,  `RESP1 → p1_x = 30`

For sl 43, cart writes GRP0/GRP1 = `$00, $00, $3c, $3c, $3c, $3c, $3c`
at cc 9/33/57/90/99/108/117.

**Expected jutari render at sl 43 col 23 (=  P0 copy 0 bit 5 set):**
  - sets.p0 includes col 23 with GRP0=$3c (after cc 57's deferred
    write activates at cc ≈ 70)
  - render_pixel returns COLUP0=$00 (black)

**xitari row 9 col 23 = $d2** — xitari does NOT render the player at
col 23, even though our state model says it should.

**Three plausible causes:**

  1. **xitari's `computePlayerMaskTable` semantics differ from
     jutari's `_object_pixel_sets`** for NUSIZ multi-copy. xitari may
     use a SINGLE mask precomputed at the most recent `RESP0/GRP0/
     NUSIZ0` change; jutari rebuilds the full multi-copy set per
     mid-scanline GRP swap. The defer-list activation order differs.
  2. **xitari's GRP0 latch has a per-pixel delay** that jutari skips.
     The xitari mask table indexes by `(POSP0 & 0x03, NUSIZ & 0x07,
     scale, 160 - (POSP0 & 0xFC))` — there may be a "from-the-last-
     latched-position-only" rendering that jutari doesn't honour.
  3. **`COLUP0 / COLUP1` apply IMMEDIATELY in jutari** but go through
     xitari's per-pixel sub-cycle latch differently — at sl 43 cc 23
     xitari's `myCOLUP0` might still be $0c (set at sl 14), letting
     the player render as $0c which blends with the $0c BG segments
     in the digit.

Next-agent direction: read `xitari/emucore/TIA.cxx` (~line 2400 area
where `myCurrentP0Mask` is built per RESP/NUSIZ/GRP change) and
compare against `jutari/src/tia/TIA.jl::_object_pixel_sets`. The
deferred-write block in `tia_poke!` queues per-register writes but
the rendering uses a SINGLE post-write `sets.p0` covering ALL three
copies — xitari's per-copy mask may not work that way.

### 🔥 Task #86 ROOT CAUSE LOCATED (2026-06-11) — xitari RESP* skip-first-copy semantic

Read `xitari/emucore/TIA.cxx` line 2238 (`case 0x10: // Reset Player 0`):
xitari's RESP0 handler does NOT just set `myPOSP0 = newx`. It also
consults `ourPlayerPositionResetWhenTable[NUSIZ_lo][oldx][newx]`:

  - **case +1** ("RESP fires during display of a copy"): `myPOSP0 =
    newx`, AND `myCurrentP0Mask` set to the **`enable=1` mask variant
    that skips copy 0 of the multi-copy player**.
  - **case 0** ("RESP fires between copies"): same skip-first-copy
    mask.
  - **case -1** ("RESP fires during the 4-pixel `delay` slot in front
    of any copy"): `myPOSP0 = newx`, mask covers ALL copies
    (`enable=0`).

The table `ourPlayerPositionResetWhenTable` (built in `TIA::compute*`
~L920) is a 3D lookup `[mode 0..7][oldx 0..159][newx 0..159]`. For
NUSIZ low3 = 3 (three copies close — pitfall's HUD setup), each of
the three copies has a 4-pixel delay slot directly before its 8-pixel
display window; RESP newx anywhere outside those three delay windows
triggers skip-first-copy.

**Why jutari mis-renders the HUD digits:** `_object_pixel_sets` /
`_player_set` always emit ALL three copies for NUSIZ=$13 — there is
no "enable" parameter. Pitfall's RESP0 at sl 41 cc 72 (newx=18, oldx
likely far from any delay slot) puts xitari in skip-first-copy mode
for the rest of the frame; jutari renders copy 0 (cols 18-25) too,
which is what darkens cols 20-23 with COLUP0=$00 and breaks the
digit topology.

**Fix outline** (next agent — implement carefully so pong stays
bit-exact):

  1. Add `skip_first_copy_p0::Bool` + `skip_first_copy_p1::Bool` to
     `TIAState` (default `false`).
  2. Add a `_resp_when_case(nusiz_lo, oldx, newx) -> Int` helper that
     mirrors xitari's table build (TIA.cxx lines 938-1010). Returns
     -1 / 0 / 1.
  3. In `tia_poke!` for W_RESP0 / W_RESP1, compute
     `newx = _resp_player_position(beam_cc)`, read
     `oldx = tia.p0_x` (or p1_x), call the helper. Set
     `skip_first_copy_p0 = (case != -1)`.
  4. In `_player_set` (line 1176), if `skip_first_copy_p*` is true,
     drop the first entry of `copy_offsets` before iterating.
  5. Same in `_overlay_player!` (legacy whole-scanline path).
  6. Same for missile if xitari has a similar rule for missiles
     (check `case 0x14`/`0x15` in xitari).

**Risk:** pong is currently BIT-EXACT and uses NUSIZ multi-copy for
its score digits. If pong's RESP0/RESP1 land in delay slots → case
-1 → no skip → no change → pong stays bit-exact. If they land in
display/between → skip-first-copy → pong's score visual would change.
Verify by regenerating pong screen fixture AFTER applying the fix
and diffing against the existing 0-px-residual pin.

Pitfall draws the score+timer digits with PLAYER multi-copy
(NUSIZ0=$13 / NUSIZ1=$13 = three close copies, no scaling) and
writes new GRP0/GRP1 values BETWEEN each copy's rendered cc range
(the classic "draw three digits with one sprite" technique). For
each scanline of the digit, the cart issues 7 GRP writes at
cc 9 / 33 / 57 / 90 / 99 / 108 / 117 with DIFFERENT values per
copy. xitari's per-cycle render picks up each value at its
correct sub-scanline cc — three different digit shapes per row.
jutari's deferred-write block queues all 7 GRP writes; the per-cc
loop in `tia_advance!` should also pick up each at its
activation_clock — but the rendered output shows only ONE pattern
applied to all three copies, suggesting the deferred GRP writes
aren't being interleaved with copy positions correctly. This is
the same class of bug as the original pre-#84 GRP defer issue;
it may be a residual that #84 closed for single-copy sprites but
not for NUSIZ multi-copy.

**Root cause: NOT YET LOCALIZED.** Possible directions:

  - xitari may treat some address differently as a TIA mirror that
    jutari treats as RAM (need direct A0-A12 decoding comparison
    against `xitari/emucore/System.cxx`).
  - Pitfall may use a mid-scanline COLUBK strobe via WSYNC + sub-
    instruction timing that jutari deferred-writes doesn't capture.
  - Pitfall may use the CTRLPF SCOREMODE bit (D1) in a way jutari
    doesn't honor — currently `CTRLPF = $01` (just D0 reflect), but
    that should still render BG = COLUBK for PF-clear.

**Player sprite misalignment** (also user-reported): on the noop_10
fixture, all mid-screen pixels match between engines — the misalignment
likely shows up only during motion in the 60s random-action video.
The HUD divergence likely also affects how the player sprite renders
(e.g. brown background bleed into the player's torso outline). Same
investigation will likely localise both.

Task #86 created to track. PXC1 RAM stays bit-exact; PXC2 cross-port
unaffected (both jutari and jaxtari render the same wrong HUD).

### 🔬 Task #80 + #83 INVESTIGATION round 2 (2026-06-11)

**#80 — seaquest boot off-by-1, per-frame probe:**

`tools/seaquest_boot_probe.jl` runs the 60 NOOP + 4 RESET boot burn
step by step and prints `tia.frame` / `RAM[$01]` after each frame. Key
observations:

  - After 60 NOOPs jutari is at `tia.frame=60, RAM[$01]=$3b` (= 59).
    So 60 frames produce 59 increments — the FIRST NOOP frame after
    `console_reset!` does NOT increment RAM[$01].
  - The 4 RESET frames each increment RAM[$01], ending at $3f (= 63).

So jutari does **63 cart-increments in 64 boot frames** (skips 1).
xitari does **62 cart-increments in 64 boot frames** (skips 2).
xitari skips ONE EXTRA frame that jutari doesn't. Most likely the
**first RESET-pressed frame** — xitari's `Event::ConsoleReset → SWCHB`
propagation might lag jutari's immediate `set_swchb_input!`, so xitari's
first reset frame still sees `reset_not_pressed` on cart's
SWCHB read while jutari already sees `reset_pressed`. Seaquest's
cart-frame loop probably branches on the SWCHB read to choose
"increment counter" vs "enter reset routine".

**Plausible quick fix (untested):** in `env_reset!`, set
`console_switches!(reset_pressed=true)` AFTER the first reset-frame
runs. That makes jutari's first reset frame run with
reset_not_pressed, matching xitari. Worth trying; risk = other ROMs
that rely on the current timing may regress.

**#83 — VBLANK transition timing:**

Jutari's `vblank_active` flips to false 7 scanlines earlier than
xitari's. xitari renders incrementally so VBLANK status is consulted
per-pixel; jutari renders whole-scanline so once vblank_active is
false, the WHOLE scanline renders in the visible branch. With the
per-cycle render refactor (the architectural work the original task
#83 plan calls for), the HMOVE-blank flag would persist through the
xitari-equivalent VBLANK scanlines and land on display row 0 like
xitari.

Without that refactor, every targeted patch lands on a scanline that
gets cropped out by Y_START=34, leaving the visible row-0 still
missing its comb. Both #80 and #83 share this "jutari renders per
scanline, xitari per cycle" root-cause class — closing them properly
needs the same refactor, not piecemeal patches.

### 🔬 Task #80 INVESTIGATION (2026-06-11) — xitari VSYNC pulse-width gate

While auditing the VSYNC handler for seaquest's `RAM[$01] = $3f` (jutari)
vs `$3e` (xitari) boot-end off-by-1, I looked at xitari `case 0x00`
(TIA.cxx:2010-2031):

  - On 0→1 rising edge: `myVSYNCFinishClock = clock + 228` (= 1 scanline
    threshold).
  - On 1→0 falling edge: ONLY count as frame end if
    `clock >= myVSYNCFinishClock`. Sub-scanline VSYNC pulses are
    SILENTLY IGNORED.

jutari currently ends a frame on ANY 1→0 falling edge regardless of
pulse width — which on the surface looks like the cause of the off-by-1.

**Result of the attempt:** I added a `vsync_finish_cycles` field +
threshold gate. After regen, seaquest's `RAM[$01]` was STILL `$3f`
— the pulse-width gate didn't move it. So the boot-end off-by-1 is
NOT a transient sub-scanline VSYNC pulse; it's a different drift
(maybe a 1-CPU-cycle alignment of the first frame, or first-instruction
RIOT INTIM read timing).

The patch ALSO broke 3 jutari unit tests in `runtests.jl` (lines
2240-2280) — those tests call `tia_poke!(W_VSYNC, 0x02)` then
`tia_poke!(W_VSYNC, 0x00)` back-to-back without advancing cycles, so
the unit-test VSYNC pulse has 0-cycle duration and the threshold blocks
it. Reverted the change so the threshold isn't in tree. If a future
attempt brings it back, update the tests to advance `tia.total_cycles`
between the rise and fall (e.g. via `tia_advance!(tia, 80)`).

The boot-end off-by-1 is still open; needs different investigation
(per-bus-op diff vs xitari trace_dump in the boot-burn region).

### 🔬 Task #83 INVESTIGATION (2026-06-11) — HMOVE row-0 comb localized but NOT closed

**Symptom:** pong row 0 cols 0-7 paint as $34 (BG) in jutari but $00
(BLACK / HMOVE-blank) in xitari. 8 px residual, deterministic across
all frames.

**What was tried + what we learned:**

  1. **VBLANK clear bug fix (landed):** previously
     `tia.hmove_blank_pending = false` ran after EVERY scanline
     (VBLANK and visible). xitari's `TIA::updateFrame` only clears
     `myHMOVEBlankEnabled` from within the visible-render branch. The
     jutari unconditional clear meant an HMOVE written during the
     VBLANK phase lost its blank by the time the first visible
     scanline rendered. Fix: moved the clear inside the
     `if !tia.vblank_active` branch. xitari-faithful and correct, but
     **does not close pong's 8 px residual** (the underlying bug is
     elsewhere — see #2).
  2. **HMOVE comb fires at the wrong scanline:** `tools/pong_hmove_probe.jl`
     scans the post-frame framebuffer for scanlines with `cols 0-7 == 0`.
     In jutari, the comb lands on **internal scanline 27** — which is
     cropped out by `Y_START = 34` so the user never sees it. xitari's
     comb lands on **internal scanline 34** (= display row 0), the
     pixel pattern the conformance test compares.
  3. **PXC1 RAM is bit-exact** for pong, so the CPU is in lockstep. The
     cart writes HMOVE + VBLANK transitions at identical CPU cycles in
     both engines. The 7-scanline drift must therefore come from how
     each engine maps `(CPU cycle) → (current scanline + beam_cc)` at
     the moment of the relevant write — specifically, jutari's
     `vblank_active` toggles to false earlier (relative to absolute
     scanline number within the frame) than xitari's "first visible
     scanline" boundary in `updateFrame`. The HMOVE-blank flag then
     fires on jutari's earlier-but-cropped-out scanline 27.

**Why a small patch doesn't close it:** the actual fix requires
  the per-cycle render refactor described under the original task
  #83 plan — render scanlines as they happen rather than batching at
  end-of-frame, so VBLANK transitions and the HMOVE-blank flag stay
  in lock-step with xitari's `myClockWhenFrameStarted` / scanline
  numbering.

**Numbers:** pong screen residual stays at **8 px** (unchanged from
post-#85). VBLANK-clear fix landed alongside the #85 RESMP fix in
the same commit. Task #83 remains OPEN.

### 🏆 Task #85 CLOSED (2026-06-11) — RESMP* gate fix (pong 24→8 px)

**Symptom:** pong rows 35-37 cols 16-19 (color $38, COLUP0) and
cols 140-143 (color $c8, COLUP1) painted in jutari; xitari paints BG.
4-px width + COLUP0/COLUP1 colors = M0/M1 missile sprites.

**Probe:** Inspected `tools/pong_phantom_probe.jl` — at the
phantom scanlines: NUSIZ0=NUSIZ1=$20 (missile MWID=4), ENAM0=ENAM1=0
at end of frame, **RESMP0=RESMP1=$06** (bit-1 set → "reset to player,
hide"), m0_x=140 / m1_x=7 (matching the phantom right/left columns).
Confirms missiles being painted when RESMP* says they should be hidden.

**Root cause:** Neither `jutari/src/tia/TIA.jl` nor
`jaxtari/jaxtari/tia/system.py` honored the **RESMP0/RESMP1** register
($28/$29). Both declared the address constants but the render path
never gated on them. Xitari `case 0x1D/0x1E` (TIA.cxx:2525, 2536) and
`case 0x28/0x29` both compute `myEnabledObjects` as `ENAM* && !RESMP*`
— RESMP*=$02 makes the missile invisible (locked to its player's
center, no pixels rendered).

**Fix** (this commit):
  - `jutari/src/tia/TIA.jl::_missile_set` and `_overlay_missile!`:
    early return if `(RESMP* & 0x02) != 0`.
  - `jaxtari/jaxtari/tia/system.py::_missile_set` and `_overlay_missile`:
    same symmetric gate.

**Not implemented** (deferred):
  - RESMP* 1→0 transition reposition (xitari snaps `myPOSM0 = (POSP0
    + middle) % 160` where `middle ∈ {4,8,16}` by NUSIZ low 3 bits).
    Adding this had no measurable effect on the pong residual (still
    8 px) and risked regressing other ROMs.
  - RESMP* defer to `pending_writes`. Tried — gave identical screen
    diffs as the immediate-update path, so the gate alone suffices
    for the current scoreboard.

**Numbers** (post-fix, after task #84 GRP defer was already in):

  | ROM            | Pre-#84 | Post-#84 (true base) | Post-#85 |
  |----------------|---------|----------------------|----------|
  | pong           | 32      | 24                   | **8**    |
  | breakout       | 0       | 0                    | 0        |
  | space_invaders | 12      | 42                   | 42       |
  | pitfall        | 322     | 553                  | 553      |
  | seaquest       | 1104    | 1043                 | 1043     |
  | enduro         | 1197    | 774                  | 774      |

Pong residual now **8 px = only the row-0 HMOVE comb** (task #83 —
the only Phase C item still open). All jutari runtests pass.

Notes on the SI/pitfall/seaquest/enduro numbers: task #84's GRP defer
shifted the "true baseline" — SI/pitfall worsened by exposing other
render-timing mismatches (likely PF/sprite layering or NUSIZ-wide
+1 quirks), while seaquest/enduro improved. The pinned screen
fixtures had stayed at their pre-#84 values until now, so the test
pin update + fixture regen lands in the same commit as the #85 fix.

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

### 🔬 Task #96 dead-end logged (2026-06-13) — seaquest 904 is NOT a settings mismatch

After #95 (enduro 516→249 by adding the missing RomSettings), hypothesized
seaquest's 904-px PXC-S residual was the same class of bug. DISPROVED:
jaxtari seaquest renders **904 px with both GenericRomSettings AND
SeaquestRomSettings** (measured identical). Reason: seaquest's
`getStartingActions` is empty (`[]`), so the settings choice doesn't
change its boot or screen — unlike pitfall (UP) / enduro (FIRE). So
seaquest's 904 is a GENUINE render/timing divergence, to be chased on its
own with the bus-trace method (and, per #95's lesson, only after
confirming action-stream + boot parity between the two sides). Adding a
jutari `SeaquestRomSettings` would help scoring/terminal parity (jutari
lacks the type) but would NOT move the screen number. Downgraded to low
priority.
