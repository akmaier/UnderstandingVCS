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

### 🎯 Task #103 — SOLVED (elevator_action): cart was F8SC (Superchip), mis-detected as F8 → 53→0 b/f under NOOP (2026-06-15)

The deepest residual, cracked by a per-bus-op trace that bottomed out at the cart layer.

**Pinpoint chain (each step halved the search):** elevator 53 b/f under NOOP → boot-end
RAM is bit-exact EXCEPT **2 bytes RAM[$20]/[$21]** ($00/$00 jutari vs $22/$30 xitari) →
those are loaded `LDA $F0DC,X; STA $A0` / `LDA $F0DA,X; STA $A1` → **$F0DC/$F0DA are
Superchip RAM read ports** (write $F000-$F07F, read $F080-$F0FF) → elevator is an **F8SC**
cart. jutari read ROM (zeros) there instead of SC-RAM → $20/$21 wrong → cascaded to 53
b/f (the $2c-$3e table, the frame-1 BMI at $1874, the INTIM 14-vs-9 poll were all
downstream).

**Root cause.** (1) The #100 Cart.jl rewrite DROPPED jutari's earlier F8SC support
(left only a "Deferred: SC variants" comment). (2) NEITHER port AUTO-DETECTED Superchip
carts — both default 8K→F8 (jaxtari kept the F8SC *impl* but required an explicit
`kind=`). xitari's `autodetectType` checks `isProbablySC`→F8SC FIRST for 8K/16K/32K.

**Fix.**
- jutari `Cart.jl`: re-implemented F8SC/F6SC/F4SC — KIND_F8SC/F6SC/F4SC, a 128 B
  `sc_ram` CartState field, read window $1080-$10FF / write window $1000-$107F (alias
  `& 0x7F`), SC banking via the F8/F6/F4 hotspots, and `_is_probably_sc` +
  SC-first autodetect (8K→F8SC, 16K→F6SC, 32K→F4SC). cart_poke! now takes the value.
- jaxtari `cart/system.py`: added `_is_probably_sc` + SC-first autodetect (impl already
  existed).

**Result.** elevator **53→0 b/f under NOOP-30 (BIT-EXACT)**; 1 b/f under the sweep's
breakout_random stream (a frame-0 action edge case, same class as skiing). Verified ONLY
elevator matches isProbablySC among all 64 ROMs (scanned) → zero false positives, zero
regression risk to the other 63. The earlier "frame-timing drift / partial-frame model"
hypothesis was WRONG (a trace-measurement artifact) — the real cause was a missing cart
mapper, far simpler. Sweep target: 61/64 (elevator's sweep cell drops 54→1).

---

### 🔬 Task #103 — skiing is NOOP-bit-exact (boot fixed); surround SELECT/RESET dead-end (2026-06-14)

**skiing — boot SOLVED, sweep residual is an action-stream artifact.** After the 16×
DOWN getStartingActions fix, skiing is **bit-exact under NOOP-30** (boot/idle
conformance is perfect). Its remaining sweep number (83 b/f) appears ONLY under the
sweep's `breakout_random_actions` stream, first diverging at **frame 20 when FIRE
(action 1) is applied** — a narrow skiing-specific action-handling edge case, NOT a
boot or shared-core divergence. (The 64-ROM sweep applies breakout's random actions to
every ROM; for non-breakout games that exercises arbitrary inputs. Other games are 0
under the same stream, so this is skiing-only game logic under FIRE.) Treating boot
conformance as the bar, skiing is effectively closed.

**surround — SELECT/RESET starting actions help frame-0 but unmask a deeper bug
(REVERTED).** surround's xitari `getStartingActions` = `{SELECT, RESET}` (ALE codes
46, 40 → Event::ConsoleSelect/Reset → SWCHB switch presses; SELECT picks the game
variation). I implemented console-switch starting actions in `env_reset!` + a
`SurroundRomSettings`. It dropped surround's **frame-0** divergence 16→9, but the
**max-diff across frames rose 16→18** — pressing SELECT puts surround in the correct
game variation (matching xitari), which then exposes a deeper per-frame emulation bug
that diverges MORE than the wrong-variation default. Since the sweep gate regressed
(16→18) and I couldn't close the deeper bug, I **reverted** the surround change to keep
the gate clean (60/64). The console-switch-starting-action machinery is correct and
worth re-applying once the deeper surround residual (16 b/f under NOOP+generic, a
genuine boot-phase divergence) is found. qbert (56) and elevator_action (53) likewise
diverge under NOOP — genuine boot-phase residuals (qbert = xitari sub-instruction
"sliver" frame at the boot→step transition; elevator = title-vs-demo state machine),
both needing the xitari TIA partial-frame model — deferred.

**UPDATE 2026-06-15 — boot-end-RAM compare localizes the last two (after elevator F8SC
landed; sweep 61/64):**
  - **qbert: boot-end RAM is BIT-EXACT** (0 diff bytes). Its 56 b/f is ENTIRELY in
    frame-1+ — confirming the sub-instruction "sliver" frame at the boot→step transition
    (xitari emits a ~9-cycle partial frame that jutari folds into the next). Pure
    shared-core frame-boundary / partial-frame (`myClockWhenFrameStarted`) issue —
    highest-risk change; deferred to a supervised session.
  - **surround: 7 boot-end diff bytes** ($64/$6c/$71/$79/$7d-$7f). These are the missing
    `getStartingActions = {SELECT, RESET}` (xitari presses them during resetGame to pick
    the game variation; jutari-generic doesn't). The console-switch-starting-action
    machinery (implemented + reverted above) fixes boot-end but unmasks a deeper per-frame
    timing bug (max-diff 16→18) — same frame-model family. Two-part fix (SELECT/RESET +
    partial-frame), deferred together.
  - **Bottom line:** the last two residuals both hinge on the xitari TIA partial-frame /
    frame-boundary-cycle model. That single shared-core change (highest regression risk to
    the 61 bit-exact games) is the next big lever; it likely closes both. The
    cpu_tia_cycle_trace + boot-end-RAM-compare tooling is in place to drive it.

**elevator_action (53 b/f under NOOP) — deeper characterization + RIOT-timer-base
hypothesis RULED OUT.** With the 16× FIRE getStartingActions fix it's 98→53. The
divergence is a DETERMINISTIC init-branch: at boot-end jutari has CONSTANTS where
xitari has computed tables — e.g. RAM[$2c..$3e] = xitari `c3 99 6f 45 1b 91 67 3d 13
e9` (a descending ramp, step -0x2A) vs jutari constant `e7`/`b5`; the attract-demo
object table $20-$21 is `22 30` in xitari, `00 00` in jutari. So jutari never runs
elevator's table-fill / demo-init routine — its CPU takes a different branch during
boot. The title loop reads ONLY INTIM + RAM (no TIA/collision/INPT), and RAM is
bit-exact except the table OUTPUT, so the branch input is INTIM.
**RIOT-timer-base ruled out:** the agent suspected jutari's free-running cycle base
(vs xitari's per-frame `resetCycles`) skews INTIM. But jutari's RIOT timer
(`_timer_output!`) is xitari's exact delta-based formula — INTIM = `my_timer -
(delta>>shift) - 1`, `delta = cur_cycles - cycles_when_set`. Both terms share the
same base, so the base CANCELS in the delta; free-running vs frame-reset is
immaterial to the timer value (and 60 INTIM-polling games are bit-exact). The
residual is therefore a subtle CPU↔RIOT *cycle-threading* edge case (the exact
`cur_cycles` at the mid-instruction INTIM read) OR a register read not yet
identified — needs the per-CPU-instruction boot trace (jutari console_step! vs
xitari trace_dump CpuDebug) to find the first divergent branch. Likely fix is a
shared-core cycle-threading change → HIGH risk to the 60 bit-exact games; left for a
supervised session (the diagnostic trace is safe; the fix is not).

**elevator_action — CONCLUSIVE root cause via per-bus-op trace (2026-06-15).** Built a
jutari↔xitari frame-1 bus-op diff (jutari `cpu_tia_cycle_trace.jl` — added
`elevator_action` settings — vs xitari `trace_dump --bus-trace`), normalizing the
mirror-address representation ($DDC6≡$1DC6) and collapsing jutari's cycle-accurate
6502 dummy reads/writes (RMW write-old, branch-taken dummy read) that xitari's trace
doesn't log. Findings:
  1. CONTROL FLOW is identical until the first INTIM poll — the divergence is a
     VALUE, not a branch-target, difference.
  2. First divergent register read: **INTIM ($0284) = 14 (jutari) vs 9 (xitari)** at
     the title loop's first timer poll. That 5-tick gap flips elevator's timer-gated
     branch, so the demo/table-fill routine (which populates RAM $20-$74) never runs.
  3. WHY INTIM differs: the timer is loaded (TIM64T $296=44) at scanline 0 in BOTH,
     but jutari reaches the INTIM read at **scanline 24** vs xitari's **28** — jutari's
     beam runs ~2-4 scanlines (~320 CPU cycles) BEHIND through scanlines 3-25, then
     RE-SYNCS at scanline 36. The WSYNC ($02) progression confirms it: WSYNC #0 xi
     sl=5 / ju sl=3; #1 xi 9 / ju 5; #2 xi 22 / ju 18; #3 xi 24 / ju 20; #4 xi 25 /
     ju 21; #5 BOTH 36. So it's a transient per-scanline cycle/beam drift in the early
     frame (identical instructions, different elapsed cycles → WSYNC-stall-dependent),
     i.e. a frame-boundary / CPU↔TIA cycle-accounting timing difference — the SAME
     family as qbert's sub-instruction "sliver" frame.
**Status:** root cause is a shared-core frame-timing drift (beam ~2-4 scanlines behind
in early frame-1), not a settings/feature gap. The fix is in the frame-boundary /
cycle-accounting core (xitari's partial-frame `myClockWhenFrameStarted` model) → the
single highest-risk change (touches every game's timing); likely also closes qbert +
surround. Deferred to a focused, supervised session with the full 64-ROM sweep as the
gate. Tooling (`cpu_tia_cycle_trace.jl` + the normalize/dummy-collapse diff recipe) is
in place to pinpoint the exact miscounting instruction next time.

**UPDATE 2026-06-15 — pinpoint narrowed from 53 b/f to 2 ROOT BOOT-END BYTES.** The
earlier "frame-timing drift" read was a measurement artifact (xitari's bus-trace derives
sl/cc from frame-relative `mySystem->cycles()` reset each frame; jutari logs the real
preserved beam — so the beam columns aren't comparable, and jutari's cycle-accurate 6502
dummy reads/writes further offset the streams). Comparing the actual **boot-end RAM**:
jutari ≡ xitari at boot-end EXCEPT **2 bytes — RAM[$20]=$00/$21=$00 (jutari) vs $22/$30
(xitari)**. Everything else (the $2c-$3e table — ALL ZERO at boot-end in BOTH; the
frame-1 BMI at $1874 on RAM[$3C+X]; the INTIM 14-vs-9 poll) CASCADES from these 2 bytes
during frame 1. So elevator is NOT a frame-timing/partial-frame bug after all — it's a
single conditional write to $20/$21 during the 80-frame boot (60 NOOP + 4 RESET + 16
FIRE) that xitari performs ($22/$30) and jutari skips/zeroes. jutari pokes $A0/$A1 (=RAM
$20/$21) =0 repeatedly through the boot; xitari's last boot write is $22/$30. NEXT STEP:
trace xitari's BOOT (modify trace_dump to enable the bus-trace before `resetGame`, or do
an explicit-boot mode) to find the exact boot frame/instruction + the condition (a
register/INTIM read or computed value) under which xitari writes $22/$30 and jutari
doesn't. This is now a narrow 2-byte boot divergence, far more tractable than a frame
rewrite — and likely the same class as qbert (also boot-phase). Diagnostic recipe +
boot-end-RAM compare are the tools; the fix scope is one conditional write.

---

### 🎯 Task #103 — SOLVED (air_raid): VSYNC frame-end myVSYNCFinishClock hold-gate → air_raid 43→0 b/f (2026-06-14)

The characterization workflow flagged a frame-boundary cause for air_raid (1-frame
offset: jutari frame N == xitari frame N-1). The actual fix is xitari's **VSYNC
hold-gate**, which jutari lacked.

**xitari (TIA.cxx:2011-2031).** On a VSYNC write: if D1 SET, arm
`myVSYNCFinishClock = clock + 228`. A later VSYNC CLEAR only ends the frame when
`clock >= myVSYNCFinishClock` — i.e. **VSYNC must be HELD >= 1 scanline (228 color
clocks)**. jutari's old logic ended the frame on ANY VSYNC D1 1→0 edge (no hold
requirement).

**Why this fixed air_raid.** air_raid runs a 291-scanline frame and drives VSYNC
(set ~scanline 286, clear ~291). The old edge-only end interacted with the #80
max-scanline cutoff (`lines_since_frame > 290`) such that jutari's boundary landed
one scanline early → every per-frame state write in the line-290→291 overscan never
ran before the RAM snapshot → 43 cells stuck at prior-frame values. With the faithful
hold-gate, air_raid's frame ends at its real VSYNC boundary (held 5 scanlines >> 1),
matching xitari.

**Implementation (jutari `TIA.jl`).** New `vsync_finish_clock` field (default
`typemax` = disarmed). W_VSYNC handler: `frame_clock = lines_since_frame*228 +
beam_cc` (xitari's `clock - myClockWhenFrameStarted`); SET arms `frame_clock + 228`;
CLEAR sets `vsync_reset_pending` (the existing 2026-06-04 deferred reset) only if
`frame_clock >= vsync_finish_clock`. Disarmed at the max-scanline cutoff boundary too.

**Result.** air_raid **43→0 b/f** on BOTH ports — jutari sweep 59→60 bit-exact;
jaxtari air_raid verified 0 via direct RAM diff (the jaxtari W_VSYNC handler got the
same `vsync_finish_clock` hold-gate + the new `vsync_finish_clock` NamedTuple field +
the max-scanline-cutoff disarm). seaquest (#80 boot, VSYNC-less → uses max-scanline
cutoff, unaffected) + breakout + the rest stay bit-exact. surround/qbert/elevator_action
are NOT this mechanism (unchanged — separate causes). jutari `Pkg.test` green (P3f
VSYNC 36/36 — the lifecycle test now holds VSYNC >= 1 scanline + a new short-pulse
regression test). NOTE: jaxtari pytest is wedged this session (xdist/JAX env), so the
jaxtari mirror was verified via the direct RAM diff, not the suite — re-run the jaxtari
suite in a clean shell to confirm.

---

### 🎯 Task #103 — SOLVED (amidar): per-game console difficulty (SWCHB) — amidar is A/A, not B/B → 11→0 b/f, both ports (2026-06-14)

The 7-ROM characterization workflow flagged amidar as a SETTINGS issue (an agent
captured an xitari bus trace showing amidar reads **SWCHB=$ff**, with ZERO TIA reads
in the divergent frame). This contradicted an earlier triage note ("forcing jutari
SWCHB=0xFF left amidar at 11 b/f unchanged") — resolved here.

**Root cause.** xitari `Props.cxx:292-293` defaults BOTH `Console.LeftDifficulty` and
`Console.RightDifficulty` to **"B"** → `Switches.cxx` clears SWCHB bits 0x40/0x80 →
default SWCHB = **0x3F (B/B)**, which jutari/jaxtari hardcode and which is correct for
the 59 bit-exact games. BUT amidar's `stella.pro` entry **overrides** both to "A"
(`Console.LeftDifficulty "A"` / `RightDifficulty "A"`) → SWCHB = **0xFF (A/A)**.
amidar's frame-1 object-sort kernel branches on the P0/Left difficulty bit
(`LDA SWCHB; AND #$40`), so with jutari's B/B it sorted the wrong way → the value-
SWAPPED pairs at $47/$48, $4e/$4f plus $41/$56 (11 b/f).

**Why "forcing 0xFF earlier didn't help":** jutari's `env_reset!` calls
`console_switches!(reset_pressed=false)` AFTER the RESET-burn frames, which REBUILDS
the whole SWCHB byte (b=0xFF then clear difficulty bits → 0x3F), overwriting any
pre-set swchb_in. The earlier attempt set the value once and it got clobbered.

**Fix (both ports) — per-game difficulty, sourced like xitari's per-ROM properties:**
- New RomSettings hook `romsettings_difficulty(settings) -> (p0_a, p1_a)`, default
  `(false, false)` = B/B (jutari `RomSettings.jl`; jaxtari `rom_settings.py`
  `difficulty()`). `AmidarRomSettings` returns `(true, true)` = A/A.
- `env_reset!` / `StellaEnvironment.reset` now apply the difficulty BEFORE the boot
  burn AND re-assert it in every `console_switches!` call (so the RESET-burn rebuilds
  don't clobber it). Default B/B keeps the 59 bit-exact games unchanged.
- Registered `amidar.bin` in jutari_trace_dump.jl + check_trace.py settings maps;
  exported AmidarRomSettings from jaxtari `games/__init__.py`.

**Result:** amidar **11 → 0 b/f** on both ports (jutari sweep + jaxtari↔xitari diff
both bit-exact). jutari sweep **58→59** bit-exact, every other game byte-identical
(zero regressions); jutari `Pkg.test` green. This is the model for any future
per-game difficulty/TV-type/controller property divergence.

---

### 🎯 Task #103 — SOLVED (gravitar) + improved (skiing/elevator_action): getStartingActions COUNT was 1, xitari repeats 16× (2026-06-14)

Found by a 7-ROM read-only characterization workflow (one agent per residual). Three
"boot-phase" residuals share ONE bug: xitari's `getStartingActions()` for these games
repeats the action **16 times** (`for(i=0;i<16;i++) push_back(...)`), but jutari/jaxtari
returned a **single** action (the #100/#101 port transcribed the action value but not the
count). So jutari ran 15 fewer boot frames → the title/selection-screen init never
finished → RAM tables left at $00.

Verified in xitari sources:
- `Gravitar.cpp`: 16× PLAYER_A_FIRE   |  `Skiing.cpp`: 16× PLAYER_A_DOWN
- `ElevatorAction.cpp`: 16× PLAYER_A_FIRE
(audited ALL 15 getStartingActions: only these three loop; AirRaid/Asterix/BeamRider/
DoubleDunk/Enduro/Gopher/JourneyEscape/Pitfall/PrivateEye/UpNDown/YarsRevenge are all 1×
and already correct. Surround is `[SELECT, RESET]` — console-switch actions, a separate
fix, deferred.)

**Fix (both ports):**
- jutari `JoystickGames.jl`: ElevatorAction/Gravitar → `fill(1, 16)`, Skiing → `fill(5, 16)`.
- jaxtari `joystick_starts.py`: same three → `[1]*16` / `[5]*16`.

**Result:** **gravitar 93 → 0 b/f** (fully bit-exact). skiing 85 → first divergence
moved frame 0 → frame 20 (frames 0-19 now bit-exact; a *separate* residual surfaces at
frame 20). elevator_action 98 → 54 (improved; a separate title-vs-demo residual remains).
The 16× count is unambiguously xitari-correct, so it stays regardless of the leftover
residuals. No other game touched (only these three RomSettings changed). Sweep now 58/64.

---

### 🎯 Task #103 — SOLVED (frostbite): RESMP* lock→unlock missile reposition → frostbite 2→0 b/f, both ports (2026-06-14)

User: "Make sure to also work on #102, and #103." First of the 6 #103 residuals closed.

**Symptom.** frostbite RAM[$34]/[$36] = jutari $47 vs xitari $07 — a constant **D6**
(0x40) difference, stable every frame. The triage called it "collision-D6".

**Root-cause hunt (instrumented BOTH emulators).** Logged every CXM1P read:
- reg 1 = **CXM1P**, val 0x40 = **D6 = M1-P1 collision** (missile-1 vs player-1).
- xitari frame: 4 CXM1P reads, **0 with D6=1**; jutari frame: 4 reads, **2 with D6=1**.
  (All 9 of xitari's D6=1 reads are during its ~191-frame `ale.resetGame()` boot
  animation, NOT steady state.) The low 0x07 is identical in both — it's the data-bus
  noise xitari ORs into every collision read (`getDataBusState()&0x3F`) + game logic;
  only D6 genuinely differs. So **jutari over-detects M1-P1 every steady-state frame.**
- frostbite's title kernel multiplexes missile-1: it toggles **RESMP1 between 0x03
  (lock+hide) and 0xc0 (unlock+show)** and repositions via RESP1/HMOVE, mid-scanline.

**The bug.** xitari `case 0x29` (TIA.cxx:2666): on the RESMP1 **D1 1→0 (unlock)** edge
it **snaps the missile to its player centre** — `POSM1 = (POSP1 + middle) % 160`,
middle = 8/16/4 for NUSIZ size 5/7/else. jutari (and jaxtari for visible-region writes)
**omitted this reposition** (a #85 note explicitly skipped it). So a released missile
kept its stale RESM1/HMOVE position and spuriously overlapped its player → CXM1P D6.

**Fix (both ports).** Implement the xitari-faithful RESMP* 1→0 reposition, applied
**immediately at poke time** (like a RES* position write) — NOT in the deferred apply
(putting it there had regressed space_invaders +30 px / pitfall +231 px in #85):
- jutari `TIA.jl`: capture pre-store RESMP D1; new `_resmp_reposition!(tia, player,
  old_d1, val)` called from the W_RESMP0/W_RESMP1 immediate handlers.
- jaxtari `tia/system.py`: do the reposition in the RESMP defer block BEFORE returning
  the deferred store (so it fires for visible-region writes too, not just HBLANK); the
  register STORE stays deferred for per-color-clock gate timing.

**Result.** frostbite **2 → 0 b/f** on both ports. jutari 64-ROM RAM sweep **55→57**
bit-exact (frostbite + robotank/#102), with **every other game byte-identical to the
prior sweep** (air_raid 43, amidar 11, elevator_action 98, gravitar 93, qbert 56,
skiing 85, surround 16 all unchanged — zero regressions). jutari `Pkg.test()` green.

**Tooling bug found + fixed (sweep parallelism).** `jutari_xitari_ram_diff.py` used
fixed `/tmp/_xitari_actions.txt` / `_jutari_rams.jsonl` paths, so `sweep_jutari_ram.py
--jobs N>1` workers clobbered each other → garbage diffs (breakout "109", ms_pacman
ERROR, etc. — all artifacts, not real). Keyed the temp files by ROM stem + PID; the
`--jobs` speedup now actually works (the multi-threading the user asked for).

Remaining #103: amidar 11, surround 16, elevator_action 98 (NEW vs old triage list),
gravitar 93, qbert 56, skiing 85, air_raid 43 — each its own deep-dive.

---

### 🎯 Task #102 — SOLVED: FE cart mapper (Activision JSR-banking) → robotank 81→0 b/f, both ports (2026-06-14)

User: "Make sure to also work on #102, and #103." The last exotic 8K mapper. After
#100 added FE *detection* (isProbablyFE 5-byte signatures), robotank was correctly
classified as FE but still diverged **81 b/f** because the banking was wrong.

**Root cause — FE banks on A13 of the *full 16-bit CPU address*, not a hotspot.**
Activision's FE ("Front/Back E") has no bank-switch hotspot. Stella exploits the fact
that the JSR/RTS that crosses banks leaves a tell-tale high address bit on the bus:
the selected 4K half is chosen by **A13 of the un-masked address** —
`$Fxxx → A13=1 → lower bank (rom offset 0)`, `$Dxxx → A13=0 → upper bank (offset 4096)`.
xitari: `myImage[(addr&0x0FFF) + ((addr&0x2000)==0 ? 4096 : 0)]`. The poke side is a no-op.

The trap: our bus masked the address to the 4K cart window (`a = addr & 0x0FFF`) **before**
calling `cart_peek`, so A13 was already gone. A static "always upper 4K" guess failed
(robotank stayed 81). Fix = thread the **un-masked** `addr` through to `cart_peek` so the
FE branch can read A13; every other mapper ignores the extra bits (`addr & 0x0FFF`).

**Changes (both ports):**
- jutari `Cart.jl`: `KIND_FE=6`, `_FE_SIGNATURES` + `_is_probably_fe`, FE in `_autodetect_kind`
  (8K order E0→FE→F8), cart_peek FE branch uses full addr `(addr&0x0FFF)+((addr&0x2000)==0 ? 0x1000 : 0)+1`.
- jutari `Bus.jl`: `cart_peek(bus.cart, a)` → `cart_peek(bus.cart, addr)` (un-masked). poke unchanged.
- jaxtari `cart/system.py`: `KIND_FE=9`, `_FE_SIGNATURES`/`_is_probably_fe`, FE in `_autodetect_kind`,
  `_DEFAULT_BANK[FE]=0`, `_SC_EXPECTED_SIZE[FE]=8192`, cart_peek FE branch (same A13 formula).
- jaxtari `bus/system.py`: `cart_peek(bus.cart, addr_masked)` → `cart_peek(bus.cart, addr)`.

**Result:** robotank **81 → 0 b/f** on both ports. No regression — ms_pacman/jamesbond/breakout/
pong all stay 0 b/f; jutari `Pkg.test()` green; jaxtari cart/bus/F8SC/E0 suites (75 tests) green.
64-ROM sweep now **56/64** bit-exact. All four 8K mappers (F8, F8SC/F6SC/F4SC, E0, FE) covered.

---

### 🔬 Task #103 — TRIAGE of the 6 genuine ROM residuals (per-game causes characterized) (2026-06-14)

After #100 (E0 detect) + the 12 getStartingActions + canonical NTSC dumps, 6 of the 64
games still diverge on the jutari RAM sweep with correct cart + starting action. First-
divergent-frame byte analysis (all diverge at frame 0) gives distinct, per-game causes —
NOT one common bug, so each is its own focused deep-dive (seaquest #80 / enduro #99 mold).
Fixes live in the shared collision/timing/boot core → HIGH risk to the 55 bit-exact games;
gate every attempt on the full sweep.

  - **frostbite — 2 b/f (smallest, start here).** RAM[$34]/[$36] = jutari $47 vs xitari
    $07: a constant **bit-6** difference → a **collision-register** (CXxx, D6=object-
    overlap) read where jutari's collision latch sets a bit xitari doesn't. Same class as
    the pong sub-cycle collision items (#83-85). Trace: which CXxx the game reads into
    $34/$36, why jutari's D6 differs at that beam position.
  - **amidar — 11 b/f.** $47/$48 and $4e/$4f are VALUE-SWAPPED (jutari has them in the
    other order) → a write-**ordering/timing** divergence, not a wrong value.
  - **surround — 16 b/f.** 6 scattered bytes ($71/$75/$79/$7d-$7f), varied values →
    timing/logic.
  - **qbert — 56 b/f.** xitari RAM mostly $00 where jutari is populated → a **boot-phase**
    divergence (jutari has run FURTHER by frame 0). Relates to the old #52 qbert
    RESET-boot issue.
  - **skiing — 85 b/f.** 80 bytes differ → major boot-phase divergence.
  - **gravitar — 93 b/f.** xitari RAM populated ($01/$03/$04/$0a nonzero) where jutari is
    $00 → boot-phase divergence (jutari has run LESS far by frame 0 — opposite of qbert).

**Two families:** (1) localized — frostbite (collision bit), amidar/surround (write
ordering/timing); (2) boot-phase / frame-count — qbert/skiing/gravitar (one emulator is
further along by frame 0; same family as seaquest's VSYNC-less-boot #80 — suspect a
console-switch-driven boot branch [we hard-code SWCHB=0x3F] or a boot-timing difference).
NOTE (#101): the ports are NOT bit-identical across all 64 — asterix is 0 on jutari but
9 b/f on jaxtari — so some residuals must be triaged per-port. A jaxtari 64-ROM sweep
harness (mirror sweep_jutari_ram.py + --jobs) would give the jaxtari column.

**Investigation update (2026-06-14, "work on them" pass) — safe hypotheses RULED OUT;
these are genuine deep-dives:**
  - **Console-switch (difficulty) ruled out for amidar.** stella.pro gives amidar
    Console.Left/RightDifficulty=A, but forcing jutari SWCHB=0xFF (A/A) left amidar at
    **11 b/f** (unchanged) — the difficulty doesn't affect its first 30 NOOP frames. So
    amidar is a genuine write-ordering/timing residual, not a settings gap.
  - **ROM-provenance ruled out for frostbite.** frostbite = **2 b/f with BOTH** the
    canonical Activision (Iceman, AX-031) dump AND the Digitel/PAL dump — dump-robust →
    a genuine collision-D6 residual, not a PAL/clone artifact.
  - **Resolver provenance (cosmetic, not conformance).** resolve_roms.py picks some
    PAL/Brazilian dumps (Frostbite Digitel, Surround PAL) because filename-based PAL
    detection misses dumps whose name lacks "(PAL)" but whose MD5 is PAL, and a penalty
    tie favors the shorter (clone) name. Hardened the penalty (digitel/digivision/quelle/
    "4 game in one"/"rad action" + an original-publisher bonus). Doesn't change the
    bit-exact count (residuals are dump-robust) — only cleans up provenance for the paper.
  - **Bottom line:** all 6 are genuine shared-core deep-dives (collision-timing
    [frostbite], write-ordering [amidar/surround], boot-phase/frame-count
    [qbert/skiing/gravitar]). Each risks the 55 bit-exact games and needs its own focused
    session with the full sweep as the gate — not safely batchable. #104 (the one safe
    win in this pass) is fixed + committed.

---

### 🎯 Task #100 — SOLVED: E0 cart auto-detect + 12 getStartingActions → 64-ROM sweep 44→55 bit-exact; remaining triaged (2026-06-14)

User: "Fix #100, port the 12 getStartingActions and triage the 4 small ones." Closed the
20 divergent games from the 64-ROM jutari RAM-conformance sweep into three groups.

**(A) #100 — cart auto-detect was size-only.** jutari `make_cart`/jaxtari `make_cart`
mapped 8 KB → F8 unconditionally, but Parker Bros 8 KB titles use the **E0** mapper
(4×1 KB slice slots), so Tutankham / Montezuma's Revenge / James Bond were banked as F8
→ wrong code → large frame-0 divergence. Fix: content-based detection mirroring xitari
`Cartridge::autodetectType` — scan the image for the 6 MESS E0 bankswitch signatures
(`isProbablyE0`) and route 8 KB matches to E0, else F8.
  - jaxtari already had E0 *banking* (P5c) but size-only detection — added `_autodetect_kind`
    + `_is_probably_e0` (`jaxtari/cart/system.py`).
  - jutari had neither — **ported E0 banking** (slice slots, `$1FE0..$1FF7` hotspots,
    fixed slice 7) **+ detection** into `src/cart/Cart.jl`.
  - Result: jamesbond/montezuma/tutankham **0 b/f**; ms_pacman (real F8 8 KB) stays 0
    (no regression); jutari `Pkg.test` + jaxtari `test_cart`/`test_p5c_e0` pass.

**(B) 12 getStartingActions ported (jutari).** Added 12 minimal RomSettings
(`src/games/JoystickGames.jl`) overriding only `romsettings_starting_actions`, registered
in `tools/jutari_trace_dump.jl`. Action codes (xitari ale_interface.hpp): FIRE=1, UP=2,
RIGHT=3, DOWN=5, UPFIRE=10. **8 of 12 → bit-exact**: asterix, beam_rider, double_dunk,
gopher, journey_escape, private_eye, up_n_down, yars_revenge. (jaxtari mirror of the 12 is
a mechanical follow-up — the jutari sweep is the headline.)

**(C) triage of the rest (canonical NTSC dumps + correct cart + starting action, RAM-diff
30 frames):**
  - **robotank 81 b/f** — **FE** mapper (Activision JSR-driven banking, the most exotic
    8 KB mapper; 1 ALE game). Deferred — needs the FE data-bus-watch banking in both ports.
  - **air_raid 43** — only a **(PAL)** dump exists in the collection; xitari format-detect
    vs our NTSC-only timing. Not an emulation bug on the NTSC path.
  - **elevator_action 98** — only a **(Prototype)** dump exists (game was never released).
  - **Genuine small residuals** (canonical Atari/Activision NTSC dump, correct cart +
    starting action, STILL diverge — confirmed by swapping in canonical dumps): frostbite
    **2**, amidar **11**, surround **15**, qbert **56**, skiing **85**, gravitar **93**.
    These are real per-game emulation residuals (likely console-switch/difficulty reads or
    sub-cycle boot timing), each a focused deep-dive in the seaquest(#80)/enduro(#99) mold.
    Resolver hardened to prefer canonical dumps over Brazilian-clone/multicart matches.

**Net: 64-ROM jutari RAM-conformance 44 → 55 bit-exact.** Remaining 9 = 1 FE + 1 PAL-only
+ 1 prototype-only + 6 genuine residuals (own tasks).

---

### 🎯 Task #99 — SOLVED: enduro road-marker offset closed; ALL 6 ROMs bit-exact RAM + 0 px screen, both ports (2026-06-14)

**Supersedes the "DEFERRED" entry below.** After the audit pinned the object (Missile 0
left + Ball right) and the mechanism (a `beam_sc >= 76` HMOVE strobe whose object motion
was applied 1 scanline early), the guarded fix landed and verified clean. **enduro screen
33 → 0** on BOTH ports; this makes **all 6 conformance ROMs simultaneously RAM-bit-exact
AND 0 px screen** vs xitari.

**Root cause (confirmed by per-cycle trace):** enduro free-running road lines strobe HMOVE
at `beam_sc ≈ 79` (≥ 76 → the beam has crossed into line N+1) on ~636 scanlines/run.
xitari applies that late strobe's motion to line N+1, but jutari/jaxtari render whole
scanlines in `tia_advance` AFTER the poke, so applying the motion immediately moved
M0/BL on the already-completed line N — a 1-color-clock outward offset that the road's
perspective magnified into the visible "~2 scanlines early" look. The base RES/HMOVE
arithmetic was already exact (4-agent audit); this was purely the *timing* of when the
motion takes visible effect.

**Fix:** defer the object motion of a `beam_sc ≥ 76` HMOVE — mirror of #97's
`hmove_blank_pending_next` comb deferral, but for the object positions. New field
`hmove_motion_next` (the 5 per-object deltas) is parked in `tia_poke!`/`tia_poke` when
`sc ≥ 76` and applied in `tia_advance!`/`tia_advance` right after line N commits, so N+1
onward use the moved positions. Below 76 the motion still applies immediately. The
`sc ≥ 76` path is enduro-specific (established by #97), so the 5 bit-exact ROMs are
untouched.
  - jutari: `src/tia/TIA.jl` (struct field + W_HMOVE defer + tia_advance! apply).
  - jaxtari: `jaxtari/tia/system.py` (field + W_HMOVE defer + tia_advance return apply).
  - `tests/test_screen_conformance.py` enduro pin 33 → 0; enduro jutari screen fixture
    regenerated (now 0 vs xitari).

**Gates (all green):** jutari RAM-diff all 6 ROMs (NOOP, 80f) = 0; jutari screen all 6 = 0
(enduro 33→0); jutari `Pkg.test()` exit 0; jaxtari core TIA/bus/players/missiles/
collisions tests pass; jaxtari-live screen enduro=0 + seaquest=0; full PXC2 (RAM, both
engines) + screen conformance (both engines) re-run clean. RAM stayed bit-exact for all 6
(the deferral did not perturb collision-driven RAM).

---

### 🔬 Task #99 — enduro road-marker 1px offset: deep audit, NO clean fix, keep DEFERRED (2026-06-14)

User noticed a "very slight offset in enduro (jutari), one or two pixels per scanline"
in the freshly-rendered #80 video. This is the **#97-deferred render residual** (33 px
worst-frame, RAM bit-exact), now precisely characterized + audited.

**Precise signature (frame 5, jutari vs xitari screen fixture):**
  - Two **single-pixel (1 color clock) color-index-4 markers** per affected row at the
    road EDGES (perspective-widening). NOT playfield (PF draws 4-clock blocks) → ball
    or missiles.
  - **1-color-clock OUTWARD split**: jutari's LEFT marker is 1px farther left, the RIGHT
    1px farther right (2px wider apart). Because the road widens per scanline, this looks
    like a ~2-scanline vertical shift: jutari marker cols @row N == xitari @row N+2
    (e.g. xitari row102=[63,112] row104=[62,113]; jutari row102=[62,113] row104=[61,114]).
  - Only ~9 rows differ; rows 102/104 carry the markers, the rest differ elsewhere.

**4-agent code audit (xitari ref + jutari + jaxtari, ultracode workflow):** the ports
FAITHFULLY match xitari on every position path —
  - RES* formula `((cc-68)+4)%160` for ball/missiles (jutari TIA.jl:430-431, jaxtari
    system.py:553-554) == xitari TIA.cxx:2325/2343/2361.
  - HMOVE motion: `_COMPLETE_MOTION_TABLE` is the exact negation of xitari
    `ourCompleteMotionTable` (TIA.cxx:2807-2884); pos−motion == xitari pos+motion.
  - beam derivation: `beam_cc == beam_sc*3` exactly (color_clock advances 3× scanline_
    cycle), mirroring xitari's single clock feeding hpos(%228) and x=hpos/3.
  - render mapping: ports draw `pos..pos+size-1` rightward == xitari mask
    `160-(pos&0xFC)` left-edge. NOT the `160-pos` mask bug an early hypothesis guessed.
  → **No single file:line deviation explains the symmetric 1-clock outward offset.**

**Only real deviation found (latent, NOT #99's cause):** ports compute
`(beam_cc-68+4)%160` without first doing `beam_cc%228`, so a RES poke with beam_cc≥228
(instruction straddling the scanline boundary) gets a wrong position. This is a
boundary-only glitch and cannot produce a constant offset on 9 consecutive rows.

**Object PINNED via live jutari trace (env-gated instrument in tia_advance!, reverted):**
the two markers are **Missile 0 (left edge) + Ball (right edge)**, both color index 4,
1px wide (`ENAM0=2, ENABL=2, ENAM1=0, NUSIZ0=5`→missile width 1; col4 == [m0_x, bl_x]
exactly on every traced row). They march apart down the perspective ramp (m0_x decrements,
bl_x increments ~1 per 2 scanlines).

**Refined mechanism:** at display row 102 xitari paints M0@63 + BL@112; jutari paints
M0@**62** + BL@**113** — jutari is **one perspective-step AHEAD**, i.e. it applies the
per-scanline M0/BL HMOVE reposition ~1 scanline EARLIER than xitari. The base RES/HMOVE
position arithmetic is exact (audited); this is a sub-cycle HMOVE-strobe-near-scanline-
boundary PHASE issue (enduro strobes HMOVE at beam_sc~79, same class as task #97's
`hmove_blank_pending_next` comb-deferral and #98) — but for the OBJECT MOTION, not just
the blank comb. The xitari game-specific RES hacks (Dolphin/Pitfall II/Mindmaster,
TIA.cxx:2330-2333/2348-2351/2367-2397) are RULED OUT — they require exact clock-distance
+ hpos matches enduro never hits.

**DECISION: keep DEFERRED.** Rationale: (1) 33 px **render-only** cosmetic residual
(RAM bit-exact, PXC2 18/18); (2) no clean bug to fix — every audited path already matches
xitari; (3) **HIGH regression risk** — pong/breakout/space_invaders/pitfall/seaquest are
bit-exact (0 px) BECAUSE the RES*/HMOVE core matches xitari; a ±1 guess there would
move THEIR objects too and break 0 px screen conformance (render-only, so RAM gates
wouldn't even catch it).

**PRECISE NEXT STEP (dedicated session, if pursued):** object is PINNED (M0+BL), so:
  1. Per-cycle trace of enduro's HMOVE strobe + HMM0/HMBL writes around the marker
     scanlines, jutari vs an xitari per-cycle trace, to find WHICH scanline the M0/BL
     motion takes visible effect (xitari) vs jutari (~1 line early). Confirm it's the
     beam_sc~79 boundary-straddle HMOVE (the #97/#98 class) applied to object position.
  2. If so, defer the OBJECT-POSITION HMOVE motion by one scanline when the strobe lands
     past the boundary — mirror of #97's `hmove_blank_pending_next` but for m*_x/bl_x,
     not just the blank comb. This is a P3i-g cycle-core change.
  3. GATE on RAM-diff + screen all 6 ROMs both ports (5 stay 0 px / bit-exact, enduro
     markers match xitari at rows 102/104) before landing; revert on ANY of the 5
     regressing (render-only, so RAM gates alone won't catch a screen regression).

---

### 🎯 Task #80 — SOLVED: seaquest boot off-by-1 fixed, all 6 ROMs bit-exact both ports (2026-06-14)

**Supersedes the "did NOT converge" entry below.** The dedicated #80 session
found the port bug and landed the faithful fix. seaquest is now **bit-exact
(0 bytes) vs xitari** in both ports; the 5 previously-bit-exact ROMs stay 0; and
enduro's leftover **jaxtari** gap (8 b/f) closed to 0 as a bonus.

**Root cause — two independent pieces, BOTH required:**
  1. **RIOT (M6532) timer was the wrong model.** Both ports used an *eager*
     decrement model with power-on `intim=0 / prescaler_shift=0`. xitari uses a
     *lazy* model: `M6532::reset` sets `myTimer=25 / myIntervalShift=6` and INTIM/
     INSTAT are **computed on read** from the monotonic system cycle counter via
     `myTimer - (delta>>shift) - 1` (with `delta = (cycles-1) - myCyclesWhenTimerSet`)
     plus the `myTimerReadAfterInterrupt` post-expiry slow-count branch. seaquest's
     boot polls INTIM heavily (2969 reads / 6 frames) while the timer is expired, so
     the post-expiry read values must match xitari exactly.
  2. **Missing max-scanlines frame cutoff.** xitari force-ends a frame after
     `myMaximumNumberOfScanlines` (=290 NTSC, TIA.cxx:2003) even with no software
     VSYNC. seaquest's boot-init runs ~455 VSYNC-less scanlines; without the cutoff
     the burst was counted as ONE frame instead of TWO, so the cart ran one extra
     boot-loop iteration → one extra `INC RAM[$01]` → `$3f` vs xitari's `$3e`.

**Why the earlier attempt failed (see below):** the lazy port had a messy
`_timer_output!` (duplicate `timer=` lines, wrong signed cast) → boot got LONGER
(34583 cyc) not shorter. The clean port uses `reinterpret(Int32, ...)` (jutari) /
explicit `_u32`/`_i32` helpers (jaxtari) to mirror xitari's uInt32/Int32 wrap
exactly. The port bug — NOT the approach — was the blocker.

**Files (both ports + tests, one commit):**
  - jutari: `src/riot/RIOT.jl` (lazy rewrite), `src/bus/Bus.jl` (×2: thread
    `tia.total_cycles + pending_tia_cycles` into `riot_peek!`/`riot_poke!`),
    `src/tia/TIA.jl` (`lines_since_frame` field + 290 cutoff), `test/runtests.jl`
    (RIOT testsets rewritten for the lazy model).
  - jaxtari: `jaxtari/riot/system.py` (lazy rewrite, `(value, riot)` returns),
    `jaxtari/bus/system.py` (×2: thread `total_cycles + new_pending`),
    `jaxtari/tia/system.py` (`lines_since_frame` field + 290 cutoff + reset on
    VSYNC 1→0 edge), `tests/test_riot.py` (rewritten), `tests/test_pxc2_*` (seaquest
    4→0, enduro 8→0 pins).

**Measured / gates (all green):**
  - `jutari_xitari_ram_diff.py` ALL 6 ROMs (NOOP, 80 frames) vs PRISTINE rebuilt
    xitari: **0 bytes** each (seaquest, breakout, pong, space_invaders, pitfall,
    enduro).
  - jutari PXC2 fixtures regenerated → **0 bytes vs xitari** for all 6 (only
    `seaquest_noop_10_jutari.jsonl` actually changed — confirms #80 is seaquest-
    specific; enduro's jutari side was already 0).
  - jaxtari PXC2 jaxtari-vs-xitari last-frame divergence: seaquest 4→**0**, enduro
    8→**0**, other 4 unchanged at 0.
  - jutari `Pkg.test()` exit 0; jaxtari `test_riot.py` 24/24 + 172 core TIA/bus/
    CPU/P3/P4 tests pass.
  - xitari reference reverted to pristine + rebuilt clean; trace_dump rebuilt.

---

### 🧪 Task #80 — FAITHFUL-PORT ATTEMPT (ultracode) did NOT converge; reverted (2026-06-13)

Attempted the principled fix: replace jutari's eager RIOT timer with a faithful
lazy port of xitari `M6532` (reset 25/shift-6; store my_timer/interval_shift/
cycles_when_set; compute INTIM/INSTAT on read via xitari's exact formulas incl.
the `myTimerReadAfterInterrupt` post-expiry branch; thread the monotonic cycle
`tia.total_cycles + pending_tia_cycles` into `riot_peek!`/`riot_poke!`;
`riot_advance!` → no-op). Code: `jutari/src/riot/RIOT.jl` rewrite + 2 `Bus.jl`
call sites. **Reverted** — `main` is clean (jutari tests pass, breakout
bit-exact, seaquest baseline 6 bytes @ frame 0).

**Why it didn't work (measured):**
  - The lazy port took effect (seaquest boot frame-1 changed 21435 → **34583**
    cyc) but went the WRONG way — LONGER than xitari's ~22116 (291 sl) — and the
    INC still landed in frame 2 (→ $3f, not $3e). So my port returns INTIM
    values that diverge from xitari (cart's boot-delay loop iterates MORE). It's
    a port BUG, not a refutation of the approach.
  - Re-confirmed dead-ends: reset-state-only (25/6 alone overshoots to 24703 +
    still $3f); double-boot + RAM-residue + lazy timer (`/tmp/jutari_doubleboot.jl`)
    → still $3f.

**The blocker — clean alignment is impossible with the current probes.** xitari
is DOUBLE-booted by trace_dump (`ALEInterface(rom)` ctor→loadROM→reset_game [boot
1, RAM zeroed] then explicit `resetGame` [boot 2, RAM=boot-1 residue]); the
REFERENCE fixture is **boot 2**. jutari single-boots. My INTIM probes mixed the
two xitari boots, so I could never align jutari's single-boot read sequence to
xitari's boot-2 read sequence to find the FIRST divergent read.

**Confirmed sub-facts (keep):**
  - seaquest boot delay polls INTIM heavily (2969 reads / 6 frames).
  - xitari boot-delay timer is TIM64T-loaded (addr $0296, val 31, shift 6); the
    RESET-default timer (25/shift-6) is read post-expiry before that load.
  - jutari (eager, reset 0/0) loads val=192/presc-0; with reset 25/6 it loads
    val=31/presc-6 (matches xitari) — so the reset fix DOES correct the load
    divergence, but the post-expiry read values still diverge.

**PRECISE NEXT STEP (fully-online, focused session):**
  1. Instrument xitari to log ONLY **boot 2** (count `reset_game` calls; emit on
     the 2nd) — its per-read INTIM sequence (value + my_timer/shift/delta) for
     the whole boot. Env-guarded probe in `M6532::peek` case 0x04 + a call
     counter.
  2. Apply the lazy port to jutari again; log its per-read INTIM sequence.
  3. Diff read-by-read → the FIRST divergent read pins the port bug (suspects:
     `cur_cycles` base vs xitari per-frame `cycles()` with `systemCyclesReset`
     adjustment; the read_after_int `cycles_when_int_reset` timing; UInt32/Int32
     edge cases).
  4. Fix the port; then rewrite the ~30 eager-model RIOT unit tests
     (`jutari/test/runtests.jl` ~L2390-2520 assert `intim`/`prescaler_shift`/
     `cycles_since_tick`/`timer_expired` internals → must test via `riot_peek!`
     return values instead); mirror to jaxtari `riot/system.py`.
  5. GATE: `jutari_xitari_ram_diff.py` ALL 6 ROMs (seaquest→0, 4 bit-exact stay
     0) + jutari tests + jaxtari unit + PXC2 + PXC-S. Commit only if fully green.

Cost/benefit note: this is a P4-scale RIOT-timer rewrite (subsystem + ~30 tests +
both ports) for a 1-ROM HUD-flicker ($3e/$3f phase). Worth doing for fidelity,
but it is NOT a turn-sized fix — budget a dedicated session.

---

### 🎯 Task #80 — ROOT AREA FOUND: RIOT timer RESET-STATE mismatch (both ports 0/0 vs xitari 25/shift-6) (2026-06-13)

Best #80 lead yet — the boot off-by-1 lives in the **RIOT (M6532) timer**, not
frame-counting/render. Chain of evidence this session:

1. The seaquest boot delay **polls INTIM heavily** (jutari probe: 2969 reads /
   6 boot frames; first reads count down 192,185,178,… −7/read). So the cart's
   "wait N" boot delay is timer-driven.
2. xitari issues its **first VSYNC ~9000 CPU cycles (~118 scanlines, ≈1 frame)
   EARLIER** than jutari (xitari first VSYNC-on at cpucyc 12236/sl 161; jutari at
   tc 21204/sl 279). Same cart, same zeroed RAM → the boot-delay LOOP runs longer
   in jutari ⇒ its frame counter ends 1 ahead.
3. **The mismatch:** `M6532::reset()` (xitari, M6532.cxx:60-61) resets the timer
   to `myTimer=25, myIntervalShift=6` (TIM64T-equivalent, prescaler 64). BOTH
   ports reset to `intim=0, prescaler_shift=0` (jutari RIOT.jl:53-54; jaxtari
   riot/system.py:103-104). So before the cart loads its own timer, xitari's INTIM
   counts down from 25 @ presc-64 while the ports sit at 0 @ presc-1.

**Tested (jutari):** set the reset state to `intim=25, prescaler_shift=6`.
Result: seaquest boot frame 1 went 21435→24703 cyc (the reset state DOES drive
the boot delay) — but it OVERSHOT xitari's ~22116-cyc (291-sl) frame 1 and the
INC still landed in frame 2 (→ still $3f, not $3e). So the reset-state mismatch
is real and load-bearing, but **not sufficient alone**: jutari's timer-EXPIRY
semantics also diverge from xitari's. xitari's expired-timer read (M6532.cxx
case 0x04, the `timer<0` branch with `myTimerReadAfterInterrupt` /
`myCyclesWhenInterruptReset` lazy formula) differs from jutari's eager
`timer_expired` model — and the seaquest boot reads the timer while it's expired
(xitari trace: myTimer=25, shift=6, delta~3907, timer=-37). Reverted the 25/6
experiment (overshoots + unrendered risk to the 4 bit-exact ROMs).

**FIX PLAN (deferred to fully-online session — high-risk RIOT change, gate
hard):**
  (a) Match xitari's RIOT reset state exactly: `intim=25, prescaler_shift=6`
      (and confirm `cycles_since_tick`/`timer_expired` analogues == xitari's
      `myCyclesWhenTimerSet=0`, not-yet-read-after-interrupt).
  (b) Reconcile jutari/jaxtari timer-EXPIRY semantics with xitari's
      `myTimerReadAfterInterrupt` lazy formula so an expired-timer read during
      boot returns the SAME value as xitari (this is what the 25/6 overshoot
      exposed).
  (c) Decisive diagnostic to drive (b): an ALIGNED INTIM trace — same single
      boot, read-by-read — jutari vs xitari (add the env-guarded probes used this
      session: jutari `riot_peek!` reg-4 log; xitari M6532.cxx case 0x04 log).
  (d) GATE: `jutari_xitari_ram_diff.py` ALL 6 ROMs — seaquest must reach 0, the
      4 bit-exact ROMs (breakout/pong/space_invaders/pitfall/enduro) MUST stay
      bit-exact (they pass today with 0/0, so they presumably overwrite the timer
      before reading — but a reset-state change touches every boot, so confirm) —
      then jutari tests + jaxtari unit + PXC2, then full PXC-S. Revert on any
      regression (#93/#95 discipline).

This supersedes the frame-counting / VSYNC-duration / RAM-residue dead-ends below.

---

### ⚠️ Task #80 — CORRECTION: VSYNC-duration root was WRONG; RAM-residue also ruled out (2026-06-13)

Two #80 hypotheses tested and DISPROVEN this session (logging dead-ends per the
project rule). The earlier "🎯 task #80 ROOT FOUND" commit (VSYNC-on-≥1-scanline)
is **retracted** — do not implement that fix.

1. **VSYNC-on-duration — RULED OUT.** Traced jutari's seaquest-boot VSYNC writes
   (temp env-guarded probe in `tia_poke!`, reverted): the VSYNC pulse that ends
   jutari's frame 1 is `tc 21204(on)→21432(off)` = **228 CPU cycles = 3 scanlines**
   — a NORMAL pulse, well over xitari's ≥1-scanline (76-cyc) threshold, so xitari
   counts it too. (The two earlier writes at tc=18, tc=754 are VSYNC-*off* with no
   prior on — both ports ignore them.) The frame-1 boundary is NOT a short-VSYNC
   miscount.

2. **Double-boot / RAM-residue — RULED OUT.** xitari's `M6532::reset()` does NOT
   clear the 128-byte RAM (only the M6532 *constructor* zeroes it,
   M6532.cxx:37-39 vs :58), and trace_dump double-boots (`ALEInterface(rom)` ctor
   → loadROM → `reset_game` [boot 1, zeroed RAM], then explicit `resetGame()` →
   `reset_game` [boot 2, from boot-1 RAM residue]). Hypothesis: the reference
   ($3e) came from booting on residue. TESTED in jutari
   (`/tmp/jutari_doubleboot.jl`): boot 1 = $3f, boot 2 with boot-1 RAM residue
   kept = **still $3f**. So RAM residue does NOT change the result, and xitari's
   boot 1 (from ctor-zeroed RAM, == jutari's start) already shows the INC-frame-3
   / $3e pattern. The divergence is in the **single clean boot from zeroed RAM**,
   not the double-boot.

**What IS established (hard facts):** from an identical zeroed-RAM clean reset
running the same seaquest ROM, **jutari's `INC RAM[$01]` first fires in boot frame
2 (→ ends $3f); xitari's first fires in frame 3 (→ ends $3e)** — jutari is exactly
one boot-frame ahead. Per-frame (jutari): frame1=282 sl, INC frame 2. xitari:
frame1≈291 sl + an extra ≈164-sl frame 2, INC frame 3. Both see the SAME first
VSYNC pulse (228 cyc @ ~scanline 282) yet xitari reports a ~9-line-longer frame 1
plus an extra short settle frame — UNEXPLAINED by the VSYNC model above.

**Remaining unknown + exact next probe.** Need xitari's per-frame **PC** and the
VSYNC on/off **color-clock** (in xitari's `clock = cycles*3 - myClockWhenFrameStarted`
space, incl. the `startFrame` `myClockWhenFrameStarted = -clocks` carry and the
`updateFrame(clock+delay)` `delay`) for boot frames 1-3, to see exactly where
xitari cuts frame 1/2 vs jutari's PC=f69c. Add `pc()` to trace_dump's CpuDebug +
extend the env-guarded `emulate` probe (scanlines + PC + a per-VSYNC color-clock
log). This is a deeper multi-hour dive into xitari's frame-clock bookkeeping;
DEFERRED to a fully-online session (internet dropping). No fix until the framing
mechanism is actually pinned — the discipline cost of guessing here is high (#93/#95).

---

### 🗺️ SHARED-ROOT INVESTIGATION PLAN (2026-06-13) — is one ~1-cycle CPU↔TIA beam offset behind #80 + #97-residual + #98 + pitfall-INTIM?

**Hypothesis (to TEST, not assume).** The 4 remaining divergences may share one
root — jutari's sub-instruction beam position lands ~1 CPU cycle (~3 color clocks)
before xitari's at bus ops (noted "expected" in CLAUDE.md). They surface on
different paths:
  - #80 seaquest boot: jutari's cart hits `INC RAM[$01]` a frame early (frame 2 vs
    xitari frame 3) — a FRAME-BOUNDARY question (jutari counts frames on the
    software VSYNC falling edge; the 262-line scanline-wrap net was removed in
    PXC1-x).
  - #97 residual (enduro): road-border player X off by 1 color clock (RESP+HMOVE).
  - #98 (pong): ball 2 scanlines low for 2 frames (ENABL/Y scanline).
  - pitfall: INTIM read ~1 cycle early.
They might be ONE root or several. The plan tests #80 first (cleanest: RAM-level,
deterministic boot, no rendering involved) and then checks whether the same
mechanism explains the render cases.

**Phase 1 — #80 boot, pin the frame-boundary mechanism (diagnostic, SAFE).**
  1. jutari per-frame (have it, `/tmp/jutari_cycle_probe.jl`): frame 1 = 21435 cyc
     (282 scanlines), boundary PC=f69c (cart's 1st VSYNC), first INC frame 2.
  2. Get xitari's per-frame **PC + scanline** at boot frames 1–4. `CpuDebug`
     (trace_dump.cpp) exposes A/X/Y/SP/P but not PC — add
     `static uInt16 pc(const M6502& c){return c.PC;}`. For a per-frame dump during
     the boot burn (which happens inside `ALEInterface::resetGame`, before
     trace_dump regains control), use an env-guarded probe in
     `StellaEnvironment::emulate`'s joystick loop (the recipe used for the RAM
     probe — `std::getenv("XITARI_BOOTPROBE")`), dumping PC and the TIA scanline.
     Read PC via `mySystem->m6502().PC`? PC is protected — simplest is to dump
     `mySystem->cycles()` (per-frame, resets each frame) + a self-maintained
     cumulative counter, OR add a `friend`/accessor. Pick the least-invasive that
     compiles; REVERT + rebuild clean after (xitari ref must stay pristine).
  3. Compare: does xitari cut frame 1 at a different cart PC than jutari's f69c?
     Expected if xitari's 1st frame ends at a scanline-render boundary (≈262
     lines, mid-init) while jutari runs to the cart's 1st VSYNC (282 lines).

**Phase 2 — decision tree.**
  - If #80 is a FRAME-DEFINITION difference (xitari = scanline-rendered frame,
    jutari = VSYNC-edge frame) ⇒ fix is in jutari's `run_until_frame!` /
    frame-counter, NOT the P3i-g beam core. Risk localized to boot; the 4
    bit-exact ROMs are insensitive (proven) so likely safe — but re-introducing a
    scanline-wrap boundary risks the double-count bug PXC1-x removed, so it must
    be done so the VSYNC edge and the wrap can't BOTH fire for one frame.
  - If #80 is a sub-instruction beam-position error (jutari's beam_sc/beam_cc off
    by ~1 cycle at the VSYNC write) ⇒ it IS the P3i-g core, shared with #97/#98.
    Highest risk.

**Gating for ANY fix (all OFFLINE-capable except the last):**
  - `cd jutari && julia --project=. -e 'using Pkg; Pkg.test()'` (fast).
  - `tools/jutari_xitari_ram_diff.py` for ALL 6 ROMs: seaquest must reach 0;
    breakout/pong/space_invaders/pitfall/enduro must STAY bit-exact. (This is the
    decisive correctness gate and is fast + offline.)
  - jaxtari TIA unit suite + PXC2 cross-check (fast-ish, offline).
  - Full PXC-S all-6 (jaxtari env ~2 h) — the final gate before claiming done.
  REVERT immediately on any bit-exact regression (discipline from #93/#95).

**Status:** Phase 1 DONE — ROOT FOUND (see below). Fix designed; gating next.

**Phase 1 RESULT (2026-06-13) — xitari per-boot-frame scanline counts** (probe:
env-guarded `scanlines()`+RAM[$01] dump in `StellaEnvironment::emulate`, reverted
+ rebuilt clean after):
```
            frame1  frame2  frame3  frame4   RAM[$01] per frame
xitari:      291     164     262     262     00, 00, 01, 02   (extra SHORT settle)
jutari:      282     262     262     262     00, 01, 02, 03   (no extra frame)
```
xitari takes TWO settle frames (291 + 164 scanlines) before the steady 262-line
loop and its first `INC RAM[$01]` is in frame 3; jutari takes ONE (282) and INCs
in frame 2 → jutari's counter is 1 ahead for the whole run (ends $3f vs $3e).

**ROOT CAUSE (frame-DEFINITION difference, NOT the P3i-g beam core):** jutari's
VSYNC handler (`TIA.jl:514-528`) treats ANY VSYNC 1→0 falling edge (while
`vsync_active`) as a frame boundary. xitari (`TIA.cxx:2009-2030`) ends a frame on
a VSYNC-off ONLY when `clock >= myVSYNCFinishClock`, where
`myVSYNCFinishClock = (clock at VSYNC-on) + 228` — i.e. VSYNC must have been ON
for ≥1 scanline (228 color clocks). It also inits `myVSYNCFinishClock` to
`0x7FFFFFFF` (reset, TIA.cxx:240), so a VSYNC-off BEFORE any VSYNC-on is ignored.
seaquest's boot kernel emits a short VSYNC (on <1 scanline, or an off with no
prior on) near line 282 that jutari counts as a frame but xitari does not — so
jutari skips xitari's extra 164-line settle frame and runs 1 frame ahead.

This is GOOD news for risk: the fix is in jutari's frame-counter, NOT the shared
P3i-g beam core (so it does NOT touch the enduro/pong render residuals — those
remain the separate ~1-cycle beam offset). The 4 bit-exact ROMs use standard
3-scanline VSYNC pulses (≫1 scanline) so the condition won't change their
counting — but this MUST be confirmed by the RAM-diff-all-6 gate (could regress a
ROM that relies on a short-VSYNC count). NOTE: this also means #80 and the
render residuals (#97/#98/pitfall) are likely TWO different roots, not one — the
"unifying hypothesis" only holds for the render cases.

**FIX (jutari `TIA.jl` VSYNC handler):** track the clock when VSYNC goes on
(monotonic, e.g. via total_cycles or an explicit `vsync_on_clock`), and at the
1→0 edge only set `vsync_reset_pending` if VSYNC was on ≥76 CPU cycles
(=228 color clocks). Mirror in jaxtari. Gate: jutari tests + RAM-diff ALL 6 ROMs
(seaquest→0, others stay 0) + jaxtari unit + PXC2, then full PXC-S.

---

### 🔬 Task #98 CHARACTERIZED (2026-06-13) — pong ball is 2 SCANLINES low for 2 frames (not "2 px"); likely the shared ~1-cycle beam offset

Pixel diff of the cached 600-frame pong dumps (`output/pong_{xitari,jutari}_frames.raw`):
only **f459 + f460** differ, 16 px each (everything else bit-identical). Tracking
the ball (color $C8=200, cols 140-143) across the bounce:
```
f457/458: both rows 150-165
f459/460: xitari rows 160-175   jutari rows 162-177   <- jutari 2 scanlines LOW
f461/462: both rows 172-187     (re-synced)
```
So jutari renders the 16px-tall ball **2 scanlines lower** for exactly 2 frames
around a vertical step, then re-syncs — the 16 px diff is 2 rows × 4 cols at each
ball edge. RAM is bit-exact (the ball's logical Y is identical), so this is a pure
RENDER-timing artifact: the ball's ENABL/vertical-position scanline is evaluated
~2 scanlines off at this transient. The plan's "2 px" was really "2-scanline
offset".

**Unifying hypothesis (the remaining residuals share a root).** #98 (ball 2 sl),
#97's residual (enduro road-border 1-cc), #80 (seaquest boot frame-2), and
pitfall's INTIM ~1-cycle read offset all look like the **same ~1-cycle CPU↔TIA
beam-position offset** in the P3i-g cycle-accounting core, surfacing on different
register/timing paths (ENABL scanline / RESP+HMOVE x / VSYNC-frame boundary /
INTIM poll). A correct fix to that one offset would likely close several at once —
but it is the HIGHEST-risk change (touches every ROM's per-instruction timing),
so it must be done in a fully-online session gated on full PXC1+PXC2+all-6 PXC-S.
Until then these are all correctly deferred (each is ≤16 px / transient / a 1-frame
state phase).

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

**Mechanism deepening (2026-06-13, `/tmp/jutari_cycle_probe.jl`):** jutari's
per-frame cycle/PC trace —
```
post-reset: PC=f000 (reset vector)          RAM01=00
frame 1: Δ=21435 cyc (=282 scanlines) PC=f69c RAM01=00   <- init completes here
frame 2: Δ=19912 cyc (=262, normal)   PC=f69c RAM01=01   <- first INC
frame 3..: Δ=19912 each               PC=f69c RAM01=02..
```
jutari counts a frame at the software VSYNC falling edge (the 262-line scanline-
wrap "safety net" was REMOVED in PXC1-x to stop double-counting). So jutari's
frame-1 boundary is the cart's FIRST VSYNC at PC=f69c — 282 scanlines in, i.e.
init runs to completion inside frame 1 and the first `INC RAM01` lands in frame
2. xitari reaches the first INC in frame 3, so its boot "settle" spans an extra
frame. Exact xitari frame-1 boundary PC still TBD — needs a per-frame PC probe
on the xitari side (CpuDebug exposes A/X/Y/SP/P but not PC; add
`static uInt16 pc(const M6502&){return cpu.PC;}` to trace_dump's CpuDebug, or an
env-guarded PC dump in `StellaEnvironment::emulate`). That datum pins WHERE
xitari cuts frame 1 → tells exactly how to delay jutari's first INC by one
frame. FIX STILL DEFERRED (high-risk boot timing; needs full PXC1+PXC2+all-6
PXC-S gating, ~2 h for the jaxtari env — do it in a fully-online session).

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

### 🏆 Task #97 LANDED — both ports CONFIRMED (2026-06-13) — enduro HMOVE-blank comb 249 → 33 px

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
  - jaxtari ENV screen now CONFIRMED: live-env worst = **33 px**
    (per-frame `[21,23,27,27,29,33,25,33,31,31]`, identical to jutari) and
    pong/breakout/pitfall/space_invaders all **0 px** — PXC2 parity holds, no
    regression. The shared PXC-S enduro pin was held at 137 during the ~40-min
    jaxtari env run (jutari-first rule) and is now **tightened 137 → 33**.

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
  | pong           | bit-exact                | 16 px @ f459-460 | #98 CHARACTERIZED (ball 2 sl low, 2 frames) |
  | enduro         | **bit-exact (40 f)**     | 257 px*      | #97 LANDED (noop-10 249→33) |
  | seaquest       | **diverges f0** (6 bytes)| 1449 px      | #80 LOCALIZED → frame-2 partial boot |

\* enduro 257 px is the pre-#97 600-frame VIDEO worst; #97 (HMOVE-blank `_next`)
took the noop-10 PXC-S 249→33 on both ports. The 600-frame video isn't
re-measured yet but should drop similarly (same comb mechanism); the residual is
the road-border 1-cc positioning.

**Decisive recon** (`jutari_xitari_ram_diff.py` on the video streams):
  - enduro RAM bit-exact ⇒ its 257 px is a pure RENDER bug (state
    identical). Broad: 92 rows differ, hottest 53-54, 66-71, 154.
  - seaquest RAM diverges at frame 0 (`$01 $02 $60 $66 $7e $7f` — the
    task #80 boot off-by-1) ⇒ its 1449 px (hot band rows 45-53 = score/
    oxygen HUD) is largely DOWNSTREAM of state. Fix #80 first.
  - pong RAM bit-exact ⇒ 2 px at ball rows 160/161/176/177 is a pure
    ball-bounce sub-cycle render artifact.

**Order: ~~enduro (#97)~~ ✅ → seaquest state (#80, LOCALIZED) → pong (#98).**
  1. ✅ **#97 enduro DONE** — HMOVE-blank `_next` deferral landed both ports
     (noop-10 249→33, jaxtari env confirmed 33, bit-exact ROMs 0, pin tightened
     to 33). Residual 33 px = road-border 1-cc positioning (deferred, P3i-g
     core, high-risk). See "Task #97 LANDED" above.
  2. **#80 seaquest** — NOW LOCALIZED to the FRAME-2 partial-boot divergence
     (xitari's first post-reset frame is partial; jutari's is full → jutari's
     cart counter runs 1 frame ahead). Fix: reproduce xitari's partial first
     frame in jutari; gate on full PXC1+PXC2+all-6 PXC-S (high-risk boot
     timing). See "Task #80 LOCALIZED" above.
  3. **#98 pong** — CHARACTERIZED: ball renders 2 scanlines low at f459-460
     only (16 px, RAM bit-exact, transient at a vertical step). Pure render-
     timing; likely the same ~1-cycle beam offset as #97-residual/#80/pitfall
     (see "Task #98 CHARACTERIZED" above). Lowest priority — fix it via the
     shared-root fix, not a pong-specific hack.

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

### 📐 Task #106 PLAN (2026-06-15) — xitari TIA partial-frame / grey-frame model (qbert 56, surround)

**Confirmed root cause (qbert).** jutari user-frame[i] ≡ xitari frame[i+1]
BYTE-FOR-BYTE (`/tmp/qbert_offset.py`: offset+1 diff = all zeros). qbert's
entire 56 b/f is a pure frame-count offset: jutari skips xitari's first
"sliver" frame. Measured xitari per-frame bus-op counts: frame 1 = **141 ops**
(sliver), frame 2 = 710, frames 3+ = ~15700 (full). The sliver's first bus op
is at **sl=3948** → `mySystem->cycles()` was NOT reset → the previous frame
never ended (xitari `myPartialFrameFlag` stayed true through the boot's tail).

**Mechanism.** qbert's RESET-boot wait loop (task #52) is a tight poll loop
with NO TIA pokes for thousands of scanlines. xitari ends a frame ONLY at
poke time (`TIA::poke` line 2003 max-scanline cutoff; line 2022 VSYNC-clear
hold-gate) OR never within one `update()` — `m6502().execute(25000)` runs at
most **25000 INSTRUCTIONS** (M6502Low.cxx:65 `for(;number!=0;--number)`, NOT
cycles), then returns a *grey* frame (greyOutFrame, frame counter NOT
incremented, partialFrameFlag stays true, no startFrame → cycles keep
accumulating). So xitari slices the poke-less wait loop into 25000-instruction
grey `update()`s. jutari instead force-ends the frame EVERY CPU step in
`tia_advance!` (`lines_since_frame > 290`), slicing the loop into ~291-scanline
REAL frames → different slice count → +1 frame offset that persists.

**Fix (jutari, then mirror jaxtari):**
1. `Console.jl run_until_frame!`: bound to **25000 instructions** (xitari
   `execute(25000)`); on budget exhaustion return a GREY frame (no error, frame
   counter not advanced — the next `run_until_frame!` continues the same frame
   because the beam/scanline state is never reset).
2. `TIA.jl tia_poke!`: add xitari's max-scanline cutoff at POKE time
   (`lines_since_frame > 290` → set `vsync_reset_pending`, disarm
   `vsync_finish_clock`), mirroring TIA.cxx:2003-2007 (before the register
   switch). This is the ONLY place a non-VSYNC frame end may now happen.
3. `TIA.jl tia_advance!`: REMOVE the every-step `lines_since_frame > 290`
   cutoff (task #80's block). The grey-frame budget (1) prevents an infinite
   poke-less loop; the poke-time cutoff (2) ends real frames.

**Why safe for the 61 bit-exact games:** normal frames end via VSYNC within
~5-8k instructions (< 25000) and reset `lines_since_frame` every ~262 lines, so
neither the budget nor the cutoff fires — behavior unchanged. The every-step
vs poke-time cutoff differs ONLY for poke-less stretches > 290 scanlines, which
only qbert's wait loop has. RISK: #80 (seaquest) / #108 (air_raid) ended frames
via the cutoff during a VSYNC-less burst — if those bursts have an inter-poke
gap, the boundary may shift. GATE: full 64-ROM sweep must keep all 61 at 0 and
close qbert; revert if any regress.

### ✅ Task #106 SOLVED (jutari) (2026-06-15) — qbert 56→0 via the partial-frame / grey-frame model

Implemented the plan above. Three changes in jutari:
1. `Console.jl run_until_frame!`: bounded to **25000 instructions**
   (`_UPDATE_INSTRUCTION_BUDGET`, = xitari `m6502().execute(25000)`); on budget
   exhaustion it returns a GREY frame (no error, frame counter NOT advanced,
   beam/scanline/cycle state preserved) so the next call continues the same
   TIA frame — exactly xitari's `myPartialFrameFlag`-true path.
2. `TIA.jl tia_poke!`: added the max-scanlines cutoff at POKE time
   (`frame_clock ÷ 228 > 290` → set `vsync_reset_pending`, disarm the VSYNC
   hold-gate), mirroring TIA.cxx:2003-2007.
3. `TIA.jl tia_advance!`: REMOVED the per-CPU-step `lines_since_frame > 290`
   cutoff (the task #80 block). Frame ends are now decided ONLY in the poke
   handler (VSYNC hold-gate OR max-scanlines), like xitari.

**Verification (FULL 64-ROM RAM sweep, the gate):**
- qbert **56 → 0 ✅** (now byte-for-byte aligned; previously jutari[i]≡xitari[i+1]).
- **ZERO regressions**: all 61 previously-bit-exact games stay at 0, including
  the two at-risk games whose frames end via the cutoff during a VSYNC-less
  boot burst — **seaquest 0, air_raid 0** (their bursts poke every scanline, so
  poke-time ≡ every-step there). 62/64 bit-exact.
- skiing 85→84 (unchanged class — the sweep's frame-20 FIRE edge case; NOOP-
  bit-exact). surround still 16 (needs the {SELECT,RESET} starting actions —
  separate, next).
- jutari `Pkg.test` green (the task #80 test rewritten: a poke-less VSYNC-less
  ROM now correctly GREYS — frame counter does not advance — and a TIA-poking
  VSYNC-less ROM ends at the poke-time cutoff).

Mirror to jaxtari pending (immediate-increment model; verify via direct RAM
diff since jaxtari pytest is wedged).

### ✅ Task #106 SOLVED (jaxtari mirror) (2026-06-15) — partial-frame model ported

Mirrored the jutari fix to jaxtari (faithful structural copy; jaxtari uses the
*immediate* frame-increment model in `tia_poke` rather than jutari's deferred
`vsync_reset_pending`, so the cutoff increments `frame` directly there):
1. `console.py run_until_frame`: bounded to **25000 instructions**
   (`_UPDATE_INSTRUCTION_BUDGET`); grey-frame return on budget exhaustion (no
   `RuntimeError`), preserving beam state.
2. `tia/system.py tia_poke`: max-scanlines cutoff at POKE time
   (`frame_clock ÷ 228 > 290` → `frame+1, scanline=0, lines_since_frame=0`,
   disarm hold-gate), before the register switch — mirrors TIA.cxx:2003-2007.
3. `tia/system.py tia_advance`: REMOVED the per-step `lines_since_frame > 290`
   cutoff (task #80 block).

**Verification (direct jaxtari-vs-xitari RAM diff; pytest still wedged):**
- Standalone smoke test: poke-less VSYNC-less ROM GREYS (frame counter does not
  advance, beam runs to lines_since_frame=986); a TIA-poking VSYNC-less ROM ends
  at the poke-time cutoff.
- qbert **0 ✅** (fix works — was the +1 offset), seaquest **0 ✅**, air_raid
  **0 ✅** (with AirRaidRomSettings — an initial 54 was a harness bug from
  passing GenericRomSettings, not a regression). All 8-frame NOOP aligned diffs.
- This keeps the jaxtari ≡ jutari (PXC2) invariant: both now match xitari's
  partial-frame slicing.

### ✅ Task #103 PROGRESS (jutari) (2026-06-15) — skiing 84→1, surround 16→7 (illegal-action filter + PAL/SELECT-RESET)

Two parallel investigation agents pinpointed both residuals (read-only,
evidence-based). Implemented the jutari fixes; gated on the full 64-ROM sweep
(62/64 stay bit-exact, ZERO regressions incl. seaquest/air_raid/qbert).

**SKIING 84 → 1 (illegal-action injection).** xitari's `StellaEnvironment::act`
calls `noopIllegalActions` (stella_environment.cpp:189) BEFORE a user step,
converting actions for which `isLegal` is false to NOOP. Skiing is the ONLY
supported game overriding `isLegal` (Skiing.cpp:96-111: rejects the whole FIRE
family {1,10,11,12,13,14,15,16,17}). The sweep's shared breakout stream injects
FIRE at frame 20; jutari applied it verbatim (no legal-action filter) → skiing's
game state diverged 84 b/f. Fix: added `romsettings_is_legal_action` (default
true; RomSettings.jl), overrode SkiingRomSettings (JoystickGames.jl), and
filter illegal→NOOP in `env_step!` (StellaEnvironment.jl). Starting actions
bypass the filter (xitari emulate() bypasses noopIllegalActions). Residual 1 b/f
= a SEPARATE pre-existing $00 frame-counter off-by-one (present under pure NOOP).

**SURROUND 16 → 7 (PAL frame cutoff + console-switch starting actions).** Two
coupled causes: (1) missing SurroundRomSettings — xitari getStartingActions =
{SELECT,RESET} (Surround.cpp:135) selects game variation 1; jutari booted
generic (variation 0). Added a console-switch starting-actions path
(`romsettings_console_switch_starts` → routed via `console_switches!` in
env_reset!, since apply_action! can't encode codes 46/40). (2) surround is PAL
(312-line frame); jutari's hardcoded 290 max-scanlines cutoff force-split it
into 291+21 = two frames/TV-frame (half-rate counters). Added a PAL-gated
max-scanlines threshold: `max_scanlines` field on TIAState (default 290, set to
342 for `romsettings_pal` ROMs in env_reset!), and the poke-time cutoff now
compares against it. PAL-GATED, not global — seaquest (NTSC, 455-line boot burst
sliced at 290) stays 0; air_raid (PAL, VSYNC@286) is cutoff-insensitive and
stays 0. Result: frame-0 6→1 byte, per-frame rate corrected ($fd now +1/frame
matching xitari). Residual 7 b/f: (a) $fd has the right RATE but a constant +95
offset — xitari runs 60 PAL-autodetect probe frames at console construction
(Console.cxx:199) that increment surround's free-running $fd; jutari uses a
static PAL flag (no probe). (b) a frame-16 ball-movement divergence
($e6/$ea/$ed/$ef/$f2/$fc) — game state matches frames 1-15, diverges when the
ball starts moving. Both are deeper than the settings/PAL fixes; documented for
follow-up (full fix would need replicating xitari's construction-time probe +
the ball-timing bug).

Files: jutari/src/games/RomSettings.jl (3 new interfaces),
jutari/src/games/JoystickGames.jl (Skiing legal-action override +
SurroundRomSettings + PAL flags), jutari/src/env/StellaEnvironment.jl
(illegal-action filter + console-switch starts + PAL max_scanlines),
jutari/src/tia/TIA.jl (max_scanlines field + PAL-aware cutoff),
tools/jutari_trace_dump.jl (surround → SurroundRomSettings).

### 🔬 Task #103 — construction-probe replication ATTEMPTED, not viable (2026-06-15)

Per user request, attempted to drive surround/skiing to 0 by replicating
xitari's construction-time format-autodetect probe. Mechanism (xitari):
M6532 zeroes RAM only in its CONSTRUCTOR (M6532.cxx:35-39); `reset()` does NOT
zero RAM. Console.cxx:197-218 runs `mySystem->reset()` → 60 `mediaSource.update()`
probe frames (at the constructor default `myMaximumNumberOfScanlines = 262`,
TIA.cxx:46) → `mySystem->reset()` AGAIN (keeps RAM) → ALE resetGame boot. So a
free-running RAM counter the game only INCREMENTS (never reads-to-init) — e.g.
surround $fd, skiing $80 — carries the probe-frame residue in xitari but starts
at 0 in jutari (which zeroes RAM on console_reset!).

EMPIRICAL TEST (/tmp/probe_experiment.jl — power-on → N probe frames @ cutoff →
keep-RAM reset → boot → measure surround frame-1 $fd vs xitari's 0xa1=161):
- no probe:                       $fd = 0x42 (66)  [current]
- 60 probe @ max_sl=262, keep RAM: $fd = 0x60 (96)  [xitari's actual probe cutoff]
- 60 probe @ max_sl=342, keep RAM: $fd = 0x7d (125)
- 60 probe @ max_sl=290, keep RAM: $fd = 0x60 (96)

NONE reach xitari's 161 — the faithful config (262) gives only 96 (65 short).
So xitari runs ~34-65 more game-VSYNCs to user-frame-1 than 60-probe + 66-boot +
1 accounts for; the extra component is NOT visible in Console.cxx and I can't
pin it down without instrumenting xitari (which must stay pristine). For skiing
(+1) the probe would OVERSHOOT by ~30-60. So the construction probe is not the
(full) mechanism for either counter.

CONCLUSION: not implementing it. It would (a) NOT reach 0 (model incomplete),
(b) run 60+ extra frames per env_reset! (≈9 min/reset on jaxtari at ~9 s/frame),
(c) carry regression risk for the 62 bit-exact games (keep-RAM reset + cart-bank
edge cases). Keeping the committed improvements (skiing 84→1, surround 16→7;
both correct & faithful, full sweep clean). Residuals = free-running frame
counters ($80 / $fd) + surround's downstream frame-16 ball seed; ALL other
gameplay state is bit-exact. A full fix needs exact replication of xitari's
construction-time frame sequence (the unaccounted ~34 frames) — deferred.

### ✅ Task #103 — surround emulation PROVEN correct; residual is construction-counter only (2026-06-15)

Follow-up to the probe-attempt entry. An N-sweep (/tmp/probe_nsweep.jl: N probe
frames @ max_sl=262 + keep-RAM reset + boot, then compare surround frames 1-20
full 128 B RAM vs xitari) found:
  N=60  maxdiff=7   N=130 maxdiff=1   N=190 maxdiff=0   N=220 maxdiff=1
At **N=190 the entire RAM is BIT-EXACT (maxdiff=0)**. This PROVES surround's CPU/
TIA/RIOT emulation is 100% correct — the 7 b/f residual (incl. the frame-16 ball
bytes) is ENTIRELY downstream of one construction-time free-running counter ($fd):
once $fd matches, every byte matches. So surround is "emulation bit-exact modulo
the construction-time counter seed."

Why no principled fix: the faithful 60-frame probe (xitari Console.cxx:199, run
at the constructor-default 262 cutoff) gives only maxdiff=7 (≈ no change) — it
adds +30 $fd, but xitari's effective seed is +95 (needs N≈190 jutari frames).
The extra ~65 is not visible in xitari's construction source and can't be
measured without instrumenting the (pristine) xitari core's TIA myFrameCounter.
Worse, the offsets are per-game incompatible: surround needs +95 (PAL frame
splits 2:1 at the 262 probe cutoff), skiing needs +1 (NTSC, no split) — a single
uniform construction probe cannot satisfy both, and a faithful 60-frame probe
fixes neither. A magic per-game frame count (surround=190) is not principled and
isn't worth the cost (60-190 frames/reset, ≈9-30 min/reset on jaxtari) or risk.

DECISION: accept skiing 84→1 and surround 16→7 as the final state for these two.
Both are correct, xitari-faithful improvements (full 64-ROM sweep clean, 62/64
bit-exact). Residuals are construction-time free-running frame counters ($80/$fd)
+ surround's downstream frame-16 ball seed; all other gameplay state is bit-exact
(surround proven via the N-sweep). Closing #103 at 62/64.

### ✅ Task #103 (jaxtari mirror) (2026-06-15) — skiing + surround fixes ported

Mirrored the jutari skiing illegal-action filter + surround PAL/SELECT-RESET
fixes to jaxtari (parity / PXC2):
- rom_settings.py GenericRomSettings: added is_legal_action (default True),
  console_switch_starts (default []), pal (default False).
- joystick_starts.py: SkiingRomSettings.is_legal_action rejects the FIRE family;
  new SurroundRomSettings (console_switch_starts={SELECT,RESET}, pal=True);
  AirRaidRomSettings.pal=True. Exported SurroundRomSettings.
- tia/system.py: TIAState gained `max_scanlines` (default 290); poke-time cutoff
  compares against it (PAL→342).
- env/stella_environment.py: PAL max_scanlines set from settings.pal() after
  console_reset; illegal-action→NOOP filter in step(); console-switch starting
  actions (SELECT/RESET via console_switches) after the joystick starts.
- check_trace.py: surround → SurroundRomSettings.
- test_tia_vsync_vblank.py: fixed the stale `test_vsync_falling_edge` (it cleared
  VSYNC without holding ≥1 scanline; now holds 1 scanline — the #103/#108
  hold-gate. This test had been failing since the air_raid hold-gate landed,
  surfaced now that the suite runs serially with -n0).

VERIFICATION (direct jaxtari-vs-xitari RAM diff, breakout stream, 30 frames;
pytest still xdist-wedged so verified via the diff like #106):
- skiing  maxdiff=0  (jaxtari's boot matches xitari's $80 counter exactly — even
  better than jutari's residual 1; the $80 +1 is a jutari-specific boot artifact).
- surround maxdiff=7  (exact parity with jutari: per-frame 1×15, 7×9, 6×6 — same
  construction-counter residual, emulation otherwise bit-exact).
- test_tia_vsync_vblank.py: 10/10 pass.

### ✅ Task #98 SOLVED (jutari) (2026-06-15) — pong dump-pot read/write cycle-threading

Traced the pong "ball-bounce render blip" (frames 460-461, 16px) to its true
root cause: a CPU-cycle-domain bug in the paddle dump-pot (INPT0-3) model.

TRACE: pong is bit-exact (RAM+screen) for 455 frames, then RAM diverges at f457
(2 bytes $04/$3c) on a LEFT paddle move, surfacing as a 16px ball offset at
f460 then re-converging. Per-bus-op trace (xitari trace_dump --bus-trace vs
jutari cpu_tia_cycle_trace): the game polls the paddles on alternating frames;
at f457 it reads INPT1 (user paddle) and the dump-pot cap crosses threshold
(0→0x80) at scanline **195 in xitari but 197 in jutari** — 2 scanlines late.
INPT0 (f456, fixed centre resistance) matched, so it was boundary-sensitive.

ROOT CAUSE: xitari's INPT0_3 dump-pot (TIA.cxx:1877-1900) compares
`mySystem->cycles() > myDumpDisabledCycle + needed`, where BOTH cycle values
are the exact `mySystem->cycles()` at their bus access (incl. the cycles
consumed so far inside the current instruction). jutari used the bare
instruction-start `tia.total_cycles` for BOTH the INPT read AND the VBLANK-D7→0
cap-release write — missing the sub-instruction offset (`pending_tia_cycles`)
that the RIOT/INTIM read already threads (Bus.jl:307). When the threshold
crossing fell within that ~few-cycle window of an INPT poll, jutari saw the
flip one ~2-scanline poll late.

FIX (jutari): thread `pending_tia_cycles` into BOTH sides so they share xitari's
cycle domain — `tia_peek` gains `extra_cpu_cycles` (dump-pot read), `tia_poke!`
gains `extra_cpu_cycles` (records `dump_disabled_cycle = total_cycles + offset`
at the VBLANK release), both fed `bus.pending_tia_cycles` from Bus.jl. Threading
only the read side first MOVED the bug (f456→f182, same $04/$3c) — both sides
MUST be consistent. Also changed `round`→`trunc` to match xitari's `(uInt32)`
cast + jaxtari's `int()` (TIA.cxx:1890; latent, not the trigger here).

VERIFIED: pong RAM bit-exact through **465 frames** (was first-div f456) and
screen bit-exact through **500 frames** (the f460 blip gone). Full 64-ROM sweep
**62/64, zero regressions** (breakout/pong/seaquest/enduro all 0; skiing 1 /
surround 7 unchanged construction-counter residuals). jutari Pkg.test green.
This was the last known jutari render/RAM residual on the conformance ROM set.
jaxtari parity: jaxtari's dump-pot read uses bare int(total_cycles) too — it
likely needs the same both-sides threading (follow-up for PXC2).

### 🗺️ Render scoreboard + probing plan (2026-06-15) — 29/64 screen-exact

Built `tools/rom_sweep/sweep_jutari_screen.py` (the render companion to the RAM
sweep): per-frame 210x160 framebuffer diff, jutari vs xitari, 60 frames,
correct per-game RomSettings. Result: **29/64 pixel-exact**
(`results_jutari_screen.md`) — RAM is 62/64 bit-exact but the *renderer*
diverges on 35 games. Per-pixel probes show one dominant signature: **jutari
emits a colour where xitari emits black**, confined to scanline bands (rows
outside the band match → not a global shift) — i.e. a TIA output-blanking /
display-window family of bugs. Buckets + a probing methodology are written up in
**PROBING_PLAN_RENDER.md**: (A) PAL screen height (5 games, jutari NTSC-only),
(B) the VBLANK/window "draws-where-blanked" family (top band, bottom band,
full-width structural fills — likely one shared root cause), (C) qbert (RAM-exact
but 7664 px — a partial-frame/grey-frame framebuffer side-effect of #106).
Verification mp4s (xitari|jutari|diff) rendered + committed for pong/seaquest/
enduro/qbert/breakout under tools/breakout_video/output/ (the repo tracks these).

### ✅ Task #109 (render) (2026-06-15) — VBLANK output-blanking writes BLACK (29→37/64 screen)

First render-conformance fix from PROBING_PLAN_RENDER.md bucket B. xitari's
`updateFrameScanline` MEMSETS the framebuffer to 0 when `myVBLANK & 0x02`
(TIA.cxx:1121-1124) — VBLANK output-blanking actively paints black. jutari's
`tia_advance!` only SKIPPED the row in its VBLANK branch, so VBLANK-blanked
scanlines retained STALE framebuffer content from prior frames. Probe: star_gunner
clears VBLANK at sl=44 (confirmed identical in both ports via bus-trace), so its
display rows 0-9 (internal sl 34-43) are VBLANK; xitari shows black, jutari showed
last frame's pixels (ju=184 vs xi=0). Fix: in the VBLANK branch of `tia_advance!`,
write a black row to the framebuffer for completed scanlines ≥ Y_START (mirror of
the visible-branch commit). Screen scoreboard 29→**37/64** pixel-exact; cleared
star_gunner/hero/crazy_climber/frostbite/beam_rider/chopper_command/video_pinball
and slashed robotank 1353→241, solaris 482→2. ZERO regressions (the change only
writes tia.framebuffer; RAM/registers/collisions untouched — RAM sweep + Pkg.test
unaffected). Remaining divergers have OTHER causes (structural up_n_down/pacman,
qbert partial-frame, sub-scanline VBLANK edges) — next.

### ✅ Task #110 (render) (2026-06-15) — PAL display height + colour-loss (unblocks PAL screen comparison)

PROBING_PLAN_RENDER.md bucket A (PAL screen height). jutari rendered a fixed
NTSC display (210 rows from `Y_START=34`, framebuffer 244 tall, NTSC-only
262-line scanline wrap), so PAL games whose `Display.Height` is taller than 210
were **not even screen-comparable** with xitari (`sweep_jutari_screen.py` flagged
them "PAL not matched / shape mismatch"). xitari renders each ROM's
`Props Display.Height` (Props.cxx default 210; Console.cxx:213-215 bumps a PAL
game that kept the default to 250), wraps scanlines at the PAL 312-line frame,
and applies PAL/SECAM **colour-loss** (TIA.cxx:562-577: on a frame whose
predecessor had an *odd* scanline count it ORs D0 into every COLU output).

FIX (jutari, all PAL-gated so NTSC is byte-identical):
- `SCREEN_HEIGHT` 244→**312** (covers the full PAL frame so a PAL frame's
  sl 262..311 never fold onto the top of the buffer).
- 4 new per-TIA fields (`TIAState`, defaulted NTSC in `initial_tia_state`,
  overridden in `env_reset!` from `romsettings_pal`/`romsettings_screen_height`):
  `screen_height_rows` (rows `get_screen` returns from Y_START — 210 NTSC, 250
  surround/air_raid), `scanlines_per_frame` (row-wrap: 262 NTSC / 312 PAL),
  `color_loss_enabled` (PAL/SECAM), `color_loss_active` (per-frame state,
  recomputed at the VSYNC boundary from the just-ended frame's `lines_since_frame`
  parity — mirror of xitari keying on `myScanlineCountForLastFrame & 0x01`).
- render commit ORs `0x01` into each rendered pixel when `color_loss_active`
  (HMOVE-blank + VBLANK pixels stay black — they're not COLU values).
- `romsettings_screen_height` interface (default 210); surround/air_raid → 250.
- `get_screen` crops `screen_height_rows` instead of the fixed `VISIBLE_HEIGHT`.

VERIFIED: full screen scoreboard **37/64** pixel-exact (unchanged vs #109 — no
NTSC regression; every previously-exact game stays 0 px). air_raid and surround
are now **comparable** where they were "PAL not matched": air_raid → 24 px @
rows 219-223 (a genuine render residual in the PAL-extended region, deferred),
surround → 224 px (the same construction-counter free-running offset as its 7 b/f
RAM residual, a proven non-bug — task #103). RAM sweep **62/64 unchanged**
(PAL-gated, no RAM path touched); jutari Pkg.test green (the "off-screen lines"
unit test updated: with framebuffer 312 ≥ NTSC wrap 262 no NTSC scanline is
off-screen via the buffer bound, so it now asserts the *top* display gate —
scanlines < Y_START aren't committed). Remaining PAL-height games
carnival(214)/journey_escape(230)/pooyan(220) still need their own RomSettings
(distinct heights + likely a per-game YStart, since Y_START is still a fixed
const) — follow-up. jaxtari parity (PAL render + #98 dump-pot) deferred to PXC2.

### ✅ Task #110 follow-up (render) (2026-06-15) — per-game YStart closes bucket A (all 64 screen-comparable)

The 3 remaining bucket-A games (carnival/journey_escape/pooyan) have an EXPLICIT
`Display.Height` (and carnival/pooyan an explicit `Display.YStart=26`) in
stella.pro — and they are **NTSC, not PAL** (verified: each game's rendered
content stays within scanline 262 via xitari `--screen` dump, so no 312-wrap /
colour-loss needed; only the crop window differs). The blocker was that jutari's
display-start was a fixed `Y_START=34` const — and it's not merely a crop: it
also gates HMOVE-blank flag consumption to mirror xitari's `myClockStartDisplay =
myClockWhenFrameStarted + 228*myYStart` (TIA.cxx), which IS keyed on the
per-game YStart. So the faithful fix is a per-game YStart.

FIX (jutari, render-only): new `romsettings_y_start` interface (default 34) +
per-TIA `y_start_row` field (set in `env_reset!`); the two framebuffer-commit
gates (`tia.scanline >= Y_START`) and the `get_screen` crop base now use
`tia.y_start_row`. New `CarnivalRomSettings`(YStart=26,H=214) /
`PooyanRomSettings`(YStart=26,H=220); `JourneyEscapeRomSettings` gains H=230
(it already existed for its FIRE start). All 4 jutari tool settings-maps synced.

VERIFIED: all 64 games now **screen-comparable** (zero "PAL not matched" in the
scoreboard). carnival → **4 px** (@ row 27, the YStart edge) and pooyan → **1 px**
(@ row 11) are near-exact. journey_escape → 325 px, but a **structural** delta
(pixel XORs are not bit-0 → not colour-loss; horizontal object-position shifts
e.g. col 65↔115) — a genuine render divergence unrelated to height, filed under
the render long-tail. Screen scoreboard **37/64** pixel-exact (unchanged — no
NTSC regression); RAM sweep **62/64** (carnival/pooyan/journey_escape all
0 b/f — the YStart/height overrides are render-only, RAM untouched); jutari
Pkg.test green. Bucket A (PAL screen height) is now CLOSED.

### ✅ Task #111 (render) (2026-06-15) — HmoveBlanks property: battle_zone 1112→0, ms_pacman 232→0

First structural-diverger fix (user picked battle_zone). battle_zone's diff was
EVERY visible scanline (38-176) blanked black in jutari at cols 0-7 (the 8px
HMOVE-blank "comb" window) while xitari draws content there. ROOT CAUSE: xitari
gates the comb on TWO conditions (TIA.cxx:2694): `myAllowHMOVEBlanks &&
ourHMOVEBlankEnableCycles[x]`. `myAllowHMOVEBlanks` is the per-ROM
`Emulation.HmoveBlanks` property (Props.cxx default "YES"; 43 games set "NO").
**battle_zone + ms_pacman are the only two "NO" games in the 64-ROM set** — so
xitari NEVER arms their comb. jutari had the cycle-table gate
(`_HMOVE_BLANK_ENABLE_CYCLES`, correct) but was MISSING the property gate, so it
armed the comb on every strobe. battle_zone strobes HMOVE every scanline (at
cc 222) → jutari combed every row.

(Probe aside: jutari sees battle_zone's cc=222 / sl_cyc=74 strobe at `beam_sc=3`
of the NEXT line — a ~5-cycle beam-position lead — so even the cycle gate would
mis-fire (x=3→enabled vs xitari x=74→disabled). The property gate makes this moot
for battle_zone/ms_pacman; the beam_sc lead is noted as a separate latent issue
for any "YES" game that strobes near the 74/75 boundary.)

FIX (jutari, render-only): new `romsettings_hmove_blanks` interface (default
true) + per-TIA `allow_hmove_blanks` field (set in `env_reset!`); both HMOVE
comb-arm sites now AND it: `tia.allow_hmove_blanks && _hmove_blank_enabled_at(sc)`.
New `BattleZoneRomSettings` / `MsPacmanRomSettings` (hmove_blanks=false; no
starting actions — both RAM bit-exact). All 4 jutari tool settings-maps synced.

VERIFIED: **battle_zone 1112→0 ✅, ms_pacman 232→0 ✅**. Screen scoreboard
**37→39/64** pixel-exact, ZERO regressions (the gate only ANDs a default-true
flag for the 62 "YES" games; every other nonzero game holds its exact prior px,
pong/breakout still 0). RAM **62/64** unchanged (render-only); jutari Pkg.test
green. NOTE: this is a SHARED structural-diverger fix (2 games at once) — unlike
the heterogeneous up_n_down/pacman/qbert which remain distinct per-game bugs.

### ✅ Task #112 (render) (2026-06-15) — beam_sc lead (no-op); HMOVE comb consumed on first DISPLAY scanline → bowling/kangaroo 0

Two HMOVE-comb findings from chasing the "beam_sc lead" surfaced in #111.

**(1) beam_sc lead — characterized, NO fix (battle_zone-only, already fixed).**
jutari sees battle_zone's `WSYNC;STA HMOVE` strobe at line N+1 cc 0 (`x=0`,
table TRUE) while xitari sees it at line N cc 222 (`sl_cyc 74`, `x=74`, table
FALSE) — a ~6 cc / 2-cycle beam-position lead SPECIFIC to the WSYNC→HMOVE-line-
tail idiom (jutari's beam has wrapped into the next line by the post-WSYNC write).
This is the ONLY victim: kangaroo/centipede/asterix also strobe at sl_cyc 73/74
but NOT post-WSYNC, so jutari attributes them correctly (no spurious comb).
battle_zone is already fixed by #111's `HmoveBlanks=NO` property gate, so there
is no remaining conformance impact — and the underlying WSYNC/beam alignment is
foundational (the 39 px-exact + 62 RAM-exact games depend on it), so it is NOT
worth touching. Logged, not fixed.

**(2) HMOVE-comb "VBLANK-carry" (bowling 8, kangaroo 8) — SOLVED.**
jutari carried `hmove_blank_pending` to the first VISIBLE scanline (the VBLANK
branch never cleared it, #83), wrongly combing its left 8px. The REAL xitari
mechanism (read in full, not the earlier poke-driven guess): `updateFrame`
RETURNS EARLY for `clock < myClockStartDisplay` (TIA.cxx:1708) — it never renders
NOR clears the comb in the PRE-display region (scanline < YStart) — then applies
+ clears `myHMOVEBlankEnabled` on the FIRST DISPLAY scanline (>= YStart), EVEN
IF that scanline is VBLANK (the blank lands on already-black pixels, invisibly,
and the flag clears). So the comb is visible ONLY when the first display scanline
is itself visible:
- **pong**: VBLANK off at sl 27 (< YStart 34) → first display row (sl 34) is
  VISIBLE → row-0 comb shows (correct, #83 preserved).
- **bowling**: VBLANK off at sl 38 (> YStart 34) → first display rows 34-37 are
  VBLANK → comb consumed invisibly at sl 34 → sl 38 shows NO comb.

ROUND 1 (reverted): consumed the comb on EVERY scanline incl. pre-display →
killed pong's legitimate carry (pong row-0 → 8px) + pacman (+8).
ROUND 2 (LANDED): consume the comb in the VBLANK branch too, but GATED on
`scanline >= y_start_row` (the SAME gate the visible branch already uses for its
clear). Pre-display scanlines (< YStart) still skip the clear → the carry to the
first display scanline is preserved; display-region VBLANK scanlines now consume
it like xitari. One line added to `tia_advance!` (TIA.jl VBLANK branch).

VERIFIED: **bowling 8→0 ✅, kangaroo 8→0 ✅**; pong row-0 PRESERVED (0). Bonus:
**berzerk 25→21, elevator_action 24→16** (they had comb px too). Screen
scoreboard **39→41/64** pixel-exact, ZERO regressions (every other game holds
its exact px). RAM **62/64** unchanged (the change only writes the framebuffer
black-row branch + a flag); jutari Pkg.test green. pacman top rows 0-1 are a
SEPARATE "draws-where-blanked" bug (jutari draws 132 where xitari blanks), not
this comb — unchanged (3362).

### ✅ Task #113 (render) (2026-06-15) — per-game YStart for up_n_down/pacman/qbert (the "structural" divergers were a vertical OFFSET)

User asked to tackle up_n_down (10838 px, the worst diverger — hypothesised as a
"rainbow-kernel COLU/sprite-racing" bug). Deep-traced it: the 210/18 blocks are a
playfield (COLUPF=18 on COLUBK=210), the PF registers MATCHED xitari (RAM-exact),
yet jutari's rendered scanline showed the PF pattern of a DIFFERENT scanline —
xitari's display row 50 = the sl-78 PF (block @ x36-51), jutari's = the sl-82 PF
(blocks @ x16-27,x40-55), a ~4-scanline VERTICAL OFFSET. CAUSE: up_n_down's
stella.pro sets `Display.YStart "30"` but jutari rendered from the default 34, so
the whole frame was shifted 4 scanlines and every row compared against the wrong
scanline (the pervasive 203-row "shifted pattern" divergence). NOT a sprite/COLU
bug at all.

A full scan of all 64 sweep ROMs' `Display.YStart` found THREE uncovered games:
up_n_down (30), **pacman (33)**, **qbert (40)** — all defaulting to 34.

FIX (render-only, reuses the #110 `romsettings_y_start` + per-TIA `y_start_row`):
`romsettings_y_start(::UpNDownRomSettings)=30`; new render-only `PacmanRomSettings`
(33) / `QbertRomSettings` (40) (no starting actions — all RAM bit-exact; pacman ≠
ms_pacman; qbert keeps GenericRomSettings' #106 partial-frame behavior). All 4
jutari tool settings-maps synced.

VERIFIED: **pacman 3362→0 ✅** (fully fixed — its maze looked exact because a
vertically-uniform maze hides a 1-row shift; only the top/bottom edges diverged).
**up_n_down 10838→221** (the offset was 98%; the 221 px/frame residual is a
genuine render delta now visible — sprite/PF racing detail, deferred).
**qbert: total 345224→7664** — steady state now EXACT; only frame 2 diverges (the
#106 grey/partial boot frame vs xitari's `greyOutFrame`, a known single-frame boot
artifact). Screen scoreboard **41→42/64** pixel-exact, ZERO regressions (every
other game holds its exact px; breakout/pong/battle_zone 0). RAM **62/64**
unchanged (YStart is render-only); jutari Pkg.test green. LESSON: before chasing a
pervasive "shifted/structural" render divergence as a sub-cycle bug, CHECK the
per-game `Display.YStart` first.

### 🔬 Task #114 (render) (2026-06-15) — qbert frame-2 = boot-transition frame-slice artifact (DIAGNOSED, deferred — no code change)

After #113 (YStart=40), qbert is screen-exact on EVERY frame EXCEPT counted frame
2 (7664 px). Diagnosed precisely; NOT fixed (the fix would destabilize the
RAM-exact #106 partial-frame model for one cosmetic boot frame).

Findings (jutari `run_until_frame!` instrumentation + direct RAM compare):
- jutari counted frame 2 RAM == xitari counted frame 2 RAM **bit-exact**
  (`0094b894…`, 56 nonzero) — perfectly aligned, same frame, same state.
- xitari frame 2 framebuffer = the qbert board (7664 px); jutari = all black.
- jutari's counted frame 2 is a **235-instruction sliver** (~3 scanlines):
  qbert's RESET-boot wait loop (#52) is sliced into 4 grey frames by the #106
  25000-instr budget (beam free-running 201→140→79→18), then frame 60→61
  completes in 3 instrs and 61→62 in 235 — so the board (drawn the same RAM-cycle
  in both) lands in jutari's NEXT frame (63, 5370 instrs, full) while xitari's
  frame 2 already spans the full board render.
- Net: cumulative CPU execution converges (RAM bit-exact every frame), but the
  per-frame VSYNC/grey-frame BOUNDARY lands one boot-transition frame differently,
  so the frame-2 FRAMEBUFFER differs by one frame. Frames 1 and 3-60 are exact.

DECISION: deferred. Matching xitari's exact boot-transition slicing means changing
the #106 grey-frame budget / VSYNC-boundary placement — the very mechanism that
makes qbert RAM bit-exact (62/64) and powers the partial-frame family
(#103/#106). Not worth that risk for a single cosmetic boot frame. qbert's STEADY
STATE is screen-exact (screen scoreboard counts it as 7664 because the sweep diffs
the boot-transition frame). Logged for a future focused #106 frame-slicing pass.

### 🔬 Task #114 follow-up (2026-06-15) — deep Phase-A diagnosis: TIA frame structures MATCH; it's an agent-step↔screen emission alignment (still deferred)

Pursued the #106 frame-slicing fix under an approved diagnostic-first plan. Two
approaches tried and REVERTED; root cause refined; no code shipped (still 42/64
screen / 62/64 RAM, tree clean).

(1) Render-only framebuffer-accumulation (window-clamp commits by
`lines_since_frame` + grey-out) — implemented, **zero effect on qbert**, reverted.
Confirmed qbert counted frame 2 is a COMPLETE short frame (235 instrs), not a
grey/partial frame, so accumulation doesn't apply.

(2) #106 frame-boundary approach — instrumented jutari's `run_until_frame!` +
`tia_poke!` cutoff/VSYNC and read xitari's bus-trace. KEY FINDING: jutari and
xitari have **IDENTICAL TIA frame structure** at the transition —
  frame 1 = spin (3948 scanlines, no VSYNC, ended by the max-scanline CUTOFF at
    the first game poke; `lines_since_frame` accumulates across the boot grey
    frames in both),
  frame 2 = SHORT (~11-12 scanlines, ended by a real held-≥1-scanline VSYNC),
  frame 3+ = full 262-scanline board frames.
So it is NOT a frame-slicing divergence (the earlier "extra short frame"
hypothesis was wrong — both have it). The divergence is purely in the
**agent-step ↔ TIA-frame ↔ screen emission alignment**: the same board-render
lands in xitari's emitted screen 2 but jutari's emitted screen 3, with RAM
bit-exact per counted frame. The bus-trace frame counter (TIA frames) and the
`--screen` frame counter (post-boot agent-steps) are offset, so pinning the exact
emission-mapping difference needs instrumenting xitari's OWN per-update screen
emission — a build that would modify the pristine reference — and any fix still
risks the RAM-exact #106 model.

DECISION: **deferred again** (per the plan's Phase-A gate + fallback). One
cosmetic boot frame is not worth touching the pristine xitari build or risking the
RAM-exact 62/64 backbone. Diagnosis substantially deepened for a future session
that can instrument xitari's screen emission in an out-of-tree scratch build.

### ✅ Task #114 SOLVED (2026-06-15) — jutari was SINGLE-buffered; xitari is DOUBLE-buffered. qbert 7664→0, RAM unchanged.

The user pushed to fix the bit-exactness flaw even at the cost of RAM regression —
and it turned out there was NO cost: the root cause is render-only.

Instrumented xitari's per-act frame structure via a `TIADebug` friend tap in
`tools/trace_dump.cpp` (a TOOL — xitari core stays pristine; fast relink, no lib
rebuild). DECISIVE finding: jutari and xitari have IDENTICAL TIA frame structure
at qbert's boot transition (frame 1 = spin 3948 sl cutoff, frame 2 = SHORT 11-12
sl VSYNC, frame 3+ = board), AND xitari's counted frame 2 IS that 11-scanline
short frame — yet it shows the full board (screen_nz=7664). The board can't be
rendered in 11 scanlines: it comes from xitari's DOUBLE FRAMEBUFFER
(`myCurrentFrameBuffer`/`myPreviousFrameBuffer`, swapped in `startFrame`,
TIA.cxx:537-539, never cleared). The short frame renders ~12 rows into the
swapped-in buffer whose other ~250 rows still hold the boot attract board from
TWO frames earlier → board shows. **jutari had a SINGLE framebuffer**, so the
3948-scanline spin overwrote the board to black → its identical short frame
rendered black. This is a genuine port flaw, NOT a frame-slicing/timing issue (the
two earlier hypotheses were both wrong — see the follow-up above).

FIX (render-only): give jutari a double framebuffer. `TIAState` gains
`framebuffer_prev::Matrix{UInt8}` + `buffer_swap_pending::Bool`. The
`vsync_reset_pending` drain (frame COMPLETION) arms `buffer_swap_pending` (mirror
xitari's startFrame-only-on-completion; grey/partial frames don't arm it).
`run_until_frame!` (Console.jl) swaps `framebuffer ↔ framebuffer_prev` at its
START when armed — AFTER get_screen has read the just-completed frame, BEFORE
rendering the next — so a short/partial frame shows the 2-frames-ago buffer for
un-rendered rows, exactly like xitari. Normal full frames overwrite the whole
display window, so they're unchanged → no regression.

VERIFIED: **qbert 7664→0 ✅**. Screen scoreboard **42→43/64** pixel-exact, ZERO
regressions (every other game holds its exact px; pong/breakout/battle_zone/pacman
/up_n_down unchanged). RAM **62/64 BYTE-IDENTICAL** (render-only — the authorized
RAM regression was not needed). jutari Pkg.test green. LESSON: xitari/Stella is
double-buffered with swap-not-clear; a single-buffer port silently diverges on any
partial/short frame that relies on preserved content. The "keep xitari pristine /
don't regress RAM" caution was right in spirit but the user was right that
bit-exactness comes first — and here the faithful fix cost nothing.

### ✅ skiing RAM bit-exact — VSYNC hold-gate not rebased across frame reset (2026-06-15)

User: "Let's work on the remaining RAM problems." Re-opened the two RAM residuals
(skiing 1 b/f, surround 7 b/f) that #103 had deferred as "construction-counter
non-bugs." skiing turned out to be a GENUINE jutari frame-slicing bug, NOT a
construction counter (the #103 triage conflated it with surround).

SYMPTOM: skiing `RAM[$00]` (a free-running per-frame counter, the game re-inits it
to 0 at boot so the construction probe is irrelevant) was constant **+1** vs
xitari at every frame (xitari=$50, jutari=$51 over 30 frames). xitari boot_end
$00=79, jutari=80.

ROOT CAUSE (per-frame trace of $00 + VSYNC events through the boot): jutari
double-increments $00 on boot frame 2 (0→2 instead of 0→1) — one
`run_until_frame!` call swallowed TWO game-frames. Mechanism:
1. The long boot frame ends via the max-scanlines cutoff at lsf=291. In the SAME
   instruction the game STROBES VSYNC, arming `vsync_finish_clock` (vfc) =
   frame_clock+228 = 66585 with lsf still 291.
2. The `vsync_reset_pending` drain resets `lines_since_frame` → 0 (beam color_clock
   preserved) but LEFT vfc=66585 — now stale: jutari's `frame_clock =
   lines_since_frame*228 + beam_cc` is FRAME-RELATIVE, so resetting lsf shifted the
   whole frame_clock domain DOWN by 291*228 while vfc stayed in the old domain.
3. Next frame: the game's legitimate VSYNC-clear at lsf=3 (frame_clock=693) is
   tested `693 >= vfc=66585` → REJECTED (gate thinks VSYNC wasn't held ≥1 line),
   so the frame doesn't end. The game runs a 2nd full frame (INC $00 again),
   re-arms vfc=59973, clears at lsf=265 → ends. Two frames → one call → +1 $00.

xitari never hits this: `myVSYNCFinishClock` and the frame clock are BOTH in the
ABSOLUTE `mySystem->cycles()*3` domain — `systemCyclesReset` decrements them
together (TIA.cxx:274-278) and `startFrame` never invalidates an armed gate.

FIX (TIA.jl `tia_advance!` drain): when the frame resets, rebase an armed gate
into the new domain — `vfc -= lines_since_frame * 228` (only if vfc != typemax).
This mirrors xitari's absolute-clock semantics in jutari's frame-relative scheme.
NO-OP for normal frames: both end paths (VSYNC-clear, max-scanlines cutoff) disarm
vfc to typemax BEFORE the drain, so this only ever rebases a gate re-armed during
the very instruction that ended a frame (the boot cutoff→VSYNC transition).

VERIFIED: skiing $00 now 79→80→81→… exactly matching xitari; skiing RAM
**1 b/f → 0 (BIT-EXACT, 30 frames)**. Full 64-ROM RAM sweep **62→63/64**, ZERO
regressions (every previously-bit-exact game holds). jutari Pkg.test green.

surround (7 b/f) was UNCHANGED by this fix — confirmed it is a DIFFERENT mechanism:
`RAM[$7d]` is a TRUE free-running counter from power-on (the game never re-inits
it), so it carries xitari's construction-probe seed (Console.cxx:199, 60 frames +
PAL boot). xitari $7d=$a1=161; jutari (no probe)=$42=66. The #103 N-sweep proved
surround's emulation is otherwise bit-exact (N=190 jutari probe frames → maxdiff
0). A principled fix needs jutari to replicate xitari's construction probe +
keep-RAM reset — bigger and riskier (touches every game's boot); left as the one
remaining RAM residual at 63/64.

### 🔬 surround RAM residual — construction-probe seed CHARACTERIZED, fix not viable (2026-06-15)

Follow-up to the skiing fix. surround's 7 b/f all reduce to ONE byte: RAM[$7d], a
free-running counter the game NEVER re-inits, so it carries xitari's
construction-time format-autodetect probe seed (Console.cxx:197-218: 60
`mediaSource().update()` at the TIA-ctor-default 262 cutoff, then a keep-RAM
`mySystem->reset()`). jutari skipped the probe AND zeroes RAM every reset → $7d
starts 0.

DECISIVE measurement (TRACE_PROBE_DEBUG friend-tap in tools/trace_dump.cpp — a
TOOL; xitari core stays pristine — reading getRAM()[$7d] at construction
checkpoints):
- xitari POST-CONSTRUCT (after the 60-frame probe): $7d = **95**
- xitari POST-RESETGAME (after the 60+4+SELECT/RESET boot): $7d = **160** (→161 f0)
So probe contributes 95, boot +65. (This also corrects the #103 entry, which
compared jutari's probe-ONLY $7d=96 against xitari's boot-END 161 and wrongly
concluded "65 short" — the 65 is just the boot jutari ALSO runs after the probe.)

ATTEMPT (reverted): added a `construction_probe` to env_reset! (probe 60 frames
@ 262 + keep-RAM reset via a new `console_reset!(keep_ram=true)`). Result: jutari
post-probe $7d = **30**, not 95 — and surround boot-end only moved 66→96 (still
7 b/f). The gap is ENTIRELY in the probe: jutari's `run_until_frame!` advances
$7d by 30 over 60 probe calls; xitari's `update()` advances it by 95 — ~3× more
game-loops per probe frame. surround is a 312-line PAL game running the probe at
the 262 cutoff (format not yet detected); jutari slices/runs its PAL attract loop
very differently from xitari's `m6502().execute(25000)`-per-update() there. A
probe-config sweep (maxsl ∈ {262,290,342}, spf ∈ {262,312,524}) peaks at 59
(@342), never 95. (The skiing vfc-rebase fix did not help — different mechanism.)

DECISION: REVERTED the construction-probe (it does NOT reach 95 and adds risk to
the 63 bit-exact games for nothing). surround stays the SOLE RAM residual at
63/64. It is a known-benign, non-growing, 1-counter divergence on emulation the
#103 N-sweep PROVED otherwise bit-exact (N=190 probe frames → maxdiff 0). A real
fix needs jutari's PAL attract-loop probe-frame slicing/instruction-budget to
match xitari's update() exactly — deep, PAL-specific, high regression risk; not
worth it for 7 non-growing bytes. Left precisely characterized for a future pass.

### ✅✅ surround RAM BIT-EXACT — xitari's probe + DOUBLE-boot, the seed was a red herring (2026-06-15)

CORRECTS the "fix not viable" entry above. User: "investigate the game-loop
difference and make a plan how to fix this difference. Taking a risk is ok,
because we have version control." The investigation overturned the earlier
conclusion entirely — surround is now BIT-EXACT (64/64 RAM).

The "game-loop difference" (jutari probe $7d=30 vs xitari 95) was a RED HERRING.
Friend-tap instrumentation of the construction probe loop (temporary fprintf in
xitari Console.cxx:201, reverted after; libxitari rebuilt) showed xitari's 60-frame
probe loop ends at $7d=**30 — IDENTICAL to jutari**. jutari's probe was never wrong.

The real cause is STRUCTURAL: xitari runs the boot (`StellaEnvironment::reset`)
TWICE before episode 1's first action:
  1. Console ctor: 60-frame format probe @ 262 cutoff + keep-RAM `mySystem->reset()`  → $7d 0→30
  2. `loadROM` → `reset_game`  (boot #1)                                              → $7d 30→95
  3. explicit `ALEInterface::resetGame`  (boot #2)                                    → $7d 95→160
(measured via getRAM()[$7d] at each checkpoint: 30 / 95 / 160.) jutari ran only
boot #2 from a cold, RAM-zeroed start → $7d=66, a 94-short seed for the one byte
surround never re-inits. Every other game re-inits its RAM at boot, so the missing
probe + boot washed out — which is exactly why 62→63/64 were already bit-exact and
the divergence hid in surround alone.

FIX (StellaEnvironment.jl): factor the boot body into `_boot_burn!`; `env_reset!`
now mirrors xitari's full construction sequence when `construction_probe=true`
(default): cold reset → 60-frame probe @ 262 → keep-RAM reset → `_boot_burn!`
(boot #1) → keep-RAM reset → `_boot_burn!` (boot #2). `console_reset!` gained a
`keep_ram` flag (xitari's M6532 zeroes RAM only in its ctor, not on reset). Each
keep-RAM reset still resets the TIA to fresh (zeroed framebuffer + counters), so
boot #2 renders from the same clean state as the old single boot — no render change.

VERIFIED bit-exact at every stage (jutari $7d: 30 → 95 → 160, == xitari). Full
64-ROM RAM sweep **63→64/64 BIT-EXACT** (surround 7→0, skiing still 0, all 62
others hold). jutari Pkg.test green. Screen sweep **43→44/64 pixel-exact** —
surround's 224px screen residual ALSO collapsed to 0 (its render divergence was
the same boot-seed all along; once RAM is bit-exact the screen is too), the ONLY
row that changed (zero render regressions). The earlier "deep, PAL-specific, not
worth it" assessment was WRONG — the bit-exactness-over-caution lesson again: a
"deferred" residual was a real, fully-fixable structural flaw once the friend-tap
pinned the true mechanism. **jutari RAM is now 64/64 BIT-EXACT vs xitari.**

### 🔬 render harness (tools/render_diff) + tutankham root-cause: CPU↔TIA sub-instruction phase (2026-06-16)

Built the per-scanline xitari-vs-jutari render diff harness (tools/render_diff.jl
+ .py, + a zero-cost `_RENDER_PROBE` hook in TIA.jl) to replace error-prone inline
probes. It does the screen-row→TIA-scanline mapping from the env's real
`y_start_row` (the by-hand version is what derailed earlier attempts) and prints
both rendered rows, jutari's full per-scanline render state (decoded regs + object
sets + per-poke pending-write ACTIVATION clocks), and xitari's bus-trace pokes
(with the PF `cc+delay` activation), side by side.

FIRST USE pinned tutankham's 80px band precisely. Row 103 (TIA sl 137) is
racing-the-beam PLAYFIELD: mid-scanline pokes CTRLPF@cc141, PF0@cc162, PF1@cc192,
PF2@cc222. xitari's PF delay (d={4,5,2,3}[(x/3)&3], TIA.cxx:1994) activates them at
164/196/224; jutari activates at 176/204/236 — **+8..12 color clocks late** → the
PF changes land ~3 columns right of xitari and cascade across scanlines.

ROOT CAUSE (poke-probe breakdown): the delay FORMULA is correct (jutari's
`_pf_dynamic_delay` == xitari's d). The error is the write's `beam_cc`: jutari's
instruction-start `color_clock` = 162 already EQUALS xitari's write cc (162), and
jutari then adds `extra_cpu_cycles*3` (=9) → beam_cc=171, so the activation is
computed from 171 not 162. i.e. jutari's TIA beam is ~2-3 CPU cycles AHEAD of the
CPU vs xitari for this instruction stream — a CPU↔TIA sub-instruction PHASE error.

KEY: this phase error is INVISIBLE to the RAM sweep (RAM is CPU-only; the beam
phase only manifests in racing-the-beam render), which is exactly why all 64 RAM
games are bit-exact while the render long-tail (tutankham, and likely the other
mid-scanline PF/sprite divergences) persists. The fix is deep per-cycle CPU↔TIA
cycle-threading (the #58/#63 class) — shared logic that breakout/pong's
pixel-exact RESP*/HMOVE/PF depend on, so it MUST be gated on the full screen+RAM
sweep (those are the canaries). Not attempted blind; left precisely targeted with
the harness as the tool to verify any fix.

### ⚠️ CORRECTION (2026-06-16): jutari's PF/deferred-write activation timing is CORRECT — there is NO CPU↔TIA phase bug

Direct xitari instrumentation (temporary fprintf in TIA::poke's PF-delay branch,
TIA.cxx:1994, printing the TRUE frame-relative `x = (clock - myClockWhenFrameStarted)
%228`, delay, and activation; reverted + libxitari rebuilt) overturns the prior
"jutari activates PF +12cc late" diagnosis. On tutankham sl137 xitari's REAL
activations are PF0→**176**, PF1→**204** — **identical to jutari's** (176/204).

The earlier "+12" was a measurement ARTIFACT: the `--bus-trace` color_clock is
`cpu_cycles*3 % 228`, which is offset from xitari's true frame-relative beam
position `(clock - myClockWhenFrameStarted) % 228` by the per-frame `startFrame`
carry (~+9 on this frame; xitari's bus-trace cc=162 ⇒ true x=171 ⇒ activate 176).
**Lesson for the harness: the `--bus-trace` cc/scanline are CPU-cycle-derived and
are offset from xitari's TRUE rendered beam position by the startFrame carry — do
NOT use bus-trace cc for precise PF/sprite render-timing; use xitari's frame-
relative `x` (XI_PF_DBG-style instrumentation) or jutari's own pending activations
(which ARE correct).** jutari's pending activations matched xitari exactly here.

So tutankham's 80px band is NOT a per-poke timing bug. With PF0@176/PF1@204
matching and PF2's write wrapping past the scanline end in BOTH (activate 236 →
applies to the next scanline), the real divergence is the OLD PF2 carried into
sl137 (the cols-96-111 band = right-half reflected PF2 bits 0-3, rendered at
cc164-179, BEFORE PF0@176) — i.e. a cross-scanline racing-the-beam PF-register
carry difference, to be chased with the harness across consecutive scanlines.
Net: do NOT touch the (correct) beam_cc / deferred-write timing.

### 🔬 tutankham root-cause via completed harness (XI_POKE_DUMP) — WSYNC/boundary beam phase (2026-06-16)

Completed the render harness: added an env-gated `XI_POKE_DUMP` to xitari
TIA::poke (local diagnostic in the git-excluded xitari, like the existing
System bus-trace hook) that prints every poke's TRUE frame-relative activation
(`(clock - myClockWhenFrameStarted)` + delay, wrapped) — the reliable PF/sprite
timing reference the `--bus-trace` cc could NOT give (it's cpu_cycles-derived,
offset by the startFrame carry). tools/render_diff.py now prints xitari's true
activations beside jutari's pending activations. (Harness degrades gracefully if
the xitari dump isn't built.)

This pinned tutankham's 80px band DEFINITIVELY. On sl137 the deferred-write
DELAYS match (CTRLPF→150, PF0→176, PF1→204 identical in both). The divergence is
the PF2=0 write: **xitari activates it at frame-relative x=3 — the START of sl137
(post-WSYNC) — so the whole visible scanline has PF2=0 → cols 96-111 (reflected
PF2 bits 0-3) render background. jutari activates the SAME write at cc236 (END of
sl137), so cols 96-111 keep the carried PF2=0x0f → playfield.** A ~one-scanline
(228cc) phase gap (jutari beam_cc 231 vs xitari x=3=231%228) for a TIA write that
lands right at a WSYNC/scanline boundary. So it is NOT the (verified-correct)
deferred-write delay — it's how jutari's `beam_cc` maps a post-WSYNC / boundary
write to the new scanline's start. Shared WSYNC/beam logic (breakout/pong are
pixel-exact through WSYNC), so any fix must be gated on the full screen+RAM sweep.
Harness is now the verifier for it.

---

## #115 (2026-06-16) — RESMP per-color-clock deferral → ice_hockey SCREEN exact (44→45/64)

**SOLVED.** ice_hockey rendered 5px/frame wrong (first @ frame 1, rows 87-103):
xitari draws a 4px missile-0 (NUSIZ0=0x20 → 4-clock width, COLUP0=0x02) at
cols 72-75; jutari showed background there. The harness color-attribution +
XI_POKE_DUMP pinned it: xitari's `RESMP0=0x02` store activates at frame-relative
**cc=210** (poke-delay table entry 0x28 = 0), but the missile is painted at
**cc=140** (col 72). xitari's `TIA::poke` runs `updateFrame(clock+0)` (renders up
to the beam) THEN re-gates `myEnabledObjects` on `myENAM0 && !myRESMP0` — so the
missile painted left of the RESMP write stays visible; only pixels right of cc210
are suppressed. jutari applied RESMP **immediately at poke time**, so
`_missile_set`'s `(RESMP & 0x02)` gate suppressed the missile for the WHOLE
scanline.

**Fix:** defer RESMP0/RESMP1 to their activation color clock exactly like
ENAM/GRP (TIA.jl `tia_poke!` deferred-write set + `_apply_pending_write!`). The
render-loop drain recomputes `cached_sets` after each pending write, so the
suppress re-gate now threads per-color-clock. The 1→0 (lock→release) reposition
(`_resmp_reposition!`, #103 frostbite) moves into the activation-time apply; the
HBLANK path (`beam_cc < 68`) still applies immediately, preserving frostbite.

RESMP touches only TIA render state (missile enable + position), never RAM
directly — so the risk was the narrow collision-read-to-RAM path. **Gated:** RAM
**64/64 bit-exact** (zero change), screen **44→45/64** (ice_hockey 5→0; all 44
prior exacts unchanged), jutari Pkg.test green (frostbite/pong-score-stripe RESMP
cases included). asterix's 1px missile is NOT this — it's a carried-over ENAM0
state with no RESMP/ENAM poke on the divergent scanline (separate bug).

Found via the render-divergence diagnosis workflow (20 parallel color-attribution
agents). The same run surfaced the next high-confidence target: **air_raid** is an
INPT0 default-bit bug (jutari defaults analog pots INPT0-3 to D7=1; xitari joystick
controllers drive them D7=0 — air_raid's kernel reads INPT0 via `LDA $58` into
COLUP0/P1, so jutari paints player pixels 0x98 vs xitari 0x18, 24px/frame).

---

## #115b (2026-06-16) — joystick INPT0-3 idle LOW → air_raid 24→2px

air_raid rendered 24px/frame wrong (rows 219-223, all frames). The
color-attribution workflow (HIGH confidence) pinned it: jutari defaulted the
analog pot pins INPT0-3 to D7=1 ($80), but xitari's `Joystick::read(AnalogPin)`
returns `maximumResistance` → `TIA::INPT0_3` yields D7=0 (TIA.cxx:1877-1885).
air_raid's bottom-band kernel does `LDA $58` (INPT0 via TIA mirror) → `STA
COLUP0/COLUP1`, so jutari's D7=1 painted the player pixels 0x98 vs xitari 0x18.

**Fix (controller-aware):** keep the raw TIA default at $80 (paddle-idle), but for
joystick games (`!romsettings_uses_paddles`) override INPT0-3 to $00 at each
`_boot_burn!` (StellaEnvironment.jl). Paddle games (pong/breakout) keep $80 + the
dump-pot model — a first attempt at a *global* $00 default regressed pong's PXC1
trace (paddles idle HIGH), so the override must be controller-scoped.

**Gated:** RAM 64/64 bit-exact (zero change), jutari Pkg.test green (incl. pong
PXC1 + the INPT-defaults testset). Screen: air_raid 24→2px (rows 219-222 now
exact); the residual **2px at row 223** is a separate bottom-band sub-issue (color
0x19 vs 0xe1, not a D7 bit) — air_raid still counts as non-exact (screen stays
45/64), TODO. No other game changed.
